#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Inference Benchmark Script for LSTM vs Transformer Decoders
Evaluates inference performance with various optimization techniques.
Scientific evaluation focused on single-sequence inference (batch_size=1).
"""

import sys
import time
import statistics
import argparse
from pathlib import Path
import torch
import torch.nn as nn
from torch.amp import autocast
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any
from contextlib import contextmanager
import warnings
import psutil
import gc

# Add project root to path for imports
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.feature_config import (
    DETECTION_INPUT_DIM_PER_BOX,
    MAX_DETECTIONS_PER_FRAME,
    SEQUENCE_LENGTH,
    TELEMETRY_INPUT_DIM,
    PREDICTION_TASKS,
)
from src.model.factory import create_model_variant


def setup_device():
    """Simplified device setup without problematic imports."""
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print("🚀 Using CUDA device")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
        print("🍎 Using MPS device")
    else:
        device = torch.device("cpu")
        print("💻 Using CPU device")
    return device


# Suppress specific warnings for cleaner output
warnings.filterwarnings("ignore", category=torch.jit.TracerWarning)
warnings.filterwarnings("ignore", category=UserWarning, module="torch._functorch")


class MemoryProfiler:
    """Memory usage profiling for GPU/MPS and CPU during model inference."""

    def __init__(self, device: torch.device):
        self.device = device
        self.is_cuda = device.type == "cuda"
        self.is_mps = device.type == "mps"
        self.device_name = self._get_device_name()

    def _get_device_name(self) -> str:
        """Get human-readable device name."""
        if self.is_cuda:
            return "GPU (CUDA)"
        elif self.is_mps:
            return "GPU (MPS)"
        else:
            return "CPU"

    def get_memory_snapshot(self) -> Dict[str, float]:
        """Capture current memory usage state."""
        snapshot = {}

        # Device Memory (GPU/MPS)
        if self.is_cuda:
            snapshot["device_allocated_mb"] = (
                torch.cuda.memory_allocated(self.device) / 1024**2
            )
            snapshot["device_cached_mb"] = (
                torch.cuda.memory_reserved(self.device) / 1024**2
            )
        elif self.is_mps:
            try:
                snapshot["device_allocated_mb"] = (
                    torch.mps.current_allocated_memory() / 1024**2
                )
                snapshot["device_cached_mb"] = (
                    torch.mps.current_allocated_memory()
                    / 1024**2  # MPS doesn't separate cached
                )
            except AttributeError:
                # Fallback for older PyTorch versions
                snapshot["device_allocated_mb"] = 0.0
                snapshot["device_cached_mb"] = 0.0
        else:
            # For CPU, we track process memory as "device memory"
            process = psutil.Process()
            memory_info = process.memory_info()
            snapshot["device_allocated_mb"] = memory_info.rss / 1024**2
            snapshot["device_cached_mb"] = memory_info.vms / 1024**2

        # CPU Memory (always track for comparison)
        process = psutil.Process()
        memory_info = process.memory_info()
        snapshot["cpu_rss_mb"] = memory_info.rss / 1024**2

        return snapshot

    def reset_peak_memory(self):
        """Reset peak memory statistics for clean measurement."""
        if self.is_cuda:
            torch.cuda.reset_peak_memory_stats(self.device)
        elif self.is_mps:
            try:
                torch.mps.empty_cache()
            except AttributeError:
                pass  # Not available in all PyTorch versions
        gc.collect()

    def get_device_name(self) -> str:
        """Return device name for labels."""
        return self.device_name


class InferenceBenchmark:
    """Scientific inference benchmark for multimodal architectures."""

    def __init__(self, device: str = "auto", output_dir: str = "evaluation/benchmark"):
        self.device = setup_device() if device == "auto" else torch.device(device)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Simplified benchmark configuration - focus on single sequence inference
        self.batch_size = 1  # Fixed for real-time inference scenarios
        self.warmup_runs = 20
        self.benchmark_runs = 100
        self.optimization_modes = [
            "baseline",
            "compile",
            "mixed_precision",
        ]
        self.profiler = MemoryProfiler(self.device)

        print(f"🔧 Benchmark initialized on device: {self.device}")
        print(f"📁 Output directory: {self.output_dir}")
        print(f"📦 Batch size: {self.batch_size} (single sequence inference)")

    def create_test_models(self) -> Dict[str, nn.Module]:
        """Create LSTM and Transformer decoder variants for benchmarking."""
        base_config = {
            "telemetry_input_dim": TELEMETRY_INPUT_DIM,
            "detection_input_dim_per_box": DETECTION_INPUT_DIM_PER_BOX,
            "embedding_dim": 64,
            "hidden_dim": 128,
            "attention_num_heads": 8,
            "decoder_num_layers": 4,
            "dropout_prob": 0.0,  # Disable dropout for consistent inference timing
            "prediction_tasks": PREDICTION_TASKS,
            "max_detections": MAX_DETECTIONS_PER_FRAME,
            "max_seq_length": SEQUENCE_LENGTH,
        }

        models = {}

        # LSTM variant
        lstm_config = base_config.copy()
        lstm_config.update(
            {"encoder_type": "simple", "fusion_type": "concat", "decoder_type": "lstm"}
        )
        models["LSTM"] = create_model_variant(lstm_config)

        # Transformer variant
        transformer_config = base_config.copy()
        transformer_config.update(
            {
                "encoder_type": "simple",
                "fusion_type": "concat",
                "decoder_type": "transformer",
            }
        )
        models["Transformer"] = create_model_variant(transformer_config)

        # Move to device and set to eval mode
        for name, model in models.items():
            model = model.to(self.device)
            model.eval()
            models[name] = model

            # Count parameters
            total_params = sum(p.numel() for p in model.parameters())
            decoder_params = sum(p.numel() for p in model.output_decoder.parameters())
            print(
                f"📊 {name}: {total_params:,} total params, {decoder_params:,} decoder params"
            )

        return models

    def apply_optimizations(self, model: nn.Module, optimization: str) -> nn.Module:
        """Apply specified optimization to model."""
        if optimization == "baseline":
            return model
        elif optimization == "compile":
            try:
                return torch.compile(model, mode="reduce-overhead", fullgraph=False)
            except Exception as e:
                print(f"⚠️ torch.compile failed: {e}, falling back to baseline")
                return model
        elif optimization in ["mixed_precision"]:
            return model
        else:
            raise ValueError(f"Unknown optimization: {optimization}")

    def generate_test_data(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Generate synthetic test data for single sequence benchmarking."""
        telemetry = torch.randn(
            self.batch_size,
            SEQUENCE_LENGTH,
            TELEMETRY_INPUT_DIM,
            device=self.device,
            dtype=torch.float32,
        )

        detections = torch.randn(
            self.batch_size,
            SEQUENCE_LENGTH,
            MAX_DETECTIONS_PER_FRAME,
            DETECTION_INPUT_DIM_PER_BOX,
            device=self.device,
            dtype=torch.float32,
        )

        # Create realistic detection mask (some objects per frame)
        mask = torch.zeros(
            self.batch_size,
            SEQUENCE_LENGTH,
            MAX_DETECTIONS_PER_FRAME,
            device=self.device,
            dtype=torch.bool,
        )

        # Add some valid detections per frame (realistic scenario)
        for s in range(SEQUENCE_LENGTH):
            num_objects = torch.randint(2, 6, (1,)).item()  # 2-5 objects per frame
            mask[0, s, :num_objects] = True

        return telemetry, detections, mask

    @contextmanager
    def timer(self):
        """Context manager for precise timing."""
        if self.device.type == "cuda":
            torch.cuda.synchronize()

        start_time = time.perf_counter()
        yield

        if self.device.type == "cuda":
            torch.cuda.synchronize()

        end_time = time.perf_counter()
        self.last_duration = end_time - start_time

    def benchmark_single_configuration(
        self, model: nn.Module, optimization: str, verbose: bool = False
    ) -> Dict[str, float]:
        """Benchmark a single model configuration."""

        # Apply optimizations
        optimized_model = self.apply_optimizations(model, optimization)

        # Generate test data
        telemetry, detections, mask = self.generate_test_data()

        # Enable mixed precision if required
        use_mixed_precision = "mixed" in optimization

        # Reset memory tracking
        self.profiler.reset_peak_memory()
        baseline_memory = self.profiler.get_memory_snapshot()

        # Warmup runs
        if verbose:
            print(f"   🔥 Warmup ({self.warmup_runs} runs)...")

        for _ in range(self.warmup_runs):
            with torch.no_grad():
                if use_mixed_precision:
                    with autocast(device_type=self.device.type):
                        _ = optimized_model(telemetry, detections, mask)
                else:
                    _ = optimized_model(telemetry, detections, mask)

        # Memory snapshot after warmup
        post_warmup_memory = self.profiler.get_memory_snapshot()

        # Benchmark runs
        if verbose:
            print(f"   ⏱️ Benchmarking ({self.benchmark_runs} runs)...")

        inference_times = []

        for _ in range(self.benchmark_runs):
            with torch.no_grad():
                if use_mixed_precision:
                    with autocast(device_type=self.device.type):
                        with self.timer():
                            _ = optimized_model(telemetry, detections, mask)
                else:
                    with self.timer():
                        _ = optimized_model(telemetry, detections, mask)

                inference_times.append(self.last_duration * 1000)  # Convert to ms

        # Calculate statistics
        stats = {
            "mean_ms": statistics.mean(inference_times),
            "median_ms": statistics.median(inference_times),
            "std_ms": (
                statistics.stdev(inference_times) if len(inference_times) > 1 else 0.0
            ),
            "min_ms": min(inference_times),
            "max_ms": max(inference_times),
            "p95_ms": np.percentile(inference_times, 95),
            "p99_ms": np.percentile(inference_times, 99),
            "device_memory_mb": post_warmup_memory["device_allocated_mb"],
            "cpu_memory_mb": post_warmup_memory["cpu_rss_mb"],
        }

        return stats

    def run_comprehensive_benchmark(self) -> pd.DataFrame:
        """Run comprehensive benchmark across all configurations."""
        print("🚀 Starting comprehensive inference benchmark...")

        # Create models
        models = self.create_test_models()

        # Initialize results storage
        results = []

        total_configs = len(models) * len(self.optimization_modes)
        current_config = 0

        for model_name, model in models.items():
            print(f"\n📊 Benchmarking {model_name} decoder...")

            for optimization in self.optimization_modes:
                current_config += 1
                print(
                    f"⚙️ Optimization: {optimization} ({current_config}/{total_configs})"
                )

                try:
                    stats = self.benchmark_single_configuration(
                        model, optimization, verbose=True
                    )

                    # Store results
                    result = {
                        "model": model_name,
                        "optimization": optimization,
                        **stats,
                    }
                    results.append(result)

                    print(
                        f"   ✅ Mean: {stats['mean_ms']:.2f}ms ± {stats['std_ms']:.2f}ms"
                    )

                except Exception as e:
                    print(f"   ❌ Failed: {e}")

        # Convert to DataFrame
        df = pd.DataFrame(results)

        # Save raw results
        csv_path = self.output_dir / "benchmark_results.csv"
        df.to_csv(csv_path, index=False)
        print(f"💾 Raw results saved to: {csv_path}")

        return df

    def create_scientific_visualizations(self, df: pd.DataFrame):
        """Create clean scientific visualizations."""
        print("📊 Creating scientific visualizations...")

        # Set scientific style
        plt.style.use("default")

        # Scientific color scheme (plasma-like purple-yellow)
        colors = ["#440154", "#31688e", "#35b779", "#fde725"]  # Viridis palette

        # Filter valid results
        valid_df = df.dropna(subset=["mean_ms"])

        if valid_df.empty:
            print("❌ No valid data for visualization")
            return

        # Create figure with 2x2 subplots
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))
        fig.suptitle(
            "Inference Performance Analysis: LSTM vs Transformer",
            fontsize=14,
            fontweight="bold",
        )

        # 1. Mean inference time comparison
        models = valid_df["model"].unique()
        optimizations = valid_df["optimization"].unique()

        x = np.arange(len(optimizations))
        width = 0.35

        for i, model in enumerate(models):
            model_data = valid_df[valid_df["model"] == model]
            means = [
                (
                    model_data[model_data["optimization"] == opt]["mean_ms"].iloc[0]
                    if not model_data[model_data["optimization"] == opt].empty
                    else 0
                )
                for opt in optimizations
            ]
            stds = [
                (
                    model_data[model_data["optimization"] == opt]["std_ms"].iloc[0]
                    if not model_data[model_data["optimization"] == opt].empty
                    else 0
                )
                for opt in optimizations
            ]

            ax1.bar(
                x + i * width,
                means,
                width,
                label=model,
                color=colors[i],
                alpha=0.8,
                yerr=stds,
                capsize=3,
            )

        ax1.set_xlabel("Optimization")
        ax1.set_ylabel("Inference Time (ms)")
        ax1.set_ylim(0, valid_df["mean_ms"].max() * 1.2)
        ax1.set_title("Mean Inference Time by Optimization")
        ax1.set_xticks(x + width / 2)
        ax1.set_xticklabels(optimizations, rotation=45, ha="right")
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # 2. Memory usage comparison
        device_name = self.profiler.get_device_name()
        for i, model in enumerate(models):
            model_data = valid_df[valid_df["model"] == model]
            device_memory = [
                (
                    model_data[model_data["optimization"] == opt][
                        "device_memory_mb"
                    ].iloc[0]
                    if not model_data[model_data["optimization"] == opt].empty
                    else 0
                )
                for opt in optimizations
            ]

            ax2.bar(
                x + i * width,
                device_memory,
                width,
                label=model,
                color=colors[i],
                alpha=0.8,
            )

        ax2.set_xlabel("Optimization")
        ax2.set_ylabel(f"Memory (MB, {device_name})")
        ax2.set_ylim(0, valid_df["device_memory_mb"].max() * 1.2)
        ax2.set_title(f"{device_name} Memory Usage")
        ax2.set_xticks(x + width / 2)
        ax2.set_xticklabels(optimizations, rotation=45, ha="right")
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        # 3. Performance distribution (violin plot)
        performance_data = []
        labels = []
        violin_colors = [
            "#440154",  # Dark purple
            "#5D1A86",  # Medium purple
            "#7B2382",  # Light purple
            "#1F4E79",  # Dark blue
            "#2E5F8A",  # Medium blue
            "#3E709B",  # Light blue
        ]
        for model in models:
            for opt in [
                "baseline",
                "compile",
                "mixed_precision",
            ]:  # Compare baseline vs best
                subset = valid_df[
                    (valid_df["model"] == model) & (valid_df["optimization"] == opt)
                ]
                if not subset.empty:
                    # Create synthetic distribution from mean and std for visualization
                    mean_val = subset["mean_ms"].iloc[0]
                    std_val = subset["std_ms"].iloc[0]
                    # Generate points for violin plot
                    points = np.random.normal(mean_val, std_val, 100)
                    performance_data.append(points)
                    labels.append(f"{model}\n{opt}")

        if performance_data:
            parts = ax3.violinplot(
                performance_data,
                positions=range(len(performance_data)),
                showmeans=True,
                showmedians=True,
            )

            # Color the violins
            for i, pc in enumerate(parts["bodies"]):
                pc.set_facecolor(violin_colors[i % len(violin_colors)])
                pc.set_alpha(0.7)

            ax3.set_xticks(range(len(labels)))
            ax3.set_xticklabels(labels, rotation=45, ha="right")
            ax3.set_ylim(0, valid_df["mean_ms"].max() * 1.2)
            ax3.set_ylabel("Inference Time (ms)")
            ax3.set_title("Performance Distribution")
            ax3.grid(True, alpha=0.3)

        # 4. Optimization effectiveness (speedup factors)
        baseline_data = valid_df[valid_df["optimization"] == "baseline"]

        speedup_data = []
        speedup_labels = []

        for model in models:
            model_baseline = baseline_data[baseline_data["model"] == model]["mean_ms"]
            if not model_baseline.empty:
                baseline_time = model_baseline.iloc[0]

                for opt in ["compile", "mixed_precision"]:
                    opt_data = valid_df[
                        (valid_df["model"] == model) & (valid_df["optimization"] == opt)
                    ]
                    if not opt_data.empty:
                        opt_time = opt_data["mean_ms"].iloc[0]
                        speedup = baseline_time / opt_time
                        speedup_data.append(speedup)
                        speedup_labels.append(f"{model}\n{opt}")

        if speedup_data:
            bars = ax4.bar(
                range(len(speedup_data)),
                speedup_data,
                color=[colors[i % len(colors)] for i in range(len(speedup_data))],
                alpha=0.8,
            )

            ax4.axhline(y=1.0, color="red", linestyle="--", alpha=0.7, label="Baseline")
            ax4.set_xticks(range(len(speedup_labels)))
            ax4.set_xticklabels(speedup_labels, rotation=45, ha="right")
            ax4.set_ylabel("Speedup Factor")
            ax4.set_ylim(0, max(speedup_data) * 1.2)
            ax4.set_title("Optimization Effectiveness")
            ax4.legend()
            ax4.grid(True, alpha=0.3)

            # Add value labels on bars
            for i, (bar, value) in enumerate(zip(bars, speedup_data)):
                height = bar.get_height()
                ax4.text(
                    bar.get_x() + bar.get_width() / 2.0,
                    height + 0.01,
                    f"{value:.2f}x",
                    ha="center",
                    va="bottom",
                    fontsize=9,
                )

        plt.tight_layout()

        # Save plot
        plot_path = self.output_dir / "inference_benchmark_scientific.png"
        plt.savefig(plot_path, dpi=300, bbox_inches="tight")
        plt.close()

        print(f"📊 Scientific visualizations saved to: {plot_path}")

    def print_scientific_summary(self, df: pd.DataFrame):
        """Print concise scientific summary."""
        print("\n" + "=" * 80)
        print("🔬 INFERENCE PERFORMANCE ANALYSIS")
        print("=" * 80)

        valid_df = df.dropna(subset=["mean_ms"])

        if valid_df.empty:
            print("❌ No valid results to analyze")
            return

        print(f"\nConfiguration: Single sequence inference (batch_size=1)")
        print(f"Sequence length: {SEQUENCE_LENGTH} frames")
        print(f"Prediction tasks: {len(PREDICTION_TASKS)} tasks")

        # Baseline performance
        print("\n📊 Baseline Performance:")
        baseline_data = valid_df[valid_df["optimization"] == "baseline"]
        for _, row in baseline_data.iterrows():
            print(f"   {row['model']}: {row['mean_ms']:.2f} ± {row['std_ms']:.2f} ms")

        # Best configurations
        print("\n🚀 Best Optimized Performance:")
        for model in valid_df["model"].unique():
            model_data = valid_df[valid_df["model"] == model]
            best_config = model_data.loc[model_data["mean_ms"].idxmin()]
            baseline_time = baseline_data[baseline_data["model"] == model][
                "mean_ms"
            ].iloc[0]
            speedup = baseline_time / best_config["mean_ms"]

            print(
                f"   {model}: {best_config['mean_ms']:.2f} ± {best_config['std_ms']:.2f} ms"
            )
            print(f"      Optimization: {best_config['optimization']}")
            print(f"      Speedup: {speedup:.2f}x ({(speedup-1)*100:+.1f}%)")

        # Memory usage
        device_name = self.profiler.get_device_name()
        print(f"\n💾 Memory Usage ({device_name}):")
        for model in valid_df["model"].unique():
            model_baseline = baseline_data[baseline_data["model"] == model]
            if not model_baseline.empty:
                device_mem = model_baseline["device_memory_mb"].iloc[0]
                print(f"   {model}: {device_mem:.1f} MB")

        print("\n" + "=" * 80)


def main():
    """Main entry point for inference benchmark."""
    parser = argparse.ArgumentParser(
        description="Scientific inference performance benchmark"
    )
    parser.add_argument("--device", type=str, default="cpu", help="Device to use")
    parser.add_argument(
        "--output-dir",
        type=str,
        default="evaluation/benchmark",
        help="Output directory",
    )
    parser.add_argument(
        "--warmup-runs", type=int, default=20, help="Number of warmup runs"
    )
    parser.add_argument(
        "--benchmark-runs", type=int, default=100, help="Number of benchmark runs"
    )

    args = parser.parse_args()

    # Initialize benchmark
    benchmark = InferenceBenchmark(device=args.device, output_dir=args.output_dir)

    # Override configuration if specified
    benchmark.warmup_runs = args.warmup_runs
    benchmark.benchmark_runs = args.benchmark_runs

    try:
        # Run comprehensive benchmark
        results_df = benchmark.run_comprehensive_benchmark()

        # Create scientific visualizations
        benchmark.create_scientific_visualizations(results_df)

        # Print scientific summary
        benchmark.print_scientific_summary(results_df)

        print("\n✅ Scientific benchmark completed successfully!")
        return True

    except Exception as e:
        print(f"❌ Benchmark failed: {e}")
        import traceback

        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
