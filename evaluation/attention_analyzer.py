#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Transformer Attention Pattern Analysis for Multimodal Brake Prediction
Analyzes attention patterns in the TransformerOutputDecoder for scientific evaluation.
"""

import sys
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import json
from tqdm import tqdm
import argparse

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.feature_config import (
    DETECTION_INPUT_DIM_PER_BOX,
    MAX_DETECTIONS_PER_FRAME,
    PREDICTION_TASKS,
    SEQUENCE_LENGTH,
    TELEMETRY_INPUT_DIM,
)
from src.model.factory import create_model_variant
from training.datasets.data_loaders import create_multimodal_dataloader


class TransformerAttentionAnalyzer:
    """
    Analyzes attention patterns in transformer-based decoder for brake prediction.
    """
    
    def __init__(self, model_path: str, data_dir: str, output_dir: str, device: str = "auto"):
        self.model_path = Path(model_path)
        self.data_dir = Path(data_dir)
        self.output_dir = Path(output_dir)
        
        # Setup device
        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
            
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize components
        self.model = None
        self.dataloader = None
        self.architecture_name = None
        
        print(f"🔬 Transformer Attention Analyzer initialized")
        print(f"   📁 Model: {self.model_path}")
        print(f"   📊 Data: {self.data_dir}")
        print(f"   💾 Output: {self.output_dir}")
        print(f"   🔧 Device: {self.device}")
    
    def load_model(self):
        """Load model and check if it uses transformer decoder."""
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model checkpoint not found: {self.model_path}")
        
        # Load checkpoint
        checkpoint = torch.load(self.model_path, map_location="cpu")
        self.architecture_name = checkpoint.get("arch_name", "unknown")
        
        # Check if architecture uses transformer decoder
        if "transformer" not in self.architecture_name:
            raise ValueError(f"Model {self.architecture_name} doesn't use transformer decoder")
        
        # Extract tasks from checkpoint
        tasks = []
        for key in checkpoint["model_state_dict"].keys():
            if "task_heads" in key and "weight" in key:
                task_name = key.split("task_heads.")[1].split(".")[0]
                if task_name not in tasks:
                    tasks.append(task_name)
        
        # Recreate model
        encoder_type, fusion_type, decoder_type = self.architecture_name.split("_")
        
        model_config = {
            "encoder_type": encoder_type,
            "fusion_type": fusion_type,
            "decoder_type": decoder_type,
            "telemetry_input_dim": TELEMETRY_INPUT_DIM,
            "detection_input_dim_per_box": DETECTION_INPUT_DIM_PER_BOX,
            "embedding_dim": 64,
            "hidden_dim": 128,
            "attention_num_heads": 8,
            "decoder_num_layers": 5,
            "dropout_prob": 0,
            "prediction_tasks": tasks,
            "max_detections": MAX_DETECTIONS_PER_FRAME,
            "max_seq_length": SEQUENCE_LENGTH,
        }
        
        self.model = create_model_variant(model_config)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model = self.model.to(self.device)
        self.model.eval()
        
        print(f"✅ Loaded transformer model: {self.architecture_name}")
        print(f"   🎯 Tasks: {tasks}")
        
        return tasks
    
    def setup_dataloader(self, tasks: List[str]):
        """Setup test dataloader."""
        self.dataloader = create_multimodal_dataloader(
            h5_file_path=str(self.data_dir / "test.h5"),
            batch_size=16,  # Smaller batch for attention analysis
            shuffle=False,  # Deterministic for reproducible analysis
            num_workers=4,
            pin_memory=True,
            load_into_memory=True,
            target_horizons=tasks,
        )
        
        print(f"✅ Test dataloader ready: {len(self.dataloader)} batches")
    
    def extract_attention_weights(self, num_samples: int = 100) -> Dict:
        """
        Extract attention weights from transformer decoder using attention patching.
        
        Args:
            num_samples: Number of samples to analyze
            
        Returns:
            Dictionary with attention weights and metadata
        """
        print(f"🔍 Extracting attention weights from {num_samples} samples...")
        
        # Storage for attention weights
        attention_weights_storage = []
        
        def patch_attention(module):
            """Patch MultiheadAttention to return attention weights."""
            forward_orig = module.forward
            
            def forward_wrapper(*args, **kwargs):
                kwargs['need_weights'] = True
                kwargs['average_attn_weights'] = False
                return forward_orig(*args, **kwargs)
            
            module.forward = forward_wrapper
        
        def save_attention_hook(module, input, output):
            """Hook to save attention weights."""
            if len(output) == 2:  # (attn_output, attn_weights)
                attn_output, attn_weights = output
                attention_weights_storage.append(attn_weights.detach().cpu())
        
        # Apply patches and hooks to all MultiheadAttention modules in transformer decoder
        hooks = []
        original_forwards = []
        
        for module in self.model.output_decoder.transformer_encoder.modules():
            if isinstance(module, torch.nn.MultiheadAttention):
                # Store original forward for restoration
                original_forwards.append((module, module.forward))
                
                # Apply patch
                patch_attention(module)
                
                # Register hook
                hook = module.register_forward_hook(save_attention_hook)
                hooks.append(hook)
        
        # Store data
        all_attention_weights = []
        all_predictions = []
        all_targets = []
        sample_metadata = []
        
        samples_processed = 0
        
        try:
            with torch.no_grad():
                for batch_idx, batch in enumerate(tqdm(self.dataloader, desc="Extracting attention")):
                    if samples_processed >= num_samples:
                        break
                    
                    # Move to device
                    telemetry = batch["telemetry_seq"].to(self.device)
                    detections = batch["detection_seq"].to(self.device)
                    mask = batch["detection_mask"].to(self.device)
                    targets = {k: v.to(self.device) for k, v in batch["targets"].items()}
                    
                    # Clear attention storage
                    attention_weights_storage.clear()
                    
                    # Forward pass - attention weights will be captured by hooks
                    predictions = self.model(telemetry, detections, mask)
                    
                    # Organize attention weights by layer
                    # Each layer's MultiheadAttention produces one set of weights
                    layer_attention_weights = attention_weights_storage.copy()
                    
                    # Store data for each sample in batch
                    batch_size = telemetry.shape[0]
                    for i in range(min(batch_size, num_samples - samples_processed)):
                        # Extract attention weights for this sample from each layer
                        sample_attention_weights = [aw[i] for aw in layer_attention_weights]
                        
                        all_attention_weights.append(sample_attention_weights)
                        all_predictions.append({k: v[i].cpu() for k, v in predictions.items()})
                        all_targets.append({k: v[i].cpu() for k, v in targets.items()})
                        sample_metadata.append({
                            "batch_idx": batch_idx,
                            "sample_idx": i,
                            "sequence_length": telemetry.shape[1],
                        })
                        samples_processed += 1
        
        finally:
            # Restore original forward methods and remove hooks
            for hook in hooks:
                hook.remove()
            
            for module, original_forward in original_forwards:
                module.forward = original_forward
        
        print(f"✅ Extracted attention weights from {samples_processed} samples")
        
        return {
            "attention_weights": all_attention_weights,
            "predictions": all_predictions,
            "targets": all_targets,
            "metadata": sample_metadata,
            "model_info": {
                "architecture": self.architecture_name,
                "num_layers": len(all_attention_weights[0]),
                "num_heads": all_attention_weights[0][0].shape[1],
                "sequence_length": SEQUENCE_LENGTH,
            }
        }
    
    def analyze_temporal_attention_patterns(self, attention_data: Dict) -> Dict:
        """
        Analyze temporal attention patterns across sequence positions.
        
        Args:
            attention_data: Extracted attention weights and predictions
            
        Returns:
            Analysis results
        """
        print("📈 Analyzing temporal attention patterns...")
        
        attention_weights = attention_data["attention_weights"]
        predictions = attention_data["predictions"]
        targets = attention_data["targets"]
        
        num_layers = attention_data["model_info"]["num_layers"]
        num_heads = attention_data["model_info"]["num_heads"]
        seq_len = attention_data["model_info"]["sequence_length"]
        
        # Aggregate attention patterns
        layer_attention_means = []
        head_attention_means = []
        
        for layer_idx in range(num_layers):
            layer_attentions = []
            head_attentions = []
            
            for sample_idx in range(len(attention_weights)):
                # Shape: (num_heads, seq_len, seq_len)
                sample_attention = attention_weights[sample_idx][layer_idx]
                layer_attentions.append(sample_attention.mean(dim=0))  # Average over heads
                head_attentions.append(sample_attention)  # Keep head dimension
            
            # Average over samples
            layer_attention_means.append(torch.stack(layer_attentions).mean(dim=0))
            head_attention_means.append(torch.stack(head_attentions).mean(dim=0))
        
        # Analyze attention focus patterns
        attention_focus_analysis = {}
        
        for layer_idx in range(num_layers):
            layer_attn = layer_attention_means[layer_idx]  # (seq_len, seq_len)
            
            # Focus on last position (used for prediction)
            last_pos_attention = layer_attn[-1, :]  # Attention from last position to all positions
            
            attention_focus_analysis[f"layer_{layer_idx}"] = {
                "temporal_distribution": last_pos_attention.tolist(),
                "peak_attention_position": int(torch.argmax(last_pos_attention)),
                "attention_entropy": float(-torch.sum(last_pos_attention * torch.log(last_pos_attention + 1e-8))),
                "recent_bias": float(last_pos_attention[-5:].sum()),  # Last 5 positions
                "early_bias": float(last_pos_attention[:5].sum()),    # First 5 positions
            }
        
        return {
            "layer_attention_means": layer_attention_means,
            "head_attention_means": head_attention_means,
            "attention_focus_analysis": attention_focus_analysis,
            "sequence_positions": list(range(seq_len)),
        }
    
    def analyze_task_specific_attention(self, attention_data: Dict) -> Dict:
        """
        Analyze how attention patterns differ for different prediction tasks.
        
        Args:
            attention_data: Extracted attention weights and predictions
            
        Returns:
            Task-specific attention analysis
        """
        print("🎯 Analyzing task-specific attention patterns...")
        
        attention_weights = attention_data["attention_weights"]
        predictions = attention_data["predictions"]
        targets = attention_data["targets"]
        
        task_attention_analysis = {}
        
        for task_name in PREDICTION_TASKS:
            if task_name not in predictions[0]:
                continue
            
            # Separate samples by prediction outcome
            positive_samples = []
            negative_samples = []
            
            for i, (pred, target) in enumerate(zip(predictions, targets)):
                pred_prob = torch.sigmoid(pred[task_name]).item()
                actual_label = target[task_name].item()
                
                # Categorize based on prediction confidence and correctness
                if pred_prob > 0.7 and actual_label == 1:  # High confidence correct positive
                    positive_samples.append(i)
                elif pred_prob < 0.3 and actual_label == 0:  # High confidence correct negative
                    negative_samples.append(i)
            
            # Analyze attention differences
            if len(positive_samples) > 5 and len(negative_samples) > 5:
                pos_attention = self._average_attention_for_samples(
                    attention_weights, positive_samples
                )
                neg_attention = self._average_attention_for_samples(
                    attention_weights, negative_samples
                )
                
                # Calculate attention difference
                attention_diff = pos_attention - neg_attention
                
                task_attention_analysis[task_name] = {
                    "positive_samples": len(positive_samples),
                    "negative_samples": len(negative_samples),
                    "attention_difference": attention_diff.tolist(),
                    "max_difference_position": int(torch.argmax(torch.abs(attention_diff))),
                    "positive_attention_entropy": float(self._calculate_entropy(pos_attention)),
                    "negative_attention_entropy": float(self._calculate_entropy(neg_attention)),
                }
        
        return task_attention_analysis
    
    def _average_attention_for_samples(self, attention_weights: List, sample_indices: List) -> torch.Tensor:
        """Average attention weights for given sample indices (last layer, last position)."""
        selected_attentions = []
        for idx in sample_indices:
            # Use last layer, average over heads, last position attention
            last_layer_attn = attention_weights[idx][-1].mean(dim=0)  # Average over heads
            last_pos_attn = last_layer_attn[-1, :]  # Last position attention
            selected_attentions.append(last_pos_attn)
        
        return torch.stack(selected_attentions).mean(dim=0)
    
    def _calculate_entropy(self, attention_weights: torch.Tensor) -> float:
        """Calculate entropy of attention distribution."""
        return -torch.sum(attention_weights * torch.log(attention_weights + 1e-8)).item()
    
    def create_attention_visualizations(self, temporal_analysis: Dict, task_analysis: Dict):
        """Create comprehensive attention pattern visualizations."""
        print("📊 Creating attention visualizations...")
        
        # Set style
        plt.style.use("seaborn-v0_8")
        
        # Create figure with subplots
        fig = plt.figure(figsize=(20, 12))
        gs = fig.add_gridspec(3, 4, hspace=0.3, wspace=0.3)
        
        # 1. Temporal attention patterns by layer
        ax1 = fig.add_subplot(gs[0, :2])
        num_layers = len(temporal_analysis["layer_attention_means"])
        seq_positions = temporal_analysis["sequence_positions"]
        seq_len = len(seq_positions)
        
        for layer_idx in range(num_layers):
            attention_dist = temporal_analysis["attention_focus_analysis"][f"layer_{layer_idx}"]["temporal_distribution"]
            ax1.plot(seq_positions, attention_dist, label=f"Layer {layer_idx + 1}", linewidth=2)
        
        ax1.set_xlabel("Sequence Position")
        ax1.set_ylabel("Attention Weight")
        ax1.set_title("Temporal Attention Distribution by Layer")
        ax1.legend()
        
        # Set proper ticks and grid for discrete time positions
        x_ticks = list(range(0, seq_len, 2))
        ax1.set_xticks(x_ticks)
        ax1.set_xlim(0, seq_len - 1)
        ax1.grid(True, alpha=0.3)
        
        # 2. Attention entropy by layer
        ax2 = fig.add_subplot(gs[0, 2])
        layer_entropies = [
            temporal_analysis["attention_focus_analysis"][f"layer_{i}"]["attention_entropy"]
            for i in range(num_layers)
        ]
        ax2.bar(range(num_layers), layer_entropies, color='skyblue')
        ax2.set_xlabel("Layer")
        ax2.set_ylabel("Attention Entropy")
        ax2.set_title("Attention Entropy by Layer")
        ax2.set_xticks(range(num_layers))
        ax2.set_xticklabels([f"L{i+1}" for i in range(num_layers)])
        
        # 3. Recent vs Early bias
        ax3 = fig.add_subplot(gs[0, 3])
        recent_bias = [
            temporal_analysis["attention_focus_analysis"][f"layer_{i}"]["recent_bias"]
            for i in range(num_layers)
        ]
        early_bias = [
            temporal_analysis["attention_focus_analysis"][f"layer_{i}"]["early_bias"]
            for i in range(num_layers)
        ]
        
        x = np.arange(num_layers)
        width = 0.35
        ax3.bar(x - width/2, recent_bias, width, label='Recent (Last 5)', color='orange')
        ax3.bar(x + width/2, early_bias, width, label='Early (First 5)', color='blue')
        ax3.set_xlabel("Layer")
        ax3.set_ylabel("Attention Sum")
        ax3.set_title("Temporal Bias Analysis")
        ax3.set_xticks(x)
        ax3.set_xticklabels([f"L{i+1}" for i in range(num_layers)])
        ax3.legend()
        
        # 4. Attention heatmap for last layer
        ax4 = fig.add_subplot(gs[1, :2])
        last_layer_attn = temporal_analysis["layer_attention_means"][-1].numpy()
        
        im = ax4.imshow(last_layer_attn, cmap='Blues', aspect='auto')
        ax4.set_xlabel("Key Position")
        ax4.set_ylabel("Query Position")
        ax4.set_title(f"Attention Matrix - Layer {num_layers}")
        
        # Set proper ticks for discrete positions
        tick_positions = list(range(0, seq_len, 2))
        tick_labels = [str(i) for i in tick_positions]
        ax4.set_xticks(tick_positions)
        ax4.set_xticklabels(tick_labels)
        ax4.set_yticks(tick_positions)
        ax4.set_yticklabels(tick_labels)
        
        plt.colorbar(im, ax=ax4, shrink=0.8)
        
        # 5. Task-specific attention differences
        if task_analysis:
            ax5 = fig.add_subplot(gs[1, 2:])
            
            task_names = list(task_analysis.keys())
            if task_names:
                # Show attention differences for each task
                for i, task_name in enumerate(task_names[:3]):  # Show max 3 tasks
                    if "attention_difference" in task_analysis[task_name]:
                        attention_diff = task_analysis[task_name]["attention_difference"]
                        ax5.plot(seq_positions, attention_diff, 
                                label=f"{task_name} (Δ Attention)", linewidth=2)
                
                ax5.set_xlabel("Sequence Position")
                ax5.set_ylabel("Attention Difference (Positive - Negative)")
                ax5.set_title("Task-Specific Attention Differences")
                ax5.legend()
                
                # Set proper ticks and grid for discrete positions
                ax5.set_xticks(x_ticks)
                ax5.set_xlim(0, seq_len - 1)
                ax5.grid(True, alpha=0.3)
                ax5.axhline(y=0, color='black', linestyle='--', alpha=0.5)
        
        # 6. Summary statistics table
        ax6 = fig.add_subplot(gs[2, :])
        ax6.axis('off')
        
        # Create summary table
        summary_data = []
        for layer_idx in range(num_layers):
            layer_analysis = temporal_analysis["attention_focus_analysis"][f"layer_{layer_idx}"]
            summary_data.append([
                f"Layer {layer_idx + 1}",
                f"{layer_analysis['peak_attention_position']}",
                f"{layer_analysis['attention_entropy']:.3f}",
                f"{layer_analysis['recent_bias']:.3f}",
                f"{layer_analysis['early_bias']:.3f}"
            ])
        
        table = ax6.table(
            cellText=summary_data,
            colLabels=["Layer", "Peak Position", "Entropy", "Recent Bias", "Early Bias"],
            cellLoc="center",
            loc="center",
            bbox=[0.1, 0.2, 0.8, 0.6]
        )
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        ax6.set_title("Attention Pattern Summary", pad=20)
        
        plt.tight_layout()
        
        # Save plot
        viz_path = self.output_dir / "transformer_attention_analysis.png"
        plt.savefig(viz_path, dpi=400, bbox_inches="tight")
        plt.close()
        
        print(f"✅ Attention visualizations saved to {viz_path}")
    
    def save_analysis_results(self, temporal_analysis: Dict, task_analysis: Dict):
        """Save detailed analysis results to JSON."""
        results = {
            "model_architecture": self.architecture_name,
            "analysis_type": "transformer_attention_patterns",
            "temporal_analysis": {
                "attention_focus_analysis": temporal_analysis["attention_focus_analysis"],
                "sequence_length": len(temporal_analysis["sequence_positions"]),
            },
            "task_analysis": task_analysis,
            "summary": {
                "num_layers_analyzed": len(temporal_analysis["layer_attention_means"]),
                "tasks_analyzed": list(task_analysis.keys()) if task_analysis else [],
            }
        }
        
        # Convert tensors to lists for JSON serialization
        def convert_tensors(obj):
            if isinstance(obj, torch.Tensor):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {k: convert_tensors(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_tensors(item) for item in obj]
            else:
                return obj
        
        results = convert_tensors(results)
        
        # Save results
        results_path = self.output_dir / "attention_analysis_results.json"
        with open(results_path, "w") as f:
            json.dump(results, f, indent=2)
        
        print(f"✅ Analysis results saved to {results_path}")
    
    def run_analysis(self, num_samples: int = 100):
        """Run complete attention pattern analysis."""
        print(f"🚀 Starting transformer attention analysis...")
        print(f"   📊 Architecture: {self.architecture_name}")
        print(f"   🔍 Samples: {num_samples}")
        
        # Load model and setup dataloader
        tasks = self.load_model()
        self.setup_dataloader(tasks)
        
        # Extract attention weights
        attention_data = self.extract_attention_weights(num_samples)
        
        # Analyze temporal patterns
        temporal_analysis = self.analyze_temporal_attention_patterns(attention_data)
        
        # Analyze task-specific patterns
        task_analysis = self.analyze_task_specific_attention(attention_data)
        
        # Create visualizations
        self.create_attention_visualizations(temporal_analysis, task_analysis)
        
        # Save results
        self.save_analysis_results(temporal_analysis, task_analysis)
        
        # Print key findings
        self._print_key_findings(temporal_analysis, task_analysis)
        
        print("🎉 Attention analysis completed successfully!")
    
    def _print_key_findings(self, temporal_analysis: Dict, task_analysis: Dict):
        """Print key findings from attention analysis."""
        print("\n" + "=" * 60)
        print("🔬 TRANSFORMER ATTENTION ANALYSIS RESULTS")
        print("=" * 60)
        
        num_layers = len(temporal_analysis["layer_attention_means"])
        
        print(f"\n📊 Temporal Attention Patterns:")
        for layer_idx in range(num_layers):
            analysis = temporal_analysis["attention_focus_analysis"][f"layer_{layer_idx}"]
            print(f"   Layer {layer_idx + 1}:")
            print(f"     Peak attention at position: {analysis['peak_attention_position']}")
            print(f"     Attention entropy: {analysis['attention_entropy']:.3f}")
            print(f"     Recent bias (last 5): {analysis['recent_bias']:.3f}")
            print(f"     Early bias (first 5): {analysis['early_bias']:.3f}")
        
        if task_analysis:
            print(f"\n🎯 Task-Specific Attention Differences:")
            for task_name, analysis in task_analysis.items():
                print(f"   {task_name}:")
                print(f"     Positive samples: {analysis['positive_samples']}")
                print(f"     Negative samples: {analysis['negative_samples']}")
                print(f"     Max difference at position: {analysis['max_difference_position']}")


def main():
    """Main entry point for attention analysis."""
    parser = argparse.ArgumentParser(description="Analyze transformer attention patterns")
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Path to transformer model checkpoint"
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="data/datasets/multimodal/s1",
        help="Dataset directory"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        help="Output directory (default: model directory + /attention_analysis)"
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=4000,
        help="Number of samples to analyze"
    )
    
    args = parser.parse_args()
    
    # Validate model path
    model_path = Path(args.model)
    if not model_path.exists():
        model_path = Path("data/models/multimodal") / model_path
    if not model_path.exists() or model_path.is_dir():
        model_path = model_path / "best_model.pt"
    
    if not model_path.exists():
        print(f"❌ Model not found: {args.model}")
        return False
    
    # Set output directory
    if args.output_dir is None:
        args.output_dir = model_path.parent / "attention_analysis"
    
    print(f"🎯 Transformer Attention Analysis Configuration:")
    print(f"   🤖 Model: {model_path}")
    print(f"   📊 Data: {args.data_dir}")
    print(f"   📁 Output: {args.output_dir}")
    print(f"   🔍 Samples: {args.num_samples}")
    
    try:
        # Initialize analyzer
        analyzer = TransformerAttentionAnalyzer(
            model_path=str(model_path),
            data_dir=args.data_dir,
            output_dir=args.output_dir
        )
        
        # Run analysis
        analyzer.run_analysis(num_samples=args.num_samples)
        
        return True
        
    except Exception as e:
        print(f"❌ Analysis failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)