#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
False Negative Intensity Analysis Script
Analyzes the intensity characteristics of False Negative predictions using pre-computed intensities.
"""

import sys
import json
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional
import torch
import torch.nn.functional as F
from tqdm import tqdm
import h5py
from scipy import stats

# Add project root to path for imports
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.feature_config import (
    COAST_THRESHOLD_PERCENT,
    DETECTION_INPUT_DIM_PER_BOX,
    MAX_DETECTIONS_PER_FRAME,
    SEQUENCE_LENGTH,
    TELEMETRY_INPUT_DIM,
    TELEMETRY_FEATURES,
    SAMPLING_RATE_HZ,
)
from src.model.factory import create_model_variant
from training.datasets.data_loaders import create_multimodal_dataloader


class FalseNegativeAnalyzer:
    """
    Analyzes the intensity characteristics of False Negative predictions.
    """

    def __init__(
        self,
        model_path: str,
        data_dir: str,
        output_dir: str,
        device: str = "auto",
    ):
        self.model_path = Path(model_path)
        self.data_dir = Path(data_dir)
        self.output_dir = Path(output_dir)

        # Setup device
        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Will be populated during initialization
        self.model = None
        self.tasks = None
        self.test_loader = None
        self.architecture_name = None

        print(f"🔬 False Negative Intensity Analyzer initialized")
        print(f"   📁 Model: {self.model_path}")
        print(f"   📊 Data: {self.data_dir}")
        print(f"   💾 Output: {self.output_dir}")
        print(f"   🔧 Device: {self.device}")

    def load_model_and_extract_tasks(self) -> Tuple[List[str], str]:
        """Load model checkpoint and extract tasks."""
        print("🤖 Loading model checkpoint...")

        if not self.model_path.exists():
            raise FileNotFoundError(f"Model checkpoint not found: {self.model_path}")

        # Load checkpoint
        checkpoint = torch.load(self.model_path, map_location="cpu")
        architecture_name = checkpoint.get("arch_name", "unknown_architecture")

        # Extract tasks from model state dict
        tasks = []
        for key in checkpoint["model_state_dict"].keys():
            if "task_heads" in key and "weight" in key:
                task_name = key.split("task_heads.")[1].split(".")[0]
                if task_name not in tasks:
                    tasks.append(task_name)

        if not tasks:
            raise ValueError("No tasks found in model checkpoint")

        tasks = sorted(tasks)
        print(f"✅ Model loaded: {architecture_name}")
        print(f"   🎯 Tasks found: {tasks}")

        # Recreate model architecture
        encoder_type, fusion_type, decoder_type = architecture_name.split("_")

        model_config = {
            "encoder_type": encoder_type,
            "fusion_type": fusion_type,
            "decoder_type": decoder_type,
            "telemetry_input_dim": TELEMETRY_INPUT_DIM,
            "detection_input_dim_per_box": DETECTION_INPUT_DIM_PER_BOX,
            "embedding_dim": 64,
            "hidden_dim": 128,
            "attention_num_heads": 8,
            "decoder_num_layers": 4,
            "dropout_prob": 0,
            "prediction_tasks": tasks,
            "max_detections": MAX_DETECTIONS_PER_FRAME,
            "max_seq_length": SEQUENCE_LENGTH,
        }

        self.model = create_model_variant(model_config)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model = self.model.to(self.device)
        self.model.eval()

        self.tasks = tasks
        self.architecture_name = architecture_name

        return tasks, architecture_name

    def setup_test_dataloader(self):
        """Setup test dataloader."""
        print("📚 Setting up test dataloader...")

        self.test_loader = create_multimodal_dataloader(
            h5_file_path=str(self.data_dir / "test.h5"),
            batch_size=64,  # Smaller batch size for detailed analysis
            shuffle=False,
            num_workers=4,
            pin_memory=True,
            target_horizons=self.tasks,
            load_intensities=True,  # Load pre-computed intensities
        )

        print(f"✅ Test dataloader ready: {len(self.test_loader)} batches")

    def generate_predictions_with_intensities(self) -> Dict[str, Any]:
        """
        Generate predictions along with corresponding intensity data.
        
        Returns:
            Dictionary with predictions, targets, and intensity data
        """
        print("🔮 Generating predictions with intensity data...")

        all_predictions = {task: [] for task in self.tasks}
        all_targets = {task: [] for task in self.tasks}
        all_intensities = {}
        all_sample_indices = []
        sample_idx = 0

        with torch.no_grad():
            for batch in tqdm(self.test_loader, desc="Inference with Intensities"):
                # Move data to device
                telemetry = batch["telemetry_seq"].to(self.device, non_blocking=True)
                detections = batch["detection_seq"].to(self.device, non_blocking=True)
                mask = batch["detection_mask"].to(self.device, non_blocking=True)

                # Forward pass
                predictions = self.model(telemetry, detections, mask)

                # Collect predictions and targets
                for task in self.tasks:
                    pred_probs = torch.sigmoid(predictions[task]).squeeze(-1)
                    all_predictions[task].append(pred_probs.cpu().numpy())
                    all_targets[task].append(batch["targets"][task].numpy())

                # Collect intensity data
                if "intensities" in batch:
                    for intensity_key, intensity_values in batch["intensities"].items():
                        if intensity_key not in all_intensities:
                            all_intensities[intensity_key] = []
                        all_intensities[intensity_key].append(intensity_values.numpy())

                # Track sample indices for this batch
                batch_size = telemetry.shape[0]
                batch_indices = list(range(sample_idx, sample_idx + batch_size))
                all_sample_indices.extend(batch_indices)
                sample_idx += batch_size

        # Concatenate all batches
        results = {
            "predictions": {},
            "targets": {},
            "intensities": {},
            "sample_indices": all_sample_indices,
        }

        for task in self.tasks:
            results["predictions"][task] = np.concatenate(all_predictions[task])
            results["targets"][task] = np.concatenate(all_targets[task])

        for intensity_key in all_intensities:
            results["intensities"][intensity_key] = np.concatenate(all_intensities[intensity_key])

        print(f"✅ Data collected for {len(results['sample_indices'])} samples")
        print(f"   📊 Available intensity types: {list(results['intensities'].keys())}")
        
        return results

    def load_optimal_thresholds(self) -> Dict[str, float]:
        """
        Load optimal thresholds from evaluation results, or use defaults.
        """
        # Try to load from existing evaluation
        eval_summary_path = self.model_path.parent / "evaluation" / "evaluation_summary.json"
        
        if eval_summary_path.exists():
            print(f"📊 Loading optimal thresholds from {eval_summary_path}")
            with open(eval_summary_path, 'r') as f:
                eval_data = json.load(f)
            
            thresholds = {}
            for task in self.tasks:
                if task in eval_data.get("summary_metrics", {}):
                    thresholds[task] = eval_data["summary_metrics"][task]["optimal_performance"]["f1_optimal"]["threshold"]
                else:
                    thresholds[task] = 0.5  # Default
            
            print(f"   ✅ Loaded thresholds: {thresholds}")
        else:
            print("⚠️ No evaluation results found, using default thresholds")
            thresholds = {task: 0.5 for task in self.tasks}

        return thresholds

    def extract_event_intensities_from_dataset(
        self,
        targets: Dict[str, np.ndarray],
        intensities: Dict[str, np.ndarray],
        sample_indices: List[int]
    ) -> Dict[str, Dict[str, List[float]]]:
        """
        Extract event intensities directly from the dataset for positive cases.
        
        Args:
            targets: Target labels per task
            intensities: Pre-computed intensity values from dataset
            sample_indices: Sample indices for tracking
            
        Returns:
            Dictionary with event intensities per task
        """
        print("🎯 Extracting event intensities from dataset...")
        
        extracted_intensities = {}
        
        for task in self.tasks:
            extracted_intensities[task] = {
                "brake_force": [],
                "accelerator_pos": [],
                "sample_indices": [],
                "event_occurred": [],
            }
            
            # Parse horizon from task name (e.g., "brake_1s" -> 1)
            horizon_str = task.split("_")[1].replace("s", "")
            
            brake_force_key = f"brake_force_{horizon_str}s"
            acc_pos_key = f"acc_pos_{horizon_str}s"
            
            # Check if intensity keys exist in dataset
            if brake_force_key not in intensities:
                print(f"⚠️ Missing intensity key: {brake_force_key}")
                continue
            if acc_pos_key not in intensities:
                print(f"⚠️ Missing intensity key: {acc_pos_key}")
                continue
            
            task_targets = targets[task]
            brake_force_values = intensities[brake_force_key]
            acc_pos_values = intensities[acc_pos_key]
            
            # Extract intensities for positive cases only
            positive_indices = np.where(task_targets == 1)[0]
            
            for idx in positive_indices:
                if idx < len(brake_force_values) and idx < len(acc_pos_values):
                    brake_force = float(brake_force_values[idx])
                    acc_pos = float(acc_pos_values[idx])
                    
                    if "brake" in task:
                        brake_force = max(0, brake_force)  
                        extracted_intensities[task]["brake_force"].append(brake_force)
                        extracted_intensities[task]["accelerator_pos"].append(0.0)  # Not relevant
                    elif "coast" in task:
                        # Coast intensity: lower accelerator position = higher coast intensity
                        coast_intensity = max(0, COAST_THRESHOLD_PERCENT - acc_pos) / COAST_THRESHOLD_PERCENT * 100.0
                        extracted_intensities[task]["accelerator_pos"].append(coast_intensity)
                        extracted_intensities[task]["brake_force"].append(0.0)  # Not relevant
                    
                    extracted_intensities[task]["sample_indices"].append(sample_indices[idx])
                    extracted_intensities[task]["event_occurred"].append(True)
        
        # Print summary statistics
        for task in self.tasks:
            if task in extracted_intensities:
                n_events = len(extracted_intensities[task]["sample_indices"])
                print(f"   📊 {task}: {n_events} positive events with intensities")
        
        print("✅ Event intensities extracted from dataset")
        return extracted_intensities

    def analyze_false_negatives(
        self,
        predictions: Dict[str, np.ndarray],
        targets: Dict[str, np.ndarray],
        intensities: Dict[str, Dict[str, List[float]]],
        thresholds: Dict[str, float],
    ) -> Dict[str, Dict[str, Any]]:
        """
        Analyze False Negative cases and their intensity characteristics.
        """
        print("🕵️ Analyzing False Negative characteristics...")

        analysis_results = {}

        for task in self.tasks:
            y_true = targets[task]
            y_pred_probs = predictions[task]
            threshold = thresholds[task]
            y_pred = (y_pred_probs >= threshold).astype(int)

            # Identify different prediction outcomes
            true_positives = (y_true == 1) & (y_pred == 1)
            false_negatives = (y_true == 1) & (y_pred == 0)

            # Get intensity data for this task
            if task not in intensities:
                print(f"⚠️ No intensity data available for task: {task}")
                continue
                
            task_intensities = intensities[task]
            
            if "brake" in task:
                intensity_values = np.array(task_intensities["brake_force"])
                intensity_name = "brake_force"
            else:  # coast task
                intensity_values = np.array(task_intensities["accelerator_pos"])
                intensity_name = "coast_intensity"

            # Map sample indices to intensity values
            sample_to_intensity = {}
            for i, sample_idx in enumerate(task_intensities["sample_indices"]):
                sample_to_intensity[sample_idx] = intensity_values[i]

            # Extract intensities for TP and FN cases
            tp_intensities = []
            fn_intensities = []
            tp_predictions = []
            fn_predictions = []

            positive_indices = np.where(y_true == 1)[0]
            for idx in positive_indices:
                if idx in sample_to_intensity:
                    intensity = sample_to_intensity[idx]
                    prediction_score = y_pred_probs[idx]
                    
                    if true_positives[idx]:
                        tp_intensities.append(intensity)
                        tp_predictions.append(prediction_score)
                    elif false_negatives[idx]:
                        fn_intensities.append(intensity)
                        fn_predictions.append(prediction_score)

            # Statistical analysis
            analysis = {
                "task": task,
                "threshold": threshold,
                "intensity_name": intensity_name,
                "counts": {
                    "total_positives": int(y_true.sum()),
                    "true_positives": int(true_positives.sum()),
                    "false_negatives": int(false_negatives.sum()),
                    "tp_with_intensity": len(tp_intensities),
                    "fn_with_intensity": len(fn_intensities),
                },
                "intensity_statistics": {},
                "prediction_score_statistics": {},
            }

            if tp_intensities and fn_intensities:
                # Intensity statistics
                analysis["intensity_statistics"] = {
                    "tp_mean": float(np.mean(tp_intensities)),
                    "tp_std": float(np.std(tp_intensities)),
                    "tp_median": float(np.median(tp_intensities)),
                    "fn_mean": float(np.mean(fn_intensities)),
                    "fn_std": float(np.std(fn_intensities)),
                    "fn_median": float(np.median(fn_intensities)),
                }

                # Prediction score statistics
                analysis["prediction_score_statistics"] = {
                    "tp_pred_mean": float(np.mean(tp_predictions)),
                    "tp_pred_std": float(np.std(tp_predictions)),
                    "fn_pred_mean": float(np.mean(fn_predictions)),
                    "fn_pred_std": float(np.std(fn_predictions)),
                }

                # Statistical tests
                if len(tp_intensities) > 1 and len(fn_intensities) > 1:
                    # Mann-Whitney U test (non-parametric)
                    statistic, p_value = stats.mannwhitneyu(
                        tp_intensities, fn_intensities, alternative='two-sided'
                    )
                    analysis["statistical_test"] = {
                        "test": "Mann-Whitney U",
                        "statistic": float(statistic),
                        "p_value": float(p_value),
                        "significant": p_value < 0.05,
                    }

                # Store raw data for visualization
                analysis["raw_data"] = {
                    "tp_intensities": tp_intensities,
                    "fn_intensities": fn_intensities,
                    "tp_predictions": tp_predictions,
                    "fn_predictions": fn_predictions,
                }

            analysis_results[task] = analysis

        return analysis_results

    def create_intensity_visualizations(
        self, analysis_results: Dict[str, Dict[str, Any]]
    ):
        """Create comprehensive visualizations for intensity analysis."""
        print("📊 Creating intensity visualizations...")

        # Calculate layout
        n_tasks = len(self.tasks)
        if n_tasks <= 2:
            fig_rows, fig_cols = 2, n_tasks
            fig_width = 7 * n_tasks
        else:
            fig_rows, fig_cols = 2, (n_tasks + 1) // 2
            fig_width = 14

        fig_height = 12
        fig, axes = plt.subplots(fig_rows, fig_cols, figsize=(fig_width, fig_height))
        
        if n_tasks == 1:
            axes = np.array([[axes], [axes]])  # Make it 2D
        elif fig_rows == 1:
            axes = axes.reshape(1, -1)
        elif fig_cols == 1:
            axes = axes.reshape(-1, 1)

        # Color scheme
        tp_color = '#2E8B57'  # Sea Green
        fn_color = '#DC143C'  # Crimson

        for i, (task, analysis) in enumerate(analysis_results.items()):
            row = i // fig_cols
            col = i % fig_cols

            if "raw_data" not in analysis:
                # Skip if no data
                continue

            tp_intensities = analysis["raw_data"]["tp_intensities"]
            fn_intensities = analysis["raw_data"]["fn_intensities"]
            tp_predictions = analysis["raw_data"]["tp_predictions"]
            fn_predictions = analysis["raw_data"]["fn_predictions"]
            intensity_name = analysis["intensity_name"]

            # Top plot: Intensity distributions
            ax_top = axes[0, col] if fig_cols > 1 else axes[0]
            
            if tp_intensities and fn_intensities:
                # Histogram comparison
                bins = np.linspace(
                    min(min(tp_intensities), min(fn_intensities)),
                    max(max(tp_intensities), max(fn_intensities)),
                    20
                )

                ax_top.hist(tp_intensities, bins=bins, alpha=0.7, 
                           label=f'True Positives (n={len(tp_intensities)})',
                           color=tp_color, density=True)
                ax_top.hist(fn_intensities, bins=bins, alpha=0.7, 
                           label=f'False Negatives (n={len(fn_intensities)})',
                           color=fn_color, density=True)

                # Add mean lines
                ax_top.axvline(np.mean(tp_intensities), color=tp_color, linestyle='--',
                              label=f'TP Mean: {np.mean(tp_intensities):.2f}')
                ax_top.axvline(np.mean(fn_intensities), color=fn_color, linestyle='--',
                              label=f'FN Mean: {np.mean(fn_intensities):.2f}')

                ax_top.set_xlabel(f'{intensity_name.replace("_", " ").title()}')
                ax_top.set_ylabel('Density')
                ax_top.set_title(f'{task.upper()}: Event Intensity Distribution')
                ax_top.legend(fontsize=8)
                ax_top.grid(True, alpha=0.3)

            # Bottom plot: Prediction Score vs Intensity
            ax_bottom = axes[1, col] if fig_cols > 1 else axes[1]
            
            if tp_intensities and fn_intensities:
                ax_bottom.scatter(tp_intensities, tp_predictions, alpha=0.6,
                                 color=tp_color, label=f'True Positives', s=30)
                ax_bottom.scatter(fn_intensities, fn_predictions, alpha=0.6,
                                 color=fn_color, label=f'False Negatives', s=30)

                # Add threshold line
                threshold = analysis["threshold"]
                ax_bottom.axhline(threshold, color='black', linestyle='-', alpha=0.8,
                                 label=f'Threshold: {threshold:.3f}')

                ax_bottom.set_xlabel(f'{intensity_name.replace("_", " ").title()}')
                ax_bottom.set_ylabel('Prediction Score')
                ax_bottom.set_title(f'{task.upper()}: Prediction Score vs Intensity')
                ax_bottom.legend(fontsize=8)
                ax_bottom.grid(True, alpha=0.3)

        # Remove empty subplots
        for i in range(len(self.tasks), fig_rows * fig_cols):
            row = i // fig_cols
            col = i % fig_cols
            if row < fig_rows and col < fig_cols:
                fig.delaxes(axes[row, col])

        plt.tight_layout()

        # Save plot
        viz_path = self.output_dir / "false_negative_intensity_analysis.png"
        plt.savefig(viz_path, dpi=300, bbox_inches='tight')
        plt.close()

        print(f"✅ Intensity visualizations saved to {viz_path}")

    def create_summary_boxplots(self, analysis_results: Dict[str, Dict[str, Any]]):
        """Create summary boxplots comparing TP vs FN intensities."""
        print("📊 Creating summary boxplots...")

        # Color scheme
        tp_color = '#2E8B57'  # Sea Green
        fn_color = '#DC143C'  # Crimson

        # Prepare data for boxplots
        plot_data = []
        for task, analysis in analysis_results.items():
            if "raw_data" not in analysis:
                continue

            tp_intensities = analysis["raw_data"]["tp_intensities"]
            fn_intensities = analysis["raw_data"]["fn_intensities"]
            intensity_name = analysis["intensity_name"]

            # Add TP data
            for intensity in tp_intensities:
                plot_data.append({
                    'task': task,
                    'type': 'True Positive',
                    'intensity': intensity,
                    'intensity_type': intensity_name
                })

            # Add FN data
            for intensity in fn_intensities:
                plot_data.append({
                    'task': task,
                    'type': 'False Negative',
                    'intensity': intensity,
                    'intensity_type': intensity_name
                })

        if not plot_data:
            print("⚠️ No data available for boxplots")
            return

        df = pd.DataFrame(plot_data)

        # Create figure
        fig, ax = plt.subplots(1, 1, figsize=(12, 8))

        # Create boxplot
        sns.boxplot(data=df, x='task', y='intensity', hue='type', ax=ax, 
                   palette=[tp_color, fn_color])

        ax.set_title('Event Intensity: True Positives vs False Negatives', fontsize=14)
        ax.set_xlabel('Task', fontsize=12)
        ax.set_ylabel('Event Intensity', fontsize=12)
        ax.grid(True, alpha=0.3)

        # Rotate x-axis labels if needed
        plt.xticks(rotation=45)
        plt.tight_layout()

        # Save boxplot
        boxplot_path = self.output_dir / "intensity_comparison_boxplots.png"
        plt.savefig(boxplot_path, dpi=300, bbox_inches='tight')
        plt.close()

        print(f"✅ Summary boxplots saved to {boxplot_path}")

    def generate_intensity_report(
        self, analysis_results: Dict[str, Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Generate summary report for intensity analysis."""
        print("📋 Generating intensity analysis report...")

        report = {
            "architecture": self.architecture_name,
            "analysis_type": "false_negative_intensity",
            "tasks_analyzed": list(self.tasks),
            "key_findings": {},
            "statistical_summary": {},
        }

        for task, analysis in analysis_results.items():
            if "intensity_statistics" not in analysis:
                continue

            intensity_stats = analysis["intensity_statistics"]
            pred_stats = analysis["prediction_score_statistics"]

            # Calculate effect size (Cohen's d)
            tp_mean, tp_std = intensity_stats["tp_mean"], intensity_stats["tp_std"]
            fn_mean, fn_std = intensity_stats["fn_mean"], intensity_stats["fn_std"]

            if tp_std > 0 and fn_std > 0:
                pooled_std = np.sqrt((tp_std**2 + fn_std**2) / 2)
                cohens_d = (tp_mean - fn_mean) / pooled_std
            else:
                cohens_d = 0.0

            task_findings = {
                "intensity_difference": {
                    "tp_mean": tp_mean,
                    "fn_mean": fn_mean,
                    "difference": tp_mean - fn_mean,
                    "effect_size_cohens_d": cohens_d,
                },
                "prediction_score_difference": {
                    "tp_pred_mean": pred_stats["tp_pred_mean"],
                    "fn_pred_mean": pred_stats["fn_pred_mean"],
                    "difference": pred_stats["tp_pred_mean"] - pred_stats["fn_pred_mean"],
                },
                "sample_sizes": {
                    "true_positives": analysis["counts"]["tp_with_intensity"],
                    "false_negatives": analysis["counts"]["fn_with_intensity"],
                },
            }

            # Add statistical test results if available
            if "statistical_test" in analysis:
                task_findings["statistical_significance"] = analysis["statistical_test"]

            # Interpretation
            if abs(cohens_d) < 0.2:
                effect_interpretation = "negligible"
            elif abs(cohens_d) < 0.5:
                effect_interpretation = "small"
            elif abs(cohens_d) < 0.8:
                effect_interpretation = "medium"
            else:
                effect_interpretation = "large"

            task_findings["interpretation"] = {
                "effect_size_category": effect_interpretation,
                "intensity_pattern": "higher" if tp_mean > fn_mean else "lower",
                "clinical_significance": abs(tp_mean - fn_mean) > 0.1,  # Domain-specific threshold
            }

            report["key_findings"][task] = task_findings

        return report

    def _convert_numpy_types(self, obj):
        """Recursively convert numpy types to native Python types for JSON serialization."""
        if isinstance(obj, dict):
            return {key: self._convert_numpy_types(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [self._convert_numpy_types(item) for item in obj]
        elif isinstance(obj, np.bool_):
            return bool(obj)
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return obj

    def save_results(
        self, analysis_results: Dict[str, Dict[str, Any]], report: Dict[str, Any]
    ):
        """Save analysis results to files."""
        print("💾 Saving intensity analysis results...")

        # Clean raw data for JSON serialization and convert numpy types
        clean_analysis = {}
        for task, analysis in analysis_results.items():
            clean_analysis[task] = {k: v for k, v in analysis.items() if k != "raw_data"}

        # Convert numpy types before saving
        clean_analysis = self._convert_numpy_types(clean_analysis)
        report_clean = self._convert_numpy_types(report)

        # Save detailed analysis
        analysis_path = self.output_dir / "false_negative_intensity_analysis.json"
        with open(analysis_path, "w") as f:
            json.dump(clean_analysis, f, indent=2)

        # Save summary report
        report_path = self.output_dir / "intensity_analysis_report.json"
        with open(report_path, "w") as f:
            json.dump(report_clean, f, indent=2)

        print(f"✅ Results saved:")
        print(f"   📊 Detailed analysis: {analysis_path}")
        print(f"   📋 Summary report: {report_path}")

    def analyze(self):
        """Run complete false negative intensity analysis."""
        print(f"🚀 Starting False Negative Intensity Analysis")
        print(f"   🎯 Model: {self.architecture_name}")

        # 1. Load model and setup
        self.load_model_and_extract_tasks()
        self.setup_test_dataloader()

        # 2. Generate predictions with pre-computed intensities
        results = self.generate_predictions_with_intensities()

        # 3. Load optimal thresholds
        thresholds = self.load_optimal_thresholds()

        # 4. Extract event intensities from dataset
        intensities = self.extract_event_intensities_from_dataset(
            results["targets"], results["intensities"], results["sample_indices"]
        )

        # 5. Analyze false negatives
        analysis_results = self.analyze_false_negatives(
            results["predictions"], results["targets"], intensities, thresholds
        )

        # 6. Create visualizations
        self.create_intensity_visualizations(analysis_results)
        self.create_summary_boxplots(analysis_results)

        # 7. Generate report
        report = self.generate_intensity_report(analysis_results)

        # 8. Save results
        self.save_results(analysis_results, report)

        # 9. Print key findings
        self._print_key_findings(report)

        print("🎉 False Negative Intensity Analysis completed!")

    def _print_key_findings(self, report: Dict[str, Any]):
        """Print key findings to console."""
        print("\n" + "=" * 60)
        print(f"🔬 FALSE NEGATIVE INTENSITY ANALYSIS")
        print("=" * 60)

        for task, findings in report["key_findings"].items():
            print(f"\n📊 {task.upper()}:")
            
            intensity_diff = findings["intensity_difference"]
            print(f"   Event Intensity - TP: {intensity_diff['tp_mean']:.3f}, FN: {intensity_diff['fn_mean']:.3f}")
            print(f"   Difference: {intensity_diff['difference']:.3f} (Effect size: {intensity_diff['effect_size_cohens_d']:.3f})")
            
            pred_diff = findings["prediction_score_difference"]
            print(f"   Prediction Score - TP: {pred_diff['tp_pred_mean']:.3f}, FN: {pred_diff['fn_pred_mean']:.3f}")
            
            interp = findings["interpretation"]
            print(f"   Pattern: FN events have {interp['intensity_pattern']} intensity ({interp['effect_size_category']} effect)")
            
            if "statistical_significance" in findings:
                stat_test = findings["statistical_significance"]
                significance = "significant" if stat_test["significant"] else "not significant"
                print(f"   Statistical test: {significance} (p={stat_test['p_value']:.4f})")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Analyze False Negative intensity characteristics"
    )
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Path to model checkpoint",
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="data/datasets/multimodal/s1",
        help="Dataset directory containing test.h5",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        help="Output directory (default: model_dir/fn_intensity_analysis)",
    )

    args = parser.parse_args()

    # Handle path resolution
    data_dir = Path(args.data_dir)
    if not data_dir.exists():
        data_dir = Path("data/datasets/multimodal") / data_dir

    model_path = Path(args.model)
    if not (model_path.exists() and model_path.is_file()):
        model_path = Path("data/models/multimodal") / model_path
    if not (model_path.exists() and model_path.is_file()):
        model_path = model_path / "best_model.pt"

    # Set default output directory
    if args.output_dir is None:
        args.output_dir = Path(model_path).parent / "fn_intensity_analysis"

    print(f"🎯 False Negative Intensity Analysis Configuration:")
    print(f"   🤖 Model: {model_path}")
    print(f"   📊 Data: {data_dir}")
    print(f"   📁 Output: {args.output_dir}")

    # Initialize and run analyzer
    analyzer = FalseNegativeAnalyzer(
        model_path=model_path, data_dir=data_dir, output_dir=args.output_dir
    )

    analyzer.analyze()
    return True


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)