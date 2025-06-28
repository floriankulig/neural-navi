#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Single Model Evaluation Script
Evaluates a single multimodal architecture on test data with scientific metrics.
"""

import sys
import json
import argparse
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Tuple, Any
import torch
import torch.nn.functional as F
from sklearn.metrics import (
    precision_recall_curve,
    average_precision_score,
    roc_curve,
    auc,
    matthews_corrcoef,
    f1_score,
    balanced_accuracy_score,
    confusion_matrix,
    precision_score,
    recall_score,
    accuracy_score,
    brier_score_loss,
)
from sklearn.calibration import calibration_curve
from tqdm import tqdm


# Add project root to path for imports
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "training"))

# from utils.feature_config import
from utils.feature_config import (
    DETECTION_INPUT_DIM_PER_BOX,
    MAX_DETECTIONS_PER_FRAME,
    SEQUENCE_LENGTH,
    TELEMETRY_INPUT_DIM,
)
from model.factory import create_model_variant
from datasets.data_loaders import create_multimodal_dataloader


class ModelEvaluator:
    """
    Comprehensive evaluation of single multimodal architecture.
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

        print(f"🔬 Model Evaluator initialized")
        print(f"   📁 Model: {self.model_path}")
        print(f"   📊 Data: {self.data_dir}")
        print(f"   💾 Output: {self.output_dir}")
        print(f"   🔧 Device: {self.device}")

    def load_model_and_extract_tasks(self) -> Tuple[List[str], str]:
        """
        Load model checkpoint and extract tasks from model architecture.

        Returns:
            Tuple of (task_list, architecture_name)
        """
        print("🤖 Loading model checkpoint...")

        if not self.model_path.exists():
            raise FileNotFoundError(f"Model checkpoint not found: {self.model_path}")

        # Load checkpoint
        checkpoint = torch.load(self.model_path, map_location="cpu")

        # Extract architecture name
        architecture_name = checkpoint.get("arch_name", "unknown_architecture")

        # Extract tasks from model state dict
        tasks = []
        for key in checkpoint["model_state_dict"].keys():
            if "task_heads" in key and "weight" in key:
                # Format: "output_decoder.task_heads.brake_1s.0.weight"
                task_name = key.split("task_heads.")[1].split(".")[0]
                if task_name not in tasks:
                    tasks.append(task_name)

        if not tasks:
            raise ValueError("No tasks found in model checkpoint")

        tasks = sorted(tasks)  # Consistent ordering

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
            "dropout_prob": 0.15,
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
        """Setup test dataloader with same preprocessing as training."""
        print("📚 Setting up test dataloader...")

        self.test_loader = create_multimodal_dataloader(
            h5_file_path=str(self.data_dir / "test.h5"),
            batch_size=256,  # Large batch size for evaluation
            shuffle=False,  # Deterministic evaluation
            num_workers=8,
            pin_memory=True,
            target_horizons=self.tasks,
        )

        print(f"✅ Test dataloader ready: {len(self.test_loader)} batches")

    def generate_predictions(self) -> Dict[str, Dict[str, np.ndarray]]:
        """
        Generate predictions on entire test set.

        Returns:
            Dictionary with predictions and targets per task
        """
        print("🔮 Generating predictions on test set...")

        all_predictions = {task: [] for task in self.tasks}
        all_targets = {task: [] for task in self.tasks}

        with torch.no_grad():
            for batch in tqdm(self.test_loader, desc="Inference"):
                # Move data to device
                telemetry = batch["telemetry_seq"].to(self.device, non_blocking=True)
                detections = batch["detection_seq"].to(self.device, non_blocking=True)
                mask = batch["detection_mask"].to(self.device, non_blocking=True)

                # Forward pass
                predictions = self.model(telemetry, detections, mask)

                # Collect predictions and targets
                for task in self.tasks:
                    # Apply sigmoid to get probabilities
                    pred_probs = torch.sigmoid(predictions[task]).squeeze(-1)
                    all_predictions[task].append(pred_probs.cpu().numpy())
                    all_targets[task].append(batch["targets"][task].numpy())

        # Concatenate all batches
        results = {"predictions": {}, "targets": {}}

        for task in self.tasks:
            results["predictions"][task] = np.concatenate(all_predictions[task])
            results["targets"][task] = np.concatenate(all_targets[task])

        print(
            f"✅ Predictions generated for {len(results['targets'][self.tasks[0]])} samples"
        )
        return results

    def find_optimal_thresholds(
        self, predictions: Dict[str, np.ndarray], targets: Dict[str, np.ndarray]
    ) -> Dict[str, Dict[str, float]]:
        """
        Find optimal thresholds for each task based on different metrics.

        Returns:
            Dictionary with optimal thresholds per task and metric
        """
        print("🎯 Finding optimal thresholds...")

        optimal_thresholds = {}

        for task in self.tasks:
            y_true = targets[task]
            y_scores = predictions[task]

            # Generate candidate thresholds
            thresholds = np.linspace(0.1, 0.9, 81)  # Fine-grained search

            best_thresholds = {}
            best_scores = {}

            # Optimize for different metrics
            metrics_to_optimize = ["f1", "mcc", "balanced_accuracy"]

            for metric_name in metrics_to_optimize:
                best_score = -1
                best_threshold = 0.5

                for threshold in thresholds:
                    y_pred = (y_scores >= threshold).astype(int)

                    if metric_name == "f1":
                        score = f1_score(y_true, y_pred, zero_division=0)
                    elif metric_name == "mcc":
                        score = matthews_corrcoef(y_true, y_pred)
                    elif metric_name == "balanced_accuracy":
                        score = balanced_accuracy_score(y_true, y_pred)

                    if score > best_score:
                        best_score = score
                        best_threshold = threshold

                best_thresholds[metric_name] = best_threshold
                best_scores[metric_name] = best_score

            optimal_thresholds[task] = best_thresholds

            print(f"   📊 {task}:")
            for metric, threshold in best_thresholds.items():
                print(
                    f"      {metric}: {threshold:.3f} (score: {best_scores[metric]:.3f})"
                )

        return optimal_thresholds

    def calculate_comprehensive_metrics(
        self,
        predictions: Dict[str, np.ndarray],
        targets: Dict[str, np.ndarray],
        optimal_thresholds: Dict[str, Dict[str, float]],
    ) -> Dict[str, Dict[str, Any]]:
        """
        Calculate comprehensive evaluation metrics for each task.
        """
        print("📈 Calculating comprehensive metrics...")

        results = {}

        for task in self.tasks:
            y_true = targets[task]
            y_scores = predictions[task]

            task_results = {}

            # Threshold-independent metrics
            precision, recall, pr_thresholds = precision_recall_curve(y_true, y_scores)
            fpr, tpr, roc_thresholds = roc_curve(y_true, y_scores)

            task_results["pr_auc"] = average_precision_score(y_true, y_scores)
            task_results["roc_auc"] = auc(fpr, tpr)
            task_results["brier_score"] = brier_score_loss(y_true, y_scores)

            # Threshold-dependent metrics for each optimization criterion
            for threshold_type in ["f1", "mcc", "balanced_accuracy"]:
                threshold = optimal_thresholds[task][threshold_type]
                y_pred = (y_scores >= threshold).astype(int)

                # Calculate confusion matrices (raw and normalized)
                cm_raw = confusion_matrix(y_true, y_pred)
                cm_normalized_true = confusion_matrix(
                    y_true, y_pred, normalize="true"
                )  # Row-wise (Recall)
                cm_normalized_pred = confusion_matrix(
                    y_true, y_pred, normalize="pred"
                )  # Column-wise (Precision)
                cm_normalized_all = confusion_matrix(
                    y_true, y_pred, normalize="all"
                )  # Overall proportions

                metrics = {
                    "threshold": threshold,
                    "accuracy": accuracy_score(y_true, y_pred),
                    "precision": precision_score(y_true, y_pred, zero_division=0),
                    "recall": recall_score(y_true, y_pred, zero_division=0),
                    "f1": f1_score(y_true, y_pred, zero_division=0),
                    "mcc": matthews_corrcoef(y_true, y_pred),
                    "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
                    "confusion_matrix": {
                        "raw": cm_raw.tolist(),
                        "normalized_true": cm_normalized_true.tolist(),  # Shows recall per class
                        "normalized_pred": cm_normalized_pred.tolist(),  # Shows precision per class
                        "normalized_all": cm_normalized_all.tolist(),  # Shows overall proportions
                    },
                }

                task_results[f"metrics_{threshold_type}_optimal"] = metrics

            # Store curves for visualization
            task_results["curves"] = {
                "precision_recall": {
                    "precision": precision.tolist(),
                    "recall": recall.tolist(),
                },
                "roc": {"fpr": fpr.tolist(), "tpr": tpr.tolist()},
            }

            # Class distribution
            task_results["class_distribution"] = {
                "positive": int(y_true.sum()),
                "negative": int(len(y_true) - y_true.sum()),
                "positive_ratio": float(y_true.mean()),
            }

            results[task] = task_results

        print("✅ Metrics calculated for all tasks")
        return results

    def analyze_calibration(
        self, predictions: Dict[str, np.ndarray], targets: Dict[str, np.ndarray]
    ) -> Dict[str, Dict[str, Any]]:
        """
        Analyze prediction calibration using reliability diagrams.
        """
        print("🎯 Analyzing prediction calibration...")

        calibration_results = {}

        for task in self.tasks:
            y_true = targets[task]
            y_scores = predictions[task]

            # Calculate calibration curve
            fraction_of_positives, mean_predicted_value = calibration_curve(
                y_true, y_scores, n_bins=10, strategy="uniform"
            )

            # Calculate calibration metrics
            calibration_error = np.abs(
                fraction_of_positives - mean_predicted_value
            ).mean()

            calibration_results[task] = {
                "calibration_curve": {
                    "fraction_of_positives": fraction_of_positives.tolist(),
                    "mean_predicted_value": mean_predicted_value.tolist(),
                },
                "calibration_error": float(calibration_error),
                "prediction_histogram": {
                    "bins": np.histogram(y_scores, bins=20)[1].tolist(),
                    "counts": np.histogram(y_scores, bins=20)[0].tolist(),
                },
            }

        print("✅ Calibration analysis completed")
        return calibration_results

    def create_visualizations(
        self,
        predictions: Dict[str, np.ndarray],
        targets: Dict[str, np.ndarray],
        metrics: Dict[str, Dict[str, Any]],
        calibration: Dict[str, Dict[str, Any]],
    ):
        """Create comprehensive visualizations with improved layout."""
        print("📊 Creating visualizations...")

        # Set style
        plt.style.use("seaborn-v0_8")

        # Define distinct color palette for better visibility
        num_tasks = len(self.tasks)
        colors = plt.cm.Set1(np.linspace(0, 1, num_tasks))

        # Calculate layout requirements
        if num_tasks <= 4:
            total_rows = 2  # 1 row for main plots + 1 row for confusion matrices
            main_height_ratio = 1
            cm_height_ratio = 1
        elif num_tasks <= 6:
            total_rows = 3  # 1 row for main plots + 2 rows for confusion matrices
            main_height_ratio = 1
            cm_height_ratio = 0.8
        elif num_tasks <= 8:
            total_rows = 3  # 1 row for main plots + 2 rows for confusion matrices
            main_height_ratio = 1.2
            cm_height_ratio = 0.8
        else:
            total_rows = 4  # 1 row for main plots + 3 rows for confusion matrices
            main_height_ratio = 1.5
            cm_height_ratio = 0.7

        # Create figure with adaptive height
        fig_height = (
            8 + (total_rows - 2) * 3
        )  # Base height + extra for additional CM rows
        fig = plt.figure(figsize=(20, fig_height))

        # Create main grid based on total rows needed
        if total_rows == 2:
            gs = fig.add_gridspec(
                2,
                4,
                height_ratios=[main_height_ratio, cm_height_ratio],
                hspace=0.3,
                wspace=0.3,
            )
            cm_start_row = 1
        elif total_rows == 3:
            gs = fig.add_gridspec(
                3,
                4,
                height_ratios=[main_height_ratio, cm_height_ratio, cm_height_ratio],
                hspace=0.3,
                wspace=0.3,
            )
            cm_start_row = 1
        else:  # total_rows == 4
            gs = fig.add_gridspec(
                4,
                4,
                height_ratios=[
                    main_height_ratio,
                    cm_height_ratio,
                    cm_height_ratio,
                    cm_height_ratio,
                ],
                hspace=0.3,
                wspace=0.3,
            )
            cm_start_row = 1

        # === TOP ROW: Main Metrics (16:9 aspect ratio) ===

        # 1. PR Curves
        ax1 = fig.add_subplot(gs[0, 0])
        for i, task in enumerate(self.tasks):
            precision = metrics[task]["curves"]["precision_recall"]["precision"]
            recall = metrics[task]["curves"]["precision_recall"]["recall"]
            pr_auc = metrics[task]["pr_auc"]
            ax1.plot(
                recall,
                precision,
                label=f"{task} (AUC={pr_auc:.3f})",
                linewidth=2,
                color=colors[i],
            )
        ax1.set_xlabel("Recall", fontsize=9)
        ax1.set_ylabel("Precision", fontsize=9)
        ax1.set_title("Precision-Recall Curves", fontsize=10)
        ax1.legend(fontsize=6)
        ax1.grid(True, alpha=0.3)
        ax1.tick_params(labelsize=8)
        ax1.set_aspect(0.65, adjustable="box")

        # 2. ROC Curves
        ax2 = fig.add_subplot(gs[0, 1])
        for i, task in enumerate(self.tasks):
            fpr = metrics[task]["curves"]["roc"]["fpr"]
            tpr = metrics[task]["curves"]["roc"]["tpr"]
            roc_auc = metrics[task]["roc_auc"]
            ax2.plot(
                fpr,
                tpr,
                label=f"{task} (AUC={roc_auc:.3f})",
                linewidth=2,
                color=colors[i],
            )
        ax2.plot([0, 1], [0, 1], "k--", alpha=0.5)
        ax2.set_xlabel("False Positive Rate", fontsize=9)
        ax2.set_ylabel("True Positive Rate", fontsize=9)
        ax2.set_title("ROC Curves", fontsize=10)
        ax2.legend(fontsize=6)
        ax2.grid(True, alpha=0.3)
        ax2.tick_params(labelsize=8)
        ax2.set_aspect(0.65, adjustable="box")

        # 3. Prediction Score Distributions (same color per task, different styles)
        ax3 = fig.add_subplot(gs[0, 2])

        for i, task in enumerate(self.tasks):
            y_true = targets[task]
            y_scores = predictions[task]

            # Separate positive and negative samples
            pos_scores = y_scores[y_true == 1]
            neg_scores = y_scores[y_true == 0]

            # Use same color per task, different styles for pos/neg
            task_color = colors[i]

            # Positive: filled histogram
            ax3.hist(
                pos_scores,
                bins=30,
                alpha=0.6,
                label=f"{task} (Pos)",
                density=True,
                color=task_color,
                # linewidth=1,
            )

            # # Negative: dashed outline only
            ax3.hist(
                neg_scores,
                bins=30,
                alpha=0.4,
                label=f"{task} (Neg)",
                density=True,
                histtype="step",
                color=task_color,
                linewidth=1,
                linestyle="--",
            )

        ax3.set_xlabel("Prediction Score", fontsize=9)
        ax3.set_xlim(0, 1.01)
        ax3.set_ylabel("Density", fontsize=9)
        ax3.set_title("Score Distributions", fontsize=10)
        ax3.legend(fontsize=6, ncol=2)
        ax3.grid(True, alpha=0.3)
        ax3.tick_params(labelsize=8)
        # ax3.set_aspect(0.65, adjustable="box")

        # 4. Calibration Plots
        ax4 = fig.add_subplot(gs[0, 3])
        for i, task in enumerate(self.tasks):
            cal_data = calibration[task]["calibration_curve"]
            frac_pos = cal_data["fraction_of_positives"]
            mean_pred = cal_data["mean_predicted_value"]
            cal_error = calibration[task]["calibration_error"]

            ax4.plot(
                mean_pred,
                frac_pos,
                "o-",
                label=f"{task} (CE={cal_error:.3f})",
                linewidth=2,
                color=colors[i],
                markersize=3,
            )

        ax4.plot([0, 1], [0, 1], "k--", alpha=0.5, label="Perfect Calibration")
        ax4.set_xlabel("Mean Predicted Probability", fontsize=9)
        ax4.set_ylabel("Fraction of Positives", fontsize=9)
        ax4.set_title("Reliability Diagram", fontsize=10)
        ax4.legend(fontsize=6)
        ax4.grid(True, alpha=0.3)
        ax4.tick_params(labelsize=8)
        ax4.set_aspect(0.65, adjustable="box")

        # === BOTTOM AREA: Confusion Matrices (flexible multi-row layout) ===

        # Calculate optimal grid layout for confusion matrices
        if num_tasks <= 2:
            cm_rows, cm_cols = 1, num_tasks
        elif num_tasks <= 4:
            cm_rows, cm_cols = 1, num_tasks  # Keep single row for up to 4 tasks
        elif num_tasks <= 6:
            cm_rows, cm_cols = 2, 3  # 2 rows, 3 columns
        elif num_tasks <= 8:
            cm_rows, cm_cols = 2, 4  # 2 rows, 4 columns
        else:
            cm_rows, cm_cols = 3, 4  # 3 rows, 4 columns for many tasks

        # Create confusion matrix grid spanning multiple rows if needed
        if cm_rows == 1:
            gs_bottom = gs[cm_start_row, :].subgridspec(
                1, cm_cols, hspace=0.4, wspace=0.3
            )
        elif cm_rows == 2:
            gs_bottom = gs[cm_start_row : cm_start_row + 2, :].subgridspec(
                2, cm_cols, hspace=0.4, wspace=0.3
            )
        else:  # cm_rows == 3
            gs_bottom = gs[cm_start_row : cm_start_row + 3, :].subgridspec(
                3, cm_cols, hspace=0.4, wspace=0.3
            )

        for i, task in enumerate(self.tasks):
            row = i // cm_cols
            col = i % cm_cols

            ax_cm = fig.add_subplot(gs_bottom[row, col])
            cm_norm = np.array(
                metrics[task]["metrics_f1_optimal"]["confusion_matrix"][
                    "normalized_true"
                ]
            )

            sns.heatmap(
                cm_norm,
                annot=True,
                fmt=".2f",
                cmap="Blues",
                xticklabels=["Neg", "Pos"],
                yticklabels=["Neg", "Pos"],
                vmin=0,
                vmax=1,
                ax=ax_cm,
                cbar_kws={"shrink": 0.8},
                square=True,  # Force square cells
                annot_kws={"size": 9},  # Smaller annotation font
            )

            ax_cm.set_title(f"{task} - Confusion Matrix", fontsize=9)
            ax_cm.set_ylabel("True Label", fontsize=8)
            ax_cm.set_xlabel("Predicted Label", fontsize=8)
            ax_cm.tick_params(labelsize=7)

            # Ensure minimum 1:1 aspect ratio (never taller than wide)
            ax_cm.set_aspect(1.0, adjustable="box")

        plt.tight_layout()

        # Save plot with higher DPI for better quality
        viz_path = self.output_dir / "evaluation_plots.png"
        plt.savefig(viz_path, dpi=400, bbox_inches="tight")
        plt.close()

        print(f"✅ Visualizations saved to {viz_path}")

    def generate_summary_report(
        self, metrics: Dict[str, Dict[str, Any]], calibration: Dict[str, Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Generate concise summary report."""
        print("📋 Generating summary report...")

        summary = {
            "architecture": self.architecture_name,
            "tasks": self.tasks,
            "test_samples": len(
                list(metrics.values())[0]["curves"]["precision_recall"]["precision"]
            ),
            "summary_metrics": {},
            "detection_capability_analysis": {},
        }

        # Summary metrics per task
        for task in self.tasks:
            task_summary = {
                "class_distribution": metrics[task]["class_distribution"],
                "threshold_independent": {
                    "pr_auc": metrics[task]["pr_auc"],
                    "roc_auc": metrics[task]["roc_auc"],
                    "brier_score": metrics[task]["brier_score"],
                },
                "optimal_performance": {
                    "f1_optimal": {
                        "threshold": metrics[task]["metrics_f1_optimal"]["threshold"],
                        "f1": metrics[task]["metrics_f1_optimal"]["f1"],
                        "precision": metrics[task]["metrics_f1_optimal"]["precision"],
                        "recall": metrics[task]["metrics_f1_optimal"]["recall"],
                        "mcc": metrics[task]["metrics_f1_optimal"]["mcc"],
                    }
                },
                "calibration": {
                    "calibration_error": calibration[task]["calibration_error"]
                },
            }
            summary["summary_metrics"][task] = task_summary

            # Detection capability analysis
            pr_auc = metrics[task]["pr_auc"]
            pos_ratio = metrics[task]["class_distribution"]["positive_ratio"]

            # Random baseline for PR-AUC equals positive class ratio
            pr_auc_improvement = pr_auc / pos_ratio if pos_ratio > 0 else 0

            capability = {
                "has_detection_capability": bool(
                    pr_auc > pos_ratio * 1.1
                ),  # 10% better than random
                "pr_auc_vs_random_baseline": pr_auc_improvement,
                "performance_category": self._categorize_performance(pr_auc, pos_ratio),
            }
            summary["detection_capability_analysis"][task] = capability

        return summary

    def _categorize_performance(self, pr_auc: float, pos_ratio: float) -> str:
        """Categorize model performance level."""
        improvement = pr_auc / pos_ratio if pos_ratio > 0 else 0

        if improvement < 1.1:
            return "No Detection Capability"
        elif improvement < 2.0:
            return "Weak Detection Capability"
        elif improvement < 3.0:
            return "Moderate Detection Capability"
        else:
            return "Strong Detection Capability"

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
        self,
        metrics: Dict[str, Dict[str, Any]],
        calibration: Dict[str, Dict[str, Any]],
        summary: Dict[str, Any],
    ):
        """Save all results to files."""
        print("💾 Saving results...")

        # Convert numpy types before saving
        metrics_clean = self._convert_numpy_types(metrics)
        calibration_clean = self._convert_numpy_types(calibration)
        summary_clean = self._convert_numpy_types(summary)

        # Save detailed metrics
        metrics_path = self.output_dir / "detailed_metrics.json"
        with open(metrics_path, "w") as f:
            json.dump(metrics_clean, f, indent=2)

        # Save calibration results
        calibration_path = self.output_dir / "calibration_analysis.json"
        with open(calibration_path, "w") as f:
            json.dump(calibration_clean, f, indent=2)

        # Save summary report
        summary_path = self.output_dir / "evaluation_summary.json"
        with open(summary_path, "w") as f:
            json.dump(summary_clean, f, indent=2)

        print(f"✅ Results saved:")
        print(f"   📊 Detailed metrics: {metrics_path}")
        print(f"   🎯 Calibration: {calibration_path}")
        print(f"   📋 Summary: {summary_path}")

    def evaluate(self):
        """Run complete evaluation pipeline."""
        print(f"🚀 Starting evaluation of {self.model_path.name}")

        # 1. Load model and extract tasks
        self.load_model_and_extract_tasks()

        # 2. Setup test dataloader
        self.setup_test_dataloader()

        # 3. Generate predictions
        results = self.generate_predictions()

        # 4. Find optimal thresholds
        optimal_thresholds = self.find_optimal_thresholds(
            results["predictions"], results["targets"]
        )

        # 5. Calculate comprehensive metrics
        metrics = self.calculate_comprehensive_metrics(
            results["predictions"], results["targets"], optimal_thresholds
        )

        # 6. Analyze calibration
        calibration = self.analyze_calibration(
            results["predictions"], results["targets"]
        )

        # 7. Create visualizations
        self.create_visualizations(
            results["predictions"], results["targets"], metrics, calibration
        )

        # 8. Generate summary
        summary = self.generate_summary_report(metrics, calibration)

        # 9. Save results
        self.save_results(metrics, calibration, summary)

        # 10. Print key findings
        self._print_key_findings(summary)

        print("🎉 Evaluation completed successfully!")
        return summary

    def _print_key_findings(self, summary: Dict[str, Any]):
        """Print key findings to console."""
        print("\n" + "=" * 60)
        print(f"🔬 EVALUATION SUMMARY: {summary['architecture']}")
        print("=" * 60)

        for task in summary["tasks"]:
            task_data = summary["summary_metrics"][task]
            capability = summary["detection_capability_analysis"][task]

            print(f"\n📊 {task.upper()}:")
            print(
                f"   Class Distribution: {task_data['class_distribution']['positive_ratio']:.1%} positive"
            )
            print(f"   PR-AUC: {task_data['threshold_independent']['pr_auc']:.3f}")
            print(
                f"   Best F1: {task_data['optimal_performance']['f1_optimal']['f1']:.3f}"
            )
            print(f"   Detection Capability: {capability['performance_category']}")
            print(
                f"   vs Random Baseline: {capability['pr_auc_vs_random_baseline']:.1f}x improvement"
            )


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Evaluate single multimodal architecture"
    )
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Path to model checkpoint (e.g., data/models/multimodal/simple_concat_lstm/best_model.pt)",
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="s1",
        help="Dataset directory containing test.h5",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        help="Output directory (default: same as model directory)",
    )

    args = parser.parse_args()

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
        args.output_dir = Path(model_path).parent / "evaluation"

    print(f"🎯 Model Evaluation Configuration:")
    print(f"   🤖 Model: {model_path}")
    print(f"   📊 Data: {data_dir}")
    print(f"   📁 Output: {args.output_dir}")

    # Initialize and run evaluator
    evaluator = ModelEvaluator(
        model_path=model_path, data_dir=data_dir, output_dir=args.output_dir
    )

    summary = evaluator.evaluate()

    return summary is not None


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
