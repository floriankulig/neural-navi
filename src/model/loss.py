"""
Simplified Unified Focal Loss System for Neural-Navi
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict
from pathlib import Path
import sys

# Add project root to path for imports
script_dir = Path(__file__).parent
project_root = script_dir.parent.parent  # Go up two levels: model/ -> src/ -> root/
sys.path.insert(0, str(project_root))

# Import prediction tasks from config
from src.utils.feature_config import PREDICTION_TASKS

# ====================================
# CONFIGURATION CONSTANTS
# ====================================

# Task Importance Weights for Multi-Task Learning
TASK_WEIGHTS = {
    "brake_1s": 1.0,
    "brake_2s": 0.95,
    "coast_1s": 0.85,
    "coast_2s": 0.8,
}

# Task-specific Focal Loss Configuration
# Optimized per task based on:
# - Imbalance severity (brake events: 1:61 ratio, coast events: 1:11 ratio)
# - Safety criticality (brake = safety-critical, coast = efficiency)
# - Prediction horizon (1s = precise, 2s = longer horizon, less aggressive)
FOCAL_CONFIG = {
    "brake_1s": {"alpha": 0.2, "gamma": 3},  # Extreme imbalance, safety-critical
    "brake_2s": {"alpha": 0.2, "gamma": 2.75},  # High imbalance, safety-critical
    "coast_1s": {"alpha": 0.3, "gamma": 2},  # Moderate imbalance, efficiency
    "coast_2s": {"alpha": 0.3, "gamma": 1.75},  # Moderate imbalance, longer horizon
}

# Training Monitoring
LOG_PER_TASK_LOSSES = True
LOG_FOCAL_BEHAVIOR = True


# ====================================
# FOCAL LOSS IMPLEMENTATION
# ====================================


class FocalLoss(nn.Module):
    """
    Focal Loss for addressing class imbalance through adaptive example weighting.

    Formula: FL(p_t) = -α_t * (1 - p_t)^γ * log(p_t)

    Args:
        alpha: Class balancing weight (addresses frequency imbalance)
        gamma: Focusing parameter (addresses easy vs hard example imbalance)
    """

    def __init__(self, alpha: float, gamma: float):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Compute focal loss for binary classification.

        Args:
            logits: Raw predictions from model (before sigmoid) [batch_size]
            targets: Binary ground truth labels (0 or 1) [batch_size]

        Returns:
            Scalar focal loss value
        """
        # Convert targets to float at the beginning
        targets = targets.float()
        
        # Convert logits to probabilities
        probs = torch.sigmoid(logits)

        # Calculate base cross-entropy loss (per sample, no reduction)
        ce_loss = F.binary_cross_entropy_with_logits(
            logits, targets, reduction="none"  # targets.float() nicht mehr nötig
        )

        # Calculate p_t: probability of the correct class
        p_t = probs * targets + (1 - probs) * (1 - targets)

        # Apply focal weight: (1 - p_t)^gamma
        focal_weight = (1 - p_t) ** self.gamma

        # Apply class balancing weight alpha
        alpha_weight = self.alpha * targets + (1 - self.alpha) * (1 - targets)

        # Combine all components: α * (1-p_t)^γ * CE_loss
        focal_loss = alpha_weight * focal_weight * ce_loss

        return focal_loss.mean()


# ====================================
# UNIFIED MULTI-TASK FOCAL LOSS
# ====================================


class UnifiedMultiTaskFocalLoss(nn.Module):
    """
    Unified multi-task loss using focal loss for all tasks.
    All tasks use the same focal loss configuration for consistency.
    """

    def __init__(self):
        super().__init__()

        # Create focal loss criterion for each task
        self.criterions = nn.ModuleDict()

        for task_name in PREDICTION_TASKS:
            task_config = FOCAL_CONFIG.get(task_name, {"alpha": 0.25, "gamma": 2.0})
            self.criterions[task_name] = FocalLoss(
                alpha=task_config["alpha"], gamma=task_config["gamma"]
            )

            if LOG_FOCAL_BEHAVIOR:
                print(
                    f"📊 {task_name}: FocalLoss(α={task_config['alpha']}, γ={task_config['gamma']})"
                )

    def forward(
        self, predictions: Dict[str, torch.Tensor], targets: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """
        Compute multi-task focal loss.

        Args:
            predictions: Model predictions per task {task_name: logits}
            targets: Ground truth per task {task_name: binary_labels}

        Returns:
            Dictionary with per-task losses and total weighted loss
        """
        losses = {}
        total_loss = 0.0

        # Compute loss for each active task
        for task_name in PREDICTION_TASKS:
            if task_name in predictions and task_name in targets:
                # Get predictions and targets for this task
                pred = predictions[task_name].squeeze(-1)
                target = targets[task_name]

                # Compute focal loss for this task
                task_loss = self.criterions[task_name](pred, target)
                losses[f"focal_loss_{task_name}"] = task_loss

                # Weight by task importance and add to total
                task_weight = TASK_WEIGHTS.get(task_name, 1.0)
                weighted_loss = task_weight * task_loss
                total_loss += weighted_loss

                if LOG_PER_TASK_LOSSES:
                    losses[f"weighted_loss_{task_name}"] = weighted_loss

        # Store total loss
        losses["loss_total"] = total_loss

        return losses

    def analyze_batch_behavior(
        self, predictions: Dict[str, torch.Tensor], targets: Dict[str, torch.Tensor]
    ) -> Dict:
        """
        Analyze focal loss behavior on current batch for debugging/monitoring.
        """
        analysis = {}

        for task_name in PREDICTION_TASKS:
            if task_name not in predictions or task_name not in targets:
                continue

            pred = predictions[task_name].squeeze(-1)
            target = targets[task_name]

            # Get probabilities and separate by class
            probs = torch.sigmoid(pred)
            pos_mask = target == 1
            neg_mask = target == 0

            task_analysis = {
                "total_samples": len(target),
                "positive_samples": pos_mask.sum().item(),
                "negative_samples": neg_mask.sum().item(),
            }

            # Analyze probability distributions
            if pos_mask.any():
                pos_probs = probs[pos_mask]
                task_analysis.update(
                    {
                        "pos_prob_mean": pos_probs.mean().item(),
                        "pos_confident_count": (pos_probs > 0.8).sum().item(),
                        "pos_uncertain_count": (pos_probs < 0.6).sum().item(),
                    }
                )

            if neg_mask.any():
                neg_probs = probs[neg_mask]
                task_analysis.update(
                    {
                        "neg_prob_mean": neg_probs.mean().item(),
                        "neg_confident_count": (neg_probs < 0.2).sum().item(),
                        "neg_uncertain_count": (neg_probs > 0.4).sum().item(),
                    }
                )

            analysis[task_name] = task_analysis

        return analysis


# ====================================
# FACTORY FUNCTION
# ====================================


def create_unified_focal_loss() -> UnifiedMultiTaskFocalLoss:
    """
    Factory function to create the unified focal loss system.

    Returns:
        Configured multi-task focal loss function
    """
    if LOG_FOCAL_BEHAVIOR:
        print("🎯 Neural-Navi Unified Focal Loss Configuration:")
        print(f"   📋 Active Tasks: {PREDICTION_TASKS}")
        print(f"   ⚖️ Task Weights: {TASK_WEIGHTS}")
        print(f"   🔥 Focal Loss Settings:")
        for task in PREDICTION_TASKS:
            config = FOCAL_CONFIG.get(task, {"alpha": 0.25, "gamma": 2.0})
            weight = TASK_WEIGHTS.get(task, 1.0)
            print(
                f"      {task}: α={config['alpha']}, γ={config['gamma']}, task_weight={weight}"
            )

    return UnifiedMultiTaskFocalLoss()


# ====================================
# TESTING
# ====================================

if __name__ == "__main__":
    print("🧪 Testing Unified Focal Loss System")
    print("=" * 50)

    # Test loss creation
    loss_fn = create_unified_focal_loss()

    # Test with dummy data
    batch_size = 16
    dummy_predictions = {task: torch.randn(batch_size, 1) for task in PREDICTION_TASKS}
    dummy_targets = {
        task: torch.randint(0, 2, (batch_size,)) for task in PREDICTION_TASKS
    }

    # Compute loss
    losses = loss_fn(dummy_predictions, dummy_targets)

    print(f"\n📊 Test Results:")
    for loss_name, loss_value in losses.items():
        if isinstance(loss_value, torch.Tensor):
            print(f"   {loss_name}: {loss_value.item():.4f}")
        else:
            print(f"   {loss_name}: {loss_value}")
