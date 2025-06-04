"""
Unified normalization utilities for telemetry and detection features.
"""

import torch
import numpy as np
from typing import Dict, Tuple, Optional


# ====================================
# TELEMETRY NORMALIZATION
# ====================================

# Telemetry feature ranges for German highway driving
TELEMETRY_RANGES = {
    "SPEED": (0.0, 150.0),  # km/h - German Autobahn max
    "RPM": (650.0, 4500.0),  # RPM - typical passenger car range
    "ACCELERATOR_POS_D": (0.0, 100.0),  # % - already normalized
    "ENGINE_LOAD": (0.0, 100.0),  # % - already normalized
}

# Feature names in order (GEAR handled separately with one-hot)
CONTINUOUS_TELEMETRY_FEATURES = ["SPEED", "RPM", "ACCELERATOR_POS_D", "ENGINE_LOAD"]
GEAR_CLASSES = 7  # 0=neutral, 1-6=gears


def normalize_telemetry_features(
    telemetry_seq: torch.Tensor,
    feature_ranges: Dict[str, Tuple[float, float]] = None,
    use_gear_onehot: bool = True,
    in_place: bool = False,
) -> torch.Tensor:
    """
    Normalize telemetry features with proper GEAR one-hot encoding.

    Args:
        telemetry_seq: Telemetry tensor of shape (..., 5)
                      Features: [SPEED, RPM, ACCELERATOR_POS_D, ENGINE_LOAD, GEAR]
        feature_ranges: Custom ranges per feature
        use_gear_onehot: Whether to one-hot encode GEAR (recommended)
        in_place: Whether to modify input tensor in-place

    Returns:
        Normalized tensor with shape (..., 4) or (..., 10) depending on gear encoding
        - Without one-hot: [norm_speed, norm_rpm, norm_accel, norm_load]
        - With one-hot: [norm_speed, norm_rpm, norm_accel, norm_load, gear_0, gear_1, ..., gear_6]
    """
    if feature_ranges is None:
        feature_ranges = TELEMETRY_RANGES

    if not in_place:
        telemetry_seq = telemetry_seq.clone()

    # Extract continuous features (first 4) and gear (last)
    continuous_features = telemetry_seq[
        ..., :4
    ]  # [SPEED, RPM, ACCELERATOR_POS_D, ENGINE_LOAD]
    gear_values = telemetry_seq[..., 4]  # [GEAR]

    # Normalize continuous features
    for i, feature_name in enumerate(CONTINUOUS_TELEMETRY_FEATURES):
        if feature_name in feature_ranges:
            min_val, max_val = feature_ranges[feature_name]

            # Min-Max normalization: (x - min) / (max - min)
            continuous_features[..., i] = (continuous_features[..., i] - min_val) / (
                max_val - min_val
            )

            # Clamp to [0, 1] to handle outliers
            continuous_features[..., i] = torch.clamp(continuous_features[..., i], 0, 1)

    if use_gear_onehot:
        # One-hot encode GEAR values
        # Clamp gear values to valid range [0, 6]
        gear_values = torch.clamp(gear_values.long(), 0, GEAR_CLASSES - 1)

        # Create one-hot encoding
        original_shape = gear_values.shape
        gear_onehot = torch.zeros(
            *original_shape,
            GEAR_CLASSES,
            dtype=torch.float32,
            device=gear_values.device,
        )
        gear_onehot.scatter_(-1, gear_values.unsqueeze(-1), 1)

        # Concatenate continuous features with one-hot gear
        normalized_telemetry = torch.cat([continuous_features, gear_onehot], dim=-1)
    else:
        # Just normalize gear as continuous (not recommended)
        gear_normalized = gear_values / (GEAR_CLASSES - 1)  # [0, 6] -> [0, 1]
        gear_normalized = torch.clamp(gear_normalized, 0, 1)

        normalized_telemetry = torch.cat(
            [continuous_features, gear_normalized.unsqueeze(-1)], dim=-1
        )

    return normalized_telemetry


def get_telemetry_input_dim(use_gear_onehot: bool = True) -> int:
    """Get the input dimension for telemetry after normalization."""
    continuous_dim = len(CONTINUOUS_TELEMETRY_FEATURES)  # 4
    gear_dim = GEAR_CLASSES if use_gear_onehot else 1  # 7 or 1
    return continuous_dim + gear_dim


# ====================================
# DETECTION NORMALIZATION
# ====================================


def normalize_detection_features(
    detection_seq: torch.Tensor,
    detection_mask: torch.Tensor,
    img_width: int = 1920,
    img_height: int = 575,
    in_place: bool = False,
) -> torch.Tensor:
    """
    Normalize detection features to [0, 1] range.

    Expected input format: [confidence, x1, y1, x2, y2, area]
    """
    if not in_place:
        detection_seq = detection_seq.clone()

    # Get valid detections mask
    valid_mask = detection_mask.unsqueeze(-1)

    # Only process if there are detections
    has_detections = detection_mask.any()
    if not has_detections:
        return detection_seq

    # Confidence (feature 0) - should already be [0, 1]
    detection_seq[..., 0] = torch.clamp(detection_seq[..., 0], 0, 1)

    # X coordinates (features 1 and 3: x1, x2)
    detection_seq[..., 1] = detection_seq[..., 1] / img_width  # x1
    detection_seq[..., 3] = detection_seq[..., 3] / img_width  # x2

    # Y coordinates (features 2 and 4: y1, y2)
    detection_seq[..., 2] = detection_seq[..., 2] / img_height  # y1
    detection_seq[..., 4] = detection_seq[..., 4] / img_height  # y2

    # Area (feature 5)
    total_image_area = img_width * img_height
    detection_seq[..., 5] = detection_seq[..., 5] / total_image_area

    # Clamp all coordinates and area to [0, 1]
    detection_seq[..., 1:] = torch.clamp(detection_seq[..., 1:], 0, 1)

    # Zero out invalid detections
    detection_seq = detection_seq * valid_mask

    return detection_seq


# ====================================
# VALIDATION UTILITIES
# ====================================


def validate_normalized_features(
    telemetry_seq: torch.Tensor,
    detection_seq: torch.Tensor,
    detection_mask: torch.Tensor,
    name: str = "features",
) -> bool:
    """
    Validate that all features are properly normalized.

    Returns:
        True if all values are in valid ranges
    """
    all_valid = True

    # Check telemetry features
    if telemetry_seq.numel() > 0:
        tel_min = telemetry_seq.min().item()
        tel_max = telemetry_seq.max().item()

        if tel_min < -0.01 or tel_max > 1.01:  # Small tolerance
            print(f"⚠️ {name} telemetry outside [0,1]: [{tel_min:.3f}, {tel_max:.3f}]")
            all_valid = False

    # Check detection features (only valid detections)
    if detection_seq.numel() > 0 and detection_mask.any():
        valid_detections = detection_seq[detection_mask]
        det_min = valid_detections.min().item()
        det_max = valid_detections.max().item()

        if det_min < -0.01 or det_max > 1.01:
            print(f"⚠️ {name} detections outside [0,1]: [{det_min:.3f}, {det_max:.3f}]")
            all_valid = False

    # Check for NaN/Inf
    if torch.isnan(telemetry_seq).any() or torch.isnan(detection_seq).any():
        print(f"⚠️ NaN values detected in {name}")
        all_valid = False

    if torch.isinf(telemetry_seq).any() or torch.isinf(detection_seq).any():
        print(f"⚠️ Inf values detected in {name}")
        all_valid = False

    return all_valid


def get_feature_statistics(
    telemetry_seq: torch.Tensor,
    detection_seq: torch.Tensor,
    detection_mask: torch.Tensor,
    use_gear_onehot: bool = True,
) -> Dict:
    """
    Get comprehensive statistics about all features.
    """
    stats = {"telemetry": {}, "detections": {}, "summary": {}}

    # Telemetry statistics
    continuous_features = CONTINUOUS_TELEMETRY_FEATURES
    for i, feature_name in enumerate(continuous_features):
        feature_values = telemetry_seq[..., i]
        stats["telemetry"][feature_name] = {
            "min": feature_values.min().item(),
            "max": feature_values.max().item(),
            "mean": feature_values.mean().item(),
            "std": feature_values.std().item(),
        }

    # Gear statistics
    if use_gear_onehot:
        gear_start_idx = len(continuous_features)
        gear_onehot = telemetry_seq[..., gear_start_idx : gear_start_idx + GEAR_CLASSES]
        gear_distribution = gear_onehot.sum(
            dim=tuple(range(len(gear_onehot.shape) - 1))
        )
        stats["telemetry"]["GEAR_distribution"] = {
            f"gear_{i}": gear_distribution[i].item() for i in range(GEAR_CLASSES)
        }
    else:
        gear_values = telemetry_seq[..., -1]
        stats["telemetry"]["GEAR"] = {
            "min": gear_values.min().item(),
            "max": gear_values.max().item(),
            "mean": gear_values.mean().item(),
            "std": gear_values.std().item(),
        }

    # Detection statistics (only valid detections)
    if detection_mask.any():
        valid_detections = detection_seq[detection_mask]
        detection_feature_names = ["confidence", "x1", "y1", "x2", "y2", "area"]

        for i, feature_name in enumerate(detection_feature_names):
            if valid_detections.numel() > 0:
                feature_values = valid_detections[:, i]
                stats["detections"][feature_name] = {
                    "min": feature_values.min().item(),
                    "max": feature_values.max().item(),
                    "mean": feature_values.mean().item(),
                    "std": feature_values.std().item(),
                }

    # Summary
    stats["summary"] = {
        "telemetry_shape": list(telemetry_seq.shape),
        "detection_shape": list(detection_seq.shape),
        "use_gear_onehot": use_gear_onehot,
        "telemetry_input_dim": get_telemetry_input_dim(use_gear_onehot),
    }

    return stats


# ====================================
# EXAMPLE USAGE AND TESTING
# ====================================

if __name__ == "__main__":
    print("🧪 Testing telemetry normalization with GEAR one-hot encoding...")

    # Create sample data with realistic ranges
    batch_size, seq_len = 4, 10
    sample_telemetry = torch.tensor(
        [
            [120.0, 2500.0, 45.0, 35.0, 4.0],  # Highway driving, Gear 4
            [80.0, 1800.0, 20.0, 25.0, 3.0],  # City driving, Gear 3
            [200.0, 4500.0, 80.0, 60.0, 5.0],  # Fast driving, Gear 5
            [0.0, 800.0, 0.0, 5.0, 0.0],  # Idle/neutral, Gear 0
        ]
    ).expand(batch_size, seq_len, -1)

    print("Original telemetry sample:")
    print(f"  Shape: {sample_telemetry.shape}")
    print(f"  Sample values: {sample_telemetry[0, 0]}")

    # Test with one-hot encoding
    normalized_onehot = normalize_telemetry_features(
        sample_telemetry, use_gear_onehot=True
    )
    print(f"\nNormalized with one-hot GEAR:")
    print(f"  Shape: {normalized_onehot.shape}")
    print(f"  Continuous features [0:4]: {normalized_onehot[0, 0, :4]}")
    print(f"  GEAR one-hot [4:11]: {normalized_onehot[0, 0, 4:]}")

    # Test without one-hot encoding
    normalized_continuous = normalize_telemetry_features(
        sample_telemetry, use_gear_onehot=False
    )
    print(f"\nNormalized without one-hot GEAR:")
    print(f"  Shape: {normalized_continuous.shape}")
    print(f"  All features: {normalized_continuous[0, 0]}")

    # Test input dimensions
    print(f"\nInput dimensions:")
    print(f"  With one-hot GEAR: {get_telemetry_input_dim(use_gear_onehot=True)}")
    print(f"  Without one-hot GEAR: {get_telemetry_input_dim(use_gear_onehot=False)}")

    print("\n✅ Telemetry normalization test completed!")
