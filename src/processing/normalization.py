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
GEAR_CLASSES = 6  # 0=neutral, 1-5=gears


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
        # Clamp gear values to valid range [0, 5]
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
        gear_normalized = gear_values / (GEAR_CLASSES - 1)  # [0, 5] -> [0, 1]
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
    n_vehicle_classes: int = None,
) -> torch.Tensor:
    """
    Normalize detection features to [0, 1] range.

    Args:
        detection_seq: Detection tensor (..., max_detections, detection_features)
                      Supports both formats:
                      - With class_id: [class_id, confidence, x1, y1, x2, y2, area] (7 features)
                      - Without class_id: [confidence, x1, y1, x2, y2, area] (6 features)
        detection_mask: Boolean mask (..., max_detections) indicating valid detections
        img_width: Image width for coordinate normalization
        img_height: Image height for coordinate normalization
        in_place: Whether to modify input tensor in-place
        n_vehicle_classes: Number of vehicle classes for one-hot encoding (if class_id present)

    Returns:
        Normalized detection tensor. If class_id was present, it's converted to one-hot encoding.
        Final format: [confidence, x1, y1, x2, y2, area, class_0, class_1, ..., class_n] or
                     [confidence, x1, y1, x2, y2, area] depending on input format
    """
    if not in_place:
        detection_seq = detection_seq.clone()

    # Only process if there are valid detections
    if not detection_mask.any():
        return detection_seq

    # Get dimensions
    *batch_dims, max_detections, feature_dim = detection_seq.shape

    # Expected feature order after class_id removal: [confidence, x1, y1, x2, y2, area]
    if feature_dim != 6:
        print(
            f"WARNING: Expected 6 features without classes after class_id removal, got {feature_dim}"
        )
        if feature_dim == 7 and n_vehicle_classes is None:
            print(
                "INFO: Input seems to still contain class_id but no number of classes to onehot-encode specified - this should be handled earlier"
            )

    # If class_id is present, convert to one-hot encoding
    if feature_dim == 7 and n_vehicle_classes is not None:
        # Extract class_id and confidence
        class_ids = detection_seq[..., 0].long()
        confidence = detection_seq[..., 1]
        coordinates = detection_seq[..., 2:6]  # [x1, y1, x2, y2]
        area = detection_seq[..., 6]  # area
        # Create one-hot encoding for class_id
        class_onehot = torch.zeros(
            *batch_dims,
            max_detections,
            n_vehicle_classes,
            dtype=torch.float32,
            device=detection_seq.device,
        )
        class_onehot.scatter_(-1, class_ids.unsqueeze(-1), 1)
        # Concatenate confidence, coordinates, area and one-hot class
        detection_seq = torch.cat(
            [class_onehot, confidence.unsqueeze(-1), coordinates, area.unsqueeze(-1)],
            dim=-1,
        )

    # If no n_classes for one-hot but 7 features, we just leave it as it is but start the normalization
    # at the confidence (then 1 idx)
    CONF_IDX = n_vehicle_classes if n_vehicle_classes else 1 if feature_dim == 7 else 0

    # 1. Confidence normalization (feature CONF_IDX) - should already be [0, 1]
    detection_seq[..., CONF_IDX] = torch.clamp(detection_seq[..., CONF_IDX], 0, 1)

    # 2. Coordinate normalization - only apply to valid coordinates
    if not (img_width > 0 and img_height > 0):
        return

    # X coordinates (features 1 and 3: x1, x2)
    detection_seq[..., CONF_IDX + 1] = (
        detection_seq[..., CONF_IDX + 1] / img_width
    )  # x1
    if feature_dim > 3:
        detection_seq[..., CONF_IDX + 3] = (
            detection_seq[..., CONF_IDX + 3] / img_width
        )  # x2

    # Y coordinates (features 2 and 4: y1, y2)
    if feature_dim > 2:
        detection_seq[..., CONF_IDX + 2] = (
            detection_seq[..., CONF_IDX + 2] / img_height
        )  # y1
    if feature_dim > 4:
        detection_seq[..., CONF_IDX + 4] = (
            detection_seq[..., CONF_IDX + 4] / img_height
        )  # y2

    # Area normalization (feature 5)
    if feature_dim > 5:
        total_image_area = img_width * img_height
        if total_image_area > 0:
            detection_seq[..., CONF_IDX + 5] = (
                detection_seq[..., CONF_IDX + 5] / total_image_area
            )

    # 3. Clamp all coordinates and area to [0, 1] to handle outliers
    detection_seq[..., :] = torch.clamp(detection_seq[..., :], 0, 1)

    # 4. Zero out invalid detections (this preserves padding)
    # Create expanded mask for broadcasting: (..., max_detections, 1)
    valid_mask = detection_mask.unsqueeze(-1)
    # The mask broadcasting: (..., max_detections, 1) * (..., max_detections, features)
    detection_seq = detection_seq * valid_mask

    return detection_seq


# ====================================
# EXAMPLE USAGE AND TESTING
# ====================================

if __name__ == "__main__":
    print("🧪 Testing telemetry normalization with GEAR one-hot encoding...")

    # Create sample data with realistic ranges
    batch_size, seq_len = 4, 10
    sample_telemetry_base = torch.tensor(
        [
            [120.0, 2500.0, 45.0, 35.0, 4.0],  # Highway driving, Gear 4
            [80.0, 1800.0, 20.0, 25.0, 3.0],  # City driving, Gear 3
            [200.0, 4500.0, 80.0, 60.0, 5.0],  # Fast driving, Gear 5
            [0.0, 800.0, 0.0, 5.0, 0.0],  # Idle/neutral, Gear 0
        ]
    )  # Shape: (4, 5)

    # Reshape to (batch_size, 1, features) then expand to (batch_size, seq_len, features)
    sample_telemetry = sample_telemetry_base.unsqueeze(1).expand(
        batch_size, seq_len, -1
    )

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

    print("\n🧪 Testing detection normalization...")

    # --- Test Case 1: With class_id and n_vehicle_classes ---
    print("\n--- Test Case 1: With class_id and n_vehicle_classes ---")
    batch_size, max_dets, num_features_with_class = 2, 3, 7
    img_w, img_h = 1920, 575
    n_classes = 3

    sample_detections_with_class = torch.tensor(
        [
            # Batch 1
            [
                [0.0, 0.9, 100.0, 50.0, 200.0, 150.0, 10000.0],  # Valid, class 0
                [1.0, 0.8, 300.0, 100.0, 400.0, 250.0, 15000.0],  # Valid, class 1
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],  # Padding
            ],
            # Batch 2
            [
                [2.0, 0.95, 500.0, 200.0, 700.0, 400.0, 40000.0],  # Valid, class 2
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],  # Padding
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],  # Padding
            ],
        ],
        dtype=torch.float32,
    )

    sample_mask_with_class = torch.tensor(
        [[True, True, False], [True, False, False]], dtype=torch.bool
    )

    print("Original detections (with class_id):")
    print(f"  Shape: {sample_detections_with_class.shape}")
    print(f"  Sample values (Batch 0, Det 0): {sample_detections_with_class[0, 0]}")

    normalized_detections_onehot = normalize_detection_features(
        sample_detections_with_class,
        sample_mask_with_class,
        img_width=img_w,
        img_height=img_h,
        n_vehicle_classes=n_classes,
    )
    print(f"\nNormalized detections (with one-hot class):")
    # Expected: n_classes + 1 (conf) + 4 (coords) + 1 (area) = 3 + 1 + 4 + 1 = 9 features
    print(f"  Shape: {normalized_detections_onehot.shape}")
    print(
        f"  Sample values (Batch 0, Det 0): {normalized_detections_onehot[0, 0]}"
    )  # Check one-hot, conf, coords, area
    print(f"  Sample values (Batch 0, Det 1): {normalized_detections_onehot[0, 1]}")
    print(
        f"  Padding check (Batch 0, Det 2): {normalized_detections_onehot[0, 2]}"
    )  # Should be all zeros

    # --- Test Case 2: With class_id but n_vehicle_classes = None ---
    print("\n--- Test Case 2: With class_id but n_vehicle_classes = None ---")
    # Using the same sample_detections_with_class
    normalized_detections_no_onehot = normalize_detection_features(
        sample_detections_with_class.clone(),  # Use clone to avoid in-place modification issues
        sample_mask_with_class,
        img_width=img_w,
        img_height=img_h,
        n_vehicle_classes=None,
    )
    print(f"\nNormalized detections (class_id as is, not one-hot):")
    print(f"  Shape: {normalized_detections_no_onehot.shape}")  # Should be (2,3,7)
    print(
        f"  Sample values (Batch 0, Det 0): {normalized_detections_no_onehot[0, 0]}"
    )  # Class ID should be 0.0, conf, coords, area normalized
    print(
        f"  Padding check (Batch 0, Det 2): {normalized_detections_no_onehot[0, 2]}"
    )  # Should be all zeros

    # --- Test Case 3: Without class_id (6 features) ---
    print("\n--- Test Case 3: Without class_id (6 features) ---")
    num_features_no_class = 6
    sample_detections_no_class = torch.tensor(
        [
            # Batch 1
            [
                [0.9, 100.0, 50.0, 200.0, 150.0, 10000.0],  # Valid
                [0.8, 300.0, 100.0, 400.0, 250.0, 15000.0],  # Valid
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],  # Padding
            ],
        ],
        dtype=torch.float32,
    )  # Shape (1, 3, 6)

    sample_mask_no_class = torch.tensor([[True, True, False]], dtype=torch.bool)

    print("Original detections (without class_id):")
    print(f"  Shape: {sample_detections_no_class.shape}")
    print(f"  Sample values (Batch 0, Det 0): {sample_detections_no_class[0, 0]}")

    normalized_detections_no_class_input = normalize_detection_features(
        sample_detections_no_class,
        sample_mask_no_class,
        img_width=img_w,
        img_height=img_h,
        n_vehicle_classes=None,  # n_vehicle_classes is irrelevant here
    )
    print(f"\nNormalized detections (input had no class_id):")
    print(f"  Shape: {normalized_detections_no_class_input.shape}")  # Should be (1,3,6)
    print(
        f"  Sample values (Batch 0, Det 0): {normalized_detections_no_class_input[0, 0]}"
    )
    print(
        f"  Padding check (Batch 0, Det 2): {normalized_detections_no_class_input[0, 2]}"
    )

    print("\n✅ Detection normalization test completed!")
