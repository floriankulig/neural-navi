"""
Detection utilities for normalization and preprocessing.
Used both in training and inference pipelines.
"""

import torch
import numpy as np
from typing import Optional, Tuple


def normalize_detection_features(
    detection_seq: torch.Tensor, 
    detection_mask: torch.Tensor,
    img_width: int = 1920,
    img_height: int = 575,  
    in_place: bool = False
) -> torch.Tensor:
    """
    Normalize detection features to [0, 1] range for stable training.
    
    Expected input format: [confidence, x1, y1, x2, y2, area]
    - confidence: already [0, 1]
    - x1, x2: absolute pixel coordinates, normalize by img_width
    - y1, y2: absolute pixel coordinates, normalize by img_height  
    - area: absolute pixel area, normalize by (img_width * img_height)
    
    Args:
        detection_seq: Detection tensor of shape (..., max_detections, 6)
        detection_mask: Boolean mask of shape (..., max_detections)
        img_width: Image width for normalization
        img_height: Image height for normalization (consider ROI)
        in_place: Whether to modify input tensor in-place
        
    Returns:
        Normalized detection tensor
    """
    if not in_place:
        detection_seq = detection_seq.clone()
    
    # Get valid detections mask
    valid_mask = detection_mask.unsqueeze(-1)  # Add feature dimension
    
    # Only process non-zero detections to avoid unnecessary computation
    has_detections = detection_mask.any()
    if not has_detections:
        return detection_seq
    
    # Confidence (feature 0) - should already be [0, 1], just clamp
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
    
    # Clamp all coordinates and area to [0, 1] to handle edge cases
    detection_seq[..., 1:] = torch.clamp(detection_seq[..., 1:], 0, 1)
    
    # Zero out invalid detections  
    detection_seq = detection_seq * valid_mask
    
    return detection_seq


def denormalize_detection_features(
    normalized_detections: torch.Tensor,
    img_width: int = 1920,
    img_height: int = 575
) -> torch.Tensor:
    """
    Denormalize detection features back to absolute coordinates.
    Useful for inference and visualization.
    
    Args:
        normalized_detections: Normalized detection tensor
        img_width: Image width for denormalization
        img_height: Image height for denormalization
        
    Returns:
        Denormalized detection tensor with absolute coordinates
    """
    denormalized = normalized_detections.clone()
    
    # X coordinates (features 1 and 3: x1, x2)
    denormalized[..., 1] = denormalized[..., 1] * img_width   # x1
    denormalized[..., 3] = denormalized[..., 3] * img_width   # x2
    
    # Y coordinates (features 2 and 4: y1, y2)
    denormalized[..., 2] = denormalized[..., 2] * img_height  # y1  
    denormalized[..., 4] = denormalized[..., 4] * img_height  # y2
    
    # Area (feature 5)
    total_image_area = img_width * img_height
    denormalized[..., 5] = denormalized[..., 5] * total_image_area
    
    return denormalized


def validate_detection_range(detection_seq: torch.Tensor, name: str = "detections") -> bool:
    """
    Validate that detection features are in expected ranges.
    
    Args:
        detection_seq: Detection tensor to validate
        name: Name for logging
        
    Returns:
        True if all values are in valid ranges
    """
    if detection_seq.numel() == 0:
        return True
    
    # Check for NaN/Inf
    if torch.isnan(detection_seq).any():
        print(f"⚠️ NaN values detected in {name}")
        return False
    
    if torch.isinf(detection_seq).any():
        print(f"⚠️ Inf values detected in {name}")
        return False
    
    # Check ranges for normalized features
    min_val = detection_seq.min().item()
    max_val = detection_seq.max().item()
    
    # For normalized features, all values should be in [0, 1]
    if min_val < -0.01 or max_val > 1.01:  # Small tolerance for floating point
        print(f"⚠️ {name} values outside [0,1]: [{min_val:.3f}, {max_val:.3f}]")
        return False
    
    return True


def get_detection_statistics(detection_seq: torch.Tensor, detection_mask: torch.Tensor) -> dict:
    """
    Get statistics about detection features for debugging.
    
    Args:
        detection_seq: Detection tensor
        detection_mask: Detection validity mask
        
    Returns:
        Dictionary with statistics
    """
    # Only consider valid detections
    valid_detections = detection_seq[detection_mask]
    
    if valid_detections.numel() == 0:
        return {"num_valid": 0}
    
    feature_names = ["confidence", "x1", "y1", "x2", "y2", "area"]
    stats = {"num_valid": valid_detections.shape[0]}
    
    for i, feature_name in enumerate(feature_names):
        feature_values = valid_detections[:, i]
        stats[feature_name] = {
            "min": feature_values.min().item(),
            "max": feature_values.max().item(), 
            "mean": feature_values.mean().item(),
            "std": feature_values.std().item()
        }
    
    return stats