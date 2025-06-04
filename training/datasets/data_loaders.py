"""
PyTorch DataLoaders with automatic telemetry AND detection normalization.
"""

import sys
from pathlib import Path
import h5py
import torch
import numpy as np
import json
from torch.utils.data import Dataset, DataLoader
from typing import Dict, Any, Optional, Tuple, List

# Add project root to path for imports
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

# Import unified normalization utilities
from processing.normalization import (
    normalize_telemetry_features,
    normalize_detection_features,
    validate_normalized_features,
    get_telemetry_input_dim,
    CONTINUOUS_TELEMETRY_FEATURES,
    GEAR_CLASSES,
)


class MultimodalDataset(Dataset):
    """
    Dataset with automatic telemetry AND detection normalization.
    Supports GEAR one-hot encoding for proper categorical handling.
    """

    def __init__(
        self,
        h5_file_path: str,
        load_into_memory: bool = False,
        use_class_features: bool = True,
        target_horizons: List[str] = None,
        auto_normalize: bool = True,
        normalize_telemetry: bool = True,
        use_gear_onehot: bool = True,
        img_width: int = 1920,
        img_height: int = 575,
    ):
        """
        Initialize the multimodal dataset with full feature normalization.

        Args:
            h5_file_path: Path to HDF5 file
            load_into_memory: Whether to load entire dataset into RAM
            use_class_features: Whether to include class_id in detection features
            target_horizons: List of target horizon names to load
            auto_normalize: Whether to automatically normalize detection features
            normalize_telemetry: Whether to normalize telemetry features to [0,1]
            use_gear_onehot: Whether to one-hot encode GEAR (recommended for categorical)
            img_width: Image width for normalization
            img_height: Image height for normalization (consider ROI cropping)
        """
        self.h5_file_path = Path(h5_file_path)
        self.load_into_memory = load_into_memory
        self.use_class_features = use_class_features
        self.auto_normalize = auto_normalize
        self.normalize_telemetry = normalize_telemetry
        self.use_gear_onehot = use_gear_onehot
        self.img_width = img_width
        self.img_height = img_height

        if target_horizons is None:
            target_horizons = ["brake_1s", "brake_2s", "coast_1s", "coast_2s"]
        self.target_horizons = target_horizons

        if not self.h5_file_path.exists():
            raise FileNotFoundError(f"HDF5 file not found: {self.h5_file_path}")

        # Load dataset info
        self._load_dataset_info()

        # Load data into memory if requested
        if self.load_into_memory:
            self._load_data_into_memory()
        else:
            self._data_cache = {}

        print(f"📚 Loaded dataset: {self.h5_file_path.name}")
        print(f"   📊 Sequences: {self.num_sequences}")
        print(f"   💾 Memory loading: {self.load_into_memory}")
        print(f"   🏷️ Use class features: {self.use_class_features}")
        print(f"   🎯 Target horizons: {self.target_horizons}")
        print(f"   🔧 Auto normalize detections: {self.auto_normalize}")
        print(f"   🔧 Normalize telemetry: {self.normalize_telemetry}")
        print(f"   🎲 GEAR one-hot encoding: {self.use_gear_onehot}")
        if self.auto_normalize:
            print(f"   📐 Detection normalization: {self.img_width}x{self.img_height}")
        if self.normalize_telemetry:
            expected_telemetry_dim = get_telemetry_input_dim(self.use_gear_onehot)
            print(
                f"   📊 Expected telemetry dim: {expected_telemetry_dim} (4 continuous + {GEAR_CLASSES if self.use_gear_onehot else 1} gear)"
            )

    def _safe_decode(self, value):
        """Safely decode bytes to string, or return string if already decoded."""
        if isinstance(value, bytes):
            return value.decode("utf-8")
        elif isinstance(value, np.bytes_):
            return value.decode("utf-8")
        else:
            return str(value)

    def _load_dataset_info(self):
        """Load dataset metadata and configuration."""
        with h5py.File(self.h5_file_path, "r") as f:
            # Get basic info
            self.num_sequences = f["info"].attrs["num_sequences"]
            self.sequence_length = f["info"].attrs["sequence_length"]
            self.max_detections = f["info"].attrs["max_detections"]
            self.telemetry_dim = f["info"].attrs["telemetry_dim"]
            self.detection_dim_per_box = f["info"].attrs["detection_dim_per_box"]

            # Get feature names
            self.telemetry_features = [
                self._safe_decode(f) for f in f["info"].attrs["telemetry_features"]
            ]

            # Get available label names
            self.available_labels = list(f["labels"].keys())

            # Validate target horizons
            missing_horizons = set(self.target_horizons) - set(self.available_labels)
            if missing_horizons:
                raise ValueError(f"Missing target horizons: {missing_horizons}")

        # Calculate actual dimensions after processing
        if self.use_class_features:
            self.actual_detection_dim = self.detection_dim_per_box
        else:
            # Remove class_id (first feature)
            self.actual_detection_dim = self.detection_dim_per_box - 1

        # Calculate telemetry dimension after normalization
        self.actual_telemetry_dim = get_telemetry_input_dim(self.use_gear_onehot)

    def _load_data_into_memory(self):
        """Load entire dataset into memory for faster access."""
        print("💾 Loading dataset into memory...")

        with h5py.File(self.h5_file_path, "r") as f:
            # Load main data arrays
            raw_telemetry = torch.from_numpy(f["telemetry"][:]).float()
            self.detection_data = torch.from_numpy(f["detections"][:]).float()
            self.detection_masks = torch.from_numpy(f["detection_masks"][:])

            # Load labels
            self.labels_data = {}
            for horizon in self.target_horizons:
                self.labels_data[horizon] = torch.from_numpy(f["labels"][horizon][:])

            # Load metadata
            self.recording_names = [
                self._safe_decode(name) for name in f["metadata"]["recording_names"][:]
            ]

        # Apply telemetry normalization to entire dataset if in memory
        if self.normalize_telemetry:
            print("🔧 Applying telemetry normalization to dataset...")
            self.telemetry_data = normalize_telemetry_features(
                raw_telemetry,
                use_gear_onehot=self.use_gear_onehot,
                in_place=False,
            )
            print(
                f"✅ Telemetry normalization applied (shape: {raw_telemetry.shape} → {self.telemetry_data.shape})"
            )

            if self.use_gear_onehot:
                print(f"   🎲 GEAR one-hot encoded: {GEAR_CLASSES} classes")
        else:
            self.telemetry_data = raw_telemetry

        # Apply detection normalization to entire dataset if in memory
        if self.auto_normalize:
            print("🔧 Applying detection normalization to dataset...")
            self.detection_data = normalize_detection_features(
                self.detection_data,
                self.detection_masks,
                img_width=self.img_width,
                img_height=self.img_height,
                in_place=True,
            )
            print("✅ Detection normalization applied")

        print(f"✅ Loaded {self.num_sequences} sequences into memory")

    def _get_data_from_file(
        self, idx: int
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
        """Load single sample from HDF5 file."""
        # Use simple caching to avoid repeated file access
        if idx in self._data_cache:
            return self._data_cache[idx]

        with h5py.File(self.h5_file_path, "r") as f:
            # Load raw telemetry
            raw_telemetry = torch.from_numpy(f["telemetry"][idx]).float()

            # Load detections
            detections = torch.from_numpy(f["detections"][idx]).float()

            # Load detection mask
            detection_mask = torch.from_numpy(f["detection_masks"][idx])

            # Load labels
            labels = {}
            for horizon in self.target_horizons:
                labels[horizon] = torch.from_numpy(np.array(f["labels"][horizon][idx]))

        # Apply telemetry normalization if enabled
        if self.normalize_telemetry:
            telemetry = normalize_telemetry_features(
                raw_telemetry,
                use_gear_onehot=self.use_gear_onehot,
                in_place=False,
            )
        else:
            telemetry = raw_telemetry

        # Apply detection normalization if enabled
        if self.auto_normalize:
            detections = normalize_detection_features(
                detections,
                detection_mask,
                img_width=self.img_width,
                img_height=self.img_height,
                in_place=False,
            )

        # Simple LRU-style cache (keep last 100 items)
        if len(self._data_cache) > 100:
            # Remove oldest item
            oldest_key = next(iter(self._data_cache))
            del self._data_cache[oldest_key]

        self._data_cache[idx] = (telemetry, detections, detection_mask, labels)
        return self._data_cache[idx]

    def _process_detection_features(self, detections: torch.Tensor) -> torch.Tensor:
        """Process detection features based on configuration."""
        if not self.use_class_features:
            # Remove class_id (first feature) and keep [confidence, x1, y1, x2, y2, area]
            detections = detections[..., 1:]

        return detections

    def __len__(self) -> int:
        """Return dataset size."""
        return self.num_sequences

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get single sample from dataset."""
        if self.load_into_memory:
            # Get from memory
            telemetry = self.telemetry_data[idx]
            detections = self.detection_data[idx]
            detection_mask = self.detection_masks[idx]

            labels = {}
            for horizon in self.target_horizons:
                labels[horizon] = self.labels_data[horizon][idx]
        else:
            # Get from file
            telemetry, detections, detection_mask, labels = self._get_data_from_file(
                idx
            )

        # Process detection features
        detections = self._process_detection_features(detections)

        return {
            "telemetry_seq": telemetry,
            "detection_seq": detections,
            "detection_mask": detection_mask,
            "targets": labels,
        }

    def get_sample_info(self, idx: int) -> Dict[str, Any]:
        """Get metadata for a specific sample."""
        with h5py.File(self.h5_file_path, "r") as f:
            recording_name = self._safe_decode(f["metadata"]["recording_names"][idx])
            start_idx = int(f["metadata"]["start_indices"][idx])
            end_idx = int(f["metadata"]["end_indices"][idx])
            timestamps = json.loads(self._safe_decode(f["metadata"]["timestamps"][idx]))

        return {
            "recording_name": recording_name,
            "start_idx": start_idx,
            "end_idx": end_idx,
            "timestamps": timestamps,
        }

    def get_label_statistics(self) -> Dict[str, Dict[str, float]]:
        """Calculate label statistics for the dataset."""
        stats = {}

        if self.load_into_memory:
            # Calculate from loaded data
            for horizon in self.target_horizons:
                labels = self.labels_data[horizon]
                positive = labels.sum().item()
                total = len(labels)
                stats[horizon] = {
                    "positive": positive,
                    "negative": total - positive,
                    "ratio": positive / total,
                    "total": total,
                }
        else:
            # Calculate from file
            with h5py.File(self.h5_file_path, "r") as f:
                for horizon in self.target_horizons:
                    labels = f["labels"][horizon][:]
                    positive = np.sum(labels)
                    total = len(labels)
                    stats[horizon] = {
                        "positive": int(positive),
                        "negative": int(total - positive),
                        "ratio": float(positive / total),
                        "total": int(total),
                    }

        return stats

    def get_feature_statistics(self, sample_size: int = 1000) -> Dict:
        """Get statistics about normalized features for validation."""
        if sample_size > len(self):
            sample_size = len(self)

        # Sample random indices
        indices = torch.randperm(len(self))[:sample_size]

        telemetry_samples = []
        detection_samples = []
        detection_masks = []

        for idx in indices:
            sample = self[idx.item()]
            telemetry_samples.append(sample["telemetry_seq"])
            detection_samples.append(sample["detection_seq"])
            detection_masks.append(sample["detection_mask"])

        # Stack samples
        all_telemetry = torch.stack(telemetry_samples)
        all_detections = torch.stack(detection_samples)
        all_masks = torch.stack(detection_masks)

        stats = {"telemetry": {}, "detections": {}, "summary": {}}

        # Telemetry statistics
        continuous_features = CONTINUOUS_TELEMETRY_FEATURES
        for i, feature_name in enumerate(continuous_features):
            feature_values = all_telemetry[..., i]
            stats["telemetry"][feature_name] = {
                "min": feature_values.min().item(),
                "max": feature_values.max().item(),
                "mean": feature_values.mean().item(),
                "std": feature_values.std().item(),
            }

        # GEAR statistics
        if self.use_gear_onehot:
            gear_start_idx = len(continuous_features)
            gear_onehot = all_telemetry[
                ..., gear_start_idx : gear_start_idx + GEAR_CLASSES
            ]
            gear_distribution = gear_onehot.sum(
                dim=tuple(range(len(gear_onehot.shape) - 1))
            )
            stats["telemetry"]["GEAR_distribution"] = {
                f"gear_{i}": int(gear_distribution[i].item())
                for i in range(GEAR_CLASSES)
            }
        else:
            gear_values = all_telemetry[..., -1]
            stats["telemetry"]["GEAR"] = {
                "min": gear_values.min().item(),
                "max": gear_values.max().item(),
                "mean": gear_values.mean().item(),
                "std": gear_values.std().item(),
            }

        # Detection statistics (only valid detections)
        if all_masks.any():
            valid_detections = all_detections[all_masks]
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

        stats["summary"] = {
            "sample_size": sample_size,
            "use_gear_onehot": self.use_gear_onehot,
            "telemetry_input_dim": self.actual_telemetry_dim,
            "detection_input_dim": self.actual_detection_dim,
            "normalization_valid": self._validate_normalization(
                all_telemetry, all_detections, all_masks
            ),
        }

        return stats

    def _validate_normalization(
        self, telemetry: torch.Tensor, detections: torch.Tensor, masks: torch.Tensor
    ) -> bool:
        """Validate that all features are properly normalized."""
        return validate_normalized_features(
            telemetry, detections, masks, "dataset_sample"
        )


def create_multimodal_dataloader(
    h5_file_path: str,
    batch_size: int = 32,
    shuffle: bool = True,
    num_workers: int = 8,
    pin_memory: bool = True,
    load_into_memory: bool = False,
    use_class_features: bool = True,
    target_horizons: List[str] = None,
    auto_normalize: bool = True,
    normalize_telemetry: bool = True,
    use_gear_onehot: bool = True,  # NEW: Enable GEAR one-hot encoding
    img_width: int = 1920,
    img_height: int = 575,  # ROI height
    **kwargs,
) -> DataLoader:
    """
    Create DataLoader for multimodal training with full feature normalization.

    Args:
        h5_file_path: Path to HDF5 dataset file
        batch_size: Batch size for training
        shuffle: Whether to shuffle data
        num_workers: Number of worker processes
        pin_memory: Whether to pin memory for GPU transfer
        load_into_memory: Whether to load dataset into memory
        use_class_features: Whether to include class_id in detection features
        target_horizons: List of target horizon names
        auto_normalize: Whether to automatically normalize detection features
        normalize_telemetry: Whether to normalize telemetry features
        use_gear_onehot: Whether to one-hot encode GEAR (recommended)
        img_width: Image width for normalization
        img_height: Image height for normalization (consider ROI)
        **kwargs: Additional DataLoader arguments

    Returns:
        Configured DataLoader
    """
    # Create dataset
    dataset = MultimodalDataset(
        h5_file_path=h5_file_path,
        load_into_memory=load_into_memory,
        use_class_features=use_class_features,
        target_horizons=target_horizons,
        auto_normalize=auto_normalize,
        normalize_telemetry=normalize_telemetry,
        use_gear_onehot=use_gear_onehot,
        img_width=img_width,
        img_height=img_height,
    )

    # Create DataLoader
    dataloader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True,  # For consistent batch sizes in DDP
        **kwargs,
    )

    return dataloader


def calculate_class_weights(dataset: MultimodalDataset) -> Dict[str, torch.Tensor]:
    """Calculate class weights with extreme value capping."""
    stats = dataset.get_label_statistics()
    class_weights = {}

    for horizon, _ in stats.items():
        pos_weight = 2 if "brake" in horizon else 1
        weights = torch.tensor([1.0, pos_weight])
        class_weights[horizon] = weights
        print(f"📊 {horizon} class weights: neg={weights[0]:.3f}, pos={weights[1]:.3f}")

    return class_weights


# Feature validation and statistics utilities
def validate_dataset_normalization(
    dataloader: DataLoader, num_batches: int = 10
) -> bool:
    """Validate that dataset features are properly normalized."""
    print("🔍 Validating dataset normalization...")

    all_valid = True
    batch_count = 0

    for batch in dataloader:
        telemetry = batch["telemetry_seq"]
        detections = batch["detection_seq"]
        masks = batch["detection_mask"]

        if not validate_normalized_features(
            telemetry, detections, masks, f"batch_{batch_count}"
        ):
            all_valid = False

        batch_count += 1
        if batch_count >= num_batches:
            break

    if all_valid:
        print("✅ All features properly normalized to [0, 1]")
    else:
        print("❌ Normalization issues detected!")

    return all_valid


def print_feature_ranges(dataloader: DataLoader, num_batches: int = 5):
    """Print actual feature ranges for inspection."""
    print("📊 Feature ranges in dataset:")

    tel_mins = []
    tel_maxs = []
    det_mins = []
    det_maxs = []

    batch_count = 0
    for batch in dataloader:
        telemetry = batch["telemetry_seq"]
        detections = batch["detection_seq"]
        masks = batch["detection_mask"]

        # Collect telemetry ranges
        tel_mins.append(
            telemetry.min(dim=0)[0].min(dim=0)[0]
        )  # Min across batch and time
        tel_maxs.append(
            telemetry.max(dim=0)[0].max(dim=0)[0]
        )  # Max across batch and time

        # Collect detection ranges (only valid detections)
        if masks.any():
            valid_detections = detections[masks]
            det_mins.append(valid_detections.min(dim=0)[0])
            det_maxs.append(valid_detections.max(dim=0)[0])

        batch_count += 1
        if batch_count >= num_batches:
            break

    # Aggregate statistics
    if tel_mins:
        overall_tel_min = torch.stack(tel_mins).min(dim=0)[0]
        overall_tel_max = torch.stack(tel_maxs).max(dim=0)[0]

        print("  Telemetry features:")
        continuous_features = CONTINUOUS_TELEMETRY_FEATURES
        for i, feature_name in enumerate(continuous_features):
            print(
                f"    {feature_name}: [{overall_tel_min[i]:.3f}, {overall_tel_max[i]:.3f}]"
            )

        # GEAR features (one-hot or continuous)
        gear_start = len(continuous_features)
        if overall_tel_min.shape[0] > gear_start + 1:  # One-hot case
            print(
                f"    GEAR_one_hot: [{overall_tel_min[gear_start:].min():.3f}, {overall_tel_max[gear_start:].max():.3f}]"
            )
        else:  # Continuous case
            print(
                f"    GEAR: [{overall_tel_min[gear_start]:.3f}, {overall_tel_max[gear_start]:.3f}]"
            )

    if det_mins:
        overall_det_min = torch.stack(det_mins).min(dim=0)[0]
        overall_det_max = torch.stack(det_maxs).max(dim=0)[0]

        detection_feature_names = ["confidence", "x1", "y1", "x2", "y2", "area"]
        print("  Detection features:")
        for i, feature_name in enumerate(detection_feature_names):
            print(
                f"    {feature_name}: [{overall_det_min[i]:.3f}, {overall_det_max[i]:.3f}]"
            )
