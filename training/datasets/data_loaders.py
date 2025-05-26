"""
PyTorch DataLoaders for multimodal training data.
Efficient loading of HDF5 datasets with caching and optimization.
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


class MultimodalDataset(Dataset):
    """
    Dataset for multimodal braking prediction training.
    Combines telemetry sequences with YOLO detection results.
    """

    def __init__(
        self,
        h5_file_path: str,
        load_into_memory: bool = False,
        use_class_features: bool = True,
        target_horizons: List[str] = None,
    ):
        """
        Initialize the multimodal dataset.

        Args:
            h5_file_path: Path to HDF5 file
            load_into_memory: Whether to load entire dataset into RAM
            use_class_features: Whether to include class_id in detection features
            target_horizons: List of target horizon names to load (default: all brake horizons)
        """
        self.h5_file_path = Path(h5_file_path)
        self.load_into_memory = load_into_memory
        self.use_class_features = use_class_features

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
                f.decode("utf-8") for f in f["info"].attrs["telemetry_features"]
            ]

            # Get available label names
            self.available_labels = list(f["labels"].keys())

            # Validate target horizons
            missing_horizons = set(self.target_horizons) - set(self.available_labels)
            if missing_horizons:
                raise ValueError(f"Missing target horizons: {missing_horizons}")

        # Calculate actual detection dimension based on class feature usage
        if self.use_class_features:
            self.actual_detection_dim = self.detection_dim_per_box
        else:
            # Remove class_id (first feature)
            self.actual_detection_dim = self.detection_dim_per_box - 1

    def _load_data_into_memory(self):
        """Load entire dataset into memory for faster access."""
        print("💾 Loading dataset into memory...")

        with h5py.File(self.h5_file_path, "r") as f:
            # Load main data arrays
            self.telemetry_data = torch.from_numpy(f["telemetry"][:]).float()
            self.detection_data = torch.from_numpy(f["detections"][:]).float()
            self.detection_masks = torch.from_numpy(f["detection_masks"][:])

            # Load labels
            self.labels_data = {}
            for horizon in self.target_horizons:
                self.labels_data[horizon] = torch.from_numpy(f["labels"][horizon][:])

            # Load metadata
            self.recording_names = [
                name.decode("utf-8") for name in f["metadata"]["recording_names"][:]
            ]

        print(f"✅ Loaded {self.num_sequences} sequences into memory")

    def _get_data_from_file(
        self, idx: int
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
        """Load single sample from HDF5 file."""
        # Use simple caching to avoid repeated file access
        if idx in self._data_cache:
            return self._data_cache[idx]

        with h5py.File(self.h5_file_path, "r") as f:
            # Load telemetry
            telemetry = torch.from_numpy(f["telemetry"][idx]).float()

            # Load detections
            detections = torch.from_numpy(f["detections"][idx]).float()

            # Load detection mask
            detection_mask = torch.from_numpy(f["detection_masks"][idx])

            # Load labels
            labels = {}
            for horizon in self.target_horizons:
                labels[horizon] = torch.from_numpy(np.array(f["labels"][horizon][idx]))

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
        """
        Get single sample from dataset.

        Args:
            idx: Sample index

        Returns:
            Dictionary with sample data:
            - telemetry_seq: [seq_len, telemetry_dim]
            - detection_seq: [seq_len, max_detections, detection_dim]
            - detection_mask: [seq_len, max_detections]
            - targets: Dict with label tensors
        """
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
            recording_name = f["metadata"]["recording_names"][idx].decode("utf-8")
            start_idx = int(f["metadata"]["start_indices"][idx])
            end_idx = int(f["metadata"]["end_indices"][idx])
            timestamps = json.loads(f["metadata"]["timestamps"][idx].decode("utf-8"))

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


def create_multimodal_dataloader(
    h5_file_path: str,
    batch_size: int = 32,
    shuffle: bool = True,
    num_workers: int = 8,
    pin_memory: bool = True,
    load_into_memory: bool = False,
    use_class_features: bool = True,
    target_horizons: List[str] = None,
    **kwargs,
) -> DataLoader:
    """
    Create DataLoader for multimodal training.

    Args:
        h5_file_path: Path to HDF5 dataset file
        batch_size: Batch size for training
        shuffle: Whether to shuffle data
        num_workers: Number of worker processes
        pin_memory: Whether to pin memory for GPU transfer
        load_into_memory: Whether to load dataset into memory
        use_class_features: Whether to include class_id in detection features
        target_horizons: List of target horizon names
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
    """
    Calculate class weights for imbalanced datasets.

    Args:
        dataset: MultimodalDataset instance

    Returns:
        Dictionary with class weights for each target horizon
    """
    stats = dataset.get_label_statistics()
    class_weights = {}

    for horizon, stat in stats.items():
        # Calculate inverse frequency weights
        positive_ratio = stat["ratio"]
        negative_ratio = 1 - positive_ratio

        if positive_ratio > 0 and negative_ratio > 0:
            # Inverse frequency weighting
            pos_weight = negative_ratio / positive_ratio
            weights = torch.tensor(
                [1.0, pos_weight]
            )  # [negative_weight, positive_weight]
        else:
            # Fallback for edge cases
            weights = torch.tensor([1.0, 1.0])

        class_weights[horizon] = weights

        print(f"📊 {horizon} class weights: neg={weights[0]:.3f}, pos={weights[1]:.3f}")

    return class_weights


# Example usage and testing
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Test multimodal DataLoader")
    parser.add_argument("--h5-file", type=str, required=True, help="Path to HDF5 file")
    parser.add_argument("--batch-size", type=int, default=4, help="Batch size")
    parser.add_argument("--load-memory", action="store_true", help="Load into memory")
    parser.add_argument(
        "--no-class-features",
        action="store_true",
        default=True,
        help="Exclude class features",
    )

    args = parser.parse_args()

    # Test DataLoader
    print("🧪 Testing MultimodalDataLoader...")

    dataloader = create_multimodal_dataloader(
        h5_file_path=args.h5_file,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=2,
        load_into_memory=args.load_memory,
        use_class_features=not args.no_class_features,
    )

    # Test a few batches
    for i, batch in enumerate(dataloader):
        print(f"\n📦 Batch {i+1}:")
        print(f"   🔢 Telemetry shape: {batch['telemetry_seq'].shape}")
        print(f"   🎯 Detection shape: {batch['detection_seq'].shape}")
        print(f"   🎭 Mask shape: {batch['detection_mask'].shape}")
        print(f"   🏷️ Targets: {list(batch['targets'].keys())}")

        for target_name, target_tensor in batch["targets"].items():
            positive_count = target_tensor.sum().item()
            print(
                f"      {target_name}: {target_tensor.shape}, {positive_count}/{len(target_tensor)} positive"
            )

        if i >= 2:  # Test only first 3 batches
            break

    # Test class weights
    print("\n⚖️ Testing class weights...")
    class_weights = calculate_class_weights(dataloader.dataset)

    print("✅ DataLoader test completed!")
