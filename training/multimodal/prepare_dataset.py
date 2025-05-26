#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Dataset Preparation Pipeline for Multimodal Training
Prepares sliding-window sequences from recordings and splits into train/val/test sets.
"""

import os
import sys
import h5py
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import argparse
import logging
import json
import ast
from typing import Dict, List, Optional, Tuple, Any
from collections import defaultdict
from sklearn.model_selection import train_test_split
import random

# Add project root to path for imports
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from src.utils.config import RECORDING_OUTPUT_PATH

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler("prepare_dataset.log"), logging.StreamHandler()],
)
logger = logging.getLogger(__name__)

# Dataset configuration
SEQUENCE_LENGTH = 20  # 20 frames = 10 seconds at 2Hz
SEQUENCE_STRIDE = 5  # 5 frames = 2.5 seconds stride
MAX_DETECTIONS = 12  # Maximum objects per frame
TRAIN_SPLIT = 0.70
VAL_SPLIT = 0.15
TEST_SPLIT = 0.15

# Feature dimensions
TELEMETRY_FEATURES = ["SPEED", "RPM", "ACCELERATOR_POS_D", "ENGINE_LOAD", "GEAR"]
TELEMETRY_DIM = len(TELEMETRY_FEATURES)
DETECTION_DIM_PER_BOX = 7  # [class_id, confidence, x1, y1, x2, y2, area]

# Output paths
DEFAULT_OUTPUT_DIR = "data/datasets/multimodal"


class DatasetPreparator:
    """
    Prepares multimodal dataset from recorded driving data.
    """

    def __init__(
        self,
        output_dir: str = DEFAULT_OUTPUT_DIR,
        sequence_length: int = SEQUENCE_LENGTH,
        sequence_stride: int = SEQUENCE_STRIDE,
        max_detections: int = MAX_DETECTIONS,
        random_seed: int = 42,
    ):
        self.output_dir = Path(output_dir)
        self.sequence_length = sequence_length
        self.sequence_stride = sequence_stride
        self.max_detections = max_detections
        self.random_seed = random_seed

        # Set random seeds for reproducibility
        random.seed(random_seed)
        np.random.seed(random_seed)

        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"🎯 Dataset Preparator initialized")
        logger.info(f"   📁 Output directory: {self.output_dir}")
        logger.info(f"   📏 Sequence length: {self.sequence_length} frames")
        logger.info(f"   👣 Sequence stride: {self.sequence_stride} frames")
        logger.info(f"   🎯 Max detections: {self.max_detections}")
        logger.info(f"   🎲 Random seed: {self.random_seed}")

    def _validate_recording(self, recording_dir: Path) -> bool:
        """
        Validate that recording has all required files.

        Args:
            recording_dir: Path to recording directory

        Returns:
            True if valid
        """
        required_files = ["telemetry.csv", "annotations.csv", "future_labels.csv"]

        for file_name in required_files:
            if not (recording_dir / file_name).exists():
                logger.warning(f"⚠️ Missing {file_name} in {recording_dir.name}")
                return False

        return True

    def _load_recording_data(self, recording_dir: Path) -> Optional[Dict]:
        """
        Load all data for a single recording.

        Args:
            recording_dir: Path to recording directory

        Returns:
            Dictionary with loaded data or None if failed
        """
        try:
            # Load telemetry data
            telemetry_df = pd.read_csv(recording_dir / "telemetry.csv")

            # Load annotations
            annotations_df = pd.read_csv(recording_dir / "annotations.csv")

            # Load future labels
            labels_df = pd.read_csv(recording_dir / "future_labels.csv")

            # Validate data consistency
            if len(telemetry_df) != len(labels_df):
                logger.error(
                    f"❌ Length mismatch in {recording_dir.name}: "
                    f"telemetry={len(telemetry_df)}, labels={len(labels_df)}"
                )
                return None

            # Merge annotations with telemetry on timestamp
            merged_df = telemetry_df.merge(
                annotations_df[
                    [
                        "Time",
                        "vehicle_count",
                        "class_ids",
                        "confidences",
                        "bboxes",
                        "areas",
                    ]
                ],
                on="Time",
                how="left",
            )

            # Merge with future labels
            merged_df = merged_df.merge(labels_df, on="Time", how="left")

            # Fill missing detection data with empty lists
            merged_df["vehicle_count"] = (
                merged_df["vehicle_count"].fillna(0).astype(int)
            )
            merged_df["class_ids"] = merged_df["class_ids"].fillna("[]")
            merged_df["confidences"] = merged_df["confidences"].fillna("[]")
            merged_df["bboxes"] = merged_df["bboxes"].fillna("[]")
            merged_df["areas"] = merged_df["areas"].fillna("[]")

            logger.debug(
                f"✅ Loaded data for {recording_dir.name}: {len(merged_df)} samples"
            )

            return {
                "recording_name": recording_dir.name,
                "data": merged_df,
                "length": len(merged_df),
            }

        except Exception as e:
            logger.error(f"❌ Failed to load data for {recording_dir.name}: {e}")
            return None

    def _parse_detection_lists(self, row: pd.Series) -> List[List[float]]:
        """
        Parse detection data from DataFrame row.

        Args:
            row: DataFrame row with detection data

        Returns:
            List of detection vectors [class_id, confidence, x1, y1, x2, y2, area]
        """
        try:
            # Parse string representations of lists
            class_ids = (
                ast.literal_eval(row["class_ids"]) if row["class_ids"] != "[]" else []
            )
            confidences = (
                ast.literal_eval(row["confidences"])
                if row["confidences"] != "[]"
                else []
            )
            bboxes = ast.literal_eval(row["bboxes"]) if row["bboxes"] != "[]" else []
            areas = ast.literal_eval(row["areas"]) if row["areas"] != "[]" else []

            # Ensure all lists have same length
            num_detections = min(
                len(class_ids), len(confidences), len(bboxes), len(areas)
            )

            detections = []
            for i in range(num_detections):
                if len(bboxes[i]) == 4:  # Valid bbox [x1, y1, x2, y2]
                    detection = [
                        float(class_ids[i]),  # class_id
                        float(confidences[i]),  # confidence
                        float(bboxes[i][0]),  # x1
                        float(bboxes[i][1]),  # y1
                        float(bboxes[i][2]),  # x2
                        float(bboxes[i][3]),  # y2
                        float(areas[i]),  # area
                    ]
                    detections.append(detection)

            return detections

        except Exception as e:
            logger.debug(f"⚠️ Failed to parse detections: {e}")
            return []

    def _create_detection_tensor(self, detections: List[List[float]]) -> np.ndarray:
        """
        Create padded detection tensor for a single frame.

        Args:
            detections: List of detection vectors

        Returns:
            Padded numpy array of shape (max_detections, detection_dim)
        """
        # Initialize with zeros (padding)
        detection_tensor = np.zeros(
            (self.max_detections, DETECTION_DIM_PER_BOX), dtype=np.float32
        )

        # Fill with actual detections (up to max_detections)
        num_detections = min(len(detections), self.max_detections)
        for i in range(num_detections):
            detection_tensor[i] = detections[i]

        return detection_tensor

    def _create_detection_mask(self, detections: List[List[float]]) -> np.ndarray:
        """
        Create detection mask indicating valid detections.

        Args:
            detections: List of detection vectors

        Returns:
            Boolean mask array of shape (max_detections,)
        """
        mask = np.zeros(self.max_detections, dtype=bool)
        num_detections = min(len(detections), self.max_detections)
        mask[:num_detections] = True
        return mask

    def _extract_sequences_from_recording(self, recording_data: Dict) -> List[Dict]:
        """
        Extract sliding window sequences from recording data.

        Args:
            recording_data: Dictionary with recording data

        Returns:
            List of sequence dictionaries
        """
        df = recording_data["data"]
        recording_name = recording_data["recording_name"]

        sequences = []

        # Generate sliding windows
        for start_idx in range(
            0, len(df) - self.sequence_length + 1, self.sequence_stride
        ):
            end_idx = start_idx + self.sequence_length
            sequence_df = df.iloc[start_idx:end_idx].copy()

            # Extract telemetry features
            telemetry_seq = sequence_df[TELEMETRY_FEATURES].values.astype(np.float32)

            # Extract detection features and masks
            detection_seq = np.zeros(
                (self.sequence_length, self.max_detections, DETECTION_DIM_PER_BOX),
                dtype=np.float32,
            )
            detection_mask = np.zeros(
                (self.sequence_length, self.max_detections), dtype=bool
            )

            for i, (_, row) in enumerate(sequence_df.iterrows()):
                detections = self._parse_detection_lists(row)
                detection_seq[i] = self._create_detection_tensor(detections)
                detection_mask[i] = self._create_detection_mask(detections)

            # Extract labels (using last frame's future labels for proper prediction)
            last_row = sequence_df.iloc[
                -1
            ]  # Use last frame for "predict future from now"
            labels = {}
            for horizon in [1, 2, 3, 4, 5]:
                brake_col = f"brake_{horizon}s"
                coast_col = f"coast_{horizon}s"

                # Handle missing columns gracefully
                labels[brake_col] = bool(last_row.get(brake_col, False))
                labels[coast_col] = bool(last_row.get(coast_col, False))

            # Create sequence dictionary
            sequence = {
                "recording_name": recording_name,
                "start_idx": start_idx,
                "end_idx": end_idx,
                "telemetry_seq": telemetry_seq,
                "detection_seq": detection_seq,
                "detection_mask": detection_mask,
                "labels": labels,
                "timestamps": sequence_df["Time"].tolist(),
            }

            sequences.append(sequence)

        logger.debug(f"✅ Extracted {len(sequences)} sequences from {recording_name}")
        return sequences

    def _calculate_brake_event_ratio(self, sequences: List[Dict]) -> float:
        """
        Calculate the ratio of sequences with brake events (for stratification).

        Args:
            sequences: List of sequence dictionaries

        Returns:
            Ratio of sequences with any brake event
        """
        brake_sequences = 0
        for seq in sequences:
            # Check if any brake horizon is True
            has_brake = any(
                seq["labels"].get(f"brake_{h}s", False) for h in [1, 2, 3, 4, 5]
            )
            if has_brake:
                brake_sequences += 1

        ratio = brake_sequences / len(sequences) if sequences else 0.0
        logger.info(
            f"📊 Brake event ratio: {brake_sequences}/{len(sequences)} ({ratio:.1%})"
        )
        return ratio

    def _split_recordings_stratified(
        self, all_recording_data: List[Dict]
    ) -> Tuple[List[str], List[str], List[str]]:
        """
        Split recordings into train/val/test sets with sequence-aware stratification.

        Args:
            all_recording_data: List of recording data dictionaries

        Returns:
            Tuple of (train_recordings, val_recordings, test_recordings)
        """
        # Calculate detailed statistics for each recording
        recording_stats = []
        for recording_data in all_recording_data:
            sequences = self._extract_sequences_from_recording(recording_data)
            brake_ratio = self._calculate_brake_event_ratio(sequences)

            # Count actual brake sequences (more informative than ratio)
            brake_sequences = sum(
                1
                for seq in sequences
                if any(seq["labels"].get(f"brake_{h}s", False) for h in [1, 2, 3, 4, 5])
            )

            recording_stats.append(
                {
                    "name": recording_data["recording_name"],
                    "brake_ratio": brake_ratio,
                    "num_sequences": len(sequences),
                    "brake_sequences": brake_sequences,
                    "normal_sequences": len(sequences) - brake_sequences,
                }
            )

        # Sort by number of brake sequences for informed splitting
        recording_stats.sort(key=lambda x: x["brake_sequences"], reverse=True)

        # Initialize splits
        train_recordings = []
        val_recordings = []
        test_recordings = []

        # Track sequence counts per split
        train_brake_count = 0
        val_brake_count = 0
        test_brake_count = 0

        train_total_count = 0
        val_total_count = 0
        test_total_count = 0

        # Distribute recordings to balance brake sequences across splits
        for i, recording in enumerate(recording_stats):
            # Calculate current ratios
            total_brake = train_brake_count + val_brake_count + test_brake_count
            total_sequences = train_total_count + val_total_count + test_total_count

            if total_sequences == 0:
                # First recording goes to train
                target_split = "train"
            else:
                # Calculate current split ratios
                train_ratio = (
                    train_total_count / total_sequences if total_sequences > 0 else 0
                )
                val_ratio = (
                    val_total_count / total_sequences if total_sequences > 0 else 0
                )
                test_ratio = (
                    test_total_count / total_sequences if total_sequences > 0 else 0
                )

                # Determine target split based on current imbalance
                if train_ratio < TRAIN_SPLIT - 0.05:  # Allow 5% tolerance
                    target_split = "train"
                elif val_ratio < VAL_SPLIT - 0.05:
                    target_split = "val"
                elif test_ratio < TEST_SPLIT - 0.05:
                    target_split = "test"
                else:
                    # Use random assignment with weights
                    remaining_train = max(0, TRAIN_SPLIT - train_ratio)
                    remaining_val = max(0, VAL_SPLIT - val_ratio)
                    remaining_test = max(0, TEST_SPLIT - test_ratio)

                    weights = [remaining_train, remaining_val, remaining_test]
                    total_weight = sum(weights)

                    if total_weight > 0:
                        rand_val = np.random.random() * total_weight
                        if rand_val < weights[0]:
                            target_split = "train"
                        elif rand_val < weights[0] + weights[1]:
                            target_split = "val"
                        else:
                            target_split = "test"
                    else:
                        target_split = "train"  # Default fallback

            # Assign to target split
            if target_split == "train":
                train_recordings.append(recording["name"])
                train_brake_count += recording["brake_sequences"]
                train_total_count += recording["num_sequences"]
            elif target_split == "val":
                val_recordings.append(recording["name"])
                val_brake_count += recording["brake_sequences"]
                val_total_count += recording["num_sequences"]
            else:  # test
                test_recordings.append(recording["name"])
                test_brake_count += recording["brake_sequences"]
                test_total_count += recording["num_sequences"]

        # Calculate final statistics
        total_recordings = len(recording_stats)
        total_brake_sequences = sum(r["brake_sequences"] for r in recording_stats)
        total_all_sequences = sum(r["num_sequences"] for r in recording_stats)

        logger.info(f"📊 Sequence-aware dataset split:")
        logger.info(
            f"   🚆 Train: {len(train_recordings)} recordings ({len(train_recordings)/total_recordings:.1%})"
        )
        logger.info(
            f"      📈 {train_total_count} sequences ({train_total_count/total_all_sequences:.1%})"
        )
        logger.info(
            f"      🛑 {train_brake_count} brake sequences ({train_brake_count/total_brake_sequences:.1%})"
        )

        logger.info(
            f"   🔍 Val: {len(val_recordings)} recordings ({len(val_recordings)/total_recordings:.1%})"
        )
        logger.info(
            f"      📈 {val_total_count} sequences ({val_total_count/total_all_sequences:.1%})"
        )
        logger.info(
            f"      🛑 {val_brake_count} brake sequences ({val_brake_count/total_brake_sequences:.1%})"
        )

        logger.info(
            f"   🧪 Test: {len(test_recordings)} recordings ({len(test_recordings)/total_recordings:.1%})"
        )
        logger.info(
            f"      📈 {test_total_count} sequences ({test_total_count/total_all_sequences:.1%})"
        )
        logger.info(
            f"      🛑 {test_brake_count} brake sequences ({test_brake_count/total_brake_sequences:.1%})"
        )

        # Validate split quality
        train_brake_ratio = (
            train_brake_count / train_total_count if train_total_count > 0 else 0
        )
        val_brake_ratio = (
            val_brake_count / val_total_count if val_total_count > 0 else 0
        )
        test_brake_ratio = (
            test_brake_count / test_total_count if test_total_count > 0 else 0
        )

        logger.info(f"📊 Brake sequence ratios:")
        logger.info(f"   🚆 Train: {train_brake_ratio:.3f}")
        logger.info(f"   🔍 Val: {val_brake_ratio:.3f}")
        logger.info(f"   🧪 Test: {test_brake_ratio:.3f}")

        return train_recordings, val_recordings, test_recordings

    def _save_sequences_to_hdf5(
        self, sequences: List[Dict], output_file: Path, split_name: str
    ) -> Dict:
        """
        Save sequences to HDF5 file for efficient loading.

        Args:
            sequences: List of sequence dictionaries
            output_file: Path to output HDF5 file
            split_name: Name of the split (train/val/test)

        Returns:
            Dictionary with dataset statistics
        """
        logger.info(f"💾 Saving {len(sequences)} sequences to {output_file}")

        if not sequences:
            logger.error(f"❌ No sequences to save for {split_name}")
            return {}

        # Prepare data arrays
        telemetry_data = np.array(
            [seq["telemetry_seq"] for seq in sequences], dtype=np.float32
        )
        detection_data = np.array(
            [seq["detection_seq"] for seq in sequences], dtype=np.float32
        )
        detection_masks = np.array(
            [seq["detection_mask"] for seq in sequences], dtype=bool
        )

        # Prepare labels
        label_data = {}
        for horizon in [1, 2, 3, 4, 5]:
            brake_col = f"brake_{horizon}s"
            coast_col = f"coast_{horizon}s"

            label_data[brake_col] = np.array(
                [seq["labels"][brake_col] for seq in sequences], dtype=bool
            )
            label_data[coast_col] = np.array(
                [seq["labels"][coast_col] for seq in sequences], dtype=bool
            )

        # Metadata
        metadata = {
            "recording_names": [seq["recording_name"] for seq in sequences],
            "start_indices": [seq["start_idx"] for seq in sequences],
            "end_indices": [seq["end_idx"] for seq in sequences],
            "timestamps": [seq["timestamps"] for seq in sequences],
        }

        # Save to HDF5
        with h5py.File(output_file, "w") as f:
            # Data arrays
            f.create_dataset("telemetry", data=telemetry_data, compression="gzip")
            f.create_dataset("detections", data=detection_data, compression="gzip")
            f.create_dataset(
                "detection_masks", data=detection_masks, compression="gzip"
            )

            # Labels
            labels_group = f.create_group("labels")
            for label_name, label_array in label_data.items():
                labels_group.create_dataset(
                    label_name, data=label_array, compression="gzip"
                )

            # Metadata
            meta_group = f.create_group("metadata")
            meta_group.create_dataset(
                "recording_names",
                data=[s.encode("utf-8") for s in metadata["recording_names"]],
            )
            meta_group.create_dataset("start_indices", data=metadata["start_indices"])
            meta_group.create_dataset("end_indices", data=metadata["end_indices"])

            # Store timestamps as JSON strings
            timestamp_strings = [json.dumps(ts) for ts in metadata["timestamps"]]
            meta_group.create_dataset(
                "timestamps", data=[s.encode("utf-8") for s in timestamp_strings]
            )

            # Dataset info
            info_group = f.create_group("info")
            info_group.attrs["num_sequences"] = len(sequences)
            info_group.attrs["sequence_length"] = self.sequence_length
            info_group.attrs["max_detections"] = self.max_detections
            info_group.attrs["telemetry_dim"] = TELEMETRY_DIM
            info_group.attrs["detection_dim_per_box"] = DETECTION_DIM_PER_BOX
            info_group.attrs["telemetry_features"] = [
                f.encode("utf-8") for f in TELEMETRY_FEATURES
            ]

        # Calculate statistics
        stats = {
            "num_sequences": len(sequences),
            "shape_telemetry": telemetry_data.shape,
            "shape_detections": detection_data.shape,
            "shape_masks": detection_masks.shape,
            "label_stats": {},
        }

        # Label statistics
        for label_name, label_array in label_data.items():
            positive_count = np.sum(label_array)
            stats["label_stats"][label_name] = {
                "positive": int(positive_count),
                "negative": int(len(label_array) - positive_count),
                "ratio": float(positive_count / len(label_array)),
            }

        logger.info(f"✅ Saved {split_name} dataset:")
        logger.info(f"   📊 Sequences: {stats['num_sequences']}")
        logger.info(f"   📏 Telemetry shape: {stats['shape_telemetry']}")
        logger.info(f"   🎯 Detection shape: {stats['shape_detections']}")
        logger.info(f"   🎭 Mask shape: {stats['shape_masks']}")

        return stats

    def prepare_dataset(
        self,
        recordings_dir: str = RECORDING_OUTPUT_PATH,
        max_recordings: Optional[int] = None,
    ) -> bool:
        """
        Main method to prepare the complete dataset.

        Args:
            recordings_dir: Path to recordings directory
            max_recordings: Maximum number of recordings to process

        Returns:
            True if successful
        """
        recordings_path = Path(recordings_dir)

        if not recordings_path.exists():
            logger.error(f"❌ Recordings directory not found: {recordings_path}")
            return False

        # Get all valid recording directories
        logger.info("🔍 Scanning for valid recordings...")
        recording_dirs = []
        for d in sorted(recordings_path.iterdir()):
            if d.is_dir() and "-" in d.name and self._validate_recording(d):
                recording_dirs.append(d)

        if not recording_dirs:
            logger.error(f"❌ No valid recordings found in {recordings_path}")
            return False

        # Limit recordings if specified
        if max_recordings:
            recording_dirs = recording_dirs[:max_recordings]

        logger.info(f"✅ Found {len(recording_dirs)} valid recordings")

        # Load all recording data
        logger.info("📚 Loading recording data...")
        all_recording_data = []
        for recording_dir in tqdm(recording_dirs, desc="Loading recordings"):
            recording_data = self._load_recording_data(recording_dir)
            if recording_data:
                all_recording_data.append(recording_data)

        if not all_recording_data:
            logger.error("❌ No valid recording data loaded")
            return False

        logger.info(f"✅ Loaded data from {len(all_recording_data)} recordings")

        # Split recordings into train/val/test
        logger.info("🎲 Splitting recordings...")
        train_recordings, val_recordings, test_recordings = (
            self._split_recordings_stratified(all_recording_data)
        )

        # Extract sequences for each split
        all_stats = {}

        for split_name, recording_names in [
            ("train", train_recordings),
            ("val", val_recordings),
            ("test", test_recordings),
        ]:
            logger.info(f"🔄 Processing {split_name} split...")

            # Get recording data for this split
            split_recording_data = [
                rd
                for rd in all_recording_data
                if rd["recording_name"] in recording_names
            ]

            # Extract sequences
            all_sequences = []
            for recording_data in tqdm(
                split_recording_data, desc=f"Extracting {split_name} sequences"
            ):
                sequences = self._extract_sequences_from_recording(recording_data)
                all_sequences.extend(sequences)

            if not all_sequences:
                logger.warning(f"⚠️ No sequences extracted for {split_name} split")
                continue

            # Save to HDF5
            output_file = self.output_dir / f"{split_name}.h5"
            stats = self._save_sequences_to_hdf5(all_sequences, output_file, split_name)
            all_stats[split_name] = stats

        # Save dataset configuration
        config = {
            "sequence_length": self.sequence_length,
            "sequence_stride": self.sequence_stride,
            "max_detections": self.max_detections,
            "telemetry_features": TELEMETRY_FEATURES,
            "telemetry_dim": TELEMETRY_DIM,
            "detection_dim_per_box": DETECTION_DIM_PER_BOX,
            "splits": {
                "train": len(train_recordings),
                "val": len(val_recordings),
                "test": len(test_recordings),
            },
            "statistics": all_stats,
            "random_seed": self.random_seed,
        }

        config_file = self.output_dir / "dataset_config.json"
        with open(config_file, "w") as f:
            json.dump(config, f, indent=2)

        logger.info(f"💾 Saved dataset configuration to {config_file}")

        # Final summary
        total_sequences = sum(
            stats.get("num_sequences", 0) for stats in all_stats.values()
        )
        logger.info(f"🎉 Dataset preparation completed!")
        logger.info(f"   📊 Total sequences: {total_sequences}")
        logger.info(f"   📁 Output directory: {self.output_dir}")
        logger.info(f"   ⚙️ Configuration: {config_file}")

        return True


def main():
    """Main entry point for dataset preparation pipeline."""
    parser = argparse.ArgumentParser(description="Prepare multimodal dataset")
    parser.add_argument(
        "--recordings-dir",
        type=str,
        default=RECORDING_OUTPUT_PATH,
        help="Path to recordings directory",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=DEFAULT_OUTPUT_DIR,
        help="Output directory for prepared dataset",
    )
    parser.add_argument(
        "--sequence-length",
        type=int,
        default=SEQUENCE_LENGTH,
        help="Length of sequences in frames",
    )
    parser.add_argument(
        "--sequence-stride",
        type=int,
        default=SEQUENCE_STRIDE,
        help="Stride between sequences in frames",
    )
    parser.add_argument(
        "--max-detections",
        type=int,
        default=MAX_DETECTIONS,
        help="Maximum number of detections per frame",
    )
    parser.add_argument(
        "--max-recordings",
        type=int,
        help="Maximum number of recordings to process (for testing)",
    )
    parser.add_argument(
        "--random-seed",
        type=int,
        default=42,
        help="Random seed for reproducible splits",
    )

    args = parser.parse_args()

    try:
        # Initialize preparator
        preparator = DatasetPreparator(
            output_dir=args.output_dir,
            sequence_length=args.sequence_length,
            sequence_stride=args.sequence_stride,
            max_detections=args.max_detections,
            random_seed=args.random_seed,
        )

        # Prepare dataset
        success = preparator.prepare_dataset(
            recordings_dir=args.recordings_dir, max_recordings=args.max_recordings
        )

        sys.exit(0 if success else 1)

    except Exception as e:
        logger.error(f"❌ Pipeline failed with error: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
