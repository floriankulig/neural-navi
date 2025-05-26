#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Future Label Generation Pipeline for Multimodal Training
Generates future labels for braking and coasting events from telemetry data.
"""

import os
import sys
import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm
import argparse
import logging
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta

# Add project root to path for imports
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from src.utils.config import RECORDING_OUTPUT_PATH, TIME_FORMAT_LOG

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler("generate_labels.log"), logging.StreamHandler()],
)
logger = logging.getLogger(__name__)

# Label generation configuration
SAMPLING_RATE_HZ = 2.0  # 2Hz sampling rate
SAMPLING_INTERVAL_S = 1.0 / SAMPLING_RATE_HZ  # 0.5 seconds per frame
COAST_THRESHOLD_PERCENT = 10.0  # Coast when accelerator < 10%
FUTURE_HORIZONS = [1, 2, 3, 4, 5]  # Future horizons in seconds

# Column mappings for telemetry data
EXPECTED_COLUMNS = [
    "Time",
    "SPEED",
    "RPM",
    "ACCELERATOR_POS_D",
    "ENGINE_LOAD",
    "BRAKE_SIGNAL",
    "GEAR",
    "BRAKE_FORCE",
]


class FutureLabelGenerator:
    """
    Generates future labels for braking and coasting events from telemetry data.
    """

    def __init__(self, coast_threshold: float = COAST_THRESHOLD_PERCENT):
        self.coast_threshold = coast_threshold
        self.sampling_rate = SAMPLING_RATE_HZ
        self.sampling_interval = SAMPLING_INTERVAL_S

        logger.info(f"🎯 Future Label Generator initialized")
        logger.info(f"   📊 Sampling rate: {self.sampling_rate} Hz")
        logger.info(f"   ⏱️ Sampling interval: {self.sampling_interval}s")
        logger.info(f"   🛑 Coast threshold: {self.coast_threshold}%")
        logger.info(f"   🔮 Future horizons: {FUTURE_HORIZONS}s")

    def _parse_timestamp(self, timestamp_str: str) -> Optional[datetime]:
        """
        Parse timestamp string to datetime object.

        Args:
            timestamp_str: Timestamp string in TIME_FORMAT_LOG format

        Returns:
            Datetime object or None if parsing failed
        """
        try:
            # Remove the last digit from microseconds (format has 6 digits, we need 6)
            # TIME_FORMAT_LOG is "%Y-%m-%d %H-%M-%S-%f" but we save without last digit
            return datetime.strptime(timestamp_str, "%Y-%m-%d %H-%M-%S-%f")
        except ValueError:
            try:
                # Fallback: try without microseconds
                return datetime.strptime(timestamp_str, "%Y-%m-%d %H-%M-%S")
            except ValueError:
                logger.warning(f"⚠️ Failed to parse timestamp: {timestamp_str}")
                return None

    def _calculate_frame_offsets(self, horizons: List[int]) -> Dict[int, int]:
        """
        Calculate frame offsets for future horizons.

        Args:
            horizons: List of future horizons in seconds

        Returns:
            Dictionary mapping horizon to frame offset
        """
        frame_offsets = {}
        for horizon in horizons:
            # Calculate number of frames for this horizon
            frames = int(horizon * self.sampling_rate)
            frame_offsets[horizon] = frames
            logger.debug(f"🔢 Horizon {horizon}s = {frames} frames")

        return frame_offsets

    def _validate_telemetry_data(self, df: pd.DataFrame) -> bool:
        """
        Validate telemetry DataFrame structure and content.

        Args:
            df: Telemetry DataFrame

        Returns:
            True if valid, False otherwise
        """
        # Check required columns
        missing_cols = set(EXPECTED_COLUMNS) - set(df.columns)
        if missing_cols:
            logger.error(f"❌ Missing columns: {missing_cols}")
            return False

        # Check for empty DataFrame
        if df.empty:
            logger.error("❌ Empty telemetry DataFrame")
            return False

        # Check for valid timestamps
        valid_timestamps = df["Time"].apply(
            lambda x: self._parse_timestamp(str(x)) is not None
        )
        invalid_count = (~valid_timestamps).sum()

        if invalid_count > 0:
            logger.warning(f"⚠️ Found {invalid_count} invalid timestamps")

        # Check for required data types
        try:
            df["BRAKE_SIGNAL"] = df["BRAKE_SIGNAL"].astype(bool)
            df["ACCELERATOR_POS_D"] = pd.to_numeric(
                df["ACCELERATOR_POS_D"], errors="coerce"
            )
        except Exception as e:
            logger.error(f"❌ Data type conversion failed: {e}")
            return False

        logger.debug(f"✅ Telemetry validation passed ({len(df)} rows)")
        return True

    def _clean_telemetry_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Clean and prepare telemetry data for processing.

        Args:
            df: Raw telemetry DataFrame

        Returns:
            Cleaned DataFrame
        """
        # Create a copy to avoid modifying original
        df_clean = df.copy()

        # Parse timestamps and add datetime column
        df_clean["DateTime"] = df_clean["Time"].apply(
            lambda x: self._parse_timestamp(str(x))
        )

        # Remove rows with invalid timestamps
        df_clean = df_clean.dropna(subset=["DateTime"])

        # Sort by timestamp
        df_clean = df_clean.sort_values("DateTime").reset_index(drop=True)

        # Convert data types
        df_clean["BRAKE_SIGNAL"] = df_clean["BRAKE_SIGNAL"].astype(bool)
        df_clean["ACCELERATOR_POS_D"] = pd.to_numeric(
            df_clean["ACCELERATOR_POS_D"], errors="coerce"
        )

        # Remove rows with invalid accelerator data
        df_clean = df_clean.dropna(subset=["ACCELERATOR_POS_D"])

        logger.debug(f"🧹 Cleaned data: {len(df)} -> {len(df_clean)} rows")
        return df_clean

    def _generate_future_labels_for_recording(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Generate future labels for a single recording.

        Args:
            df: Cleaned telemetry DataFrame

        Returns:
            DataFrame with future labels
        """
        # Calculate frame offsets
        frame_offsets = self._calculate_frame_offsets(FUTURE_HORIZONS)

        # Initialize result DataFrame
        result_data = {"Time": df["Time"].tolist(), "DateTime": df["DateTime"].tolist()}

        # Add columns for each horizon
        for horizon in FUTURE_HORIZONS:
            result_data[f"brake_{horizon}s"] = [False] * len(df)
            result_data[f"coast_{horizon}s"] = [False] * len(df)

        # Generate labels for each row
        for i in range(len(df)):
            for horizon in FUTURE_HORIZONS:
                offset = frame_offsets[horizon]
                future_idx = i + offset

                # Check if future index is within bounds
                if future_idx < len(df):
                    # Brake label: BRAKE_SIGNAL == True at future time
                    future_brake = df.iloc[future_idx]["BRAKE_SIGNAL"]
                    result_data[f"brake_{horizon}s"][i] = bool(future_brake)

                    # Coast label: ACCELERATOR_POS_D < threshold at future time
                    future_accel = df.iloc[future_idx]["ACCELERATOR_POS_D"]
                    result_data[f"coast_{horizon}s"][i] = (
                        float(future_accel) < self.coast_threshold
                    )
                else:
                    # Future index out of bounds - set to False
                    result_data[f"brake_{horizon}s"][i] = False
                    result_data[f"coast_{horizon}s"][i] = False

        return pd.DataFrame(result_data)

    def _calculate_label_statistics(self, labels_df: pd.DataFrame) -> Dict:
        """
        Calculate statistics for generated labels.

        Args:
            labels_df: DataFrame with future labels

        Returns:
            Dictionary with statistics
        """
        stats = {
            "total_samples": len(labels_df),
            "brake_events": {},
            "coast_events": {},
            "label_distribution": {},
        }

        for horizon in FUTURE_HORIZONS:
            brake_col = f"brake_{horizon}s"
            coast_col = f"coast_{horizon}s"

            brake_count = labels_df[brake_col].sum()
            coast_count = labels_df[coast_col].sum()

            stats["brake_events"][horizon] = {
                "count": int(brake_count),
                "percentage": float(brake_count / len(labels_df) * 100),
            }

            stats["coast_events"][horizon] = {
                "count": int(coast_count),
                "percentage": float(coast_count / len(labels_df) * 100),
            }

        return stats

    def process_recording(
        self, recording_dir: Path, force_overwrite: bool = False
    ) -> bool:
        """
        Process a single recording directory to generate future labels.

        Args:
            recording_dir: Path to recording directory
            force_overwrite: Whether to overwrite existing labels

        Returns:
            True if successful
        """
        logger.info(f"🎯 Processing recording: {recording_dir.name}")

        # Check if telemetry file exists
        telemetry_file = recording_dir / "telemetry.csv"
        if not telemetry_file.exists():
            logger.error(f"❌ No telemetry.csv found in {recording_dir.name}")
            return False

        # Check if labels already exist
        labels_file = recording_dir / "future_labels.csv"
        if labels_file.exists() and not force_overwrite:
            logger.info(f"⏭️ Labels already exist for {recording_dir.name}")
            return True

        try:
            # Load telemetry data
            df = pd.read_csv(telemetry_file)
            logger.debug(f"📊 Loaded telemetry: {len(df)} rows")

            # Validate data
            if not self._validate_telemetry_data(df):
                logger.error(f"❌ Telemetry validation failed for {recording_dir.name}")
                return False

            # Clean data
            df_clean = self._clean_telemetry_data(df)
            if df_clean.empty:
                logger.error(
                    f"❌ No valid data after cleaning for {recording_dir.name}"
                )
                return False

            # Generate future labels
            labels_df = self._generate_future_labels_for_recording(df_clean)

            # Calculate statistics
            stats = self._calculate_label_statistics(labels_df)

            # Save labels (exclude DateTime column for consistency)
            output_df = labels_df.drop(columns=["DateTime"])
            output_df.to_csv(labels_file, index=False)

            # Log statistics
            logger.info(f"✅ Generated labels for {recording_dir.name}")
            logger.info(f"   📊 Total samples: {stats['total_samples']}")

            for horizon in FUTURE_HORIZONS:
                brake_stats = stats["brake_events"][horizon]
                coast_stats = stats["coast_events"][horizon]
                logger.info(
                    f"   🛑 Brake {horizon}s: {brake_stats['count']} ({brake_stats['percentage']:.1f}%)"
                )
                logger.info(
                    f"   ⛵ Coast {horizon}s: {coast_stats['count']} ({coast_stats['percentage']:.1f}%)"
                )

            logger.info(f"   💾 Saved to: {labels_file}")
            return True

        except Exception as e:
            logger.error(f"❌ Failed to process {recording_dir.name}: {e}")
            import traceback

            traceback.print_exc()
            return False

    def process_all_recordings(
        self,
        recordings_dir: str = RECORDING_OUTPUT_PATH,
        force_overwrite: bool = False,
        max_recordings: Optional[int] = None,
    ) -> bool:
        """
        Process all recordings to generate future labels.

        Args:
            recordings_dir: Path to recordings directory
            force_overwrite: Whether to overwrite existing labels
            max_recordings: Maximum number of recordings to process

        Returns:
            True if all successful
        """
        recordings_path = Path(recordings_dir)

        if not recordings_path.exists():
            logger.error(f"❌ Recordings directory not found: {recordings_path}")
            return False

        # Get all recording directories
        recording_dirs = sorted(
            [
                d
                for d in recordings_path.iterdir()
                if d.is_dir() and "-" in d.name  # Timestamp format check
            ]
        )

        if not recording_dirs:
            logger.error(f"❌ No recording directories found in {recordings_path}")
            return False

        # Limit recordings if specified
        if max_recordings:
            recording_dirs = recording_dirs[:max_recordings]

        logger.info(f"🎯 Found {len(recording_dirs)} recording directories")
        logger.info(f"📋 Label generation configuration:")
        logger.info(f"   📊 Sampling rate: {self.sampling_rate} Hz")
        logger.info(f"   🛑 Coast threshold: {self.coast_threshold}%")
        logger.info(f"   🔮 Future horizons: {FUTURE_HORIZONS}s")
        logger.info(f"   🔄 Force overwrite: {force_overwrite}")

        # Process each recording
        successful = 0
        failed = 0
        total_stats = {
            "total_samples": 0,
            "brake_events": {h: 0 for h in FUTURE_HORIZONS},
            "coast_events": {h: 0 for h in FUTURE_HORIZONS},
        }

        for i, recording_dir in enumerate(recording_dirs, 1):
            logger.info(
                f"📁 Processing {i}/{len(recording_dirs)}: {recording_dir.name}"
            )

            if self.process_recording(recording_dir, force_overwrite):
                successful += 1

                # Accumulate statistics
                labels_file = recording_dir / "future_labels.csv"
                if labels_file.exists():
                    try:
                        labels_df = pd.read_csv(labels_file)
                        total_stats["total_samples"] += len(labels_df)

                        for horizon in FUTURE_HORIZONS:
                            total_stats["brake_events"][horizon] += labels_df[
                                f"brake_{horizon}s"
                            ].sum()
                            total_stats["coast_events"][horizon] += labels_df[
                                f"coast_{horizon}s"
                            ].sum()
                    except Exception as e:
                        logger.warning(
                            f"⚠️ Could not read stats from {labels_file}: {e}"
                        )
            else:
                failed += 1
                logger.error(f"❌ Failed to process {recording_dir.name}")

        # Final summary
        logger.info(f"🎉 Label generation pipeline completed!")
        logger.info(f"   ✅ Successful: {successful}")
        logger.info(f"   ❌ Failed: {failed}")
        logger.info(f"   📊 Success rate: {successful/len(recording_dirs)*100:.1f}%")

        if total_stats["total_samples"] > 0:
            logger.info(f"📈 Overall Statistics:")
            logger.info(f"   📊 Total samples: {total_stats['total_samples']}")

            for horizon in FUTURE_HORIZONS:
                brake_count = total_stats["brake_events"][horizon]
                coast_count = total_stats["coast_events"][horizon]
                brake_pct = brake_count / total_stats["total_samples"] * 100
                coast_pct = coast_count / total_stats["total_samples"] * 100

                logger.info(f"   🛑 Brake {horizon}s: {brake_count} ({brake_pct:.1f}%)")
                logger.info(f"   ⛵ Coast {horizon}s: {coast_count} ({coast_pct:.1f}%)")

        return failed == 0


def main():
    """Main entry point for future label generation pipeline."""
    parser = argparse.ArgumentParser(
        description="Generate future labels from telemetry data"
    )
    parser.add_argument(
        "--recordings-dir",
        type=str,
        default=RECORDING_OUTPUT_PATH,
        help="Path to recordings directory",
    )
    parser.add_argument(
        "--coast-threshold",
        type=float,
        default=COAST_THRESHOLD_PERCENT,
        help="Accelerator threshold for coasting detection (percentage)",
    )
    parser.add_argument(
        "--force", action="store_true", help="Overwrite existing labels"
    )
    parser.add_argument(
        "--max-recordings",
        type=int,
        help="Maximum number of recordings to process (for testing)",
    )

    args = parser.parse_args()

    try:
        # Initialize label generator
        generator = FutureLabelGenerator(coast_threshold=args.coast_threshold)

        # Run label generation pipeline
        success = generator.process_all_recordings(
            recordings_dir=args.recordings_dir,
            force_overwrite=args.force,
            max_recordings=args.max_recordings,
        )

        sys.exit(0 if success else 1)

    except Exception as e:
        logger.error(f"❌ Pipeline failed with error: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
