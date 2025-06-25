#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Auto-Annotation Pipeline for Multimodal Training
Automatically annotates recorded images using finetuned YOLO model.
"""

import os
import sys
import time
import pandas as pd
import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm
import argparse
import logging
from typing import Dict, List, Optional, Tuple
import torch

# Add project root to path for imports
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from utils.feature_config import (
    DEFAULT_VISION_MODEL,
    MULTI_CLASS_CONFIG,
    SINGLE_CLASS_CONFIG,
    YOLO_BASE_CONFIDENCE,
    YOLO_IMG_SIZE,
)
from utils.config import RECORDING_OUTPUT_PATH, TIME_FORMAT_LOG
from utils.device import setup_device
from processing.image_processor import ImageProcessor
from ultralytics import YOLO

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler("auto_annotate.log"), logging.StreamHandler()],
)
logger = logging.getLogger(__name__)

# Model and annotation configuration
SAVE_PROGRESS_INTERVAL = 100

# Custom ROIs for different recordings (from training/datasets/annotation.py)
RECORDINGS_ROIS_MAP = {
    "2024-12-19_11-50-15": (0, 320, 1920, 575),
    "2024-12-19_12-58-38": (0, 320, 1920, 575),
    "2025-04-03_19-33-31": (0, 256, 1920, 575),
    "2025-04-03_21-05-24": (0, 240, 1920, 575),
    "2025-04-03_21-53-03": (0, 245, 1920, 575),
    "2025-05-15_13-56-57": (0, 320, 1920, 575),
    "2025-05-15_15-01-48": (0, 320, 1920, 575),
    "2025-05-21_06-27-07": (0, 345, 1920, 575),
    "2025-05-21_07-18-55": (0, 345, 1920, 575),
    "2025-05-21_07-38-16": (0, 345, 1920, 575),
    "2025-05-21_07-55-21": (0, 340, 1920, 575),
    "2025-05-21_08-06-10": (0, 340, 1920, 575),
}

DEFAULT_ROI = (0, 320, 1920, 575)  # Fallback ROI


class AutoAnnotator:
    """
    Handles automatic annotation of recorded images using YOLO model.
    """

    def __init__(
        self,
        model_path: str = DEFAULT_VISION_MODEL,
        use_multiclass: bool = False,
        confidence_threshold: float = YOLO_BASE_CONFIDENCE,
        img_size: int = YOLO_IMG_SIZE,
        device: Optional[str] = None,
    ):
        self.model_path = Path(model_path)
        self.use_multiclass = use_multiclass
        self.confidence_threshold = confidence_threshold
        self.img_size = img_size

        # Setup device
        self.device = device if device else setup_device()
        logger.info(f"🔧 Using device: {self.device}")

        # Configure class settings
        self.class_config = (
            MULTI_CLASS_CONFIG if use_multiclass else SINGLE_CLASS_CONFIG
        )
        self.vehicle_classes = self._get_vehicle_classes()

        # Load model
        self.model = self._load_model()

        logger.info(
            f"🎯 Annotation mode: {'Multi-class' if use_multiclass else 'Single-class'}"
        )
        logger.info(
            f"🚗 Vehicle classes: {[self.class_config[cls]['name'] for cls in self.vehicle_classes]}"
        )

    def _load_model(self) -> YOLO:
        """Load and configure YOLO model."""
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model not found: {self.model_path}")

        logger.info(f"🤖 Loading YOLO model from: {self.model_path}")
        model = YOLO(str(self.model_path), task="detect")
        logger.info("✅ Model loaded successfully!")

        return model

    def _get_vehicle_classes(self) -> List[int]:
        """Get list of vehicle class IDs based on configuration."""
        if self.use_multiclass:
            # Exclude person class (0), keep only highway relevant vehicle classes
            return [
                1,
                2,
                3,
                4,
                5,
            ]  # car, truck, motorcycle, bus, trafficcone (might give relevant information for caution)
        else:
            # Single vehicle class
            return [0]

    def _get_roi_for_recording(self, recording_name: str) -> Tuple[int, int, int, int]:
        """Get ROI for specific recording or return default."""
        roi = RECORDINGS_ROIS_MAP.get(recording_name, DEFAULT_ROI)
        logger.debug(f"📐 ROI for {recording_name}: {roi}")
        return roi

    def _validate_recording_structure(self, recording_dir: Path) -> bool:
        """
        Validate that recording directory has expected structure.

        Args:
            recording_dir: Path to recording directory

        Returns:
            True if structure is valid
        """
        telemetry_file = recording_dir / "telemetry.csv"
        image_files = list(recording_dir.glob("*.jpg"))

        if not telemetry_file.exists():
            logger.warning(f"⚠️ No telemetry.csv found in {recording_dir.name}")
            return False

        if not image_files:
            logger.warning(f"⚠️ No image files found in {recording_dir.name}")
            return False

        # Check if annotations already exist
        annotations_file = recording_dir / "annotations.csv"
        if annotations_file.exists():
            logger.info(f"📋 Annotations already exist for {recording_dir.name}")
            return "exists"

        return True

    def _load_telemetry_timestamps(self, recording_dir: Path) -> List[str]:
        """Load timestamps from telemetry CSV."""
        telemetry_file = recording_dir / "telemetry.csv"
        try:
            df = pd.read_csv(telemetry_file)
            timestamps = df["Time"].astype(str).tolist()
            logger.debug(f"📊 Loaded {len(timestamps)} timestamps from telemetry")
            return timestamps
        except Exception as e:
            logger.error(f"❌ Failed to load telemetry timestamps: {e}")
            return []

    def _process_image(
        self,
        img_path: Path,
        roi: Tuple[int, int, int, int],
        telemetry_timestamps: List[str],
    ) -> Optional[Dict]:
        """
        Process single image and return annotation data.

        Args:
            img_path: Path to image file
            roi: Region of interest (x, y, width, height)
            telemetry_timestamps: List of valid timestamps

        Returns:
            Dictionary with annotation data or None if failed
        """
        # Extract timestamp from filename
        filename = img_path.name
        timestamp = filename.split(".")[0]

        # Check if timestamp exists in telemetry
        if timestamp not in telemetry_timestamps:
            logger.debug(f"⏭️ Timestamp {timestamp} not in telemetry, skipping")
            return None

        # Read and validate image
        img = cv2.imread(str(img_path))
        if img is None:
            logger.warning(f"❌ Failed to read image: {img_path}")
            return None

        # Crop to ROI
        img_cropped = ImageProcessor.crop_to_roi(img, roi)
        if img_cropped is None:
            logger.warning(f"❌ Failed to crop image: {img_path}")
            return None

        # Run YOLO detection
        try:
            results = self.model(
                img_cropped,
                conf=self.confidence_threshold,
                device=self.device,
                imgsz=self.img_size,
                verbose=False,
            )
        except Exception as e:
            logger.error(f"❌ YOLO inference failed for {img_path}: {e}")
            return None

        # Process detections
        class_ids = []
        confidences = []
        bboxes = []
        areas = []

        for result in results:
            boxes = result.boxes
            if boxes is None:
                continue

            for box in boxes:
                cls_id = int(box.cls.item())
                conf = float(box.conf.item())

                # Filter for vehicle classes and apply confidence threshold
                if (
                    cls_id in self.vehicle_classes
                    and cls_id in self.class_config
                    and conf >= self.class_config[cls_id]["confidence"]
                ):

                    x1, y1, x2, y2 = box.xyxy[0].tolist()
                    area = (x2 - x1) * (y2 - y1)

                    class_ids.append(cls_id)
                    confidences.append(conf)
                    bboxes.append([float(x1), float(y1), float(x2), float(y2)])
                    areas.append(float(area))

        return {
            "Time": timestamp,
            "filename": filename,
            "vehicle_count": len(class_ids),
            "class_ids": str(class_ids),
            "confidences": str(confidences),
            "bboxes": str(bboxes),
            "areas": str(areas),
        }

    def annotate_recording(
        self, recording_dir: Path, force_overwrite: bool = False
    ) -> bool:
        """
        Annotate all images in a recording directory.

        Args:
            recording_dir: Path to recording directory
            force_overwrite: Whether to overwrite existing annotations

        Returns:
            True if successful
        """
        logger.info(f"🎯 Processing recording: {recording_dir.name}")

        # Validate recording structure
        validation_result = self._validate_recording_structure(recording_dir)
        if validation_result is False:
            logger.error(f"❌ Invalid recording structure: {recording_dir.name}")
            return False
        elif validation_result == "exists" and not force_overwrite:
            logger.info(f"⏭️ Skipping {recording_dir.name} (annotations exist)")
            return True

        # Get ROI for this recording
        roi = self._get_roi_for_recording(recording_dir.name)

        # Load telemetry timestamps
        telemetry_timestamps = self._load_telemetry_timestamps(recording_dir)
        if not telemetry_timestamps:
            logger.error(f"❌ No valid telemetry timestamps for {recording_dir.name}")
            return False

        # Get all image files
        image_files = sorted(list(recording_dir.glob("*.jpg")))
        if not image_files:
            logger.error(f"❌ No image files found in {recording_dir.name}")
            return False

        logger.info(f"📸 Found {len(image_files)} images in {recording_dir.name}")

        # Initialize annotations dataframe
        annotations_df = pd.DataFrame(
            columns=[
                "Time",
                "filename",
                "vehicle_count",
                "class_ids",
                "confidences",
                "bboxes",
                "areas",
            ]
        )

        # Process images
        start_time = time.time()
        processed_count = 0
        skipped_count = 0

        for i, img_path in enumerate(
            tqdm(image_files, desc=f"Annotating {recording_dir.name}")
        ):
            annotation_data = self._process_image(img_path, roi, telemetry_timestamps)

            if annotation_data:
                annotations_df.loc[len(annotations_df)] = annotation_data
                processed_count += 1
            else:
                skipped_count += 1

            # Save progress periodically
            if (i + 1) % SAVE_PROGRESS_INTERVAL == 0 or i == len(image_files) - 1:
                annotations_path = recording_dir / "annotations.csv"
                annotations_df.to_csv(annotations_path, index=False)

        # Final save
        annotations_path = recording_dir / "annotations.csv"
        annotations_df.to_csv(annotations_path, index=False)

        # Print statistics
        elapsed_time = time.time() - start_time
        total_detections = annotations_df["vehicle_count"].sum()

        logger.info(f"✅ Completed annotation for {recording_dir.name}")
        logger.info(f"   📊 Processed: {processed_count}, Skipped: {skipped_count}")
        logger.info(f"   🚗 Total vehicles detected: {total_detections}")
        logger.info(f"   ⏱️ Processing time: {elapsed_time:.2f}s")
        logger.info(f"   🚀 Speed: {len(image_files) / elapsed_time:.2f} images/second")
        logger.info(f"   💾 Saved to: {annotations_path}")

        return True

    def annotate_all_recordings(
        self,
        recordings_dir: str = RECORDING_OUTPUT_PATH,
        force_overwrite: bool = False,
        max_recordings: Optional[int] = None,
    ) -> bool:
        """
        Annotate all recordings in the recordings directory.

        Args:
            recordings_dir: Path to recordings directory
            force_overwrite: Whether to overwrite existing annotations
            max_recordings: Maximum number of recordings to process (for testing)

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

        # Limit recordings if specified (for testing)
        if max_recordings:
            recording_dirs = recording_dirs[:max_recordings]

        logger.info(f"🎯 Found {len(recording_dirs)} recording directories")
        logger.info(f"📋 Annotation configuration:")
        logger.info(f"   🤖 Model: {self.model_path}")
        logger.info(f"   🎚️ Confidence: {self.confidence_threshold}")
        logger.info(f"   📏 Image size: {self.img_size}")
        logger.info(f"   🔄 Force overwrite: {force_overwrite}")

        # Process each recording
        successful = 0
        failed = 0

        for i, recording_dir in enumerate(recording_dirs, 1):
            logger.info(
                f"📁 Processing {i}/{len(recording_dirs)}: {recording_dir.name}"
            )

            if self.annotate_recording(recording_dir, force_overwrite):
                successful += 1
            else:
                failed += 1
                logger.error(f"❌ Failed to process {recording_dir.name}")

        # Final summary
        logger.info(f"🎉 Annotation pipeline completed!")
        logger.info(f"   ✅ Successful: {successful}")
        logger.info(f"   ❌ Failed: {failed}")
        logger.info(f"   📊 Success rate: {successful/len(recording_dirs)*100:.1f}%")

        return failed == 0


def main():
    """Main entry point for auto-annotation pipeline."""
    parser = argparse.ArgumentParser(description="Auto-annotate recordings with YOLO")
    parser.add_argument(
        "--recordings-dir",
        type=str,
        default=RECORDING_OUTPUT_PATH,
        help="Path to recordings directory",
    )
    parser.add_argument(
        "--model", type=str, default=DEFAULT_VISION_MODEL, help="Path to YOLO model"
    )
    parser.add_argument(
        "--multiclass",
        action="store_true",
        default=False,
        help="Use multi-class annotation (person, car, truck, etc.)",
    )
    parser.add_argument(
        "--confidence",
        type=float,
        default=YOLO_BASE_CONFIDENCE,
        help="Confidence threshold for detections",
    )
    parser.add_argument(
        "--img-size",
        type=int,
        default=YOLO_IMG_SIZE,
        help="Image size for YOLO inference",
    )
    parser.add_argument(
        "--force", action="store_true", help="Overwrite existing annotations"
    )
    parser.add_argument(
        "--max-recordings",
        type=int,
        help="Maximum number of recordings to process (for testing)",
    )

    args = parser.parse_args()

    model_path = (
        "data/models/yolo/" + args.model if not "/" in args.model else args.model
    )
    if not Path(model_path).exists():
        model_path += ".pt"  # Ensure .pt extension if not provided
    if not Path(model_path).exists():
        logger.error(f"❌ Model file not found: {model_path}")
        sys.exit(1)
    logger.info(f"🔧 Using model: {model_path}")

    try:
        # Initialize annotator
        annotator = AutoAnnotator(
            model_path=model_path,
            use_multiclass=args.multiclass,
            confidence_threshold=args.confidence,
            img_size=args.img_size,
            device=-1,
        )

        # Run annotation pipeline
        success = annotator.annotate_all_recordings(
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
