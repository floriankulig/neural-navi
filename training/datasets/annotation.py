#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
YOLO-based Vehicle Annotation Script for DriveRecorder Images

This script automatically annotates images in recording folders using a YOLO model
and saves annotations to CSV files. It supports class-specific confidence thresholds.
"""

import sys
import time
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import cv2

# Add parent directory to path for imports from main project
script_dir = Path(__file__).parent
project_root = script_dir.parent
sys.path.insert(0, str(project_root))

# Import custom modules
from device import setup_device
from ultralytics import YOLO
from config import (
    DEFAULT_VISION_MODEL,
    DEFAULT_IMAGE_ROI,
    TIME_FORMAT_FILES,
    TIME_FORMAT_LOG,
)
from imageprocessor import ImageProcessor


# ===== Configuration =====
# Model settings
MODEL_PATH = "yolo_best.pt"  # Path to your finetuned model
MODEL_IMG_SIZE = 640  # Image size for YOLO model

# Path settings
RECORDINGS_PATH = "deep/data/recordings"  # Path to recordings folder
RESULTS_PATH = "deep/data/annotations"  # Path to save results
SHOULD_CROP = True  # Whether to crop images to ROI

SAVE_PROGRESS_INTERVAL = 100  # Save progress every 50 images

# Vehicle classes - adjust based on your model
# Format: {class_id: {'name': 'class_name', 'confidence': threshold}}
CLASS_CONFIG = {
    # Custom model classes - uncomment if needed
    0: {"name": "person", "confidence": 0.3},
    1: {"name": "vehicle.car", "confidence": 0.3},
    2: {"name": "vehicle.truck", "confidence": 0.3},
    3: {"name": "vehicle.motorcycle", "confidence": 0.3},
    4: {"name": "vehicle.bus", "confidence": 0.3},
    5: {"name": "trafficcone", "confidence": 0.3},
}

# We use these custom ROIs because for each recording, the camera angle differs a slight bit.
# This ensures that the region of interest matches the relevant area in each dataset.
RECORDINGS_ROIS_MAP = {
    # Example: "recording_name": [x1, y1, width, height]
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


def setup():
    """Initialize device and model"""
    # Set up device for inference
    device = setup_device()
    print(f"Using device: {device}")

    # Load model
    print(f"Loading YOLO model from: {MODEL_PATH}")
    model = YOLO(MODEL_PATH, task="detect")
    print("Model loaded successfully!")

    vehicle_classes = list(CLASS_CONFIG.keys())[
        1:
    ]  # Skip class 0 (person), as it's not relevant on the autobahn
    print(f"Using custom vehicle classes: {vehicle_classes}")

    return model, device, vehicle_classes


def get_recording_dirs(recordings_path=RECORDINGS_PATH):
    """Get all recording directories"""
    recording_dirs = sorted([d for d in Path(recordings_path).iterdir() if d.is_dir()])

    if not recording_dirs:
        print(f"No recording directories found in {recordings_path}")
        return []

    print(f"Found {len(recording_dirs)} recording directories:")
    for i, dir_path in enumerate(recording_dirs):
        print(f"{i+1}: {dir_path.name}")

    return recording_dirs


def process_directory(recording_dir, model, device, vehicle_classes, crops=True):
    """
    Process all images in the given directory using the YOLO model
    and save annotations to a CSV file.

    Args:
        recording_dir: Path to the recording directory
        model: YOLO model
        device: Device to run inference on
        vehicle_classes: List of vehicle class IDs to detect

    Returns:
        Path to the saved annotations file
    """
    # Get all jpg files in the directory
    image_files = sorted(list(Path(recording_dir).glob("*.jpg")))
    # image_files = image_files[:5]  # For testing, process only the first 10 images

    telemetry_timestamps = pd.read_csv(recording_dir / "telemetry.csv")["Time"].astype(
        str
    )
    telemetry_timestamps = list(
        pd.read_csv(recording_dir / "telemetry.csv")["Time"].astype(str)
    )
    print(f"Found {len(telemetry_timestamps)} timestamps in {recording_dir}")

    if not image_files:
        print(f"No jpg files found in {recording_dir}")
        return None

    total_images = len(image_files)
    print(f"Found {total_images} jpg files in {recording_dir}")

    # Create annotations dataframe
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

    # Process each image
    start_time = time.time()
    for i, img_path in enumerate(
        tqdm(image_files, desc=f"Processing {recording_dir.name}")
    ):
        # Extract timestamp from filename (remove extension)
        filename = img_path.name
        timestamp = filename.split(".")[0]

        # Read the image
        img = cv2.imread(str(img_path))
        if img is None:
            print(f"Failed to read image: {img_path}")
            continue

        if timestamp not in telemetry_timestamps:
            print(f"Timestamp {timestamp} not found in telemetry.csv")
            continue

        # Convert BGR to RGB and crop to ROI
        # img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) # images are already saved in RGB
        if crops:
            roi_for_recording = RECORDINGS_ROIS_MAP.get(recording_dir.name, None)
            img_cropped = ImageProcessor.crop_to_roi(img, roi_for_recording)
        else:
            img_cropped = img.copy()

        if img_cropped is None:
            print(f"Failed to crop image: {img_path}")
            continue

        # Run detection with base confidence (we'll filter by class-specific confidence later)
        # Use lowest confidence threshold from CLASS_CONFIG to catch all potential detections
        base_confidence = min([cfg["confidence"] for cfg in CLASS_CONFIG.values()])
        results = model(
            img_cropped,
            conf=base_confidence,
            device=device,
            imgsz=MODEL_IMG_SIZE,
            verbose=False,
        )

        # Process detections with class-specific confidence thresholds
        class_ids = []
        confidences = []
        bboxes = []
        areas = []

        for result in results:
            boxes = result.boxes
            for box in boxes:
                cls_id = int(box.cls.item())
                conf = float(box.conf.item())

                # Only consider vehicle classes and apply class-specific confidence threshold
                if (
                    cls_id in vehicle_classes
                    and cls_id in CLASS_CONFIG
                    and conf >= CLASS_CONFIG[cls_id]["confidence"]
                ):
                    x1, y1, x2, y2 = box.xyxy[0].tolist()
                    area = (x2 - x1) * (y2 - y1)

                    class_ids.append(cls_id)
                    confidences.append(conf)
                    bboxes.append([float(x1), float(y1), float(x2), float(y2)])
                    areas.append(float(area))

        # Add row to dataframe
        annotations_df.loc[len(annotations_df)] = [
            timestamp,
            filename,
            len(class_ids),
            str(class_ids),
            str(confidences),
            str(bboxes),
            str(areas),
        ]

        # Save progress periodically
        if (i + 1) % SAVE_PROGRESS_INTERVAL == 0 or i == total_images - 1:
            annotations_path = Path(recording_dir) / "annotations.csv"
            annotations_df.to_csv(annotations_path, index=False)
            elapsed_time = time.time() - start_time

    # Save final annotations to CSV
    annotations_path = Path(recording_dir) / "annotations.csv"
    annotations_df.to_csv(annotations_path, index=False)
    print(f"✅ Annotations saved to {annotations_path}")

    # Print statistics
    elapsed_time = time.time() - start_time
    print(f"Processing time: {elapsed_time:.2f} seconds")
    print(f"Processing speed: {total_images / elapsed_time:.2f} images/second")
    print(f"Total vehicles detected: {annotations_df['vehicle_count'].sum()}")

    return annotations_path


def main():
    """Main entry point"""
    print("=" * 80)
    print("YOLO Vehicle Annotation Script")
    print("=" * 80)

    # Initialize model and device
    model, device, vehicle_classes = setup()

    # Get all recording directories
    recording_dirs = get_recording_dirs()
    # recording_dirs = recording_dirs[:1]  # For testing, process only the first directory

    if not recording_dirs:
        print("No recording directories found. Exiting.")
        return

    print(f"\nStarting annotation process for {len(recording_dirs)} directories")
    print("=" * 80)

    # Process each directory
    for i, recording_dir in enumerate(recording_dirs):
        print(
            f"\nProcessing directory {i+1}/{len(recording_dirs)}: {recording_dir.name}"
        )
        # Copy telemetry.csv and annotations.csv to the results directory
        annotations_path = process_directory(
            recording_dir, model, device, vehicle_classes, crops=SHOULD_CROP
        )
        telemetry_path = recording_dir / "telemetry.csv"

        # Save to results/[timestamp]
        results_dir = Path(RESULTS_PATH) / recording_dir.name
        results_dir.mkdir(parents=True, exist_ok=True)

        telemetry_copy_path = results_dir / "telemetry.csv"
        annotations_copy_path = results_dir / "annotations.csv"
        telemetry_copy_path.write_text(telemetry_path.read_text())
        annotations_copy_path.write_text(annotations_path.read_text())

    print("\n" + "=" * 80)
    print("Annotation process completed for all directories!")
    print("=" * 80)


if __name__ == "__main__":
    main()
