#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Vehicle detector for DriveRecorder images using YOLO
Optimized for Apple Silicon (M3)
"""

import argparse
from pathlib import Path
import cv2
import numpy as np
import matplotlib.pyplot as plt
from src.utils.device import setup_device
from ultralytics import YOLO

from src.utils.config import (
    DEFAULT_IMAGE_ROI,
    DEFAULT_VISION_MODEL,
    RECORDING_OUTPUT_PATH,
)
from src.processing.image_processor import ImageProcessor


def process_directory_interactive(
    timestamp_dir, model, device, confidence=0.25, imgsz=640, half=False
):
    """
    Process all images in the given directory using the YOLO model and show them in an interactive viewer.

    Args:
        timestamp_dir: Path to the directory containing images
        model: The YOLO model to use for detection
        device: The device to run inference on
        confidence: Confidence threshold for detections
    """
    # Get all jpg files in the directory
    image_files = sorted(list(Path(timestamp_dir).glob("*.jpg")))

    if not image_files:
        print(f"❌ No jpg files found in {timestamp_dir}")
        return

    total_images = len(image_files)
    print(f"📸 Found {total_images} jpg files in {timestamp_dir}")

    # Variables for interactive navigation
    current_idx = 0
    fig, ax = plt.subplots(figsize=(14, 10))

    # Pre-process a batch of images for faster viewing
    batch_size = 10
    processed_images = [None] * total_images
    detections_list = [None] * total_images

    def process_image(idx):
        """Process a single image and return it with detections"""
        if processed_images[idx] is not None:
            return

        img_path = image_files[idx]
        print(f"Processing image {idx+1}/{total_images}: {img_path.name}")

        img = cv2.imread(str(img_path))
        if img is None:
            print(f"Failed to read image: {img_path}")
            processed_images[idx] = None
            detections_list[idx] = []
            return

        # Convert from BGR to RGB
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_cropped = ImageProcessor.crop_to_roi(img_rgb)

        # Run detection - specify device for Apple Silicon optimization
        results = model(
            img_cropped, conf=confidence, device=device, half=half, imgsz=imgsz
        )

        # Filter for vehicles (car, truck, motorcycle, bus)
        # YOLOv11 class indices: car=2, motorcycle=3, bus=5, truck=7
        vehicle_classes = [2, 3, 5, 7]
        vehicle_classes_own = [0, 1, 2, 3, 4]  # for tuned model
        vehicle_classes = (
            vehicle_classes
            if model.ckpt_path[-4:] in ["s.pt", "n.pt"]
            else (vehicle_classes_own)
        )

        # Process results
        detections = []
        for result in results:
            boxes = result.boxes
            for box in boxes:
                cls_id = int(box.cls.item())
                if cls_id in vehicle_classes:
                    x1, y1, x2, y2 = box.xyxy[0].tolist()
                    conf = box.conf.item()
                    class_name = result.names[cls_id]
                    detections.append((class_name, conf, (x1, y1, x2, y2)))

        # Keep original image and detections for interactive viewing
        processed_images[idx] = img_rgb
        detections_list[idx] = detections

    # Process first batch of images
    start_idx = 0
    end_idx = min(batch_size, total_images)
    for i in range(start_idx, end_idx):
        process_image(i)

    def show_current_image(idx):
        """Display the current image with bounding boxes"""
        ax.clear()

        # Make sure the image is processed
        if processed_images[idx] is None:
            process_image(idx)

            # Also process the next few images for smoother navigation
            next_start = (idx + 1) % total_images
            next_end = min(next_start + batch_size, total_images)
            for next_idx in range(next_start, next_end):
                if processed_images[next_idx] is None:
                    process_image(next_idx)

        img = processed_images[idx]
        detections = detections_list[idx]

        if img is None:
            ax.text(
                0.5,
                0.5,
                f"Failed to load image: {image_files[idx].name}",
                horizontalalignment="center",
                verticalalignment="center",
            )
            return

        # Draw the image
        ax.imshow(img)

        (_, y_offset, _, _) = DEFAULT_IMAGE_ROI
        # Draw detections
        for class_name, conf, (x1, y1, x2, y2) in detections:
            x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
            rect = plt.Rectangle(
                (x1, y1 + y_offset),
                x2 - x1,
                y2 - y1,
                linewidth=2,
                edgecolor="lime",
                facecolor="none",
            )
            ax.add_patch(rect)
            ax.text(
                x1,
                y1 + y_offset - 10,
                f"{class_name} {conf:.2f}",
                color="lime",
                backgroundcolor="black",
                fontsize=10,
            )

        # Display information
        ax.set_title(
            f"Image {idx+1}/{total_images}: {image_files[idx].name} - {len(detections)} vehicles detected"
        )
        ax.axis("off")

        # Add detection count text
        if detections:
            detection_counts = {}
            for cls, _, _ in detections:
                detection_counts[cls] = detection_counts.get(cls, 0) + 1

            detection_text = "\n".join(
                [f"{cls}: {count}" for cls, count in detection_counts.items()]
            )
            ax.text(
                0.02,
                0.98,
                f"Detected objects:\n{detection_text}",
                transform=ax.transAxes,
                verticalalignment="top",
                bbox=dict(facecolor="white", alpha=0.7),
            )

        fig.canvas.draw_idle()

    # Initial display
    show_current_image(current_idx)

    # Define key event handler
    def on_key(event):
        nonlocal current_idx

        if event.key == "right" or event.key == "n":
            current_idx = (current_idx + 1) % total_images
            show_current_image(current_idx)
        elif event.key == "left" or event.key == "p":
            current_idx = (current_idx - 1) % total_images
            show_current_image(current_idx)
        elif event.key == "q" or event.key == "escape":
            plt.close(fig)

    # Connect the key event handler
    fig.canvas.mpl_connect("key_press_event", on_key)

    # Display navigation instructions
    plt.figtext(
        0.5,
        0.01,
        "Navigation: Left/Right Arrow Keys, 'n'/'p', 'q' to quit",
        horizontalalignment="center",
        fontsize=10,
    )

    plt.tight_layout()
    plt.show()


def main():
    parser = argparse.ArgumentParser(
        description="Detect vehicles in recorded images using YOLOv11n"
    )
    parser.add_argument(
        "--recordings",
        type=str,
        default=RECORDING_OUTPUT_PATH,
        help="Path to the recordings directory",
    )
    parser.add_argument(
        "--conf", type=float, default=0.25, help="Confidence threshold for detections"
    )
    parser.add_argument(
        "--model",
        type=str,
        default=DEFAULT_VISION_MODEL,
        help="YOLOv11 model to use (default: yolo11n.pt)",
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=704,
        help="imgsz (default: 704) - image size for inference",
    )
    parser.add_argument(
        "--half",
        action="store_true",
        help="Use half precision for inference (default: False)",
    )
    args = parser.parse_args()

    # Setup device (optimized for Apple Silicon M3)
    device = setup_device()

    # Load YOLOv11 model
    print(f"🔍 Loading {args.model} model...")
    model_path = (
        "data/models/yolo/" + args.model if not "/" in args.model else args.model
    )
    if not Path(model_path).exists():
        model_path = model_path + ".pt"
    if not Path(model_path).exists():
        print(f"❌ Model file {model_path} not found.")
        return
    model = YOLO(model_path)
    print("✅ Model loaded successfully.")

    # Get available timestamp directories
    recordings_path = Path(args.recordings)
    if not recordings_path.exists():
        print(f"❌ Recordings directory {args.recordings} not found.")
        return

    timestamp_dirs = sorted([d for d in recordings_path.iterdir() if d.is_dir()])

    if not timestamp_dirs:
        print(f"❌ No timestamp directories found in {args.recordings}")
        return

    print("📁 Available timestamp directories:")
    for i, dir_path in enumerate(timestamp_dirs):
        print(f"{i+1}: {dir_path.name}")

    # Let the user choose a directory
    choice = input("\nEnter the number of the directory to process (or 'q' to quit): ")
    if choice.lower() == "q":
        return

    try:
        choice_idx = int(choice) - 1
        if choice_idx < 0 or choice_idx >= len(timestamp_dirs):
            print("Invalid choice.")
            return
        selected_dir = timestamp_dirs[choice_idx]
    except ValueError:
        print("Invalid input. Please enter a number or 'q'.")
        return

    print(f"\n🔄 Processing directory: {selected_dir}")
    process_directory_interactive(
        selected_dir,
        model,
        device,
        confidence=args.conf,
        imgsz=args.imgsz,
        half=args.half,
    )

    print("✅ Processing complete.")


if __name__ == "__main__":
    main()
