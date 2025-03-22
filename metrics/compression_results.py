#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Compression Results Analysis for YOLO Detection
This script analyzes how image compression affects YOLO detection results.
"""

import os
import select
import sys
import time
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import cv2
from tqdm import tqdm
from ultralytics import YOLO

# Add parent directory to path for imports from main project
script_dir = Path(__file__).parent
project_root = script_dir.parent
sys.path.insert(0, str(project_root))

# Import project modules
from config import DEFAULT_VISION_MODEL, IMAGE_COMPRESSION_QUALITY, DEFAULT_IMAGE_ROI
from device import setup_device
from imageprocessor import ImageProcessor

# Constants
METRICS_DIR = Path("metrics")
UNCOMPRESSED_DIR = METRICS_DIR / "uncompressed_images"
RESULTS_DIR = METRICS_DIR / "results"
COMPARISON_TYPES = ["uncompressed", "compressed", "compressed_resized"]

# Vehicle classes in YOLOv11: car(2), motorcycle(3), bus(5), truck(7)
VEHICLE_CLASSES = [2, 3, 5, 7]

# Create results directory if it doesn't exist
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def load_uncompressed_images(source_dir=UNCOMPRESSED_DIR, limit=None):
    """
    Load uncompressed images from the source directory.

    Args:
        source_dir: Directory containing uncompressed images
        limit: Maximum number of images to load (None for all)

    Returns:
        List of (path, image) tuples
    """
    image_files = sorted(list(Path(source_dir).glob("*.jpg")))

    if limit:
        image_files = image_files[:limit]

    images = []
    for img_path in tqdm(image_files, desc="Loading uncompressed images"):
        img = cv2.imread(str(img_path))

        if img is not None:
            img_cropped = ImageProcessor.crop_to_roi(img)
            images.append((img_path, img_cropped))

    print(f"Loaded {len(images)} uncompressed images")
    return images


def create_compressed_versions(images, compression_quality=IMAGE_COMPRESSION_QUALITY):
    """
    Create compressed versions of images with different techniques.

    Args:
        images: List of (path, image) tuples
        compression_quality: JPEG compression quality

    Returns:
        Dictionary with keys 'uncompressed', 'compressed', 'compressed_resized'
        and values as lists of processed images
    """
    processed_images = {
        "uncompressed": images.copy(),
        "compressed": [],
        "compressed_resized": [],
    }

    # Process each image
    for img_path, img in tqdm(images, desc="Creating compressed versions"):
        # Version 1: Compressed with quality reduction only
        compressed_img = ImageProcessor.compress_image(
            img, quality=compression_quality, resize_factor=None
        )
        processed_images["compressed"].append((img_path, compressed_img))

        # Version 2: Compressed with quality and size reduction
        compressed_resized_img = ImageProcessor.compress_image(
            img,
            quality=compression_quality,
            resize_factor=0.75,  # Reduce to 75% of original size
        )
        processed_images["compressed_resized"].append(
            (img_path, compressed_resized_img)
        )

    return processed_images


def run_object_detection(processed_images, model, device, confidence=0.25):
    """
    Run YOLO object detection on all image sets and collect results.

    Args:
        processed_images: Dictionary with image sets
        model: YOLO model
        device: Device to run inference on
        confidence: Detection confidence threshold

    Returns:
        Dictionary with detection results for each image set
    """
    results = {}

    for img_type, images in processed_images.items():
        print(f"Running detection on {img_type} images...")
        img_results = []

        for img_path, img in tqdm(images):

            # Run detection
            detection_start = time.time()
            detections = model(img, conf=confidence, device=device)
            detection_time = time.time() - detection_start

            # Extract vehicle detections
            vehicles_found = []
            for result in detections:
                boxes = result.boxes
                for box in boxes:
                    cls_id = int(box.cls.item())
                    if cls_id in VEHICLE_CLASSES:
                        x1, y1, x2, y2 = box.xyxy[0].tolist()
                        conf = box.conf.item()
                        class_name = result.names[cls_id]
                        vehicles_found.append(
                            {
                                "class": class_name,
                                "confidence": conf,
                                "bbox": (x1, y1, x2, y2),
                            }
                        )

            img_results.append(
                {
                    "filename": img_path.name,
                    "vehicles_detected": len(vehicles_found),
                    "detections": vehicles_found,
                    "detection_time": detection_time,
                }
            )

        results[img_type] = img_results

    return results


def calculate_statistics(detection_results):
    """
    Calculate statistics from detection results.

    Args:
        detection_results: Dictionary with detection results for each image set

    Returns:
        DataFrame with statistics
    """
    stats = []

    for img_type, results in detection_results.items():
        # Extract metrics
        num_detections = [r["vehicles_detected"] for r in results]
        detection_times = [r["detection_time"] for r in results]

        # Calculate confidence scores for all detections
        all_confidences = []
        for r in results:
            for detection in r["detections"]:
                all_confidences.append(detection["confidence"])

        # Class distribution
        class_counts = {}
        for r in results:
            for detection in r["detections"]:
                cls = detection["class"]
                class_counts[cls] = class_counts.get(cls, 0) + 1

        # Calculate stats
        stats.append(
            {
                "image_type": img_type,
                "total_images": len(results),
                "total_detections": sum(num_detections),
                "avg_detections_per_image": np.mean(num_detections),
                "median_detections_per_image": np.median(num_detections),
                "std_detections_per_image": np.std(num_detections),
                "avg_detection_time": np.mean(detection_times),
                "median_detection_time": np.median(detection_times),
                "avg_confidence": np.mean(all_confidences) if all_confidences else 0,
                "median_confidence": (
                    np.median(all_confidences) if all_confidences else 0
                ),
                "min_confidence": min(all_confidences) if all_confidences else 0,
                "max_confidence": max(all_confidences) if all_confidences else 0,
                "class_distribution": class_counts,
            }
        )

    return pd.DataFrame(stats)


def visualize_results(
    stats_df,
    detection_results,
    processed_images=None,
    save_path=RESULTS_DIR,
    selected_model=DEFAULT_VISION_MODEL,
):
    """
    Create visualizations for detection results and save them.

    Args:
        stats_df: DataFrame with statistics
        detection_results: Dictionary with detection results
        save_path: Directory to save visualization images
    """
    # Set style
    sns.set(style="whitegrid")
    plt.rcParams.update({"font.size": 12})

    # 1. Average detections per image type
    plt.figure(figsize=(10, 6))
    ax = sns.barplot(x="image_type", y="avg_detections_per_image", data=stats_df)
    for i, row in stats_df.iterrows():
        ax.text(
            i,
            row["avg_detections_per_image"] + 0.1,
            f"{row['avg_detections_per_image']:.2f}",
            ha="center",
            va="bottom",
        )
    plt.title("Average Detections per Image Type")
    plt.ylabel("Average number of vehicles detected")
    plt.xlabel("Image Type")
    plt.tight_layout()
    plt.savefig(save_path / f"avg_detections_{selected_model}.png", dpi=300)

    # 2. Detection time comparison
    plt.figure(figsize=(10, 6))
    ax = sns.barplot(x="image_type", y="avg_detection_time", data=stats_df)
    for i, row in stats_df.iterrows():
        ax.text(
            i,
            row["avg_detection_time"] + 0.005,
            f"{row['avg_detection_time']*1000:.1f} ms",
            ha="center",
            va="bottom",
        )
    plt.title("Average Detection Time per Image Type")
    plt.ylabel("Time (seconds)")
    plt.xlabel("Image Type")
    plt.tight_layout()
    plt.savefig(save_path / f"detection_time_{selected_model}.png", dpi=300)

    # 3. Confidence score distribution
    plt.figure(figsize=(12, 6))
    all_confidences = {}

    for img_type, results in detection_results.items():
        confs = []
        for r in results:
            for detection in r["detections"]:
                confs.append(detection["confidence"])
        all_confidences[img_type] = confs

    # Plot confidence distributions
    for img_type, confidences in all_confidences.items():
        if confidences:  # Only plot if we have confidence values
            sns.kdeplot(confidences, label=img_type)

    plt.title("Distribution of Confidence Scores")
    plt.xlabel("Confidence Score")
    plt.ylabel("Density")
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path / f"confidence_distribution_{selected_model}.png", dpi=300)

    # 4. Vehicle class distribution
    plt.figure(figsize=(12, 8))

    # Extract class distributions
    class_data = []
    for i, row in stats_df.iterrows():
        img_type = row["image_type"]
        for cls, count in row["class_distribution"].items():
            class_data.append({"image_type": img_type, "class": cls, "count": count})

    if class_data:
        class_df = pd.DataFrame(class_data)
        sns.barplot(x="class", y="count", hue="image_type", data=class_df)
        plt.title("Vehicle Class Distribution by Image Type")
        plt.ylabel("Count")
        plt.xlabel("Vehicle Class")
        plt.tight_layout()
        plt.savefig(save_path / f"class_distribution_{selected_model}.png", dpi=300)

    # 5. Detection count comparison
    # Extract detection counts per image
    detection_counts = {
        img_type: [r["vehicles_detected"] for r in results]
        for img_type, results in detection_results.items()
    }

    plt.figure(figsize=(14, 7))
    positions = range(len(detection_counts[COMPARISON_TYPES[0]]))

    # Plot lines for each image type
    for img_type in COMPARISON_TYPES:
        plt.plot(
            positions, detection_counts[img_type], label=img_type, marker="o", alpha=0.7
        )

    plt.title("Vehicle Detection Count Comparison by Image")
    plt.xlabel("Image Index")
    plt.ylabel("Number of Vehicles Detected")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(save_path / f"detection_count_comparison_{selected_model}.png", dpi=300)

    # 6. File size reduction analysis
    if processed_images and len(detection_results["uncompressed"]) > 0:
        # Analyze file size differences
        plt.figure(figsize=(10, 6))

        # Get file sizes for first few images as example
        file_sizes = []
        for img_type in COMPARISON_TYPES:
            if img_type == "uncompressed":
                # Get original file size
                original_sizes = [
                    os.path.getsize(img_path) / 1024
                    for img_path, _ in processed_images[img_type][:10]
                ]
                avg_size = np.mean(original_sizes)
                file_sizes.append(
                    {"type": img_type, "size": avg_size, "percentage": 100.0}
                )
            else:
                # Calculate sizes directly in memory without writing to disk
                compressed_sizes = []
                for i, (img_path, img) in enumerate(processed_images[img_type][:10]):
                    # For compressed images, get the compressed data directly
                    if img_type == "compressed":
                        compressed_data = ImageProcessor.compress_image(
                            img, quality=IMAGE_COMPRESSION_QUALITY
                        )
                    else:  # compressed_resized
                        compressed_data = ImageProcessor.compress_image(
                            img, quality=IMAGE_COMPRESSION_QUALITY, resize_factor=0.75
                        )

                    # Calculate size in KB
                    compressed_sizes.append(len(compressed_data) / 1024)

                avg_size = np.mean(compressed_sizes)
                percentage = (avg_size / file_sizes[0]["size"]) * 100
                file_sizes.append(
                    {"type": img_type, "size": avg_size, "percentage": percentage}
                )

        # Create file size dataframe
        file_size_df = pd.DataFrame(file_sizes)

        # Plot
        ax = sns.barplot(x="type", y="size", data=file_size_df)
        for i, row in file_size_df.iterrows():
            ax.text(
                i,
                row["size"] + 5,
                f"{row['size']:.1f} KB\n({row['percentage']:.1f}%)",
                ha="center",
                va="bottom",
            )

        plt.title("Average File Size Comparison")
        plt.ylabel("Size (KB)")
        plt.xlabel("Image Type")
        plt.tight_layout()
        plt.savefig(save_path / f"file_size_comparison_{selected_model}.png", dpi=300)

    # 7. Summary table
    plt.figure(figsize=(12, 6))
    plt.axis("off")

    # Select summary columns
    summary_data = stats_df[
        [
            "image_type",
            "total_detections",
            "avg_detections_per_image",
            "avg_confidence",
            "avg_detection_time",
        ]
    ]

    # Format the data
    formatted_data = summary_data.copy()
    formatted_data["avg_detections_per_image"] = formatted_data[
        "avg_detections_per_image"
    ].map("{:.2f}".format)
    formatted_data["avg_confidence"] = formatted_data["avg_confidence"].map(
        "{:.3f}".format
    )
    formatted_data["avg_detection_time"] = formatted_data["avg_detection_time"].map(
        lambda x: f"{x*1000:.1f} ms"
    )

    # Create table
    table = plt.table(
        cellText=formatted_data.values,
        colLabels=formatted_data.columns,
        loc="center",
        cellLoc="center",
        colColours=["#f0f0f0"] * len(formatted_data.columns),
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.5)

    plt.title("Detection Performance Summary", pad=20)
    plt.tight_layout()
    plt.savefig(save_path / f"summary_table_{selected_model}.png", dpi=300)

    print(f"Saved all visualizations to {save_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Analyze how image compression affects YOLO detection results"
    )
    parser.add_argument(
        "--limit", type=int, default=None, help="Limit the number of images to process"
    )
    parser.add_argument(
        "--model",
        type=str,
        default=DEFAULT_VISION_MODEL,
        help=f"YOLO model to use (default: {DEFAULT_VISION_MODEL})",
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=0.25,
        help="Detection confidence threshold (default: 0.25)",
    )
    parser.add_argument(
        "--quality",
        type=int,
        default=IMAGE_COMPRESSION_QUALITY,
        help=f"JPEG compression quality (default: {IMAGE_COMPRESSION_QUALITY})",
    )
    parser.add_argument(
        "--resize",
        type=float,
        default=0.75,
        help="Resize factor for compressed_resized images (default: 0.75)",
    )
    args = parser.parse_args()

    print("🔍 Starting compression impact analysis on YOLO detection")

    # Set up device for inference
    device = setup_device()

    # Load YOLO model
    selected_model = args.model
    print(f"🧠 Loading _{selected_model} model...")
    model = YOLO(selected_model)
    print("✅ Model loaded successfully.")

    # Step 1: Load uncompressed images
    uncompressed_images = load_uncompressed_images(limit=args.limit)

    if not uncompressed_images:
        print(f"❌ No images found in {UNCOMPRESSED_DIR}. Exiting.")
        return

    # Step 2: Create compressed versions
    processed_images = create_compressed_versions(uncompressed_images, args.quality)

    # Step 3: Run object detection
    detection_results = run_object_detection(
        processed_images, model, device, confidence=args.conf
    )

    # Step 4: Calculate statistics
    stats_df = calculate_statistics(detection_results)

    # Print statistics summary
    print("\nStatistics Summary:")
    print(
        stats_df[
            [
                "image_type",
                "total_detections",
                "avg_detections_per_image",
                "avg_confidence",
                "avg_detection_time",
            ]
        ]
    )

    # Step 5: Visualize and save results
    visualize_results(
        stats_df, detection_results, processed_images, selected_model=selected_model
    )

    # Save full results as CSV
    csv_path = RESULTS_DIR / "statistics_summary.csv"

    # Need to convert complex dictionary to string for CSV storage
    stats_csv = stats_df.copy()
    stats_csv["class_distribution"] = stats_csv["class_distribution"].apply(str)
    stats_csv.to_csv(csv_path, index=False)

    # Save detailed HTML report
    html_path = RESULTS_DIR / "statistics_summary.html"
    with open(html_path, "w") as f:
        f.write("<html><head><title>YOLO Compression Impact Analysis</title>")
        f.write(
            "<style>body{font-family:Arial; margin:20px} table{border-collapse:collapse; width:100%} "
        )
        f.write(
            "th,td{text-align:left; padding:8px; border:1px solid #ddd} </style></head><body>"
        )
        f.write(f"<h1>YOLO Compression Impact Analysis</h1>")
        f.write(f'<p>Analysis date: {time.strftime("%Y-%m-%d %H:%M:%S")}</p>')
        f.write(f"<p>Model: {args.model}, Confidence threshold: {args.conf}</p>")
        f.write(
            f"<p>Compression quality: {args.quality}, Resize factor: {args.resize}</p>"
        )
        f.write("<h2>Summary Statistics</h2>")
        f.write(stats_csv.to_html(index=False))
        f.write("<h2>Visualizations</h2>")
        for img_name in [
            f"avg_detections_{selected_model}.png",
            f"detection_time_{selected_model}.png",
            f"confidence_distribution_{selected_model}.png",
            f"class_distribution_{selected_model}.png",
            f"detection_count_comparison_{selected_model}.png",
            f"file_size_comparison_{selected_model}.png",
            f"summary_table_{selected_model}.png",
        ]:
            img_path = f"{img_name}"
            f.write(f'<h3>{img_name[:-4].replace("_", " ").title()}</h3>')
            f.write(f'<img src="{img_path}" style="max-width:100%"><br><br>')
        f.write("</body></html>")

    print(f"✅ Analysis complete. Results saved to {RESULTS_DIR}")
    print(f"📊 Statistics summary: {csv_path}")
    print(f"📑 Detailed report: {html_path}")


if __name__ == "__main__":
    main()
