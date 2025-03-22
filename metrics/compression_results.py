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

# Set up device for inference
device = setup_device()


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
            # images.append((img_path, img))

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

    compression_times = {
        "uncompressed": [0]
        * len(images),  # No compression time for uncompressed images
        "compressed": [],
        "compressed_resized": [],
    }

    # Process each image
    for img_path, img in tqdm(images, desc="Creating compressed versions"):
        # Version 1: Compressed with quality reduction only
        compression_start = time.time()
        compressed_img = ImageProcessor.compress_image(
            img, quality=compression_quality, resize_factor=None
        )
        compression_time = time.time() - compression_start
        compression_times["compressed"].append(compression_time)
        processed_images["compressed"].append((img_path, compressed_img))

        # Version 2: Compressed with quality and size reduction
        compression_start = time.time()
        compressed_resized_img = ImageProcessor.compress_image(
            img,
            quality=compression_quality,
            resize_factor=0.75,  # Reduce to 75% of original size
        )
        compression_time = time.time() - compression_start
        compression_times["compressed_resized"].append(compression_time)
        processed_images["compressed_resized"].append(
            (img_path, compressed_resized_img)
        )

    return processed_images, compression_times


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


def calculate_statistics(detection_results, compression_times=None):
    """
    Calculate statistics from detection results.

    Args:
        detection_results: Dictionary with detection results for each image set
        compression_times: Dictionary with compression time measurements

    Returns:
        DataFrame with statistics
    """
    stats = []

    for img_type, results in detection_results.items():
        # Extract metrics
        num_detections = [r["vehicles_detected"] for r in results]
        detection_times = [r["detection_time"] for r in results]

        # Calculate compression times (if available)
        avg_compression_time = 0
        if compression_times and img_type in compression_times:
            avg_compression_time = np.mean(compression_times[img_type])

        # Extract confidence scores for all detections
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
                "avg_compression_time": avg_compression_time,
                "total_processing_time": np.mean(detection_times)
                + avg_compression_time,
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


def visualize_compression_performance(stats_df, save_path, selected_model):
    """
    Create visualizations for compression performance.

    Args:
        stats_df: DataFrame with statistics
        save_path: Directory to save visualization images
        selected_model: Name of the model used
    """
    # Set style
    sns.set(style="whitegrid")
    plt.rcParams.update({"font.size": 12})

    # Compression vs Detection Time comparison
    plt.figure(figsize=(12, 8))

    # Prepare data for stacked bar chart
    image_types = stats_df["image_type"].tolist()
    comp_times = stats_df["avg_compression_time"].values * 1000  # Convert to ms
    detect_times = stats_df["avg_detection_time"].values * 1000  # Convert to ms
    total_times = stats_df["total_processing_time"].values * 1000  # Convert to ms

    # Plot stacked bars
    width = 0.7
    ax = plt.subplot(111)
    bars1 = ax.bar(
        image_types, comp_times, width, label="Compression Time", color="#5DA5DA"
    )
    bars2 = ax.bar(
        image_types,
        detect_times,
        width,
        bottom=comp_times,
        label="Detection Time",
        color="#F15854",
    )

    # Add text annotations for each segment and total
    for i, (ct, dt, tt) in enumerate(zip(comp_times, detect_times, total_times)):
        # Only show compression time if it's significant
        if ct > 0.1:
            ax.text(
                i,
                ct / 2,
                f"{ct:.2f} ms",
                ha="center",
                va="center",
                color="white",
                fontweight="bold",
            )

        # Show detection time in center of its segment
        ax.text(
            i,
            ct + dt / 2,
            f"{dt:.2f} ms",
            ha="center",
            va="center",
            color="white",
            fontweight="bold",
        )

        # Show total time at top
        ax.text(i, tt + 0.5, f"Total: {tt:.2f} ms", ha="center", va="bottom")

    # Calculate and display efficiency metrics
    for i in range(1, len(image_types)):
        base_time = total_times[0]  # Uncompressed time
        current_time = total_times[i]
        time_saved = base_time - current_time

        if time_saved > 0:
            efficiency = (time_saved / base_time) * 100
            ax.text(
                i,
                -5,
                f"↓ {efficiency:.1f}% faster",
                ha="center",
                va="top",
                color="green",
                fontweight="bold",
            )
        else:
            efficiency = (-time_saved / base_time) * 100
            ax.text(
                i,
                -5,
                f"↑ {efficiency:.1f}% slower",
                ha="center",
                va="top",
                color="red",
                fontweight="bold",
            )

    plt.title("Image Processing Pipeline Performance")
    plt.ylabel("Time (milliseconds)")
    plt.xlabel("Image Type")
    plt.legend(loc="upper right")
    plt.tight_layout()

    # Save figure
    plt.savefig(
        save_path / f"compression_detection_time_{selected_model}_{device}.png", dpi=300
    )

    # Compression Time Performance Only
    plt.figure(figsize=(10, 6))
    ax = sns.barplot(x="image_type", y="avg_compression_time", data=stats_df)

    # Add millisecond labels
    for i, row in stats_df.iterrows():
        ms_time = row["avg_compression_time"] * 1000
        ax.text(
            i,
            row["avg_compression_time"] + 0.0001,
            f"{ms_time:.2f} ms",
            ha="center",
            va="bottom",
        )

    plt.title("Average Image Compression Time")
    plt.ylabel("Time (seconds)")
    plt.xlabel("Image Type")
    plt.tight_layout()
    plt.savefig(save_path / f"compression_time_{selected_model}_{device}.png", dpi=300)

    # Efficiency Analysis - Total processing time
    plt.figure(figsize=(10, 6))
    ax = sns.barplot(x="image_type", y="total_processing_time", data=stats_df)

    for i, row in stats_df.iterrows():
        ms_time = row["total_processing_time"] * 1000
        ax.text(
            i,
            row["total_processing_time"] + 0.0005,
            f"{ms_time:.2f} ms",
            ha="center",
            va="bottom",
        )

    plt.title("Total Processing Time (Compression + Detection)")
    plt.ylabel("Time (seconds)")
    plt.xlabel("Image Type")
    plt.tight_layout()
    plt.savefig(
        save_path / f"total_processing_time_{selected_model}_{device}.png", dpi=300
    )


def visualize_results(
    stats_df,
    detection_results,
    processed_images=None,
    compression_times=None,
    save_path=RESULTS_DIR,
    selected_model=DEFAULT_VISION_MODEL,
):
    """
    Create visualizations for detection results and save them.

    Args:
        stats_df: DataFrame with statistics
        detection_results: Dictionary with detection results
        processed_images: Dictionary with processed images
        compression_times: Dictionary with compression time measurements
        save_path: Directory to save visualization images
        selected_model: Model used for detection
    """
    # Set style
    sns.set(style="whitegrid")
    plt.rcParams.update({"font.size": 12})

    # Add compression performance visualizations
    if "avg_compression_time" in stats_df.columns:
        visualize_compression_performance(stats_df, save_path, selected_model)

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
    plt.savefig(save_path / f"avg_detections_{selected_model}_{device}.png", dpi=300)

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
    plt.savefig(save_path / f"detection_time_{selected_model}_{device}.png", dpi=300)

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
    plt.savefig(
        save_path / f"confidence_distribution_{selected_model}_{device}.png", dpi=300
    )

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
        plt.savefig(
            save_path / f"class_distribution_{selected_model}_{device}.png", dpi=300
        )

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
    plt.savefig(
        save_path / f"detection_count_comparison_{selected_model}_{device}.png", dpi=300
    )

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
                    # For compressed images, estimate file size by encoding directly
                    if img_type == "compressed":
                        # Use imencode to get compressed bytes
                        _, encoded_img = cv2.imencode(
                            ".jpg",
                            img,
                            [cv2.IMWRITE_JPEG_QUALITY, IMAGE_COMPRESSION_QUALITY],
                        )
                    else:  # compressed_resized
                        # First resize
                        height, width = img.shape[:2]
                        new_height = int(height * 0.75)
                        new_width = int(width * 0.75)
                        resized_img = cv2.resize(
                            img, (new_width, new_height), interpolation=cv2.INTER_AREA
                        )
                        # Then compress
                        _, encoded_img = cv2.imencode(
                            ".jpg",
                            resized_img,
                            [cv2.IMWRITE_JPEG_QUALITY, IMAGE_COMPRESSION_QUALITY],
                        )

                    # Calculate size in KB
                    compressed_sizes.append(len(encoded_img) / 1024)

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
        plt.savefig(
            save_path / f"file_size_comparison_{selected_model}_{device}.png", dpi=300
        )

    # 7. Summary table with compression metrics
    plt.figure(figsize=(14, 6))
    plt.axis("off")

    # Select summary columns including compression metrics
    summary_data = stats_df[
        [
            "image_type",
            "total_detections",
            "avg_detections_per_image",
            "avg_confidence",
            "avg_compression_time",
            "avg_detection_time",
            "total_processing_time",
        ]
    ]

    # Format the data for display
    formatted_data = summary_data.copy()
    formatted_data["avg_detections_per_image"] = formatted_data[
        "avg_detections_per_image"
    ].map("{:.2f}".format)
    formatted_data["avg_confidence"] = formatted_data["avg_confidence"].map(
        "{:.3f}".format
    )
    # Convert time values to milliseconds for better readability
    formatted_data["avg_compression_time"] = formatted_data["avg_compression_time"].map(
        lambda x: f"{x*1000:.2f} ms"
    )
    formatted_data["avg_detection_time"] = formatted_data["avg_detection_time"].map(
        lambda x: f"{x*1000:.2f} ms"
    )
    formatted_data["total_processing_time"] = formatted_data[
        "total_processing_time"
    ].map(lambda x: f"{x*1000:.2f} ms")

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

    plt.title("Performance Summary with Compression Metrics", pad=20)
    plt.tight_layout()
    plt.savefig(
        save_path / f"complete_summary_table_{selected_model}_{device}.png", dpi=300
    )

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
    print(f"🧠 Loading {selected_model} model...")
    model = YOLO(selected_model)
    print("✅ Model loaded successfully.")

    # Step 1: Load uncompressed images
    uncompressed_images = load_uncompressed_images(limit=args.limit)

    if not uncompressed_images:
        print(f"❌ No images found in {UNCOMPRESSED_DIR}. Exiting.")
        return

    # Step 2: Create compressed versions with timing information
    processed_images, compression_times = create_compressed_versions(
        uncompressed_images, args.quality
    )

    # Step 3: Run object detection
    detection_results = run_object_detection(
        processed_images, model, device, confidence=args.conf
    )

    # Step 4: Calculate statistics with compression times
    stats_df = calculate_statistics(detection_results, compression_times)

    # Print statistics summary including compression metrics
    print("\nStatistics Summary (with Compression Metrics):")
    print(
        stats_df[
            [
                "image_type",
                "total_detections",
                "avg_detections_per_image",
                "avg_compression_time",
                "avg_detection_time",
                "total_processing_time",
            ]
        ]
    )

    # Step 5: Visualize and save results
    visualize_results(
        stats_df,
        detection_results,
        processed_images,
        compression_times,
        selected_model=selected_model,
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

        # Include all visualization images in the report
        visualization_images = [
            f"compression_detection_time_{selected_model}_{device}.png",
            f"compression_time_{selected_model}_{device}.png",
            f"total_processing_time_{selected_model}_{device}.png",
            f"avg_detections_{selected_model}_{device}.png",
            f"detection_time_{selected_model}_{device}.png",
            f"confidence_distribution_{selected_model}_{device}.png",
            f"class_distribution_{selected_model}_{device}.png",
            f"detection_count_comparison_{selected_model}_{device}.png",
            f"file_size_comparison_{selected_model}_{device}.png",
            f"complete_summary_table_{selected_model}_{device}.png",
        ]

        for img_name in visualization_images:
            img_path = f"{img_name}"
            f.write(f'<h3>{img_name[:-4].replace("_", " ").title()}</h3>')
            f.write(f'<img src="{img_path}" style="max-width:100%"><br><br>')
        f.write("</body></html>")

    print(f"✅ Analysis complete. Results saved to {RESULTS_DIR}")
    print(f"📊 Statistics summary: {csv_path}")
    print(f"📑 Detailed report: {html_path}")


if __name__ == "__main__":
    main()
