#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
YOLO Optimization Benchmark
This script compares the inference performance of YOLO with and without optimizations.
"""

import os
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
from device import setup_device
from ultralytics import YOLO

# Add parent directory to path for imports from main project
script_dir = Path(__file__).parent
project_root = script_dir.parent
sys.path.insert(0, str(project_root))

# Import project modules
from config import DEFAULT_VISION_MODEL, DEFAULT_IMAGE_ROI
from imageprocessor import ImageProcessor

# Constants
METRICS_DIR = Path("metrics")
TEST_IMAGES_DIR = METRICS_DIR / "test_images"
RESULTS_DIR = METRICS_DIR / "optimization_results"

# Vehicle classes in YOLOv11: car(2), motorcycle(3), bus(5), truck(7)
VEHICLE_CLASSES = [2, 3, 5, 7]

# Create results directory if it doesn't exist
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# Set up device for inference
device = setup_device()


def load_test_images(source_dir=TEST_IMAGES_DIR, limit=None):
    """
    Load test images from the source directory.

    Args:
        source_dir: Directory containing test images
        limit: Maximum number of images to load (None for all)

    Returns:
        List of (path, image) tuples
    """
    # Create directory if it doesn't exist
    source_dir.mkdir(parents=True, exist_ok=True)

    # Check if the directory is empty
    image_files = sorted(list(Path(source_dir).glob("*.jpg")))

    if not image_files:
        print(f"❌ No images found in {source_dir}. Please add test images.")
        return []

    if limit:
        image_files = image_files[:limit]

    images = []
    for img_path in tqdm(image_files, desc="Loading test images"):
        img = cv2.imread(str(img_path))
        if img is not None:
            # Convert BGR to RGB for YOLO
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img_cropped = ImageProcessor.crop_to_roi(img_rgb)
            images.append((img_path, img_cropped))

    print(f"Loaded {len(images)} test images")
    return images


def run_baseline_detection(images, model, device, conf=0.25):
    """
    Run YOLO detection without optimizations.

    Args:
        images: List of (path, image) tuples
        model: YOLO model
        device: Device to run inference on
        conf: Detection confidence threshold

    Returns:
        List of detection results and inference times
    """
    results = []

    print("Running baseline detection (without optimizations)...")
    for img_path, img in tqdm(images):
        # Start timing
        start_time = time.time()

        # Run detection with default settings
        detections = model(img, conf=conf, device=device)

        # End timing
        inference_time = time.time() - start_time

        # Process results
        detected_objects = []
        for result in detections:
            boxes = result.boxes
            for box in boxes:
                cls_id = int(box.cls.item())
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                conf = box.conf.item()
                class_name = result.names[cls_id]
                detected_objects.append(
                    {
                        "class": class_name,
                        "class_id": cls_id,
                        "confidence": conf,
                        "bbox": (x1, y1, x2, y2),
                    }
                )

        # Count vehicles
        vehicles_detected = sum(
            1 for obj in detected_objects if obj["class_id"] in VEHICLE_CLASSES
        )

        # Count total objects
        total_objects = len(detected_objects)

        results.append(
            {
                "filename": img_path.name,
                "inference_time": inference_time,
                "total_objects": total_objects,
                "vehicles_detected": vehicles_detected,
                "detections": detected_objects,
            }
        )

    return results


def configure_optimized_model(model):
    """
    Configure the model with optimizations (classes filter, no verbose).

    Args:
        model: YOLO model

    Returns:
        Configured YOLO model
    """
    # Create a copy of the model to avoid modifying the original
    optimized_model = YOLO(model.ckpt_path)

    # Set model classes to only detect vehicles
    optimized_model.classes = VEHICLE_CLASSES

    # Set verbose to False
    optimized_model.verbose = False

    return optimized_model


def run_optimized_detection(images, model, device, conf=0.25):
    """
    Run YOLO detection with optimizations (classes filter, no verbose).

    Args:
        images: List of (path, image) tuples
        model: Pre-configured YOLO model with optimizations
        device: Device to run inference on
        conf: Detection confidence threshold

    Returns:
        List of detection results and inference times
    """
    results = []

    print("Running optimized detection (with class filter, no verbose)...")
    for img_path, img in tqdm(images):
        # Start timing
        start_time = time.time()

        # Run detection with the pre-configured model
        detections = model(img, conf=conf, device=device)

        # End timing
        inference_time = time.time() - start_time

        # Process results
        detected_objects = []
        for result in detections:
            boxes = result.boxes
            for box in boxes:
                cls_id = int(box.cls.item())
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                conf = box.conf.item()
                class_name = result.names[cls_id]
                detected_objects.append(
                    {
                        "class": class_name,
                        "class_id": cls_id,
                        "confidence": conf,
                        "bbox": (x1, y1, x2, y2),
                    }
                )

        # Count vehicles (should be all objects since we filtered)
        vehicles_detected = len(detected_objects)

        # Count total objects (same as vehicles since we filtered)
        total_objects = len(detected_objects)

        results.append(
            {
                "filename": img_path.name,
                "inference_time": inference_time,
                "total_objects": total_objects,
                "vehicles_detected": vehicles_detected,
                "detections": detected_objects,
            }
        )

    return results


def calculate_statistics(baseline_results, optimized_results):
    """
    Calculate statistics from detection results.

    Args:
        baseline_results: Results from baseline detection
        optimized_results: Results from optimized detection

    Returns:
        DataFrame with statistics
    """
    # Extract metrics from baseline results
    baseline_inference_times = [r["inference_time"] for r in baseline_results]
    baseline_total_objects = [r["total_objects"] for r in baseline_results]
    baseline_vehicles = [r["vehicles_detected"] for r in baseline_results]

    # Extract metrics from optimized results
    optimized_inference_times = [r["inference_time"] for r in optimized_results]
    optimized_total_objects = [r["total_objects"] for r in optimized_results]
    optimized_vehicles = [r["vehicles_detected"] for r in optimized_results]

    # Calculate confidence scores
    baseline_confidences = []
    for r in baseline_results:
        for detection in r["detections"]:
            if detection["class_id"] in VEHICLE_CLASSES:
                baseline_confidences.append(detection["confidence"])

    optimized_confidences = []
    for r in optimized_results:
        for detection in r["detections"]:
            optimized_confidences.append(detection["confidence"])

    # Create statistics DataFrame
    stats = pd.DataFrame(
        {
            "metric": [
                "Avg. Inference Time (ms)",
                "Median Inference Time (ms)",
                "Min Inference Time (ms)",
                "Max Inference Time (ms)",
                "Avg. Total Objects",
                "Avg. Vehicles Detected",
                "Total Objects Detected",
                "Total Vehicles Detected",
                "Avg. Confidence Score",
                "Median Confidence Score",
            ],
            "baseline": [
                np.mean(baseline_inference_times) * 1000,
                np.median(baseline_inference_times) * 1000,
                np.min(baseline_inference_times) * 1000,
                np.max(baseline_inference_times) * 1000,
                np.mean(baseline_total_objects),
                np.mean(baseline_vehicles),
                sum(baseline_total_objects),
                sum(baseline_vehicles),
                np.mean(baseline_confidences) if baseline_confidences else 0,
                np.median(baseline_confidences) if baseline_confidences else 0,
            ],
            "optimized": [
                np.mean(optimized_inference_times) * 1000,
                np.median(optimized_inference_times) * 1000,
                np.min(optimized_inference_times) * 1000,
                np.max(optimized_inference_times) * 1000,
                np.mean(optimized_total_objects),
                np.mean(optimized_vehicles),
                sum(optimized_total_objects),
                sum(optimized_vehicles),
                np.mean(optimized_confidences) if optimized_confidences else 0,
                np.median(optimized_confidences) if optimized_confidences else 0,
            ],
        }
    )

    # Add improvement percentage
    stats["improvement"] = (
        (stats["baseline"] - stats["optimized"]) / stats["baseline"] * 100
    ).round(2)
    # For metrics where higher is better, reverse the calculation
    higher_better_indices = [8, 9]  # Confidence scores
    stats.loc[higher_better_indices, "improvement"] = (
        (
            stats.loc[higher_better_indices, "optimized"]
            - stats.loc[higher_better_indices, "baseline"]
        )
        / stats.loc[higher_better_indices, "baseline"]
        * 100
    ).round(2)

    return stats


def visualize_results(stats, baseline_results, optimized_results, save_path):
    """
    Create visualizations for detection results and save them.

    Args:
        stats: DataFrame with statistics
        baseline_results: Results from baseline detection
        optimized_results: Results from optimized detection (classes filter, no verbose)
        save_path: Directory to save visualization images
    """
    # Set style
    sns.set(style="whitegrid")
    plt.rcParams.update({"font.size": 12})

    # 1. Inference Time Comparison
    plt.figure(figsize=(12, 6))

    # Extract inference times and convert to milliseconds
    baseline_times = [r["inference_time"] * 1000 for r in baseline_results]
    optimized_times = [r["inference_time"] * 1000 for r in optimized_results]

    # Plot inference times
    x = range(len(baseline_times))
    plt.plot(x, baseline_times, marker="o", label="Baseline", alpha=0.7)
    plt.plot(x, optimized_times, marker="x", label="Optimized", alpha=0.7)

    # Add average lines
    plt.axhline(
        y=np.mean(baseline_times),
        color="blue",
        linestyle="--",
        label=f"Baseline Avg: {np.mean(baseline_times):.2f} ms",
    )
    plt.axhline(
        y=np.mean(optimized_times),
        color="orange",
        linestyle="--",
        label=f"Optimized Avg: {np.mean(optimized_times):.2f} ms",
    )

    # Highlighting the improvement
    improvement = (
        (np.mean(baseline_times) - np.mean(optimized_times))
        / np.mean(baseline_times)
        * 100
    )
    plt.text(
        len(baseline_times) * 0.5,
        np.mean([np.mean(baseline_times), np.mean(optimized_times)]),
        f"{improvement:.2f}% Improvement",
        horizontalalignment="center",
        size=14,
        weight="bold",
        bbox=dict(facecolor="white", alpha=0.8),
    )

    plt.title("Inference Time Comparison")
    plt.xlabel("Image Index")
    plt.ylabel("Inference Time (ms)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path / "inference_time_comparison.png", dpi=300)

    # 2. Inference Time Distribution
    plt.figure(figsize=(12, 6))

    # Create a DataFrame for seaborn
    times_df = pd.DataFrame({"Baseline": baseline_times, "Optimized": optimized_times})

    # Melt the DataFrame for seaborn
    times_df_melted = pd.melt(
        times_df, var_name="Method", value_name="Inference Time (ms)"
    )

    # Plot the violin plot
    sns.violinplot(
        x="Method", y="Inference Time (ms)", data=times_df_melted, inner="box"
    )

    # Add statistical annotations
    for i, method in enumerate(["Baseline", "Optimized"]):
        times = baseline_times if method == "Baseline" else optimized_times
        plt.text(
            i,
            np.max(times) + 1,
            f"Mean: {np.mean(times):.2f} ms\nMedian: {np.median(times):.2f} ms",
            horizontalalignment="center",
        )

    plt.title("Inference Time Distribution")
    plt.tight_layout()
    plt.savefig(save_path / "inference_time_distribution.png", dpi=300)

    # 3. Object Detection Comparison
    plt.figure(figsize=(12, 6))

    # Extract object counts
    baseline_objects = [r["total_objects"] for r in baseline_results]
    baseline_vehicles = [r["vehicles_detected"] for r in baseline_results]
    optimized_vehicles = [r["vehicles_detected"] for r in optimized_results]

    # Plot object counts
    x = range(len(baseline_objects))
    plt.plot(x, baseline_objects, marker="o", label="Baseline All Objects", alpha=0.7)
    plt.plot(
        x, baseline_vehicles, marker="x", label="Baseline Vehicles Only", alpha=0.7
    )
    plt.plot(x, optimized_vehicles, marker="^", label="Optimized Vehicles", alpha=0.7)

    plt.title("Object Detection Comparison")
    plt.xlabel("Image Index")
    plt.ylabel("Number of Objects Detected")
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path / "object_detection_comparison.png", dpi=300)

    # 4. Overall Performance Metrics
    plt.figure(figsize=(14, 8))

    # Subset of metrics to display
    display_metrics = [
        "Avg. Inference Time (ms)",
        "Median Inference Time (ms)",
        "Avg. Vehicles Detected",
        "Avg. Confidence Score",
    ]

    # Filter stats for these metrics
    display_stats = stats[stats["metric"].isin(display_metrics)].copy()

    # Reshape for plotting
    display_stats_melted = pd.melt(
        display_stats,
        id_vars=["metric", "improvement"],
        value_vars=["baseline", "optimized"],
        var_name="method",
        value_name="value",
    )

    # Create grouped bar chart
    g = sns.catplot(
        data=display_stats_melted,
        kind="bar",
        x="metric",
        y="value",
        hue="method",
        palette=["#5DA5DA", "#F15854"],
        height=6,
        aspect=1.5,
    )

    # Customize the plot
    g.set_xticklabels(rotation=45, horizontalalignment="right")
    g.set_axis_labels("Metric", "Value")
    g.fig.suptitle("Performance Metrics Comparison", fontsize=16)
    g.fig.subplots_adjust(top=0.9)

    # Add improvement annotations
    ax = g.axes[0, 0]
    for i, metric in enumerate(display_stats["metric"].unique()):
        impr = display_stats[display_stats["metric"] == metric]["improvement"].values[0]
        ax.text(
            i,
            max(
                display_stats[display_stats["metric"] == metric][
                    ["baseline", "optimized"]
                ].max(axis=1)
            )
            * 1.1,
            f"{impr:.2f}%",
            ha="center",
            fontweight="bold",
        )

    # Save the figure
    g.savefig(save_path / "performance_metrics_comparison.png", dpi=300)

    # 5. Summary table with all metrics
    plt.figure(figsize=(10, 8))
    plt.axis("off")

    # Format the stats table for display
    formatted_stats = stats.copy()
    # Round numeric columns
    for col in ["baseline", "optimized"]:
        formatted_stats[col] = formatted_stats[col].round(2)
    # Add % symbol to improvement column
    formatted_stats["improvement"] = formatted_stats["improvement"].apply(
        lambda x: f"{x}%"
    )

    # Create table
    table = plt.table(
        cellText=formatted_stats.values,
        colLabels=formatted_stats.columns,
        loc="center",
        cellLoc="center",
        colColours=["#f0f0f0"] * len(formatted_stats.columns),
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.5)

    plt.title("YOLO Optimization Performance Metrics", pad=20, fontsize=14)
    plt.tight_layout()
    plt.savefig(save_path / "performance_summary_table.png", dpi=300)

    print(f"✅ Saved all visualizations to {save_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark YOLO optimizations (class filter, verbose settings)"
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
    args = parser.parse_args()

    print("🔍 Starting YOLO optimization benchmark")

    # Set up device for inference
    device = setup_device()
    print(f"📱 Using device: {device}")

    # Load YOLO model for baseline
    selected_model = args.model
    print(f"🧠 Loading baseline {selected_model} model...")
    baseline_model = YOLO(selected_model)
    print("✅ Baseline model loaded successfully.")

    # Configure optimized model with fixed settings
    print(f"🧠 Configuring optimized model with vehicle classes and no verbose...")
    optimized_model = configure_optimized_model(baseline_model)
    print("✅ Optimized model configured with:")
    print(f"   - Vehicle classes only: {VEHICLE_CLASSES}")
    print(f"   - Verbose mode: False")

    # Load test images
    images = load_test_images(limit=args.limit)
    if not images:
        print("❌ No images to process. Please add images to the test directory.")
        return

    # Run optimized detection with pre-configured model
    optimized_results = run_optimized_detection(
        images, optimized_model, device, conf=args.conf
    )

    # Run baseline detection
    baseline_results = run_baseline_detection(
        images, baseline_model, device, conf=args.conf
    )

    baseline_results = baseline_results[1:]
    optimized_results = optimized_results[1:]
    # Calculate statistics
    stats = calculate_statistics(baseline_results, optimized_results)

    # Print statistics summary
    print("\n📊 Optimization Benchmark Results:")
    pd.set_option("display.max_rows", None)
    pd.set_option("display.width", 1000)
    print(stats)

    # Visualize and save results
    visualize_results(stats, baseline_results, optimized_results, RESULTS_DIR)

    # Save results to CSV
    stats.to_csv(RESULTS_DIR / "optimization_benchmark_results.csv", index=False)

    # Create detailed HTML report
    html_path = RESULTS_DIR / "optimization_benchmark_report.html"
    with open(html_path, "w") as f:
        f.write("<html><head><title>YOLO Optimization Benchmark</title>")
        f.write(
            "<style>body{font-family:Arial; margin:20px} table{border-collapse:collapse; width:100%} "
        )
        f.write(
            "th,td{text-align:left; padding:8px; border:1px solid #ddd} </style></head><body>"
        )
        f.write("<h1>YOLO Optimization Benchmark</h1>")
        f.write(f"<p>Analysis date: {time.strftime('%Y-%m-%d %H:%M:%S')}</p>")
        f.write(f"<p>Model: {args.model}, Confidence threshold: {args.conf}</p>")

        # Add optimization description
        f.write("<h2>Optimizations Applied</h2>")
        f.write("<ul>")
        f.write(
            "<li><strong>Class Filtering:</strong> Only detecting vehicle classes [2, 3, 5, 7]</li>"
        )
        f.write("<li><strong>Verbose Mode:</strong> Disabled (verbose=False)</li>")
        f.write("</ul>")

        f.write("<h2>Performance Summary</h2>")
        f.write(stats.to_html(index=False))

        f.write("<h2>Visualizations</h2>")

        # Include all visualization images
        visualization_images = [
            "inference_time_comparison.png",
            "inference_time_distribution.png",
            "object_detection_comparison.png",
            "performance_metrics_comparison.png",
            "performance_summary_table.png",
        ]

        for img_name in visualization_images:
            f.write(
                f"<h3>{img_name.replace('_', ' ').replace('.png', '').title()}</h3>"
            )
            f.write(f'<img src="{img_name}" style="max-width:100%"><br><br>')

        f.write("</body></html>")

    print(f"✅ Benchmark complete. Results saved to {RESULTS_DIR}")
    print(
        f"📊 Statistics summary: {RESULTS_DIR / 'optimization_benchmark_results.csv'}"
    )
    print(f"📑 Detailed report: {html_path}")


if __name__ == "__main__":
    main()
