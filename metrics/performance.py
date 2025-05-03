#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Enhanced YOLO Model Benchmark
This script compares the inference performance of different YOLO models with scientific-grade metrics.
"""

import os
import sys
import time
import argparse
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import seaborn as sns
from pathlib import Path
import cv2
from tqdm import tqdm
import yaml
from datetime import datetime
from scipy import stats
import matplotlib.colors as mcolors
from typing import Dict, List, Tuple, Optional, Any, Union

# Add parent directory to path for imports from main project
script_dir = Path(__file__).parent
project_root = script_dir.parent
sys.path.insert(0, str(project_root))

# Import project modules
from device import setup_device
from ultralytics import YOLO
from config import DEFAULT_VISION_MODEL, DEFAULT_IMAGE_ROI
from imageprocessor import ImageProcessor

# Set Matplotlib style for scientific paper quality plots
plt.style.use("seaborn-v0_8-whitegrid")
plt.rcParams.update(
    {
        "font.family": "serif",
        "font.size": 12,
        "axes.titlesize": 14,
        "axes.labelsize": 12,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 10,
        "figure.titlesize": 16,
        "figure.figsize": (10, 7),
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.2,
        "figure.constrained_layout.use": True,
        "figure.constrained_layout.h_pad": 0.12,  # Erhöht für mehr Padding
        "figure.constrained_layout.w_pad": 0.12,  # Erhöht für mehr Padding
        "figure.constrained_layout.hspace": 0.12,  # Erhöht für mehr Padding
        "figure.constrained_layout.wspace": 0.12,  # Erhöht für mehr Padding
    }
)

# Constants
METRICS_DIR = Path("metrics")
TEST_IMAGES_DIR = METRICS_DIR / "test_images"
RESULTS_DIR = METRICS_DIR / "model_benchmark_results"

# Create results directory if it doesn't exist
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# Vehicle classes in YOLOv11: car(2), motorcycle(3), bus(5), truck(7)
VEHICLE_CLASSES = [2, 3, 5, 7]
VEHICLE_CLASSES_OWN = [1, 2, 3, 4]


def load_test_images(
    source_dir=TEST_IMAGES_DIR, limit=None
) -> List[Tuple[Path, np.ndarray]]:
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
            # Apply ROI crop
            img_cropped = ImageProcessor.crop_to_roi(img_rgb)
            images.append((img_path, img_cropped))

    print(f"Loaded {len(images)} test images")
    return images


def run_model_inference(
    model: YOLO,
    images: List[Tuple[Path, np.ndarray]],
    device: str,
    conf: float = 0.25,
    imgsz: int = 640,
    half: bool = False,
    warmup: bool = True,
    wait_time: float = 0.05,
) -> Dict[str, Any]:
    """
    Run inference with the specified model and collect performance metrics.

    Args:
        model: Loaded YOLO model
        images: List of (path, image) tuples
        device: Device to run inference on
        conf: Confidence threshold
        imgsz: Input image size
        half: Whether to use half precision (FP16)
        warmup: Whether to perform warmup runs
        wait_time: Time to wait between inferences (to prevent overheating)

    Returns:
        Dictionary with inference results and performance metrics
    """
    SELECTED_CLASSES = (
        VEHICLE_CLASSES
        if model.ckpt_path[-4:] in ["s.pt", "n.pt"]
        else VEHICLE_CLASSES_OWN
    )
    model_name = Path(model.ckpt_path).stem

    # Warm-up the model
    if warmup and len(images) > 0:
        print("Warming up model...")
        warmup_image = images[0][1]
        for _ in range(3):  # 3 warm-up runs
            model(warmup_image, device=device, verbose=False)

    print(f"Running inference with model: {model_name}")

    results = []
    inference_times = []
    total_objects = 0
    all_confidences = []
    all_class_ids = []
    per_class_confidences = {class_id: [] for class_id in SELECTED_CLASSES}

    for img_path, img in tqdm(images, desc="Processing images"):
        # Start timing
        start_time = time.time()

        # Run inference
        predictions = model(
            img,
            conf=conf,
            device=device,
            imgsz=imgsz,
            half=half,
            verbose=False,
        )

        # End timing
        inference_time = time.time() - start_time
        inference_times.append(inference_time)

        # Process results
        detected_objects = []

        for pred in predictions:
            boxes = pred.boxes
            for box in boxes:
                cls_id = int(box.cls.item())
                if not cls_id in SELECTED_CLASSES:
                    continue
                conf_val = box.conf.item()
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                class_name = pred.names[cls_id]

                # Add to results
                detected_objects.append(
                    {
                        "class": class_name,
                        "class_id": cls_id,
                        "confidence": conf_val,
                        "bbox": (x1, y1, x2, y2),
                        "area": (x2 - x1) * (y2 - y1),
                    }
                )

                all_confidences.append(conf_val)
                all_class_ids.append(cls_id)
                per_class_confidences[cls_id].append(conf_val)

        total_objects += len(detected_objects)

        # Sort detections by confidence (for better visualization)
        detected_objects.sort(key=lambda x: x["confidence"], reverse=True)

        results.append(
            {
                "filename": img_path.name,
                "inference_time": inference_time,
                "total_objects": len(detected_objects),
                "detections": detected_objects,
            }
        )

        # Wait between inferences if specified
        if wait_time > 0:
            time.sleep(wait_time)

    # Calculate performance metrics
    avg_inference_time = np.mean(inference_times) if inference_times else 0
    median_inference_time = np.median(inference_times) if inference_times else 0
    std_inference_time = np.std(inference_times) if inference_times else 0
    cv_inference_time = (
        (std_inference_time / avg_inference_time) if avg_inference_time > 0 else 0
    )

    avg_objects_per_image = total_objects / len(images) if images else 0

    # Confidence statistics
    avg_confidence = np.mean(all_confidences) if all_confidences else 0
    median_confidence = np.median(all_confidences) if all_confidences else 0
    min_confidence = np.min(all_confidences) if all_confidences else 0
    max_confidence = np.max(all_confidences) if all_confidences else 0
    std_confidence = np.std(all_confidences) if all_confidences else 0

    # Per-class confidence statistics
    per_class_confidence_stats = {}
    for cls_id in SELECTED_CLASSES:
        confs = per_class_confidences[cls_id]
        if confs:
            per_class_confidence_stats[cls_id] = {
                "mean": np.mean(confs),
                "median": np.median(confs),
                "min": np.min(confs),
                "max": np.max(confs),
                "std": np.std(confs),
                "count": len(confs),
            }
        else:
            per_class_confidence_stats[cls_id] = {"count": 0}

    # Process class distribution
    class_distribution = {}
    for result in results:
        for detection in result["detections"]:
            cls = detection["class"]
            cls_id = detection["class_id"]
            key = f"{cls} (id:{cls_id})"
            class_distribution[key] = class_distribution.get(key, 0) + 1

    # Calculate object size distribution
    areas = [d["area"] for r in results for d in r["detections"]]
    area_quantiles = (
        np.quantile(areas, [0, 0.25, 0.5, 0.75, 1]) if areas else np.zeros(5)
    )
    area_mean = np.mean(areas) if areas else 0

    # Calculate throughput metrics
    fps = 1.0 / avg_inference_time if avg_inference_time > 0 else 0

    summary = {
        "model_path": model.ckpt_path,
        "model_name": model_name,
        "device": str(device),
        "imgsz": imgsz,
        "confidence_threshold": conf,
        "half_precision": half,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M"),
        # Performance metrics
        "total_images": len(images),
        "avg_inference_time": avg_inference_time,
        "median_inference_time": median_inference_time,
        "std_inference_time": std_inference_time,
        "cv_inference_time": cv_inference_time,  # Coefficient of variation
        "min_inference_time": min(inference_times) if inference_times else 0,
        "max_inference_time": max(inference_times) if inference_times else 0,
        "fps": fps,
        # Object detection metrics
        "total_objects": total_objects,
        "avg_objects_per_image": avg_objects_per_image,
        # Confidence metrics
        "avg_confidence": avg_confidence,
        "median_confidence": median_confidence,
        "min_confidence": min_confidence,
        "max_confidence": max_confidence,
        "std_confidence": std_confidence,
        # Additional statistics
        "per_class_confidence": per_class_confidence_stats,
        "class_distribution": class_distribution,
        "area_quantiles": area_quantiles.tolist(),
        "area_mean": area_mean,
        "all_inference_times": inference_times,
        "all_confidences": all_confidences,
        "all_class_ids": all_class_ids,
        "classes": SELECTED_CLASSES,
        "detailed_results": results,
    }

    return summary


def create_results_folder(
    model_name: str,
    device_name: str,
    imgsz: int,
    half: bool,
    conf: float,
    test_run_id: Optional[str] = None,
) -> Path:
    """
    Create a unique folder to store the benchmark results with parameters in the name.

    Args:
        model_name: Name of the model
        device_name: Name of the device
        imgsz: Image size used
        half: Whether half precision was used
        conf: Confidence threshold
        test_run_id: Optional test run identifier

    Returns:
        Path to the results folder
    """
    precision = "fp16" if half else "fp32"

    # Format parameters for folder name
    folder_name = f"{model_name}_{device_name}_imgsz{imgsz}_{precision}_conf{conf:.2f}"

    if test_run_id:
        folder_name = f"{folder_name}_{test_run_id}"

    result_folder = RESULTS_DIR / folder_name
    result_folder.mkdir(parents=True, exist_ok=True)

    return result_folder


def generate_visualizations(summary: Dict[str, Any], save_path: Path) -> None:
    """
    Generate scientific paper quality visualizations of the benchmark results.

    Args:
        summary: Dictionary with benchmark results
        save_path: Path to save visualizations
    """
    # Colors from a qualitative color map for consistency
    colors = plt.cm.tab10.colors

    # Extract data
    model_name = summary["model_name"]
    device_name = str(summary["device"])
    imgsz = summary["imgsz"]
    half = summary["half_precision"]
    precision_str = "FP16" if half else "FP32"
    conf = summary["confidence_threshold"]

    inference_times = summary["all_inference_times"]
    inference_times_ms = [t * 1000 for t in inference_times]  # Convert to ms
    objects_detected = [r["total_objects"] for r in summary["detailed_results"]]
    confidences = summary["all_confidences"]
    class_ids = summary["all_class_ids"]

    # 1. Inference Time Distribution with scientific stats
    plt.figure(figsize=(12, 8))

    # Create main histogram with KDE
    ax = sns.histplot(
        inference_times_ms,
        kde=True,
        stat="density",
        bins=25,
        color=colors[0],
        alpha=0.7,
    )

    # Add statistical annotations with more detail
    mean_time = np.mean(inference_times_ms)
    median_time = np.median(inference_times_ms)
    std_time = np.std(inference_times_ms)
    cv = std_time / mean_time * 100  # Coefficient of variation in %

    # Fit a normal distribution to the data
    x = np.linspace(min(inference_times_ms), max(inference_times_ms), 100)
    if len(inference_times_ms) > 1:
        mu, sigma = stats.norm.fit(inference_times_ms)
        pdf = stats.norm.pdf(x, mu, sigma)
        plt.plot(
            x, pdf, "r-", linewidth=2, label=f"Normal Fit (μ={mu:.2f}, σ={sigma:.2f})"
        )

    # Add reference lines
    plt.axvline(
        x=mean_time,
        color=colors[1],
        linestyle="-",
        linewidth=2,
        label=f"Mean: {mean_time:.2f} ms",
    )
    plt.axvline(
        x=median_time,
        color=colors[2],
        linestyle="--",
        linewidth=2,
        label=f"Median: {median_time:.2f} ms",
    )

    # Add statistical summary box
    stats_text = (
        f"N = {len(inference_times_ms)}\n"
        f"Mean = {mean_time:.2f} ms\n"
        f"Median = {median_time:.2f} ms\n"
        f"Std Dev = {std_time:.2f} ms\n"
        f"CV = {cv:.2f}%\n"
        f"Range = [{min(inference_times_ms):.2f}, {max(inference_times_ms):.2f}] ms"
    )

    plt.annotate(
        stats_text,
        xy=(0.97, 0.97),
        xycoords="axes fraction",
        bbox=dict(boxstyle="round,pad=0.5", fc="white", ec="gray", alpha=0.8),
        ha="right",
        va="top",
        fontsize=10,
    )

    plt.title(
        f"Inference Time Distribution - {model_name} ({imgsz}x{imgsz}, {precision_str})"
    )
    plt.xlabel("Inference Time (ms)")
    plt.ylabel("Density")
    plt.legend(loc="upper right")
    plt.grid(True, alpha=0.3)
    # plt.tight_layout(pad=1.2, w_pad=0.5, h_pad=0.5)
    plt.savefig(
        save_path / "inference_time_distribution.png",
        bbox_inches="tight",
        pad_inches=0.2,
    )
    plt.close()

    # 2. Inference Time per Image (detailed time series analysis)
    plt.figure(figsize=(14, 8))

    # Plot inference times with trend line
    plt.plot(
        range(len(inference_times_ms)),
        inference_times_ms,
        marker="o",
        markersize=5,
        linestyle="-",
        alpha=0.7,
        color=colors[0],
        label="Inference Time",
    )

    # Add trend line
    if len(inference_times_ms) > 1:
        z = np.polyfit(range(len(inference_times_ms)), inference_times_ms, 1)
        p = np.poly1d(z)
        plt.plot(
            range(len(inference_times_ms)),
            p(range(len(inference_times_ms))),
            linestyle="--",
            color=colors[3],
            linewidth=2,
            label=f"Trend Line (Slope: {z[0]:.4f} ms/image)",
        )

    # Add moving average
    window_size = min(5, len(inference_times_ms))
    if window_size > 1:
        moving_avg = np.convolve(
            inference_times_ms, np.ones(window_size) / window_size, mode="valid"
        )
        ma_indices = range(window_size - 1, len(inference_times_ms))
        plt.plot(
            ma_indices,
            moving_avg,
            color=colors[4],
            linewidth=2,
            label=f"{window_size}-Point Moving Average",
        )

    # Add mean line and reference band (±1 std)
    plt.axhline(
        y=mean_time,
        color=colors[1],
        linestyle="-",
        linewidth=2,
        label=f"Mean: {mean_time:.2f} ms",
    )
    if len(inference_times_ms) > 1:
        plt.fill_between(
            range(len(inference_times_ms)),
            mean_time - std_time,
            mean_time + std_time,
            color=colors[1],
            alpha=0.2,
            label=f"±1 Std Dev ({std_time:.2f} ms)",
        )

    plt.title(
        f"Inference Time Series - {model_name} ({imgsz}x{imgsz}, {precision_str})"
    )
    plt.xlabel("Image Index")
    plt.ylabel("Inference Time (ms)")
    plt.legend(loc="upper right")
    plt.grid(True, alpha=0.3)
    # plt.tight_layout(pad=1.2, w_pad=0.5, h_pad=0.5)
    plt.savefig(
        save_path / "inference_time_series.png", bbox_inches="tight", pad_inches=0.2
    )
    plt.close()

    # 3. Confidence Score Distribution (advanced)
    if confidences:
        plt.figure(figsize=(12, 8))

        # Create violin plot for all confidence scores
        ax = plt.subplot(1, 2, 1)
        sns.violinplot(y=confidences, color=colors[0], inner="quartile")
        plt.title("Overall Confidence Distribution")
        plt.ylabel("Confidence Score")
        plt.grid(True, alpha=0.3)

        # Add statistical annotations
        stats_text = (
            f"N = {len(confidences)}\n"
            f"Mean = {np.mean(confidences):.3f}\n"
            f"Median = {np.median(confidences):.3f}\n"
            f"Std Dev = {np.std(confidences):.3f}\n"
            f"Range = [{min(confidences):.3f}, {max(confidences):.3f}]"
        )

        plt.annotate(
            stats_text,
            xy=(0.95, 0.05),
            xycoords="axes fraction",
            bbox=dict(boxstyle="round,pad=0.5", fc="white", ec="gray", alpha=0.8),
            ha="right",
            va="bottom",
            fontsize=10,
        )

        # Create histogram with KDE for confidence scores
        ax = plt.subplot(1, 2, 2)
        sns.histplot(confidences, kde=True, bins=20, color=colors[0])
        plt.axvline(
            x=np.mean(confidences),
            color=colors[1],
            linestyle="-",
            label=f"Mean: {np.mean(confidences):.3f}",
        )
        plt.axvline(
            x=np.median(confidences),
            color=colors[2],
            linestyle="--",
            label=f"Median: {np.median(confidences):.3f}",
        )
        plt.axvline(x=conf, color="red", linestyle="-.", label=f"Threshold: {conf:.3f}")
        plt.title("Confidence Distribution")
        plt.xlabel("Confidence Score")
        plt.ylabel("Count")
        plt.legend()
        plt.grid(True, alpha=0.3)

        plt.suptitle(f"Confidence Score Analysis - {model_name}", fontsize=16)
        # plt.tight_layout(pad=1.2, w_pad=0.5, h_pad=0.5)
        plt.savefig(
            save_path / "confidence_distribution.png",
            bbox_inches="tight",
            pad_inches=0.2,
        )
        plt.close()

        # 4. Per-Class Confidence Distribution
        if len(set(class_ids)) > 1:
            plt.figure(figsize=(14, 8))

            # Create a DataFrame for easier plotting
            conf_df = pd.DataFrame({"Confidence": confidences, "Class ID": class_ids})

            # Map class IDs to names
            class_names = {2: "Car", 3: "Motorcycle", 5: "Bus", 7: "Truck"}
            conf_df["Class"] = conf_df["Class ID"].map(
                lambda x: class_names.get(x, f"Class {x}")
            )

            # Boxplot for per-class confidence
            ax = sns.boxplot(
                x="Class",
                y="Confidence",
                hue="Class",
                data=conf_df,
                palette="tab10",
                legend=False,
            )

            # Add count annotations
            for i, class_name in enumerate(sorted(conf_df["Class"].unique())):
                count = len(conf_df[conf_df["Class"] == class_name])
                plt.annotate(
                    f"n={count}",
                    xy=(i, 0.03),
                    xycoords=("data", "axes fraction"),
                    ha="center",
                    va="bottom",
                    fontsize=10,
                )

            plt.title(f"Confidence Distribution by Class - {model_name}")
            plt.ylabel("Confidence Score")
            plt.grid(True, alpha=0.3)
            # plt.tight_layout(pad=1.2, w_pad=0.5, h_pad=0.5)
            plt.savefig(
                save_path / "per_class_confidence.png",
                bbox_inches="tight",
                pad_inches=0.2,
            )
            plt.close()

    # 5. Objects Detected per Image (advanced)
    plt.figure(figsize=(14, 8))

    # Plot detections
    plt.plot(
        range(len(objects_detected)),
        objects_detected,
        marker="o",
        markersize=5,
        linestyle="-",
        alpha=0.7,
        color=colors[0],
        label="Objects Detected",
    )

    # Add running average
    window_size = min(5, len(objects_detected))
    if window_size > 1:
        running_avg = np.convolve(
            objects_detected, np.ones(window_size) / window_size, mode="valid"
        )
        avg_indices = range(window_size - 1, len(objects_detected))
        plt.plot(
            avg_indices,
            running_avg,
            color=colors[1],
            linewidth=2,
            label=f"{window_size}-Point Moving Average",
        )

    # Add mean and std band
    mean_objects = np.mean(objects_detected)
    std_objects = np.std(objects_detected)
    plt.axhline(
        y=mean_objects,
        color=colors[2],
        linestyle="-",
        linewidth=2,
        label=f"Mean: {mean_objects:.2f} objects",
    )

    if len(objects_detected) > 1:
        plt.fill_between(
            range(len(objects_detected)),
            mean_objects - std_objects,
            mean_objects + std_objects,
            color=colors[2],
            alpha=0.2,
            label=f"±1 Std Dev ({std_objects:.2f})",
        )

    # Add statistical summary
    stats_text = (
        f"N = {len(objects_detected)}\n"
        f"Total Objects = {sum(objects_detected)}\n"
        f"Mean = {mean_objects:.2f}\n"
        f"Median = {np.median(objects_detected):.2f}\n"
        f"Std Dev = {std_objects:.2f}\n"
        f"Range = [{min(objects_detected)}, {max(objects_detected)}]"
    )

    plt.annotate(
        stats_text,
        xy=(0.97, 0.97),
        xycoords="axes fraction",
        bbox=dict(boxstyle="round,pad=0.5", fc="white", ec="gray", alpha=0.8),
        ha="right",
        va="top",
        fontsize=10,
    )

    plt.title(
        f"Objects Detected per Image - {model_name} ({imgsz}x{imgsz}, {precision_str})"
    )
    plt.xlabel("Image Index")
    plt.ylabel("Number of Objects")
    plt.legend(loc="upper right")
    plt.grid(True, alpha=0.3)
    # plt.tight_layout(pad=1.2, w_pad=0.5, h_pad=0.5)
    plt.savefig(
        save_path / "objects_detected_per_image.png",
        bbox_inches="tight",
        pad_inches=0.2,
    )
    plt.close()

    # 6. Class Distribution (enhanced)
    if summary["class_distribution"]:
        plt.figure(figsize=(12, 8))

        # Sort by count
        classes = list(summary["class_distribution"].keys())
        counts = list(summary["class_distribution"].values())
        sorted_data = sorted(zip(classes, counts), key=lambda x: x[1], reverse=True)
        classes, counts = zip(*sorted_data)

        # Calculate percentages
        total_count = sum(counts)
        percentages = [100 * count / total_count for count in counts]

        # Create a DataFrame for easier plotting
        df = pd.DataFrame(
            {"Class": classes, "Count": counts, "Percentage": percentages}
        )

        # Create bar chart with percentages
        ax = sns.barplot(
            x="Class", y="Count", data=df, hue="Class", palette="viridis", legend=False
        )

        # Add percentage annotations
        for i, p in enumerate(percentages):
            plt.annotate(
                f"{p:.1f}%",
                xy=(i, counts[i] + 0.5),
                ha="center",
                va="bottom",
                fontsize=10,
            )

        plt.title(
            f"Class Distribution - {model_name} ({imgsz}x{imgsz}, {precision_str})"
        )
        plt.xlabel("Class")
        plt.ylabel("Count")
        plt.xticks(rotation=45, ha="right")
        plt.grid(True, alpha=0.3, axis="y")
        # plt.tight_layout(pad=1.2, w_pad=0.5, h_pad=0.5)
        plt.savefig(
            save_path / "class_distribution.png", bbox_inches="tight", pad_inches=0.2
        )
        plt.close()

    # 7. Object Size Distribution
    if "area_quantiles" in summary and len(summary["area_quantiles"]) > 0:
        plt.figure(figsize=(12, 8))

        # Extract bounding box areas
        areas = [
            d["area"] for r in summary["detailed_results"] for d in r["detections"]
        ]

        if areas:
            # Create histogram of object sizes
            sns.histplot(areas, bins=30, kde=True, color=colors[0])

            # Add quartile lines
            quantiles = summary["area_quantiles"]
            labels = ["Min", "Q1", "Median", "Q3", "Max"]
            linestyles = [":", "--", "-", "--", ":"]

            for i, (q, label, ls) in enumerate(zip(quantiles, labels, linestyles)):
                plt.axvline(
                    x=q,
                    color=colors[i % len(colors)],
                    linestyle=ls,
                    label=f"{label}: {q:.0f} px²",
                )

            # Add mean line
            mean_area = summary["area_mean"]
            plt.axvline(
                x=mean_area,
                color="red",
                linestyle="-.",
                label=f"Mean: {mean_area:.0f} px²",
            )

            plt.title(f"Object Size Distribution - {model_name}")
            plt.xlabel("Bounding Box Area (pixels²)")
            plt.ylabel("Count")
            plt.legend()
            plt.grid(True, alpha=0.3)
            # plt.tight_layout(pad=1.2, w_pad=0.5, h_pad=0.5)
            plt.savefig(
                save_path / "object_size_distribution.png",
                bbox_inches="tight",
                pad_inches=0.2,
            )
            plt.close()

    fps = summary["fps"]

    # 10. Summary dashboard (combines key metrics visually)
    fig = plt.figure(figsize=(14, 10))

    # Create a 2x2 grid
    grid = plt.GridSpec(1, 2, figure=fig, wspace=0.3, hspace=0.3)

    # 1. Inference time histogram (upper left)
    ax1 = fig.add_subplot(grid[0, 0])
    sns.histplot(inference_times_ms, kde=True, bins=15, color=colors[0], ax=ax1)
    ax1.axvline(
        x=mean_time, color=colors[1], linestyle="-", label=f"Mean: {mean_time:.2f} ms"
    )
    ax1.set_title("Inference Time")
    ax1.set_xlabel("Time (ms)")
    ax1.legend()

    # 2. Class distribution (lower left)
    ax2 = fig.add_subplot(grid[0, 1])
    if summary["class_distribution"]:
        # Use only top 5 classes to avoid overcrowding
        classes = list(summary["class_distribution"].keys())
        counts = list(summary["class_distribution"].values())

        if len(classes) > 5:
            sorted_data = sorted(zip(classes, counts), key=lambda x: x[1], reverse=True)
            top_classes, top_counts = zip(*sorted_data[:5])
            other_count = sum(sorted_data[5:], key=lambda x: x[1])

            if other_count > 0:
                classes = list(top_classes) + ["Other"]
                counts = list(top_counts) + [other_count]
            else:
                classes = list(top_classes)
                counts = list(top_counts)

        # Create pie chart
        ax2.pie(
            counts,
            labels=classes,
            autopct="%1.1f%%",
            startangle=90,
            colors=plt.cm.tab10.colors[: len(classes)],
        )
        ax2.set_title(f"Class Distribution (Total: {sum(counts)})")
    else:
        ax2.text(0.5, 0.5, "No objects detected", ha="center", va="center", fontsize=12)
        ax2.set_title("Class Distribution")

    # Add metadata at the bottom
    plt.figtext(
        0.5,
        0.01,
        f"Device: {device_name} | Images: {summary['total_images']} | "
        f"Objects: {summary['total_objects']} | "
        f"Date: {summary['timestamp']}",
        ha="center",
        fontsize=10,
    )

    plt.savefig(
        save_path / "performance_dashboard.png", bbox_inches="tight", pad_inches=0.2
    )
    plt.close()

    print(f"✅ Generated {save_path.name} visualizations")


def save_results(summary: Dict[str, Any], save_path: Path) -> None:
    """
    Save the benchmark results to files with parameter-specific names.

    Args:
        summary: Dictionary with benchmark results
        save_path: Path to save results
    """
    # Extract parameters for file names
    model_name = summary["model_name"]
    device_name = str(summary["device"]).replace(":", "-")  # Sanitize device name
    imgsz = summary["imgsz"]
    precision = "fp16" if summary["half_precision"] else "fp32"
    conf = summary["confidence_threshold"]

    # Create a common prefix with parameters
    file_prefix = f"{model_name}_{device_name}_imgsz{imgsz}_{precision}_conf{conf:.2f}"

    # Save detailed results as JSON
    with open(save_path / f"{file_prefix}_detailed_results.json", "w") as f:
        # Create a copy with serializable detailed_results
        serializable_summary = summary.copy()
        serializable_summary["detailed_results"] = [
            {k: v for k, v in r.items() if k != "detections"}
            for r in summary["detailed_results"]
        ]
        # Handling numpy arrays
        serializable_summary["area_quantiles"] = [
            float(x) for x in serializable_summary["area_quantiles"]
        ]
        serializable_summary["all_inference_times"] = [
            float(x) for x in serializable_summary["all_inference_times"]
        ]
        serializable_summary["all_confidences"] = [
            float(x) for x in serializable_summary["all_confidences"]
        ]
        serializable_summary["all_class_ids"] = [
            int(x) for x in serializable_summary["all_class_ids"]
        ]

        json.dump(serializable_summary, f, indent=2)

    # Save summary as YAML
    with open(save_path / f"{file_prefix}_summary.yaml", "w") as f:
        # Create a copy without detailed_results and large arrays
        yaml_summary = {
            k: v
            for k, v in summary.items()
            if k
            not in [
                "detailed_results",
                "all_inference_times",
                "all_confidences",
                "all_class_ids",
            ]
        }
        yaml.dump(yaml_summary, f, default_flow_style=False)

    # Save inference times as CSV
    inference_df = pd.DataFrame(
        [
            {
                "image": r["filename"],
                "inference_time_ms": r["inference_time"] * 1000,
                "objects_detected": r["total_objects"],
            }
            for r in summary["detailed_results"]
        ]
    )
    inference_df.to_csv(save_path / f"{file_prefix}_inference_times.csv", index=False)

    # Save confidence scores as CSV
    if summary["all_confidences"]:
        conf_df = pd.DataFrame(
            {
                "confidence": summary["all_confidences"],
                "class_id": summary["all_class_ids"],
            }
        )
        conf_df.to_csv(save_path / f"{file_prefix}_confidence_scores.csv", index=False)

    print(f"✅ Results saved to {save_path}")


def create_report(
    summary: Dict[str, Any], save_path: Path, add_html: bool = True
) -> None:
    """
    Create a comprehensive report of the benchmark results in markdown format.

    Args:
        summary: Dictionary with benchmark results
        save_path: Path to save the report
        add_html: Whether to also create an HTML report
    """
    model_name = summary["model_name"]
    device_name = str(summary["device"])
    imgsz = summary["imgsz"]
    precision = "FP16" if summary["half_precision"] else "FP32"
    conf = summary["confidence_threshold"]

    # Create a file name with parameters
    file_prefix = f"{model_name}_{device_name}_imgsz{imgsz}_{precision}_conf{conf:.2f}"
    report_path = save_path / f"{file_prefix}_report.md"

    with open(report_path, "w") as f:
        # Title and metadata
        f.write(f"# YOLO Model Benchmark Report: {model_name}\n\n")
        f.write(f"**Date:** {summary['timestamp']}\n\n")

        # Model configuration
        f.write(f"## Model Configuration\n\n")
        f.write(f"- **Model:** {model_name}\n")
        f.write(f"- **Device:** {device_name}\n")
        f.write(f"- **Image Size:** {imgsz} x {imgsz}\n")
        f.write(f"- **Precision:** {precision}\n")
        f.write(f"- **Confidence Threshold:** {conf}\n")
        f.write(f"- **Classes:** {', '.join([str(c) for c in summary['classes']])}\n")
        f.write("\n")

        # Performance metrics
        f.write(f"## Performance Metrics\n\n")
        f.write(f"- **Total Images Processed:** {summary['total_images']}\n")
        f.write(
            f"- **Average Inference Time:** {summary['avg_inference_time'] * 1000:.2f} ms\n"
        )
        f.write(
            f"- **Median Inference Time:** {summary['median_inference_time'] * 1000:.2f} ms\n"
        )
        f.write(
            f"- **Standard Deviation:** {summary['std_inference_time'] * 1000:.2f} ms\n"
        )
        f.write(
            f"- **Coefficient of Variation:** {summary['cv_inference_time'] * 100:.2f}%\n"
        )
        f.write(
            f"- **Min Inference Time:** {summary['min_inference_time'] * 1000:.2f} ms\n"
        )
        f.write(
            f"- **Max Inference Time:** {summary['max_inference_time'] * 1000:.2f} ms\n"
        )
        f.write(f"- **Throughput:** {summary['fps']:.2f} FPS\n")

        # Detection statistics
        f.write(f"## Detection Statistics\n\n")
        f.write(f"- **Total Objects Detected:** {summary['total_objects']}\n")
        f.write(
            f"- **Average Objects per Image:** {summary['avg_objects_per_image']:.2f}\n"
        )
        f.write(f"- **Average Confidence Score:** {summary['avg_confidence']:.3f}\n")
        f.write(f"- **Median Confidence Score:** {summary['median_confidence']:.3f}\n")
        f.write(f"- **Min Confidence Score:** {summary['min_confidence']:.3f}\n")
        f.write(f"- **Max Confidence Score:** {summary['max_confidence']:.3f}\n\n")

        # Class distribution
        f.write(f"## Class Distribution\n\n")
        if summary["class_distribution"]:
            # Sort by count
            sorted_classes = sorted(
                summary["class_distribution"].items(), key=lambda x: x[1], reverse=True
            )

            f.write("| Class | Count | Percentage |\n")
            f.write("|-------|-------|------------|\n")

            for cls, count in sorted_classes:
                percentage = (
                    (count / summary["total_objects"]) * 100
                    if summary["total_objects"] > 0
                    else 0
                )
                f.write(f"| {cls} | {count} | {percentage:.2f}% |\n")
        else:
            f.write("No objects detected.\n")

        f.write("\n")

        # Per-class confidence
        if "per_class_confidence" in summary:
            f.write(f"## Per-Class Confidence\n\n")

            f.write("| Class ID | Count | Mean | Median | Min | Max | Std Dev |\n")
            f.write("|----------|-------|------|--------|-----|-----|--------|\n")

            for cls_id, stats in summary["per_class_confidence"].items():
                if stats.get("count", 0) > 0:
                    f.write(
                        f"| {cls_id} | {stats['count']} | {stats.get('mean', 0):.3f} | "
                        f"{stats.get('median', 0):.3f} | {stats.get('min', 0):.3f} | "
                        f"{stats.get('max', 0):.3f} | {stats.get('std', 0):.3f} |\n"
                    )

            f.write("\n")

        # Object size statistics
        if "area_quantiles" in summary and len(summary["area_quantiles"]) > 0:
            f.write(f"## Object Size Statistics\n\n")
            f.write(f"- **Mean Area:** {summary['area_mean']:.1f} pixels^2\n")
            f.write(
                f"- **Minimum Area:** {summary['area_quantiles'][0]:.1f} pixels^2\n"
            )
            f.write(f"- **Q1 Area:** {summary['area_quantiles'][1]:.1f} pixels^2\n")
            f.write(f"- **Median Area:** {summary['area_quantiles'][2]:.1f} pixels^2\n")
            f.write(f"- **Q3 Area:** {summary['area_quantiles'][3]:.1f} pixels^2\n")
            f.write(
                f"- **Maximum Area:** {summary['area_quantiles'][4]:.1f} pixels^2\n\n"
            )

        # Visualizations
        f.write(f"## Visualizations\n\n")

        # Add all generated visualizations
        visualization_images = [
            ("inference_time_distribution.png", "Inference Time Distribution"),
            ("inference_time_series.png", "Inference Time Series"),
            ("confidence_distribution.png", "Confidence Score Distribution"),
            ("per_class_confidence.png", "Per-Class Confidence"),
            ("objects_detected_per_image.png", "Objects Detected per Image"),
            ("class_distribution.png", "Class Distribution"),
            ("object_size_distribution.png", "Object Size Distribution"),
            ("throughput_analysis.png", "Throughput Analysis"),
            ("efficiency_horizon.png", "Efficiency Horizon"),
            ("performance_dashboard.png", "Performance Dashboard"),
        ]

        for img_file, title in visualization_images:
            if (save_path / img_file).exists():
                f.write(f"### {title}\n\n")
                f.write(f"![{title}]({img_file})\n\n")

    print(f"✅ Report saved to {report_path}")

    # Create HTML report if requested
    if add_html:
        try:
            import markdown
            from markdown.extensions.tables import TableExtension

            with open(report_path, "r") as f:
                md_content = f.read()

            html_content = markdown.markdown(md_content, extensions=[TableExtension()])

            # Add header and styling
            html_report = f"""
            <!DOCTYPE html>
            <html>
            <head>
                <title>YOLO Model Benchmark Report: {model_name}</title>
                <style>
                    body {{ font-family: 'Segoe UI', Arial, sans-serif; line-height: 1.6; max-width: 1200px; margin: 0 auto; padding: 20px; color: #333; }}
                    h1, h2, h3 {{ color: #2c3e50; margin-top: 1.5em; }}
                    h1 {{ border-bottom: 2px solid #eaecef; padding-bottom: 0.3em; }}
                    h2 {{ border-bottom: 1px solid #eaecef; padding-bottom: 0.3em; }}
                    table {{ border-collapse: collapse; width: 100%; margin: 20px 0; }}
                    th, td {{ text-align: left; padding: 12px; border-bottom: 1px solid #ddd; }}
                    th {{ background-color: #f2f3f4; font-weight: 600; }}
                    tr:nth-child(even) {{ background-color: #f9f9f9; }}
                    tr:hover {{ background-color: #f2f2f2; }}
                    img {{ max-width: 100%; height: auto; margin: 20px 0; border: 1px solid #eaecef; border-radius: 5px; box-shadow: 0 2px 5px rgba(0,0,0,0.1); }}
                    code {{ background-color: #f6f8fa; padding: 2px 5px; border-radius: 3px; font-family: 'Consolas', monospace; font-size: 0.9em; }}
                    ul, ol {{ padding-left: 25px; }}
                    li {{ margin-bottom: 5px; }}
                </style>
            </head>
            <body>
                {html_content}
                <footer style="margin-top: 50px; border-top: 1px solid #eaecef; padding-top: 20px; text-align: center; font-size: 0.8em; color: #6c757d;">
                    Generated on {datetime.now().strftime("%Y-%m-%d %H:%M:%S")} | YOLO Model Benchmark Tool
                </footer>
            </body>
            </html>
            """

            html_path = save_path / f"{file_prefix}_report.html"
            with open(html_path, "w") as f:
                f.write(html_report)

            print(f"✅ HTML report saved to {html_path}")
        except ImportError:
            print(
                "⚠️ Python markdown package not found, skipping HTML report generation"
            )
            print("   Install with: pip install markdown")


def compare_models_report(summaries: List[Dict[str, Any]], save_path: Path) -> None:
    """
    Create a scientific quality comparison report for multiple models.

    Args:
        summaries: List of dictionaries with benchmark results
        save_path: Path to save the report
    """
    if len(summaries) <= 1:
        return

    compare_path = save_path / "model_comparison"
    compare_path.mkdir(exist_ok=True)

    # Prepare comparison data
    models = [s["model_name"] for s in summaries]
    configs = [
        f"{s['imgsz']}x{s['imgsz']}, {'FP16' if s['half_precision'] else 'FP32'}"
        for s in summaries
    ]
    avg_times = [s["avg_inference_time"] * 1000 for s in summaries]  # Convert to ms
    median_times = [s["median_inference_time"] * 1000 for s in summaries]
    std_times = [s["std_inference_time"] * 1000 for s in summaries]
    fps_values = [s["fps"] for s in summaries]
    avg_objects = [s["avg_objects_per_image"] for s in summaries]
    avg_confidence = [s["avg_confidence"] for s in summaries]

    # Find base model for relative comparisons
    base_model_idx = 0  # Default to first model

    # Create a combined identifier for better labeling
    model_labels = [f"{m} ({c})" for m, c in zip(models, configs)]

    # 1. Inference Time Comparison (Enhanced)
    plt.figure(figsize=(14, 10))

    # Create combined bar chart for avg and median time
    x = np.arange(len(models))
    width = 0.35

    # Plot bars
    ax = plt.subplot(2, 1, 1)
    bar1 = ax.bar(
        x - width / 2,
        avg_times,
        width,
        label="Average Time",
        color=plt.cm.tab10.colors[0],
    )
    bar2 = ax.bar(
        x + width / 2,
        median_times,
        width,
        label="Median Time",
        color=plt.cm.tab10.colors[1],
    )

    # Add error bars
    ax.errorbar(
        x - width / 2, avg_times, yerr=std_times, fmt="none", color="black", capsize=5
    )

    # Add value labels
    for i, v in enumerate(avg_times):
        ax.text(i - width / 2, v + 0.5, f"{v:.2f}", ha="center", fontsize=9)
    for i, v in enumerate(median_times):
        ax.text(i + width / 2, v + 0.5, f"{v:.2f}", ha="center", fontsize=9)

    ax.set_xlabel("Model Configuration")
    ax.set_ylabel("Inference Time (ms)")
    ax.set_title("Inference Time Comparison")
    ax.set_xticks(x)
    ax.set_xticklabels(model_labels, rotation=45, ha="right")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Add relative comparison subplot
    ax = plt.subplot(2, 1, 2)
    relative_avg_times = [100 * (t / avg_times[base_model_idx]) for t in avg_times]
    relative_median_times = [
        100 * (t / median_times[base_model_idx]) for t in median_times
    ]

    # Calculate speedup percentage
    speedup_pct = [
        (avg_times[base_model_idx] - t) / avg_times[base_model_idx] * 100
        for t in avg_times
    ]

    # Plot relative times
    bar3 = ax.bar(
        x - width / 2,
        relative_avg_times,
        width,
        label="Relative Avg Time",
        color=plt.cm.tab10.colors[0],
        alpha=0.7,
    )
    bar4 = ax.bar(
        x + width / 2,
        relative_median_times,
        width,
        label="Relative Median Time",
        color=plt.cm.tab10.colors[1],
        alpha=0.7,
    )

    # Mark 100% line
    ax.axhline(y=100, color="red", linestyle="--", alpha=0.7, label="Baseline")

    # Add percentage labels
    for i, (pct, sp_pct) in enumerate(zip(relative_avg_times, speedup_pct)):
        if i == base_model_idx:
            ax.text(i - width / 2, pct + 2, "Baseline", ha="center", fontsize=9)
        else:
            ax.text(
                i - width / 2,
                pct + 2,
                f"{pct:.1f}%\n({'+' if sp_pct < 0 else '-'}{abs(sp_pct):.1f}%)",
                ha="center",
                fontsize=9,
            )

    for i, pct in enumerate(relative_median_times):
        if i == base_model_idx:
            ax.text(i + width / 2, pct + 2, "Baseline", ha="center", fontsize=9)
        else:
            ax.text(i + width / 2, pct + 2, f"{pct:.1f}%", ha="center", fontsize=9)

    ax.set_xlabel("Model Configuration")
    ax.set_ylabel("Relative Time (%)")
    ax.set_title(f"Relative Performance (Baseline: {model_labels[base_model_idx]})")
    ax.set_xticks(x)
    ax.set_xticklabels(model_labels, rotation=45, ha="right")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # plt.tight_layout(pad=1.2, w_pad=0.5, h_pad=0.5)
    plt.savefig(
        compare_path / "inference_time_comparison.png",
        bbox_inches="tight",
        pad_inches=0.2,
    )
    plt.close()

    # 2. Throughput Comparison (FPS)
    plt.figure(figsize=(14, 8))

    # Create a color gradient based on performance
    norm = plt.Normalize(min(fps_values), max(fps_values))
    colors = plt.cm.viridis(norm(fps_values))

    # Sort by FPS for better visualization
    sorted_indices = np.argsort(fps_values)
    sorted_models = [model_labels[i] for i in sorted_indices]
    sorted_fps = [fps_values[i] for i in sorted_indices]
    sorted_colors = colors[sorted_indices]

    # Plot FPS
    ax = plt.subplot(1, 1, 1)
    bars = ax.barh(sorted_models, sorted_fps, color=sorted_colors)

    # Add FPS labels
    for i, v in enumerate(sorted_fps):
        ax.text(v + 0.1, i, f"{v:.2f} FPS", va="center")

    # Add speed multiplier compared to slowest model
    slowest_fps = min(fps_values)
    for i, v in enumerate(sorted_fps):
        multiplier = v / slowest_fps
        if multiplier > 1.1:  # Only show if significantly faster
            ax.text(
                v / 2,
                i,
                f"{multiplier:.1f}x",
                va="center",
                ha="center",
                color="white",
                fontweight="bold",
                fontsize=10,
            )

    # Add colorbar legend
    sm = plt.cm.ScalarMappable(cmap=plt.cm.viridis, norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax)
    cbar.set_label("Frames Per Second (FPS)")

    ax.set_xlabel("Frames Per Second (FPS)")
    ax.set_title("Throughput Comparison")
    ax.grid(True, alpha=0.3)

    # plt.tight_layout(pad=1.2, w_pad=0.5, h_pad=0.5)
    plt.savefig(
        compare_path / "throughput_comparison.png", bbox_inches="tight", pad_inches=0.2
    )
    plt.close()

    # 3. Multi-Metric Comparison (Radar Chart)
    plt.figure(figsize=(14, 12))

    # Prepare metrics for radar chart
    metrics = ["FPS", "Obj/Image", "Confidence", "1/Time (ms)", "Times Std (Stability)"]

    # Normalize values to 0-1 range for radar chart
    norm_fps = (
        np.array(fps_values) / max(fps_values)
        if max(fps_values) > 0
        else np.zeros_like(fps_values)
    )
    norm_obj = (
        np.array(avg_objects) / max(avg_objects)
        if max(avg_objects) > 0
        else np.zeros_like(avg_objects)
    )
    norm_conf = (
        np.array(avg_confidence) / max(avg_confidence)
        if max(avg_confidence) > 0
        else np.zeros_like(avg_confidence)
    )
    norm_time = (
        np.array([1 / t for t in avg_times]) / max([1 / t for t in avg_times])
        if min(avg_times) > 0
        else np.zeros_like(avg_times)
    )
    # For std stability, lower is better, so invert
    norm_std = 1 - (
        np.array(std_times) / max(std_times)
        if max(std_times) > 0
        else np.zeros_like(std_times)
    )

    # Combine metrics
    metrics_values = [norm_fps, norm_obj, norm_conf, norm_time, norm_std]

    # Number of metrics
    N = len(metrics)

    # Create angle array
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    angles += angles[:1]  # Close the loop

    # Create subplot grid
    if len(models) <= 3:
        grid_size = (1, len(models))
    elif len(models) <= 6:
        grid_size = (2, 3)
    else:
        grid_size = (3, 3)

    # Create radar charts for each model
    for i, (model, label) in enumerate(zip(models, model_labels)):
        ax = plt.subplot(grid_size[0], grid_size[1], i + 1, polar=True)

        # Collect values for this model
        values = [metrics_values[j][i] for j in range(len(metrics))]
        values += values[:1]  # Close the loop

        # Plot radar
        ax.plot(
            angles,
            values,
            "o-",
            linewidth=2,
            label=label,
            color=plt.cm.tab10.colors[i % 10],
        )
        ax.fill(angles, values, alpha=0.25, color=plt.cm.tab10.colors[i % 10])

        # Set ticks and labels
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(metrics)
        ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
        ax.set_yticklabels(["20%", "40%", "60%", "80%", "100%"])
        ax.set_ylim(0, 1)

        # Set title
        ax.set_title(label, size=11)

    # Add a legend if there's space
    if len(models) < grid_size[0] * grid_size[1]:
        plt.subplot(grid_size[0], grid_size[1], len(models) + 1)
        plt.axis("off")
        plt.legend(
            [f"{m} - {c}" for m, c in zip(models, configs)], loc="center", fontsize=10
        )

    plt.suptitle("Multi-Metric Model Comparison", fontsize=16)
    # plt.tight_layout(pad=1.2, w_pad=0.5, h_pad=0.5)
    plt.savefig(
        compare_path / "multi_metric_comparison.png",
        bbox_inches="tight",
        pad_inches=0.2,
    )
    plt.close()

    # 4. Performance vs Objects Detected Scatter Plot
    plt.figure(figsize=(12, 8))

    # Create scatter plot with FPS vs Objects per image
    sc = plt.scatter(
        avg_objects, fps_values, c=avg_confidence, cmap="viridis", s=100, alpha=0.8
    )

    # Add model labels
    for i, txt in enumerate(model_labels):
        plt.annotate(
            txt,
            (avg_objects[i], fps_values[i]),
            xytext=(5, 5),
            textcoords="offset points",
        )

    plt.xlabel("Average Objects per Image")
    plt.ylabel("Frames Per Second (FPS)")
    plt.title("Performance vs Detection Capability")
    plt.grid(True, alpha=0.3)

    # Add colorbar for confidence
    cbar = plt.colorbar(sc)
    cbar.set_label("Average Confidence Score")

    # plt.tight_layout(pad=1.2, w_pad=0.5, h_pad=0.5)
    plt.savefig(
        compare_path / "performance_vs_objects.png", bbox_inches="tight", pad_inches=0.2
    )
    plt.close()

    # 5. Comprehensive Comparison Table
    plt.figure(figsize=(16, 8))
    plt.axis("off")

    # Prepare the table data
    data = []
    columns = [
        "Model",
        "Image Size",
        "Precision",
        "Avg Time (ms)",
        "Median Time (ms)",
        "Std Dev (ms)",
        "FPS",
        "Objects/Image",
        "Confidence",
    ]

    for i, s in enumerate(summaries):
        row = [
            s["model_name"],
            f"{s['imgsz']}x{s['imgsz']}",
            "FP16" if s["half_precision"] else "FP32",
            f"{s['avg_inference_time'] * 1000:.2f}",
            f"{s['median_inference_time'] * 1000:.2f}",
            f"{s['std_inference_time'] * 1000:.2f}",
            f"{s['fps']:.2f}",
            f"{s['avg_objects_per_image']:.2f}",
            f"{s['avg_confidence']:.3f}",
        ]
        data.append(row)

    # Add performance comparison row if more than one model
    if len(summaries) > 1:
        # Find best and worst performers
        best_time_idx = np.argmin(avg_times)
        worst_time_idx = np.argmax(avg_times)
        best_fps_idx = np.argmax(fps_values)
        best_objects_idx = np.argmax(avg_objects)
        best_conf_idx = np.argmax(avg_confidence)

        # Highlight in table data
        for i in range(len(data)):
            # Highlight best time
            if i == best_time_idx:
                data[i][3] = f"→ {data[i][3]} ←"
            # Highlight best FPS
            if i == best_fps_idx:
                data[i][6] = f"→ {data[i][6]} ←"
            # Highlight best objects detection
            if i == best_objects_idx:
                data[i][7] = f"→ {data[i][7]} ←"
            # Highlight best confidence
            if i == best_conf_idx:
                data[i][8] = f"→ {data[i][8]} ←"

    # Create the table
    table = plt.table(
        cellText=data,
        colLabels=columns,
        loc="center",
        cellLoc="center",
        colWidths=[0.12, 0.1, 0.08, 0.1, 0.1, 0.1, 0.08, 0.1, 0.1],
    )

    # Style the table
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.5)

    # Highlight the header row
    for (i, j), cell in table.get_celld().items():
        if i == 0:  # Header row
            cell.set_facecolor("#f2f2f2")
            cell.set_text_props(weight="bold")

    plt.title("Model Comparison Summary", pad=20, fontsize=16)
    # plt.tight_layout(pad=1.2, w_pad=0.5, h_pad=0.5)
    plt.savefig(
        compare_path / "comparison_table.png", bbox_inches="tight", pad_inches=0.2
    )
    plt.close()

    # 6. Create comparison report
    report_path = compare_path / "model_comparison_report.md"

    with open(report_path, "w") as f:
        # Title and metadata
        f.write(f"# YOLO Model Comparison Report\n\n")
        f.write(f"**Date:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

        # Models compared
        f.write(f"## Models Compared\n\n")
        for i, s in enumerate(summaries):
            precision = "FP16" if s["half_precision"] else "FP32"
            f.write(
                f"{i+1}. **{s['model_name']}** ({s['imgsz']}x{s['imgsz']}, {precision})\n"
            )
        f.write("\n")

        # Comparison summary table
        f.write(f"## Comparison Summary\n\n")

        f.write(
            "| Model | Image Size | Precision | Avg Time (ms) | Median Time (ms) | FPS | Objects/Image | Avg Confidence |\n"
        )
        f.write(
            "|-------|------------|-----------|---------------|------------------|-----|--------------|----------------|\n"
        )

        # Sort models by speed for better comparison
        sorted_summaries = sorted(summaries, key=lambda x: x["avg_inference_time"])

        for s in sorted_summaries:
            precision = "FP16" if s["half_precision"] else "FP32"
            f.write(
                f"| {s['model_name']} | {s['imgsz']}x{s['imgsz']} | {precision} | "
                f"{s['avg_inference_time'] * 1000:.2f} | {s['median_inference_time'] * 1000:.2f} | "
                f"{s['fps']:.2f} | {s['avg_objects_per_image']:.2f} | {s['avg_confidence']:.3f} |\n"
            )
        f.write("\n")

        # Performance comparison
        f.write(f"## Performance Comparison\n\n")
        f.write("### Inference Time\n\n")
        f.write("![Inference Time Comparison](inference_time_comparison.png)\n\n")

        f.write("### Throughput (FPS)\n\n")
        f.write("![Throughput Comparison](throughput_comparison.png)\n\n")

        # Multi-metric comparison
        f.write(f"## Multi-Metric Comparison\n\n")
        f.write(
            "This radar chart shows the relative performance of each model across multiple metrics:\n\n"
        )
        f.write("![Multi-Metric Comparison](multi_metric_comparison.png)\n\n")

        # Performance vs Objects
        f.write(f"## Performance vs Detection Capability\n\n")
        f.write(
            "This scatter plot shows the relationship between performance (FPS) and detection capability (objects per image):\n\n"
        )
        f.write("![Performance vs Objects](performance_vs_objects.png)\n\n")

        # Comprehensive comparison table
        f.write(f"## Comprehensive Comparison\n\n")
        f.write("This table summarizes all key metrics for each model:\n\n")
        f.write("![Comparison Table](comparison_table.png)\n\n")

        # Conclusion
        f.write(f"## Conclusion\n\n")

        # Find the best models in different categories
        fastest_idx = np.argmax([s["fps"] for s in summaries])
        fastest_model = summaries[fastest_idx]["model_name"]
        fastest_config = f"{summaries[fastest_idx]['imgsz']}x{summaries[fastest_idx]['imgsz']}, {'FP16' if summaries[fastest_idx]['half_precision'] else 'FP32'}"
        fastest_fps = summaries[fastest_idx]["fps"]

        most_detections_idx = np.argmax([s["avg_objects_per_image"] for s in summaries])
        most_detections_model = summaries[most_detections_idx]["model_name"]
        most_detections_config = f"{summaries[most_detections_idx]['imgsz']}x{summaries[most_detections_idx]['imgsz']}, {'FP16' if summaries[most_detections_idx]['half_precision'] else 'FP32'}"
        most_detections_avg = summaries[most_detections_idx]["avg_objects_per_image"]

        highest_confidence_idx = np.argmax([s["avg_confidence"] for s in summaries])
        highest_confidence_model = summaries[highest_confidence_idx]["model_name"]
        highest_confidence_config = f"{summaries[highest_confidence_idx]['imgsz']}x{summaries[highest_confidence_idx]['imgsz']}, {'FP16' if summaries[highest_confidence_idx]['half_precision'] else 'FP32'}"
        highest_confidence_avg = summaries[highest_confidence_idx]["avg_confidence"]

        # Most stable (lowest coefficient of variation)
        cv_values = [
            s["std_inference_time"] / s["avg_inference_time"] for s in summaries
        ]
        most_stable_idx = np.argmin(cv_values)
        most_stable_model = summaries[most_stable_idx]["model_name"]
        most_stable_config = f"{summaries[most_stable_idx]['imgsz']}x{summaries[most_stable_idx]['imgsz']}, {'FP16' if summaries[most_stable_idx]['half_precision'] else 'FP32'}"
        most_stable_cv = cv_values[most_stable_idx] * 100

        f.write(
            f"- **Fastest Model:** {fastest_model} ({fastest_config}) at {fastest_fps:.2f} FPS\n"
        )
        f.write(
            f"- **Most Objects Detected:** {most_detections_model} ({most_detections_config}) with {most_detections_avg:.2f} objects/image\n"
        )
        f.write(
            f"- **Highest Confidence:** {highest_confidence_model} ({highest_confidence_config}) with {highest_confidence_avg:.3f} average confidence\n"
        )
        f.write(
            f"- **Most Stable:** {most_stable_model} ({most_stable_config}) with {most_stable_cv:.2f}% coefficient of variation\n\n"
        )

        # Performance spread analysis
        if len(summaries) > 1:
            fastest_time = min([s["avg_inference_time"] for s in summaries])
            slowest_time = max([s["avg_inference_time"] for s in summaries])
            time_ratio = slowest_time / fastest_time

            f.write(f"**Performance Spread Analysis:**\n\n")
            f.write(
                f"- The fastest model is {time_ratio:.1f}x faster than the slowest model\n"
            )
            f.write(f"- Performance spread across models: {(time_ratio-1)*100:.1f}%\n")

            # Recommend models for different use cases
            f.write("\n**Recommendations:**\n\n")
            f.write(
                f"- For speed-critical applications: **{fastest_model}** ({fastest_config})\n"
            )
            f.write(
                f"- For maximum detection capability: **{most_detections_model}** ({most_detections_config})\n"
            )
            f.write(
                f"- For highest detection confidence: **{highest_confidence_model}** ({highest_confidence_config})\n"
            )
            f.write(
                f"- For consistent performance: **{most_stable_model}** ({most_stable_config})\n"
            )

    print(f"✅ Comparison report saved to {report_path}")

    # Create HTML version of comparison report
    try:
        import markdown
        from markdown.extensions.tables import TableExtension

        with open(report_path, "r") as f:
            md_content = f.read()

        html_content = markdown.markdown(md_content, extensions=[TableExtension()])

        # Add header and styling
        html_report = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>YOLO Model Comparison Report</title>
            <style>
                body {{ font-family: 'Segoe UI', Arial, sans-serif; line-height: 1.6; max-width: 1200px; margin: 0 auto; padding: 20px; color: #333; }}
                h1, h2, h3 {{ color: #2c3e50; margin-top: 1.5em; }}
                h1 {{ border-bottom: 2px solid #eaecef; padding-bottom: 0.3em; }}
                h2 {{ border-bottom: 1px solid #eaecef; padding-bottom: 0.3em; }}
                table {{ border-collapse: collapse; width: 100%; margin: 20px 0; }}
                th, td {{ text-align: left; padding: 12px; border-bottom: 1px solid #ddd; }}
                th {{ background-color: #f2f3f4; font-weight: 600; }}
                tr:nth-child(even) {{ background-color: #f9f9f9; }}
                tr:hover {{ background-color: #f2f2f2; }}
                img {{ max-width: 100%; height: auto; margin: 20px 0; border: 1px solid #eaecef; border-radius: 5px; box-shadow: 0 2px 5px rgba(0,0,0,0.1); }}
                code {{ background-color: #f6f8fa; padding: 2px 5px; border-radius: 3px; font-family: 'Consolas', monospace; font-size: 0.9em; }}
                ul, ol {{ padding-left: 25px; }}
                li {{ margin-bottom: 5px; }}
            </style>
        </head>
        <body>
            {html_content}
            <footer style="margin-top: 50px; border-top: 1px solid #eaecef; padding-top: 20px; text-align: center; font-size: 0.8em; color: #6c757d;">
                Generated on {datetime.now().strftime("%Y-%m-%d %H:%M:%S")} | YOLO Model Benchmark Tool
            </footer>
        </body>
        </html>
        """

        html_path = compare_path / "model_comparison_report.html"
        with open(html_path, "w") as f:
            f.write(html_report)

        print(f"✅ HTML comparison report saved to {html_path}")
    except ImportError:
        print("⚠️ Python markdown package not found, skipping HTML comparison report")
        print("   Install with: pip install markdown")


def parse_model_config(config_string: str) -> Dict[str, Any]:
    """
    Parse a model configuration string of format: model_path:imgsz:half:conf

    Args:
        config_string: String with model configuration

    Returns:
        Dictionary with model configuration
    """
    parts = config_string.split(":")
    model_path = parts[0]

    # Default values
    config = {"model_path": model_path, "imgsz": 640, "half": False, "conf": 0.25}

    # Update with provided values
    if len(parts) > 1 and parts[1]:
        config["imgsz"] = int(parts[1])
    if len(parts) > 2 and parts[2]:
        config["half"] = parts[2].lower() in ("true", "t", "yes", "y", "1")
    if len(parts) > 3 and parts[3]:
        config["conf"] = float(parts[3])

    return config


def main():
    parser = argparse.ArgumentParser(
        description="Enhanced YOLO model benchmark with scientific-grade metrics"
    )

    # Model parameters
    parser.add_argument(
        "--model",
        type=str,
        default="./yolo11n.pt",
        help="Path to the YOLO model (default: ./yolo11n.pt)",
    )
    parser.add_argument(
        "--models",
        type=str,
        default=None,
        help="Comma-separated list of models to benchmark with optional parameters, format: "
        "model_path[:imgsz[:half[:conf]]], e.g. yolo11n.pt:640:true:0.25,yolo11s.pt:320:false:0.3",
    )

    # Inference parameters
    parser.add_argument(
        "--imgsz", type=int, default=640, help="Input image size (default: 640)"
    )
    parser.add_argument(
        "--conf", type=float, default=0.25, help="Confidence threshold (default: 0.25)"
    )
    parser.add_argument("--half", action="store_true", help="Use half precision (FP16)")
    parser.add_argument(
        "--no-warmup", action="store_true", help="Skip model warmup runs"
    )
    parser.add_argument(
        "--wait-time",
        type=float,
        default=0.0,
        help="Time to wait between inferences in seconds (default: 0.0)",
    )

    # Run parameters
    parser.add_argument(
        "--long-run",
        action="store_true",
        help="Run long benchmark (multiple passes) to detect performance degradation",
    )
    parser.add_argument(
        "--passes",
        type=int,
        default=3,
        help="Number of passes for long-run benchmark (default: 3)",
    )
    parser.add_argument(
        "--test-run-id",
        type=str,
        default=None,
        help="Optional identifier for the test run",
    )

    # Data parameters
    parser.add_argument(
        "--limit", type=int, default=None, help="Limit the number of images to process"
    )
    parser.add_argument(
        "--image-dir",
        type=str,
        default=str(TEST_IMAGES_DIR),
        help=f"Path to test images (default: {TEST_IMAGES_DIR})",
    )

    # Output parameters
    parser.add_argument(
        "--output",
        type=str,
        default=str(RESULTS_DIR),
        help=f"Path to output directory (default: {RESULTS_DIR})",
    )
    parser.add_argument(
        "--no-html", action="store_true", help="Do not create HTML reports"
    )
    parser.add_argument(
        "--no-compare",
        action="store_true",
        help="Do not create comparison report when benchmarking multiple models",
    )

    args = parser.parse_args()

    # Set up device for inference
    device = setup_device()
    device_name = (
        "mps"
        if str(device) == "mps"
        else ("cuda" if str(device).startswith("cuda") else "cpu")
    )

    print(f"🔍 Starting enhanced YOLO model benchmark")
    print(f"📱 Using device: {device}")

    # Determine models to benchmark with their specific configurations
    models_to_benchmark = []
    if args.models:
        model_configs = args.models.split(",")
        for config_str in model_configs:
            config = parse_model_config(config_str.strip())
            models_to_benchmark.append(config)
    else:
        models_to_benchmark = [
            {
                "model_path": args.model,
                "imgsz": args.imgsz,
                "half": args.half,
                "conf": args.conf,
            }
        ]

    # Load test images
    test_images_dir = Path(args.image_dir)
    images = load_test_images(source_dir=test_images_dir, limit=args.limit)

    if not images:
        print("❌ No images to process. Please add images to the test directory.")
        return

    # Results for each model
    all_results = []

    # Benchmark each model with its specific configuration
    for model_config in models_to_benchmark:
        model_path = model_config["model_path"]
        imgsz = model_config["imgsz"]
        half = model_config["half"]
        conf = model_config["conf"]

        print(
            f"\n==== Benchmarking {model_path} (imgsz={imgsz}, half={half}, conf={conf}) ====\n"
        )

        try:
            # Load model
            model = YOLO(model_path)

            # Create results folder with uniquely identifiable name
            model_name = Path(model_path).stem
            results_folder = create_results_folder(
                model_name=model_name,
                device_name=device_name,
                imgsz=imgsz,
                half=half,
                conf=conf,
                test_run_id=args.test_run_id,
            )

            # For long run benchmarks, do multiple passes
            if args.long_run:
                print(f"Running long benchmark with {args.passes} passes...")
                all_pass_results = []

                for pass_num in range(args.passes):
                    print(f"\nPass {pass_num+1}/{args.passes}")

                    # Run model inference
                    pass_results = run_model_inference(
                        model=model,
                        images=images,
                        device=device,
                        conf=conf,
                        imgsz=imgsz,
                        half=half,
                        warmup=(pass_num == 0 and not args.no_warmup),
                        wait_time=args.wait_time,
                    )
                    all_pass_results.append(pass_results)

                    # Brief pause between passes
                    if pass_num < args.passes - 1:
                        print(f"Pausing for 5 seconds before next pass...")
                        time.sleep(5)

                # Combine results from all passes
                results = all_pass_results[0].copy()  # Start with first pass

                # Average the timing values across all passes
                all_times = [r["avg_inference_time"] for r in all_pass_results]
                results["avg_inference_time"] = np.mean(all_times)
                results["pass_times"] = all_times

                # Add thermal analysis data
                results["thermal_variation"] = (
                    np.std(all_times) / np.mean(all_times) if all_times else 0
                )
                results["time_degradation"] = (
                    (all_times[-1] - all_times[0]) / all_times[0] * 100
                    if all_times and all_times[0] > 0
                    else 0
                )

                # Add pass data to the detailed results
                results["pass_data"] = all_pass_results
            else:
                # Standard single-pass benchmark
                results = run_model_inference(
                    model=model,
                    images=images,
                    device=device,
                    conf=conf,
                    imgsz=imgsz,
                    half=half,
                    warmup=not args.no_warmup,
                    wait_time=args.wait_time,
                )

            # Save standard results
            save_results(results, results_folder)

            # Generate visualizations
            generate_visualizations(results, results_folder)

            # Create report
            create_report(results, results_folder, not args.no_html)

            # Add to results list for comparison
            all_results.append(results)

            print(f"✅ Benchmark complete for {model_path}")
            print(f"Results saved to {results_folder}")

        except Exception as e:
            print(f"❌ Error benchmarking {model_path}: {e}")
            import traceback

            traceback.print_exc()

    # Create comparison report if multiple models were benchmarked
    if len(all_results) > 1 and not args.no_compare:
        print("\n==== Creating model comparison report ====\n")
        compare_models_report(all_results, Path(args.output))

    print("\n✅ All benchmarks complete!")


if __name__ == "__main__":
    main()
