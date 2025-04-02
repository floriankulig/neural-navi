#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
YOLO Model Benchmark
This script compares the inference performance of different YOLO models.
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
import yaml
import json
import psutil
import threading
from datetime import datetime
import subprocess

# Add parent directory to path for imports from main project
script_dir = Path(__file__).parent
project_root = script_dir.parent
sys.path.insert(0, str(project_root))

# Import project modules
from device import setup_device
from ultralytics import YOLO
from config import DEFAULT_VISION_MODEL, DEFAULT_IMAGE_ROI
from imageprocessor import ImageProcessor

# Constants
METRICS_DIR = Path("metrics")
TEST_IMAGES_DIR = METRICS_DIR / "test_images"
RESULTS_DIR = METRICS_DIR / "model_benchmark_results"

# Create results directory if it doesn't exist
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


# Vehicle classes in YOLOv11: car(2), motorcycle(3), bus(5), truck(7)
VEHICLE_CLASSES = [2, 3, 5, 7]


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
            # Apply ROI crop
            img_cropped = ImageProcessor.crop_to_roi(img_rgb)
            images.append((img_path, img_cropped))

    print(f"Loaded {len(images)} test images")
    return images


def run_model_inference(
    model, images, device, conf=0.25, imgsz=640, half=False, warmup=True, int8=False
):
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

    Returns:
        Dictionary with inference results and performance metrics
    """
    model_name = Path(model.ckpt_path).stem

    # Warm-up the model
    if warmup and len(images) > 0:
        print("Warming up model...")
        warmup_image = images[0][1]
        for _ in range(3):  # 3 warm-up runs
            model(warmup_image, device=device, verbose=False, format="ncnn")

    print(f"Running inference with model: {model_name}")

    results = []
    inference_times = []
    total_objects = 0
    all_confidences = []

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
            format="ncnn",
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
                if not cls_id in VEHICLE_CLASSES:
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
                    }
                )

                all_confidences.append(conf_val)

        total_objects += len(detected_objects)

        results.append(
            {
                "filename": img_path.name,
                "inference_time": inference_time,
                "total_objects": len(detected_objects),
                "detections": detected_objects,
            }
        )

    # Calculate performance metrics
    avg_inference_time = np.mean(inference_times) if inference_times else 0
    median_inference_time = np.median(inference_times) if inference_times else 0
    avg_objects_per_image = total_objects / len(images) if images else 0
    avg_confidence = np.mean(all_confidences) if all_confidences else 0
    median_confidence = np.median(all_confidences) if all_confidences else 0

    # Process class distribution
    class_distribution = {}
    for result in results:
        for detection in result["detections"]:
            cls = detection["class"]
            class_distribution[cls] = class_distribution.get(cls, 0) + 1

    # Calculate throughput metrics
    fps = 1.0 / avg_inference_time if avg_inference_time > 0 else 0

    summary = {
        "model_path": model.ckpt_path,
        "model_name": model_name,
        "device": str(device),
        "imgsz": imgsz,
        "confidence_threshold": conf,
        "half_precision": half,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "total_images": len(images),
        "avg_inference_time": avg_inference_time,
        "median_inference_time": median_inference_time,
        "min_inference_time": min(inference_times) if inference_times else 0,
        "max_inference_time": max(inference_times) if inference_times else 0,
        "fps": fps,
        "total_objects": total_objects,
        "avg_objects_per_image": avg_objects_per_image,
        "avg_confidence": avg_confidence,
        "median_confidence": median_confidence,
        "class_distribution": class_distribution,
        "classes": VEHICLE_CLASSES,
        "detailed_results": results,
    }

    return summary


def create_results_folder(
    model_name,
    device_name,
    imgsz,
    half,
):
    """
    Create a unique folder to store the benchmark results.

    Args:
        model_name: Name of the model
        device_name: Name of the device
        imgsz: Image size used
        half: Whether half precision was used

    Returns:
        Path to the results folder
    """
    precision = "fp16" if half else "fp32"

    folder_name = f"results_{device_name}_{imgsz}_{model_name}_{precision}"
    result_folder = RESULTS_DIR / folder_name
    result_folder.mkdir(parents=True, exist_ok=True)

    return result_folder


def generate_visualizations(summary, save_path):
    """
    Generate visualizations of the benchmark results.

    Args:
        summary: Dictionary with benchmark results
        save_path: Path to save visualizations
    """
    # Set style
    sns.set(style="whitegrid")
    plt.rcParams.update({"font.size": 12})

    # Extract data
    model_name = summary["model_name"]
    device_name = str(summary["device"])
    imgsz = summary["imgsz"]
    inference_times = [
        r["inference_time"] * 1000 for r in summary["detailed_results"]
    ]  # Convert to ms
    objects_detected = [r["total_objects"] for r in summary["detailed_results"]]

    # 1. Inference Time Distribution
    plt.figure(figsize=(12, 6))
    sns.histplot(inference_times, kde=True, bins=20)
    plt.axvline(
        x=np.mean(inference_times),
        color="r",
        linestyle="--",
        label=f"Mean: {np.mean(inference_times):.2f} ms",
    )
    plt.axvline(
        x=np.median(inference_times),
        color="g",
        linestyle="--",
        label=f"Median: {np.median(inference_times):.2f} ms",
    )

    plt.title(f"Inference Time Distribution - {model_name} (imgsz={imgsz})")
    plt.xlabel("Inference Time (ms)")
    plt.ylabel("Frequency")
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path / "inference_time_distribution.png", dpi=300)
    plt.close()

    # 2. Inference Time per Image
    plt.figure(figsize=(14, 6))
    plt.plot(
        range(len(inference_times)),
        inference_times,
        marker="o",
        linestyle="-",
        alpha=0.7,
    )
    plt.axhline(
        y=np.mean(inference_times),
        color="r",
        linestyle="--",
        label=f"Mean: {np.mean(inference_times):.2f} ms",
    )

    plt.title(f"Inference Time per Image - {model_name} (imgsz={imgsz})")
    plt.xlabel("Image Index")
    plt.ylabel("Inference Time (ms)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(save_path / "inference_time_per_image.png", dpi=300)
    plt.close()

    # 3. Objects Detected per Image
    plt.figure(figsize=(14, 6))
    plt.plot(
        range(len(objects_detected)),
        objects_detected,
        marker="o",
        label="Objects",
        alpha=0.7,
    )

    plt.title(f"Objects Detected per Image - {model_name} (imgsz={imgsz})")
    plt.xlabel("Image Index")
    plt.ylabel("Number of Objects")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(save_path / "objects_detected_per_image.png", dpi=300)
    plt.close()

    # 4. Class Distribution
    if summary["class_distribution"]:
        plt.figure(figsize=(12, 8))
        classes = list(summary["class_distribution"].keys())
        counts = list(summary["class_distribution"].values())

        # Sort by count
        sorted_data = sorted(zip(classes, counts), key=lambda x: x[1], reverse=True)
        classes, counts = zip(*sorted_data)

        sns.barplot(x=list(classes), y=list(counts))
        plt.title(f"Class Distribution - {model_name} (imgsz={imgsz})")
        plt.xlabel("Class")
        plt.ylabel("Count")
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()
        plt.savefig(save_path / "class_distribution.png", dpi=300)
        plt.close()

    # 5. Summary Table
    plt.figure(figsize=(12, 8))
    plt.axis("off")

    # Select key metrics
    metrics = [
        ("Model", model_name),
        ("Device", device_name),
        ("Image Size", imgsz),
        ("Precision", "FP16" if summary["half_precision"] else "FP32"),
        ("Total Images", summary["total_images"]),
        ("Total Objects", summary["total_objects"]),
        ("Avg. Objects/Image", f"{summary['avg_objects_per_image']:.2f}"),
        ("Avg. Inference Time", f"{summary['avg_inference_time'] * 1000:.2f} ms"),
        ("Median Inference Time", f"{summary['median_inference_time'] * 1000:.2f} ms"),
        ("Min Inference Time", f"{summary['min_inference_time'] * 1000:.2f} ms"),
        ("Max Inference Time", f"{summary['max_inference_time'] * 1000:.2f} ms"),
        ("Throughput", f"{summary['fps']:.2f} FPS"),
        ("Avg. Confidence", f"{summary['avg_confidence']:.3f}"),
        ("Median Confidence", f"{summary['median_confidence']:.3f}"),
    ]

    # Create table
    table = plt.table(
        cellText=[[str(v)] for _, v in metrics],
        rowLabels=[m for m, _ in metrics],
        loc="center",
        cellLoc="center",
        colWidths=[0.5],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.5)

    plt.title(f"Performance Summary - {model_name}", pad=20)
    plt.tight_layout()
    plt.savefig(save_path / "performance_summary_table.png", dpi=300)
    plt.close()

    # 6. Throughput Analysis (FPS)
    plt.figure(figsize=(10, 6))
    fps = summary["fps"]
    metrics = ["FPS"]
    values = [fps]

    sns.barplot(x=metrics, y=values)
    plt.title(f"Throughput - {model_name} (imgsz={imgsz})")
    plt.ylabel("Frames Per Second (FPS)")

    # Add value annotation
    plt.text(0, fps, f"{fps:.2f}", ha="center", va="bottom", fontsize=12)

    plt.tight_layout()
    plt.savefig(save_path / "throughput.png", dpi=300)
    plt.close()

    print(f"✅ Visualizations saved to {save_path}")


def save_results(summary, save_path):
    """
    Save the benchmark results to files.

    Args:
        summary: Dictionary with benchmark results
        save_path: Path to save results
    """
    # Save detailed results as JSON
    with open(save_path / "detailed_results.json", "w") as f:
        # Create a copy with serializable detailed_results
        serializable_summary = summary.copy()
        serializable_summary["detailed_results"] = [
            {k: v for k, v in r.items() if k != "detections"}
            for r in summary["detailed_results"]
        ]
        serializable_summary["class_distribution"] = dict(
            serializable_summary["class_distribution"]
        )
        json.dump(serializable_summary, f, indent=2)

    # Save summary as YAML
    with open(save_path / "summary.yaml", "w") as f:
        # Create a copy without detailed_results
        yaml_summary = {k: v for k, v in summary.items() if k != "detailed_results"}
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
    inference_df.to_csv(save_path / "inference_times.csv", index=False)

    print(f"✅ Results saved to {save_path}")


def create_pipeline_report(
    pipeline_summary, system_resources, save_path, add_html=True
):
    """
    Create a comprehensive report of the pipeline simulation results.

    Args:
        pipeline_summary: Dictionary with pipeline simulation results
        system_resources: Dictionary with system resource measurements
        save_path: Path to save the report
        add_html: Whether to also create an HTML report
    """
    report_path = save_path / "pipeline_report.md"

    with open(report_path, "w") as f:
        # Title and metadata
        f.write(f"# YOLO Pipeline Simulation Report\n\n")
        f.write(f"**Date:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

        # Pipeline configuration
        f.write(f"## Pipeline Configuration\n\n")
        f.write(
            f"- **Time Budget:** {pipeline_summary['total_budget_ms']} ms (2Hz operation)\n"
        )
        f.write(f"- **Total Images Processed:** {pipeline_summary['total_images']}\n\n")

        # Performance metrics
        f.write(f"## Performance Metrics\n\n")
        f.write(
            f"- **Average YOLO Inference Time:** {pipeline_summary['avg_yolo_inference_ms']:.2f} ms\n"
        )
        f.write(
            f"- **Average Neural Model Time:** {pipeline_summary['avg_neural_model_ms']:.2f} ms\n"
        )
        f.write(
            f"- **Average Total Pipeline Time:** {pipeline_summary['avg_total_pipeline_ms']:.2f} ms\n"
        )
        f.write(
            f"- **Average Remaining Budget:** {pipeline_summary['avg_remaining_budget_ms']:.2f} ms\n\n"
        )

        # Budget analysis
        f.write(f"## Budget Analysis\n\n")
        f.write(
            f"- **Budget Overruns:** {pipeline_summary['budget_overruns']} / {pipeline_summary['total_images']} ({pipeline_summary['overrun_percentage']:.2f}%)\n\n"
        )

        # System resource usage
        if system_resources:
            f.write(f"## System Resource Usage\n\n")
            f.write(
                f"- **Average CPU Usage:** {system_resources['avg_cpu_percent']:.2f}%\n"
            )
            f.write(
                f"- **Maximum CPU Usage:** {system_resources['max_cpu_percent']:.2f}%\n"
            )
            f.write(
                f"- **Average Memory Usage:** {system_resources['avg_memory_percent']:.2f}%\n"
            )
            f.write(
                f"- **Maximum Memory Usage:** {system_resources['max_memory_percent']:.2f}%\n"
            )

            if (
                "avg_temperature" in system_resources
                and system_resources["avg_temperature"] > 0
            ):
                f.write(
                    f"- **Average CPU Temperature:** {system_resources['avg_temperature']:.2f}°C\n"
                )
                f.write(
                    f"- **Maximum CPU Temperature:** {system_resources['max_temperature']:.2f}°C\n"
                )
            f.write("\n")

        # Visualizations
        f.write(f"## Visualizations\n\n")
        f.write(f"### Pipeline Time Breakdown\n\n")
        f.write(f"![Pipeline Time Breakdown](pipeline_time_breakdown.png)\n\n")

        f.write(f"### Budget Utilization\n\n")
        f.write(f"![Budget Utilization](budget_utilization.png)\n\n")

        f.write(f"### Pipeline Time Distribution\n\n")
        f.write(f"![Pipeline Time Distribution](pipeline_time_distribution.png)\n\n")

        f.write(f"### Budget Success vs. Overruns\n\n")
        f.write(f"![Budget Overrun Analysis](budget_overrun_analysis.png)\n\n")

        if system_resources:
            f.write(f"### System Resources\n\n")
            f.write(f"![System Resources](system_resources.png)\n\n")
            f.write(f"![CPU Usage](cpu_usage.png)\n\n")
            f.write(f"![Memory Usage](memory_usage.png)\n\n")

            if (
                "avg_temperature" in system_resources
                and system_resources["avg_temperature"] > 0
            ):
                f.write(f"![CPU Temperature](temperature.png)\n\n")

        # Conclusion
        f.write(f"## Conclusion\n\n")

        if pipeline_summary["overrun_percentage"] < 5:
            f.write(
                f"The pipeline performs well within the 2Hz time budget, with only {pipeline_summary['overrun_percentage']:.2f}% of frames exceeding the budget.\n\n"
            )
            f.write(
                f"There is an average of {pipeline_summary['avg_remaining_budget_ms']:.2f} ms remaining budget per frame, which could be used for additional processing if needed.\n"
            )
        elif pipeline_summary["overrun_percentage"] < 20:
            f.write(
                f"The pipeline generally meets the 2Hz time budget, but has {pipeline_summary['overrun_percentage']:.2f}% of frames exceeding the budget.\n\n"
            )
            f.write(
                f"Consider further optimizing the models or reducing processing requirements to ensure more consistent performance.\n"
            )
        else:
            f.write(
                f"The pipeline frequently exceeds the 2Hz time budget, with {pipeline_summary['overrun_percentage']:.2f}% of frames over budget.\n\n"
            )
            f.write(
                f"Consider using a smaller/faster YOLO model, reducing image size, or simplifying the neural model to meet timing requirements.\n"
            )

    print(f"✅ Pipeline report saved to {report_path}")

    # Create HTML report if requested
    if add_html:
        try:
            import markdown

            with open(report_path, "r") as f:
                md_content = f.read()

            html_content = markdown.markdown(md_content, extensions=["tables"])

            # Add header and styling
            html_report = f"""
            <!DOCTYPE html>
            <html>
            <head>
                <title>YOLO Pipeline Simulation Report</title>
                <style>
                    body {{ font-family: Arial, sans-serif; line-height: 1.6; max-width: 1200px; margin: 0 auto; padding: 20px; }}
                    h1, h2, h3 {{ color: #333; }}
                    table {{ border-collapse: collapse; width: 100%; margin: 20px 0; }}
                    th, td {{ text-align: left; padding: 12px; border-bottom: 1px solid #ddd; }}
                    th {{ background-color: #f2f2f2; }}
                    img {{ max-width: 100%; height: auto; margin: 20px 0; border: 1px solid #ddd; }}
                </style>
            </head>
            <body>
                {html_content}
            </body>
            </html>
            """

            with open(save_path / "pipeline_report.html", "w") as f:
                f.write(html_report)

            print(
                f"✅ HTML pipeline report saved to {save_path / 'pipeline_report.html'}"
            )
        except ImportError:
            print(
                "⚠️ Python markdown package not found, skipping HTML report generation"
            )
            print("   Install with: pip install markdown")


def create_report(summary, save_path, add_html=True):
    """
    Create a comprehensive report of the benchmark results.

    Args:
        summary: Dictionary with benchmark results
        save_path: Path to save the report
        add_html: Whether to also create an HTML report
    """
    report_path = save_path / "benchmark_report.md"

    with open(report_path, "w") as f:
        model_name = summary["model_name"]
        device_name = str(summary["device"])
        imgsz = summary["imgsz"]
        precision = "FP16" if summary["half_precision"] else "FP32"
        conf = summary["confidence_threshold"]

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
        # if summary["classes"] is not None:
        #   f.write(f"- **Classes:** {summary['classes']}\n")
        # else:
        #   f.write(f"- **Classes:** All\n")
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
            f"- **Min Inference Time:** {summary['min_inference_time'] * 1000:.2f} ms\n"
        )
        f.write(
            f"- **Max Inference Time:** {summary['max_inference_time'] * 1000:.2f} ms\n"
        )
        f.write(f"- **Throughput:** {summary['fps']:.2f} FPS\n\n")

        # Detection statistics
        f.write(f"## Detection Statistics\n\n")
        f.write(f"- **Total Objects Detected:** {summary['total_objects']}\n")
        f.write(
            f"- **Average Objects per Image:** {summary['avg_objects_per_image']:.2f}\n"
        )
        f.write(f"- **Average Confidence Score:** {summary['avg_confidence']:.3f}\n")
        f.write(
            f"- **Median Confidence Score:** {summary['median_confidence']:.3f}\n\n"
        )

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

        # Visualizations
        f.write(f"## Visualizations\n\n")
        f.write(f"### Inference Time Distribution\n\n")
        f.write(f"![Inference Time Distribution](inference_time_distribution.png)\n\n")

        f.write(f"### Inference Time per Image\n\n")
        f.write(f"![Inference Time per Image](inference_time_per_image.png)\n\n")

        f.write(f"### Detection Results\n\n")
        f.write(f"![Objects Detected per Image](objects_detected_per_image.png)\n\n")

        if summary["class_distribution"]:
            f.write(f"### Class Distribution\n\n")
            f.write(f"![Class Distribution](class_distribution.png)\n\n")

        f.write(f"### Throughput Analysis\n\n")
        f.write(f"![Throughput](throughput.png)\n\n")

        f.write(f"### Performance Summary\n\n")
        f.write(f"![Performance Summary](performance_summary_table.png)\n\n")

    print(f"✅ Report saved to {report_path}")

    # Create HTML report if requested
    if add_html:
        try:
            import markdown

            with open(report_path, "r") as f:
                md_content = f.read()

            html_content = markdown.markdown(md_content, extensions=["tables"])

            # Add header and styling
            html_report = f"""
            <!DOCTYPE html>
            <html>
            <head>
                <title>YOLO Model Benchmark Report: {model_name}</title>
                <style>
                    body {{ font-family: Arial, sans-serif; line-height: 1.6; max-width: 1200px; margin: 0 auto; padding: 20px; }}
                    h1, h2, h3 {{ color: #333; }}
                    table {{ border-collapse: collapse; width: 100%; margin: 20px 0; }}
                    th, td {{ text-align: left; padding: 12px; border-bottom: 1px solid #ddd; }}
                    th {{ background-color: #f2f2f2; }}
                    img {{ max-width: 100%; height: auto; margin: 20px 0; border: 1px solid #ddd; }}
                </style>
            </head>
            <body>
                {html_content}
            </body>
            </html>
            """

            with open(save_path / "benchmark_report.html", "w") as f:
                f.write(html_report)

            print(f"✅ HTML report saved to {save_path / 'benchmark_report.html'}")
        except ImportError:
            print(
                "⚠️ Python markdown package not found, skipping HTML report generation"
            )
            print("   Install with: pip install markdown")


def compare_models_report(summaries, save_path):
    """
    Create a comparison report for multiple models.

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
    avg_times = [s["avg_inference_time"] * 1000 for s in summaries]  # Convert to ms
    median_times = [s["median_inference_time"] * 1000 for s in summaries]
    fps_values = [s["fps"] for s in summaries]
    avg_objects = [s["avg_objects_per_image"] for s in summaries]
    avg_confidence = [s["avg_confidence"] for s in summaries]

    # 1. Inference Time Comparison
    plt.figure(figsize=(12, 8))
    x = np.arange(len(models))
    width = 0.35

    plt.bar(x - width / 2, avg_times, width, label="Average Time (ms)")
    plt.bar(x + width / 2, median_times, width, label="Median Time (ms)")

    plt.xlabel("Model")
    plt.ylabel("Inference Time (ms)")
    plt.title("Inference Time Comparison")
    plt.xticks(x, models, rotation=45, ha="right")

    # Add value labels
    for i, v in enumerate(avg_times):
        plt.text(i - width / 2, v + 0.5, f"{v:.2f}", ha="center")
    for i, v in enumerate(median_times):
        plt.text(i + width / 2, v + 0.5, f"{v:.2f}", ha="center")

    plt.legend()
    plt.tight_layout()
    plt.savefig(compare_path / "inference_time_comparison.png", dpi=300)
    plt.close()

    # 2. Throughput Comparison (FPS)
    plt.figure(figsize=(12, 8))
    plt.bar(models, fps_values, color="green")
    plt.xlabel("Model")
    plt.ylabel("Frames Per Second (FPS)")
    plt.title("Throughput Comparison")
    plt.xticks(rotation=45, ha="right")

    # Add value labels
    for i, v in enumerate(fps_values):
        plt.text(i, v + 0.5, f"{v:.2f}", ha="center")

    plt.tight_layout()
    plt.savefig(compare_path / "throughput_comparison.png", dpi=300)
    plt.close()

    # 3. Detection Results Comparison
    plt.figure(figsize=(12, 8))
    plt.bar(models, avg_objects, color="blue")

    plt.xlabel("Model")
    plt.ylabel("Average Objects per Image")
    plt.title("Detection Results Comparison")
    plt.xticks(rotation=45, ha="right")

    # Add value labels
    for i, v in enumerate(avg_objects):
        plt.text(i, v + 0.1, f"{v:.2f}", ha="center")

    plt.tight_layout()
    plt.savefig(compare_path / "detection_results_comparison.png", dpi=300)
    plt.close()

    # 4. Confidence Score Comparison
    plt.figure(figsize=(12, 8))
    plt.bar(models, avg_confidence, color="purple")
    plt.xlabel("Model")
    plt.ylabel("Average Confidence Score")
    plt.title("Confidence Score Comparison")
    plt.xticks(rotation=45, ha="right")

    # Add value labels
    for i, v in enumerate(avg_confidence):
        plt.text(i, v + 0.01, f"{v:.3f}", ha="center")

    plt.tight_layout()
    plt.savefig(compare_path / "confidence_comparison.png", dpi=300)
    plt.close()

    # 5. Comparison table as image
    plt.figure(figsize=(14, 10))
    plt.axis("off")

    # Create comparison table data
    data = []
    for s in summaries:
        data.append(
            [
                s["model_name"],
                f"{s['imgsz']}",
                f"{'FP16' if s['half_precision'] else 'FP32'}",
                f"{s['avg_inference_time'] * 1000:.2f}",
                f"{s['median_inference_time'] * 1000:.2f}",
                f"{s['fps']:.2f}",
                f"{s['avg_objects_per_image']:.2f}",
                f"{s['avg_confidence']:.3f}",
            ]
        )

    # Column headers
    columns = [
        "Model",
        "Size",
        "Precision",
        "Avg Time (ms)",
        "Median Time (ms)",
        "FPS",
        "Avg Objects",
        "Avg Confidence",
    ]

    # Create table
    table = plt.table(
        cellText=data,
        colLabels=columns,
        loc="center",
        cellLoc="center",
        colWidths=[0.12] * len(columns),
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.5)

    plt.title("Model Comparison Summary", pad=20, fontsize=16)
    plt.tight_layout()
    plt.savefig(compare_path / "comparison_table.png", dpi=300)
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
            f.write(
                f"{i+1}. **{s['model_name']}** ({s['imgsz']}x{s['imgsz']}, {'FP16' if s['half_precision'] else 'FP32'})\n"
            )
        f.write("\n")

        # Comparison summary
        f.write(f"## Comparison Summary\n\n")
        f.write("![Comparison Table](comparison_table.png)\n\n")

        # Performance comparison
        f.write(f"## Performance Comparison\n\n")
        f.write("### Inference Time\n\n")
        f.write("![Inference Time Comparison](inference_time_comparison.png)\n\n")

        f.write("### Throughput (FPS)\n\n")
        f.write("![Throughput Comparison](throughput_comparison.png)\n\n")

        # Detection results comparison
        f.write(f"## Detection Results Comparison\n\n")
        f.write("### Objects Detected\n\n")
        f.write("![Detection Results Comparison](detection_results_comparison.png)\n\n")

        f.write("### Confidence Scores\n\n")
        f.write("![Confidence Comparison](confidence_comparison.png)\n\n")

        # Conclusion
        f.write(f"## Conclusion\n\n")

        # Find the fastest model
        fastest_idx = np.argmin([s["avg_inference_time"] for s in summaries])
        fastest_model = summaries[fastest_idx]["model_name"]
        fastest_fps = summaries[fastest_idx]["fps"]

        # Find the model with most detections
        most_detections_idx = np.argmax([s["avg_objects_per_image"] for s in summaries])
        most_detections_model = summaries[most_detections_idx]["model_name"]
        most_detections_avg = summaries[most_detections_idx]["avg_objects_per_image"]

        # Find the model with highest confidence
        highest_confidence_idx = np.argmax([s["avg_confidence"] for s in summaries])
        highest_confidence_model = summaries[highest_confidence_idx]["model_name"]
        highest_confidence_avg = summaries[highest_confidence_idx]["avg_confidence"]

        f.write(f"- **Fastest Model:** {fastest_model} ({fastest_fps:.2f} FPS)\n")
        f.write(
            f"- **Model with Most Detections:** {most_detections_model} ({most_detections_avg:.2f} objects/image)\n"
        )
        f.write(
            f"- **Model with Highest Confidence:** {highest_confidence_model} ({highest_confidence_avg:.3f})\n\n"
        )

    print(f"✅ Comparison report saved to {report_path}")

    # Create HTML version of comparison report
    try:
        import markdown

        with open(report_path, "r") as f:
            md_content = f.read()

        html_content = markdown.markdown(md_content)

        # Add header and styling
        html_report = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>YOLO Model Comparison Report</title>
            <style>
                body {{ font-family: Arial, sans-serif; line-height: 1.6; max-width: 1200px; margin: 0 auto; padding: 20px; }}
                h1, h2, h3 {{ color: #333; }}
                table {{ border-collapse: collapse; width: 100%; margin: 20px 0; }}
                th, td {{ text-align: left; padding: 12px; border-bottom: 1px solid #ddd; }}
                th {{ background-color: #f2f2f2; }}
                img {{ max-width: 100%; height: auto; margin: 20px 0; border: 1px solid #ddd; }}
            </style>
        </head>
        <body>
            {html_content}
        </body>
        </html>
        """

        with open(compare_path / "model_comparison_report.html", "w") as f:
            f.write(html_report)

        print(
            f"✅ HTML comparison report saved to {compare_path / 'model_comparison_report.html'}"
        )
    except ImportError:
        print("⚠️ Python markdown package not found, skipping HTML comparison report")
        print("   Install with: pip install markdown")


def monitor_system_resources(stop_event, interval=1.0, results=None):
    """
    Monitor system resources in a separate thread.

    Args:
        stop_event: Event to signal thread to stop
        interval: Sampling interval in seconds
        results: Dictionary to store results
    """
    if results is None:
        results = {"cpu_percent": [], "memory_percent": [], "temperature": []}

    def get_cpu_temperature():
        """Get CPU temperature on Raspberry Pi"""
        try:
            # This works on Raspberry Pi
            temp = subprocess.check_output(
                ["vcgencmd", "measure_temp"], universal_newlines=True
            )
            temp = float(temp.replace("temp=", "").replace("'C", ""))
            return temp
        except (subprocess.SubprocessError, FileNotFoundError):
            try:
                # Alternative method
                with open("/sys/class/thermal/thermal_zone0/temp", "r") as f:
                    temp = float(f.read().strip()) / 1000.0
                return temp
            except:
                return 0.0

    while not stop_event.is_set():
        # CPU usage (percent)
        results["cpu_percent"].append(psutil.cpu_percent(interval=0.1))

        # Memory usage (percent)
        memory = psutil.virtual_memory()
        results["memory_percent"].append(memory.percent)

        # Temperature (on Raspberry Pi)
        results["temperature"].append(get_cpu_temperature())

        # Sleep for interval
        time.sleep(interval)

    # Calculate averages and max values
    results["avg_cpu_percent"] = np.mean(results["cpu_percent"])
    results["max_cpu_percent"] = np.max(results["cpu_percent"])
    results["avg_memory_percent"] = np.mean(results["memory_percent"])
    results["max_memory_percent"] = np.max(results["memory_percent"])
    results["avg_temperature"] = np.mean(results["temperature"])
    results["max_temperature"] = np.max(results["temperature"])

    return results


def simulate_pipeline(
    model,
    images,
    device,
    conf=0.25,
    imgsz=640,
    half=False,
    total_budget_ms=500,
    neural_model_sim_time=None,
):
    """
    Simulate a complete pipeline with YOLO and a subsequent neural model.

    Args:
        model: Loaded YOLO model
        images: List of (path, image) tuples
        device: Device to run inference on
        conf: Confidence threshold
        imgsz: Input image size
        half: Whether to use half precision
        total_budget_ms: Total time budget in milliseconds (e.g., 500ms for 2Hz)
        neural_model_sim_time: Time to simulate for neural model processing
                              (None = auto, based on remaining budget)

    Returns:
        Dictionary with pipeline simulation results
    """
    print(f"Simulating complete pipeline with {total_budget_ms}ms budget (2Hz)")

    results = []
    overruns = 0
    neural_model_times = []
    total_times = []
    remaining_budget_times = []

    for img_path, img in tqdm(images, desc="Simulating pipeline"):
        # Run YOLO inference
        start_time = time.time()
        predictions = model(
            img, conf=conf, device=device, imgsz=imgsz, half=half, verbose=False
        )
        yolo_time = (time.time() - start_time) * 1000  # ms

        # Calculate remaining budget
        remaining_budget = total_budget_ms - yolo_time

        # Determine neural model processing time
        if neural_model_sim_time is None:
            # Auto mode: use 80% of remaining budget
            neural_time = max(0, remaining_budget * 0.8)
        else:
            neural_time = neural_model_sim_time

        # Simulate neural model processing
        if neural_time > 0:
            time.sleep(neural_time / 1000.0)  # Convert ms to seconds

        # Track total time
        total_time = yolo_time + neural_time

        # Check for budget overruns
        if total_time > total_budget_ms:
            overruns += 1

        # Store results
        neural_model_times.append(neural_time)
        total_times.append(total_time)
        remaining_budget_times.append(total_budget_ms - total_time)

        # Count detections
        detection_count = 0
        for pred in predictions:
            boxes = pred.boxes
            detection_count += len(boxes)

        results.append(
            {
                "filename": img_path.name,
                "yolo_time_ms": yolo_time,
                "neural_time_ms": neural_time,
                "total_time_ms": total_time,
                "remaining_budget_ms": total_budget_ms - total_time,
                "budget_overrun": total_time > total_budget_ms,
                "objects_detected": detection_count,
            }
        )

    # Calculate overall statistics
    avg_yolo_time = np.mean([r["yolo_time_ms"] for r in results])
    avg_neural_time = np.mean(neural_model_times)
    avg_total_time = np.mean(total_times)
    avg_remaining = np.mean(remaining_budget_times)
    overrun_percentage = (overruns / len(results)) * 100 if results else 0

    summary = {
        "total_budget_ms": total_budget_ms,
        "avg_yolo_inference_ms": avg_yolo_time,
        "avg_neural_model_ms": avg_neural_time,
        "avg_total_pipeline_ms": avg_total_time,
        "avg_remaining_budget_ms": avg_remaining,
        "budget_overruns": overruns,
        "overrun_percentage": overrun_percentage,
        "total_images": len(results),
        "detailed_results": results,
    }

    return summary


def visualize_pipeline_results(summary, save_path):
    """
    Generate visualizations for pipeline simulation results.

    Args:
        summary: Dictionary with pipeline simulation results
        save_path: Path to save visualizations
    """
    # Extract data
    results = summary["detailed_results"]
    total_budget = summary["total_budget_ms"]

    yolo_times = [r["yolo_time_ms"] for r in results]
    neural_times = [r["neural_time_ms"] for r in results]
    total_times = [r["total_time_ms"] for r in results]
    remaining_times = [r["remaining_budget_ms"] for r in results]
    overruns = [r["budget_overrun"] for r in results]

    # 1. Pipeline time breakdown
    plt.figure(figsize=(14, 8))
    indices = range(len(results))

    # Plot stacked bars
    plt.bar(indices, yolo_times, label="YOLO Inference", color="#5DA5DA")
    plt.bar(
        indices, neural_times, bottom=yolo_times, label="Neural Model", color="#F15854"
    )

    # Plot budget line
    plt.axhline(
        y=total_budget,
        color="r",
        linestyle="--",
        label=f"Budget Limit ({total_budget}ms)",
    )

    # Highlight overruns
    for i, overrun in enumerate(overruns):
        if overrun:
            plt.axvspan(i - 0.4, i + 0.4, alpha=0.2, color="red")

    plt.title("Pipeline Execution Time Breakdown")
    plt.xlabel("Image Index")
    plt.ylabel("Time (ms)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path / "pipeline_time_breakdown.png", dpi=300)
    plt.close()

    # 2. Budget utilization pie chart
    plt.figure(figsize=(10, 8))
    avg_yolo = summary["avg_yolo_inference_ms"]
    avg_neural = summary["avg_neural_model_ms"]
    avg_remaining = summary["avg_remaining_budget_ms"]

    # Only include remaining budget if positive
    if avg_remaining > 0:
        labels = ["YOLO", "Neural Model", "Remaining"]
        sizes = [avg_yolo, avg_neural, avg_remaining]
        colors = ["#5DA5DA", "#F15854", "#EEEEEE"]
    else:
        labels = ["YOLO", "Neural Model", "Overrun"]
        sizes = [avg_yolo, avg_neural, -avg_remaining]
        colors = ["#5DA5DA", "#F15854", "#FF9E4A"]

    plt.pie(sizes, labels=labels, autopct="%1.1f%%", colors=colors, startangle=90)
    plt.axis("equal")
    plt.title(f"Average Budget Utilization ({total_budget}ms Total)")
    plt.tight_layout()
    plt.savefig(save_path / "budget_utilization.png", dpi=300)
    plt.close()

    # 3. Histogram of total pipeline times
    plt.figure(figsize=(12, 6))
    plt.hist(total_times, bins=20, alpha=0.7, color="#5DA5DA")
    plt.axvline(
        x=total_budget,
        color="r",
        linestyle="--",
        label=f"Budget Limit ({total_budget}ms)",
    )

    plt.title("Distribution of Total Pipeline Execution Times")
    plt.xlabel("Total Time (ms)")
    plt.ylabel("Frequency")
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path / "pipeline_time_distribution.png", dpi=300)
    plt.close()

    # 4. Overrun analysis
    plt.figure(figsize=(10, 6))
    overrun_count = sum(overruns)
    success_count = len(overruns) - overrun_count

    plt.bar(
        ["Success", "Overrun"],
        [success_count, overrun_count],
        color=["#60BD68", "#F15854"],
    )
    plt.title("Budget Success vs. Overruns")
    plt.ylabel("Count")

    # Add percentage labels
    for i, v in enumerate([success_count, overrun_count]):
        plt.text(i, v + 0.5, f"{(v/len(overruns))*100:.1f}%", ha="center")

    plt.tight_layout()
    plt.savefig(save_path / "budget_overrun_analysis.png", dpi=300)
    plt.close()


def visualize_system_resources(resource_data, save_path):
    """
    Generate visualizations for system resource usage.

    Args:
        resource_data: Dictionary with system resource measurements
        save_path: Path to save visualizations
    """
    # Extract data
    cpu_data = resource_data["cpu_percent"]
    memory_data = resource_data["memory_percent"]
    temperature_data = resource_data["temperature"]
    time_points = range(len(cpu_data))

    # CPU Usage
    plt.figure(figsize=(12, 6))
    plt.plot(time_points, cpu_data, marker=".", markersize=3, label="CPU Usage")
    plt.axhline(
        y=resource_data["avg_cpu_percent"],
        color="r",
        linestyle="--",
        label=f'Average: {resource_data["avg_cpu_percent"]:.1f}%',
    )

    plt.title("CPU Usage During Benchmark")
    plt.xlabel("Time (samples)")
    plt.ylabel("CPU Usage (%)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path / "cpu_usage.png", dpi=300)
    plt.close()

    # Memory Usage
    plt.figure(figsize=(12, 6))
    plt.plot(
        time_points,
        memory_data,
        marker=".",
        markersize=3,
        color="#F15854",
        label="Memory Usage",
    )
    plt.axhline(
        y=resource_data["avg_memory_percent"],
        color="r",
        linestyle="--",
        label=f'Average: {resource_data["avg_memory_percent"]:.1f}%',
    )

    plt.title("Memory Usage During Benchmark")
    plt.xlabel("Time (samples)")
    plt.ylabel("Memory Usage (%)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path / "memory_usage.png", dpi=300)
    plt.close()

    # Temperature (if available)
    if any(temperature_data):
        plt.figure(figsize=(12, 6))
        plt.plot(
            time_points,
            temperature_data,
            marker=".",
            markersize=3,
            color="#F58700",
            label="Temperature",
        )
        plt.axhline(
            y=resource_data["avg_temperature"],
            color="r",
            linestyle="--",
            label=f'Average: {resource_data["avg_temperature"]:.1f}°C',
        )

        plt.title("CPU Temperature During Benchmark")
        plt.xlabel("Time (samples)")
        plt.ylabel("Temperature (°C)")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(save_path / "temperature.png", dpi=300)
        plt.close()

    # Combined system resources
    plt.figure(figsize=(14, 8))
    plt.plot(
        time_points,
        cpu_data,
        marker=".",
        markersize=3,
        label="CPU Usage (%)",
        color="#5DA5DA",
    )
    plt.plot(
        time_points,
        memory_data,
        marker=".",
        markersize=3,
        label="Memory Usage (%)",
        color="#F15854",
    )

    if any(temperature_data):
        # Normalize temperature to same scale for visualization
        max_temp = max(temperature_data) if temperature_data else 100
        normalized_temp = [t * 100 / max_temp for t in temperature_data]
        plt.plot(
            time_points,
            normalized_temp,
            marker=".",
            markersize=3,
            label=f"Temperature (Max: {max_temp:.1f}°C)",
            color="#F58700",
        )

    plt.title("System Resources During Benchmark")
    plt.xlabel("Time (samples)")
    plt.ylabel("Percentage (%)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path / "system_resources.png", dpi=300)
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Benchmark different YOLO models")

    # Model parameters
    parser.add_argument(
        "--model",
        type=str,
        default="./yolo11ncnn",
        help=f"Path to the YOLO model (default: {DEFAULT_VISION_MODEL})",
    )
    parser.add_argument(
        "--models",
        type=str,
        default=None,
        help="Comma-separated list of models to benchmark",
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

    # Raspberry Pi specific parameters
    parser.add_argument(
        "--simulate-pipeline",
        action="store_true",
        help="Simulate complete pipeline with neural model",
    )
    parser.add_argument(
        "--time-budget",
        type=float,
        default=500.0,
        help="Time budget in ms for each frame (default: 500ms for 2Hz)",
    )
    parser.add_argument(
        "--neural-model-time",
        type=float,
        default=None,
        help="Time in ms to simulate for neural model (default: auto)",
    )
    parser.add_argument(
        "--monitor-resources",
        action="store_true",
        help="Monitor system resources during benchmark",
    )
    parser.add_argument(
        "--long-run",
        action="store_true",
        help="Run long benchmark (multiple passes) to detect thermal throttling",
    )
    parser.add_argument(
        "--passes",
        type=int,
        default=3,
        help="Number of passes for long-run benchmark (default: 3)",
    )

    args = parser.parse_args()

    # Set up device for inference
    device = setup_device()
    device_name = (
        "mps"
        if str(device) == "mps"
        else ("cuda" if str(device).startswith("cuda") else "cpu")
    )

    print(f"🔍 Starting YOLO model benchmark")
    print(f"📱 Using device: {device}")

    # Determine models to benchmark
    models_to_benchmark = []
    if args.models:
        models_to_benchmark = [m.strip() for m in args.models.split(",")]
    else:
        models_to_benchmark = [args.model]

    # Load test images
    test_images_dir = Path(args.image_dir)
    images = load_test_images(source_dir=test_images_dir, limit=args.limit)

    if not images:
        print("❌ No images to process. Please add images to the test directory.")
        return

    # Results for each model
    all_results = []

    # Benchmark each model
    for model_path in models_to_benchmark:
        print(f"\n==== Benchmarking {model_path} ====\n")

        try:
            # Load and prepare model
            model = YOLO(model_path)

            # Create results folder with uniquely identifiable name
            model_name = Path(model_path).stem
            use_int8 = "int8" in model_name.lower()
            args.half = args.half or use_int8
            results_folder = create_results_folder(
                model_name=model_name,
                device_name=device_name,
                imgsz=args.imgsz,
                half=args.half,
            )

            # Set up system resource monitoring if requested
            resource_results = None
            stop_monitoring = threading.Event()
            if args.monitor_resources:
                print("Starting system resource monitoring...")
                resource_results = {}
                monitoring_thread = threading.Thread(
                    target=monitor_system_resources,
                    args=(stop_monitoring, 0.5, resource_results),
                )
                monitoring_thread.daemon = True
                monitoring_thread.start()

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
                        conf=args.conf,
                        imgsz=args.imgsz,
                        half=args.half,
                        warmup=(pass_num == 0 and not args.no_warmup),
                        int8=use_int8,
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
                    ((all_times[-1] - all_times[0]) / all_times[0]) * 100
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
                    conf=args.conf,
                    imgsz=args.imgsz,
                    half=args.half,
                    warmup=not args.no_warmup,
                    int8=use_int8,
                )

            # Run pipeline simulation if requested
            if args.simulate_pipeline:
                print("\nSimulating complete pipeline...")
                pipeline_results = simulate_pipeline(
                    model=model,
                    images=images,
                    device=device,
                    conf=args.conf,
                    imgsz=args.imgsz,
                    half=args.half,
                    total_budget_ms=args.time_budget,
                    neural_model_sim_time=args.neural_model_time,
                )

                # Save pipeline results
                pipeline_folder = results_folder / "pipeline_simulation"
                pipeline_folder.mkdir(exist_ok=True)

                # Generate pipeline visualizations
                visualize_pipeline_results(pipeline_results, pipeline_folder)

                # Save pipeline results to files
                with open(pipeline_folder / "pipeline_results.json", "w") as f:
                    # Create serializable copy
                    serializable_pipeline = pipeline_results.copy()
                    serializable_pipeline["detailed_results"] = [
                        {k: v for k, v in r.items()}
                        for r in pipeline_results["detailed_results"]
                    ]
                    json.dump(serializable_pipeline, f, indent=2)

            # Stop system resource monitoring if it was started
            if args.monitor_resources:
                print("Stopping system resource monitoring...")
                stop_monitoring.set()
                if monitoring_thread.is_alive():
                    monitoring_thread.join(timeout=3)

                # Generate resource usage visualizations
                if resource_results:
                    resource_folder = results_folder / "system_resources"
                    resource_folder.mkdir(exist_ok=True)
                    visualize_system_resources(resource_results, resource_folder)

                    # Save resource data
                    with open(resource_folder / "resource_data.json", "w") as f:
                        json.dump(resource_results, f, indent=2)

                    # Add system resource data to main results
                    results["system_resources"] = {
                        k: v
                        for k, v in resource_results.items()
                        if k.startswith("avg_") or k.startswith("max_")
                    }

            # Save standard results
            save_results(results, results_folder)

            # Generate visualizations
            generate_visualizations(results, results_folder)

            # Create report
            create_report(results, results_folder, not args.no_html)

            # Create pipeline report if available
            if args.simulate_pipeline:
                create_pipeline_report(
                    pipeline_results,
                    resource_results if args.monitor_resources else None,
                    pipeline_folder,
                    not args.no_html,
                )

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
        compare_models_report(all_results, RESULTS_DIR)

    print("\n✅ All benchmarks complete!")


if __name__ == "__main__":
    main()
