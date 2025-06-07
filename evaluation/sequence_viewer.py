#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Multimodal Sequence Viewer for Neural-Navi Project
==================================================

Interactive visualization tool for exploring multimodal driving sequences.
Combines original images, YOLO detections, telemetry data, and future labels
for comprehensive data analysis and scientific presentation.

Usage:
    python evaluation/sequence_viewer.py --h5-file data/datasets/multimodal/train.h5
"""

import sys
import json
import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.widgets import Button, Slider
from matplotlib.gridspec import GridSpec
import cv2
import torch
import h5py
from datetime import datetime
from tqdm import tqdm

# Add project root to path for imports
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from utils.config import DEFAULT_IMAGE_ROI, TIME_FORMAT_LOG

# ====================================
# VISUALIZATION CONFIGURATION
# ====================================

# Professional color scheme for scientific presentations
COLORS = {
    "primary": "#2E86AB",  # Blue
    "secondary": "#A23B72",  # Purple
    "accent": "#F18F01",  # Orange
    "success": "#C73E1D",  # Red
    "background": "#F5F5F5",  # Light gray
    "text": "#2C3E50",  # Dark blue-gray
    "grid": "#BDC3C7",  # Light gray
    "detection": "#00FF00",  # Bright green for bounding boxes
}

# Feature display configuration
TELEMETRY_FEATURES = ["SPEED", "RPM", "ACCELERATOR_POS_D", "ENGINE_LOAD"]
TELEMETRY_UNITS = ["km/h", "RPM", "%", "%"]
TELEMETRY_RANGES = {
    "SPEED": (0, 150),
    "RPM": (650, 4500),
    "ACCELERATOR_POS_D": (0, 100),
    "ENGINE_LOAD": (0, 100),
}

GEAR_CLASSES = 7  # 0=neutral, 1-6=gears

# Figure configuration for high-quality output
FIGURE_CONFIG = {
    "dpi": 150,
    "figsize": (20, 14),
    "style": "seaborn-v0_8-whitegrid",
    "font_size": 10,
    "title_size": 14,
    "label_size": 12,
}


class OriginalDataLoader:
    """Helper class to load original telemetry and images from recordings."""

    def __init__(self, recordings_dir: str):
        self.recordings_dir = Path(recordings_dir)
        self._telemetry_cache = {}

    def load_telemetry(self, recording_name: str) -> Optional[pd.DataFrame]:
        """Load original telemetry CSV for a recording."""
        if recording_name in self._telemetry_cache:
            return self._telemetry_cache[recording_name]

        telemetry_file = self.recordings_dir / recording_name / "telemetry.csv"

        if not telemetry_file.exists():
            print(f"Warning: Telemetry file not found: {telemetry_file}")
            return None

        try:
            df = pd.read_csv(telemetry_file)
            self._telemetry_cache[recording_name] = df
            return df
        except Exception as e:
            print(f"Error loading telemetry for {recording_name}: {e}")
            return None

    def load_image(self, recording_name: str, timestamp: str) -> Optional[np.ndarray]:
        """Load original image for a specific timestamp."""
        img_path = self.recordings_dir / recording_name / f"{timestamp}.jpg"

        if not img_path.exists():
            print(f"Warning: Image not found: {img_path}")
            return None

        img = cv2.imread(str(img_path))
        if img is not None:
            # Convert BGR to RGB for matplotlib
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return img


class MultimodalSequenceViewer:
    """
    Interactive viewer for multimodal driving sequences.

    Features:
    - Original images with detection overlays
    - Telemetry timeline visualization
    - Future label display
    - Feature normalization comparison
    - Export capabilities for scientific presentations
    """

    def __init__(self, h5_file_path: str, recordings_dir: str = "data/recordings"):
        """
        Initialize the sequence viewer.

        Args:
            h5_file_path: Path to HDF5 dataset file
            recordings_dir: Path to original recordings directory
        """
        self.h5_file_path = Path(h5_file_path)
        self.recordings_dir = Path(recordings_dir)

        # Load dataset and original data
        self._load_dataset()
        self.original_loader = OriginalDataLoader(recordings_dir)

        # Current state
        self.current_sequence = 0
        self.current_frame = 0
        self.playing = False
        self.play_speed = 1.0

        # Setup figure and interface
        self._setup_figure()
        self._setup_controls()

        # Initial display
        self.update_display()

        print("🎯 Multimodal Sequence Viewer Initialized")
        print(f"   📁 Dataset: {self.h5_file_path.name}")
        print(f"   📊 Sequences: {self.num_sequences}")
        print(
            f"   🎮 Controls: Arrow keys for navigation, Space for play/pause, 'h' for help"
        )

    def _safe_decode(self, value):
        """Safely decode bytes to string."""
        if isinstance(value, bytes):
            return value.decode("utf-8")
        elif isinstance(value, np.bytes_):
            return value.decode("utf-8")
        else:
            return str(value)

    def _load_dataset(self):
        """Load dataset information and data."""
        print("📚 Loading dataset...")

        with h5py.File(self.h5_file_path, "r") as f:
            # Basic info
            self.num_sequences = f["info"].attrs["num_sequences"]
            self.sequence_length = f["info"].attrs["sequence_length"]
            self.max_detections = f["info"].attrs["max_detections"]
            self.telemetry_dim = f["info"].attrs["telemetry_dim"]
            self.detection_dim_per_box = f["info"].attrs["detection_dim_per_box"]

            # Feature names
            self.telemetry_features = [
                self._safe_decode(f) for f in f["info"].attrs["telemetry_features"]
            ]

            # Load all data into memory for smooth interaction
            print("💾 Loading data into memory...")
            self.telemetry_data = torch.from_numpy(f["telemetry"][:]).float()
            self.detection_data = torch.from_numpy(f["detections"][:]).float()
            self.detection_masks = torch.from_numpy(f["detection_masks"][:])

            # Load labels
            self.labels_data = {}
            available_labels = list(f["labels"].keys())
            for label_name in available_labels:
                self.labels_data[label_name] = torch.from_numpy(
                    f["labels"][label_name][:]
                )

            # Load metadata
            self.recording_names = [
                self._safe_decode(name) for name in f["metadata"]["recording_names"][:]
            ]
            self.start_indices = f["metadata"]["start_indices"][:]
            self.end_indices = f["metadata"]["end_indices"][:]

            # Load timestamps
            timestamp_strings = [
                self._safe_decode(ts) for ts in f["metadata"]["timestamps"][:]
            ]
            self.timestamps = [json.loads(ts) for ts in timestamp_strings]

        print(f"✅ Loaded {self.num_sequences} sequences")

    def _setup_figure(self):
        """Setup matplotlib figure with professional layout."""
        plt.style.use(FIGURE_CONFIG["style"])
        plt.rcParams.update(
            {
                "font.size": FIGURE_CONFIG["font_size"],
                "axes.titlesize": FIGURE_CONFIG["title_size"],
                "axes.labelsize": FIGURE_CONFIG["label_size"],
                "figure.dpi": FIGURE_CONFIG["dpi"],
            }
        )

        self.fig = plt.figure(figsize=FIGURE_CONFIG["figsize"])
        self.fig.suptitle(
            "Neural-Navi: Multimodal Sequence Viewer",
            fontsize=16,
            fontweight="bold",
            color=COLORS["text"],
        )

        # Create grid layout
        gs = GridSpec(
            4,
            4,
            figure=self.fig,
            height_ratios=[3, 1, 1, 0.3],
            width_ratios=[2, 1, 1, 1],
            hspace=0.3,
            wspace=0.3,
        )

        # Main image display
        self.ax_image = self.fig.add_subplot(gs[0, :2])
        self.ax_image.set_title("Original Image + Detections", fontweight="bold")
        self.ax_image.set_aspect("equal")

        # Telemetry plots
        self.ax_speed = self.fig.add_subplot(gs[0, 2])
        self.ax_rpm = self.fig.add_subplot(gs[0, 3])
        self.ax_accel = self.fig.add_subplot(gs[1, 2])
        self.ax_load = self.fig.add_subplot(gs[1, 3])

        self.telemetry_axes = [self.ax_speed, self.ax_rpm, self.ax_accel, self.ax_load]

        # Timeline and labels
        self.ax_timeline = self.fig.add_subplot(gs[1, :2])
        self.ax_labels = self.fig.add_subplot(gs[2, :2])

        # Feature comparison
        self.ax_features = self.fig.add_subplot(gs[2, 2:])

        # Control panel
        self.ax_controls = self.fig.add_subplot(gs[3, :])
        self.ax_controls.set_xlim(0, 1)
        self.ax_controls.set_ylim(0, 1)
        self.ax_controls.axis("off")

    def _setup_controls(self):
        """Setup interactive controls."""
        # Navigation text
        control_text = (
            "Controls: ← → (frames) | ↑ ↓ (sequences) | Space (play/pause) | "
            "R (reset) | S (save) | H (help) | Q (quit)"
        )
        self.ax_controls.text(
            0.5,
            0.5,
            control_text,
            ha="center",
            va="center",
            fontsize=10,
            bbox=dict(boxstyle="round,pad=0.3", facecolor=COLORS["background"]),
        )

        # Connect keyboard events
        self.fig.canvas.mpl_connect("key_press_event", self._on_key_press)

        # Setup timer for playback
        self.timer = self.fig.canvas.new_timer(interval=500)  # 0.5s = 2Hz
        self.timer.add_callback(self._play_step)

    def _on_key_press(self, event):
        """Handle keyboard navigation."""
        if event.key == "right":
            self._next_frame()
        elif event.key == "left":
            self._prev_frame()
        elif event.key == "up":
            self._next_sequence()
        elif event.key == "down":
            self._prev_sequence()
        elif event.key == " ":  # Space
            self._toggle_playback()
        elif event.key == "r":
            self._reset_view()
        elif event.key == "s":
            self._save_current_view()
        elif event.key == "h":
            self._show_help()
        elif event.key == "q":
            plt.close(self.fig)
        elif event.key.isdigit():
            # Jump to sequence
            seq_num = int(event.key)
            if seq_num < self.num_sequences:
                self.current_sequence = seq_num
                self.current_frame = 0
                self.update_display()

    def _next_frame(self):
        """Navigate to next frame."""
        if self.current_frame < self.sequence_length - 1:
            self.current_frame += 1
            self.update_display()

    def _prev_frame(self):
        """Navigate to previous frame."""
        if self.current_frame > 0:
            self.current_frame -= 1
            self.update_display()

    def _next_sequence(self):
        """Navigate to next sequence."""
        if self.current_sequence < self.num_sequences - 1:
            self.current_sequence += 1
            self.current_frame = 0
            self.update_display()

    def _prev_sequence(self):
        """Navigate to previous sequence."""
        if self.current_sequence > 0:
            self.current_sequence -= 1
            self.current_frame = 0
            self.update_display()

    def _toggle_playback(self):
        """Toggle sequence playback."""
        if self.playing:
            self.timer.stop()
            self.playing = False
            print("⏸️ Playback paused")
        else:
            self.timer.start()
            self.playing = True
            print("▶️ Playback started")

    def _play_step(self):
        """Advance one frame during playback."""
        if self.current_frame < self.sequence_length - 1:
            self._next_frame()
        else:
            # End of sequence, stop or loop
            self.timer.stop()
            self.playing = False
            print("⏹️ End of sequence reached")

    def _reset_view(self):
        """Reset to first sequence and frame."""
        self.current_sequence = 0
        self.current_frame = 0
        if self.playing:
            self.timer.stop()
            self.playing = False
        self.update_display()
        print("🔄 View reset")

    def _save_current_view(self):
        """Save current view as high-quality image."""
        output_dir = Path("evaluation/sequence_exports")
        output_dir.mkdir(exist_ok=True)

        recording_name = self.recording_names[self.current_sequence]
        timestamp = self.timestamps[self.current_sequence][self.current_frame]

        filename = f"{recording_name}_frame_{self.current_frame:02d}_{timestamp}.png"
        filepath = output_dir / filename

        self.fig.savefig(
            filepath, dpi=300, bbox_inches="tight", facecolor="white", edgecolor="none"
        )
        print(f"💾 Saved: {filepath}")

    def _show_help(self):
        """Display help information."""
        help_text = """
        🎮 MULTIMODAL SEQUENCE VIEWER CONTROLS
        
        Navigation:
        ← → : Previous/Next frame
        ↑ ↓ : Previous/Next sequence
        0-9 : Jump to sequence number
        
        Playback:
        Space : Play/Pause
        
        Other:
        R : Reset to start
        S : Save current view (300 DPI)
        H : Show this help
        Q : Quit
        
        Current Status:
        Sequence: {}/{} 
        Frame: {}/{}
        Recording: {}
        """.format(
            self.current_sequence + 1,
            self.num_sequences,
            self.current_frame + 1,
            self.sequence_length,
            self.recording_names[self.current_sequence],
        )
        print(help_text)

    def update_display(self):
        """Update all display components."""
        # Clear all axes
        for ax in [
            self.ax_image,
            self.ax_timeline,
            self.ax_labels,
            self.ax_features,
        ] + self.telemetry_axes:
            ax.clear()

        # Update each component
        self._update_image_display()
        self._update_telemetry_display()
        self._update_timeline_display()
        self._update_labels_display()
        self._update_features_display()

        # Refresh the display
        self.fig.canvas.draw()

    def _update_image_display(self):
        """Update main image with detection overlays."""
        # Get current data
        recording_name = self.recording_names[self.current_sequence]
        timestamp = self.timestamps[self.current_sequence][self.current_frame]

        # Load original image
        img = self.original_loader.load_image(recording_name, timestamp)

        if img is None:
            self.ax_image.text(
                0.5,
                0.5,
                f"Image not found:\n{recording_name}/{timestamp}.jpg",
                ha="center",
                va="center",
                transform=self.ax_image.transAxes,
                fontsize=12,
                color="red",
            )
            self.ax_image.set_title("Image Not Found", color="red")
            return

        # Display image
        self.ax_image.imshow(img)

        # Get detections
        detections = self.detection_data[self.current_sequence][self.current_frame]
        mask = self.detection_masks[self.current_sequence][self.current_frame]

        # Convert normalized coordinates back to pixel coordinates
        valid_detections = detections[mask]

        if len(valid_detections) > 0:
            self._draw_detection_overlays(img, valid_detections)

        # Update title with info
        vehicle_count = mask.sum().item()
        self.ax_image.set_title(
            f"Frame {self.current_frame + 1}/{self.sequence_length} | "
            f"Vehicles: {vehicle_count} | "
            f"{timestamp}",
            fontweight="bold",
        )
        self.ax_image.set_xlabel(f"Recording: {recording_name}")
        self.ax_image.set_xticks([])
        self.ax_image.set_yticks([])

    def _draw_detection_overlays(self, img, detections):
        """Draw detection bounding boxes on image."""
        img_height, img_width = img.shape[:2]

        # Apply ROI offset for coordinate conversion
        roi_x, roi_y, roi_width, roi_height = DEFAULT_IMAGE_ROI

        for i, detection in enumerate(detections):
            # Convert normalized coordinates to pixel coordinates
            # Detection format: [class_id, confidence, x1, y1, x2, y2, area]
            confidence = detection[1].item()
            x1_norm, y1_norm = detection[2].item(), detection[3].item()
            x2_norm, y2_norm = detection[4].item(), detection[5].item()
            area_norm = detection[6].item()

            # Convert to pixel coordinates (considering ROI)
            x1_pixel = x1_norm * roi_width + roi_x
            y1_pixel = y1_norm * roi_height + roi_y
            x2_pixel = x2_norm * roi_width + roi_x
            y2_pixel = y2_norm * roi_height + roi_y

            # Draw bounding box
            width = x2_pixel - x1_pixel
            height = y2_pixel - y1_pixel

            rect = patches.Rectangle(
                (x1_pixel, y1_pixel),
                width,
                height,
                linewidth=2,
                edgecolor=COLORS["detection"],
                facecolor="none",
                alpha=0.8,
            )
            self.ax_image.add_patch(rect)

            # Add confidence label
            label = f"Vehicle {confidence:.2f}"
            self.ax_image.text(
                x1_pixel,
                y1_pixel - 5,
                label,
                fontsize=9,
                color=COLORS["detection"],
                bbox=dict(boxstyle="round,pad=0.2", facecolor="black", alpha=0.7),
            )

            # Add normalized coordinates info
            coord_text = f"({x1_norm:.2f},{y1_norm:.2f})-({x2_norm:.2f},{y2_norm:.2f})"
            self.ax_image.text(
                x1_pixel,
                y2_pixel + 15,
                coord_text,
                fontsize=8,
                color=COLORS["accent"],
                bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.8),
            )

    def _update_telemetry_display(self):
        """Update telemetry plots."""
        # Get current telemetry data
        telemetry_seq = self.telemetry_data[self.current_sequence]

        # Determine if GEAR is one-hot encoded
        if self.telemetry_dim == 5:
            # Original format: [SPEED, RPM, ACCEL, LOAD, GEAR]
            continuous_features = telemetry_seq[:, :4]
            gear_values = telemetry_seq[:, 4]
            use_gear_onehot = False
        else:
            # Normalized format with one-hot GEAR
            continuous_features = telemetry_seq[:, :4]
            gear_onehot = telemetry_seq[:, 4:]
            gear_values = torch.argmax(gear_onehot, dim=1)
            use_gear_onehot = True

        time_axis = np.arange(self.sequence_length) * 0.5  # 0.5s intervals

        # Plot each telemetry feature
        for i, (ax, feature_name, unit) in enumerate(
            zip(self.telemetry_axes, TELEMETRY_FEATURES, TELEMETRY_UNITS)
        ):
            values = continuous_features[:, i].numpy()

            # Plot timeline
            ax.plot(time_axis, values, color=COLORS["primary"], linewidth=2, alpha=0.8)
            ax.scatter(time_axis, values, color=COLORS["primary"], s=20, alpha=0.6)

            # Highlight current frame
            current_time = self.current_frame * 0.5
            current_value = values[self.current_frame]
            ax.scatter(
                current_time,
                current_value,
                color=COLORS["accent"],
                s=100,
                zorder=10,
                edgecolor="black",
                linewidth=2,
            )

            # Formatting
            ax.set_title(f"{feature_name}", fontweight="bold")
            ax.set_ylabel(f"{unit}")
            ax.grid(True, alpha=0.3)
            ax.set_xlim(0, (self.sequence_length - 1) * 0.5)

            # Add current value text
            ax.text(
                0.02,
                0.98,
                f"{current_value:.1f} {unit}",
                transform=ax.transAxes,
                fontsize=10,
                fontweight="bold",
                verticalalignment="top",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8),
            )

        # Add gear display
        current_gear = int(gear_values[self.current_frame].item())
        gear_text = f"Gear: {current_gear}" if current_gear > 0 else "Neutral"

        self.ax_load.text(
            0.02,
            0.02,
            gear_text,
            transform=self.ax_load.transAxes,
            fontsize=10,
            fontweight="bold",
            verticalalignment="bottom",
            bbox=dict(boxstyle="round,pad=0.3", facecolor=COLORS["accent"], alpha=0.8),
        )

    def _update_timeline_display(self):
        """Update sequence timeline overview."""
        # Create timeline visualization
        time_axis = np.arange(self.sequence_length) * 0.5

        # Timeline background
        self.ax_timeline.barh(
            0,
            (self.sequence_length - 1) * 0.5,
            height=0.4,
            color=COLORS["grid"],
            alpha=0.3,
        )

        # Current position
        current_time = self.current_frame * 0.5
        self.ax_timeline.barh(
            0, current_time, height=0.4, color=COLORS["primary"], alpha=0.8
        )

        # Add frame markers
        for i in range(0, self.sequence_length, 5):  # Every 5 frames
            x = i * 0.5
            self.ax_timeline.axvline(x, color="black", alpha=0.5, linewidth=1)
            self.ax_timeline.text(x, 0.6, f"{i}", ha="center", fontsize=8)

        # Current frame indicator
        self.ax_timeline.axvline(
            current_time, color=COLORS["accent"], linewidth=3, alpha=0.9
        )

        self.ax_timeline.set_xlim(0, (self.sequence_length - 1) * 0.5)
        self.ax_timeline.set_ylim(-0.5, 1)
        self.ax_timeline.set_xlabel("Time (seconds)")
        self.ax_timeline.set_title(
            f"Sequence Timeline (Frame {self.current_frame + 1}/{self.sequence_length})",
            fontweight="bold",
        )
        self.ax_timeline.set_yticks([])

    def _update_labels_display(self):
        """Update future labels display."""
        # Create labels overview
        label_types = [name for name in self.labels_data.keys() if name.endswith("s")]

        if not label_types:
            self.ax_labels.text(
                0.5,
                0.5,
                "No labels available",
                ha="center",
                va="center",
                transform=self.ax_labels.transAxes,
            )
            return

        # Get labels for current sequence
        y_pos = 0
        row_height = 0.8 / len(label_types)

        for i, label_name in enumerate(label_types):
            label_value = self.labels_data[label_name][self.current_sequence].item()

            # Color based on label type and value
            if "brake" in label_name:
                color = COLORS["success"] if label_value else COLORS["grid"]
                icon = "🛑" if label_value else "✅"
            elif "coast" in label_name:
                color = COLORS["accent"] if label_value else COLORS["grid"]
                icon = "⛵" if label_value else "🚗"
            else:
                color = COLORS["primary"] if label_value else COLORS["grid"]
                icon = "✓" if label_value else "✗"

            # Draw label box
            rect = patches.Rectangle(
                (i * 0.25, y_pos),
                0.2,
                row_height * 0.8,
                facecolor=color,
                alpha=0.7,
                edgecolor="black",
            )
            self.ax_labels.add_patch(rect)

            # Add text
            self.ax_labels.text(
                i * 0.25 + 0.1,
                y_pos + row_height * 0.4,
                f"{icon}\n{label_name}",
                ha="center",
                va="center",
                fontsize=9,
                fontweight="bold",
            )

            y_pos += row_height

        self.ax_labels.set_xlim(0, len(label_types) * 0.25)
        self.ax_labels.set_ylim(0, 1)
        self.ax_labels.set_title(
            "Future Labels (Prediction Targets)", fontweight="bold"
        )
        self.ax_labels.set_xticks([])
        self.ax_labels.set_yticks([])

    def _update_features_display(self):
        """Update feature normalization comparison."""
        # Get current normalized data
        normalized_telemetry = self.telemetry_data[self.current_sequence][
            self.current_frame
        ]
        normalized_detections = self.detection_data[self.current_sequence][
            self.current_frame
        ]
        detection_mask = self.detection_masks[self.current_sequence][self.current_frame]

        # Load original telemetry for comparison
        recording_name = self.recording_names[self.current_sequence]
        original_df = self.original_loader.load_telemetry(recording_name)

        if original_df is not None:
            # Find corresponding row in original data
            start_idx = self.start_indices[self.current_sequence]
            original_row_idx = start_idx + self.current_frame

            if original_row_idx < len(original_df):
                original_row = original_df.iloc[original_row_idx]

                # Create comparison table
                comparison_data = []
                for i, feature_name in enumerate(TELEMETRY_FEATURES):
                    if feature_name in original_row:
                        original_val = original_row[feature_name]
                        normalized_val = normalized_telemetry[i].item()
                        comparison_data.append(
                            [feature_name, original_val, normalized_val]
                        )

                # Display as table
                if comparison_data:
                    table_text = "Feature Normalization Comparison:\n"
                    table_text += "Feature | Original | Normalized\n"
                    table_text += "-" * 35 + "\n"

                    for feature_name, orig, norm in comparison_data:
                        table_text += (
                            f"{feature_name:<8} | {orig:>8.1f} | {norm:>8.3f}\n"
                        )

                    # Add detection info
                    valid_detections = detection_mask.sum().item()
                    table_text += (
                        f"\nDetections: {valid_detections}/{self.max_detections}"
                    )

                    self.ax_features.text(
                        0.05,
                        0.95,
                        table_text,
                        transform=self.ax_features.transAxes,
                        fontsize=9,
                        fontfamily="monospace",
                        verticalalignment="top",
                        bbox=dict(
                            boxstyle="round,pad=0.5",
                            facecolor=COLORS["background"],
                            alpha=0.8,
                        ),
                    )

        self.ax_features.set_title("Data Processing Details", fontweight="bold")
        self.ax_features.set_xticks([])
        self.ax_features.set_yticks([])
        self.ax_features.set_xlim(0, 1)
        self.ax_features.set_ylim(0, 1)


def main():
    """Main entry point for the sequence viewer."""
    parser = argparse.ArgumentParser(description="Multimodal Sequence Viewer")
    parser.add_argument(
        "--h5-file",
        type=str,
        default="data/datasets/multimodal/train.h5",
        help="Path to HDF5 dataset file",
    )
    parser.add_argument(
        "--recordings-dir",
        type=str,
        default="data/recordings",
        help="Path to original recordings directory",
    )
    parser.add_argument(
        "--sequence", type=int, default=0, help="Starting sequence index"
    )
    parser.add_argument(
        "--export-dir",
        type=str,
        default="evaluation/sequence_exports",
        help="Directory for exported images",
    )

    args = parser.parse_args()

    # Check if files exist
    h5_path = Path(args.h5_file)
    if not h5_path.exists():
        print(f"❌ HDF5 file not found: {h5_path}")
        return False

    recordings_path = Path(args.recordings_dir)
    if not recordings_path.exists():
        print(f"❌ Recordings directory not found: {recordings_path}")
        return False

    # Create export directory
    export_dir = Path(args.export_dir)
    export_dir.mkdir(parents=True, exist_ok=True)

    print("🚀 Starting Multimodal Sequence Viewer...")
    print(f"   📁 Dataset: {h5_path}")
    print(f"   📷 Recordings: {recordings_path}")
    print(f"   💾 Export directory: {export_dir}")

    try:
        # Create and run viewer
        viewer = MultimodalSequenceViewer(
            h5_file_path=str(h5_path), recordings_dir=str(recordings_path)
        )

        # Set starting sequence
        if 0 <= args.sequence < viewer.num_sequences:
            viewer.current_sequence = args.sequence
            viewer.update_display()

        # Show the interactive plot
        plt.tight_layout()
        plt.show()

        print("✅ Viewer closed successfully")
        return True

    except Exception as e:
        print(f"❌ Error running viewer: {e}")
        import traceback

        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
