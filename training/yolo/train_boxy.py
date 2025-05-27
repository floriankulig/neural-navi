#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
YOLOv12n Training Script for Boxy Dataset
"""

import sys
from pathlib import Path

# Add src and training to path for imports
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "training"))

from ultralytics import YOLO


def main():
    num_gpus = 3
    batch_size = 64 * num_gpus
    devices = [-1] * num_gpus

    # Load pre-trained model
    model = YOLO("yolo12n.pt")

    # Training hyperparameters
    hyperparameters = {
        "data": "data/datasets/dataset_nc1_hard.yaml",
        "epochs": 110,
        "close_mosaic": 10,
        "patience": 15,
        "batch": batch_size,
        "imgsz": 704,
        "multi_scale": True,
        'optimizer': 'AdamW',
        'warmup_epochs': 5,
        'lr0': 0.0075,
        'lrf': 0.011,
        'cos_lr': True,
        # Augmentations adapted
        "degrees": 6,  # Less, since camera is mounted fixed
        "perspective": 0.00025,  # Minimal
        "mixup": 0.075,  # Off - problematic for small objects
        "cutmix": 0.075,  # Off - problematic for small objects
        "copy_paste": 0.1,  # Can help!
        "close_mosaic": 10,
        "save_period": 5,
        "workers": 12,
        "cls": 0.55,
        "device": devices,
        "dropout": 0.175,
        "pretrained": True,
        "plots": True,
        "val": True,
        # "project": "runs/final",
        "name": "yolo12n1",
    }

    # Start training
    model.train(**hyperparameters)


if __name__ == "__main__":
    main()
