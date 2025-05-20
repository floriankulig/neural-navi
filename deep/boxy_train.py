#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
YOLOv12n Training Script for Boxy Dataset
"""

from ultralytics import YOLO

def main():
    # Load pre-trained model
    model = YOLO("yolo12n.pt")
    
    # Training hyperparameters
    hyperparameters = {
        'data': 'data/dataset_nc3_skip.yaml',
        'epochs': 45,
        'patience': 10,
        'batch': 240,
        'imgsz': 704,
        'multi_scale': True,
        'optimizer': 'AdamW',
        'warmup_epochs': 5,
        'lr0': 0.0075,
        'cos_lr': True,
        'degrees': 4,
        'mixup': 0.25,
        'cutmix': 0.15,
        'close_mosaic': 13,
        'save_period': 5,
        'workers': 8,
        'device': [-1, -1, -1, -1],
        'cache': True,
        'dropout': 0.15,
        'pretrained': True,
        'plots': True,
        'val': True,
        'name': 'yolo12n3_skip',
    }
    
    # Start training
    model.train(**hyperparameters)

if __name__ == "__main__":
    main()
