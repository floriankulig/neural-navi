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
        'data': 'data/dataset_nc3.yaml',
        'epochs': 60,
        'patience': 15,
        'batch': 128,
        'imgsz': 704,
        'multi_scale': True,
        'optimizer': 'AdamW',
        'warmup_epochs': 5,
        'lr0': 0.0075,
        'cos_lr': True,
        'degrees': 4,
        'mixup': 0.2,
        'cutmix': 0.1,
        'close_mosaic': 15,
        'save_period': 5,
        'workers': 8,
        'device': [-1, -1],
        'dropout': 0.15,
        'pretrained': True,
        'plots':True,
        'val':True,
        'name': 'yolo12n3_boxy',
    }
    
    # Start training
    model.train(**hyperparameters)

if __name__ == "__main__":
    main()
