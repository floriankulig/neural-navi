#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
YOLOv12n Training Script for Boxy Dataset
"""

from ultralytics import YOLO
import argparse

def main():
    parser = argparse.ArgumentParser(description='Train YOLOv12n on Boxy Dataset')
    parser.add_argument('--num_gpus', type=int, default=4, help='Number of GPUs to use for training')
    args = parser.parse_args()

    batch_size = 64 * args.num_gpus  
    devices = [-1] * args.num_gpus

    # Load pre-trained model
    model = YOLO("yolo12n.pt")
    
    # Training hyperparameters
    hyperparameters = {
        'data': 'data/dataset_nc3_skip.yaml',
        'epochs': 40,
        'patience': 10,
        'batch': batch_size,
        'imgsz': 704,
        'multi_scale': True,
        'optimizer': 'AdamW',
        'warmup_epochs': 5,
        'lr0': 0.0075,
        'cos_lr': True,
        'degrees': 4,
        'mixup': 0.25,
        'cutmix': 0.15,
        'close_mosaic': 8,
        'save_period': 5,
        'workers': 8,
        'device': devices,
        'dropout': 0.15,
        'pretrained': True,
        'plots': True,
        'val': True,
        'name': 'yolo12n3',
    }
    # model = YOLO("best.pt")
    # hyperparameters = {
    #     'data': 'data/dataset_nc3.yaml',
    #     'epochs': 15,
    #     'patience': 10,
    #     'batch': batch_size,
    #     'imgsz': 704,
    #     'multi_scale': True,
    #     'optimizer': 'AdamW',
    #     'warmup_epochs': 2,
    #     'lr0': 0.0025,
    #     'cos_lr': True,
    #     'degrees': 4,
    #     'mixup': 0.25,
    #     'cutmix': 0.15,
    #     'close_mosaic': 10,
    #     'save_period': 2,
    #     'workers': 8,
    #     'classes': [1],
    #     'device': devices,
    #     'dropout': 0.15,
    #     'pretrained': True,
    #     'plots': True,
    #     'val': True,
    #     'name': 'yolo12n3_tunefront',
    # }
    
    # Start training
    model.train(**hyperparameters)

if __name__ == "__main__":
    main()
