#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
YOLOv12n Training Script for Boxy Dataset
"""

from ultralytics import YOLO

def main():
    num_gpus = 3
    batch_size = 48 * num_gpus  
    devices = [-1] * num_gpus

    # Load pre-trained model
    model = YOLO("yolo12n.pt")
    
    # Training hyperparameters
    hyperparameters = {
        'data': 'data/dataset_nc1_hard.yaml',
        'epochs': 100,
        'close_mosaic': 10,
        'patience': 15,
        'batch': batch_size,
        'imgsz': 704,
        'multi_scale': False,
        # 'optimizer': 'AdamW',
        # 'warmup_epochs': 4,
        # 'lr0': 0.0075,
        # 'cos_lr': True,

        # Augmentations angepasst
        'degrees': 6,    # Weniger, da Kamera fest montiert
        'perspective': 0.00025,  # Minimal
        
        'mixup': 0.0,  # Aus - problematisch für kleine Objekte
        'cutmix': 0.05,   # Aus - problematisch für kleine Objekte
        'copy_paste': 0.1,  # Kann helfen!
        'close_mosaic': 10,

        'save_period':  5, 
        'workers': 12,
        'cls': 0.575,
        'device': devices,
        'dropout': 0.15,
        'pretrained': True,
        'plots': True,
        'val': True,
        'project': 'runs/final',
        'name': 'yolo12n1_auto',
    }
    
    # Start training
    model.train(**hyperparameters)

if __name__ == "__main__":
    main()
