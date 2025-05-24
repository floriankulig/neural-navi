#!/bin/bash
#SBATCH --job-name=inf22086_val_yolo
#SBATCH --gres=gpu:1                        # 1 GPU for validation
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=01:00:00
#SBATCH --output=val_yolo-%j.out
#SBATCH --error=val_yolo-%j.out

# Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    python3 -m venv venv
fi

# Activate virtual environment
source venv/bin/activate

# Install requirements
pip install ultralytics torch torchvision torchaudio

yolo val model=data/models/yolo/yolo12n1.pt data=data/datasets/dataset_nc1_hard.yaml imgsz=704 batch=48 device=-1

# Deactivate virtual environment
deactivate

echo "YOLO validation completed successfully!"