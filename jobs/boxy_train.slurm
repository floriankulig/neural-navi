#!/bin/bash
#SBATCH --job-name=inf22086_yolo12n_boxy_train
#SBATCH --gres=gpu:3   # Use 3 GPUs
#SBATCH --cpus-per-task=12
#SBATCH --mem=40G
#SBATCH --time=12:00:00
#SBATCH --output=yolo12n_train-%j.out
#SBATCH --error=yolo12n_train-%j.out

# Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    python3 -m venv venv
fi

# Activate virtual environment
source venv/bin/activate

# Install requirements
pip install ultralytics torch torchvision torchaudio

# Run training with parameters
srun python3 -u training/yolo/train_boxy.py

# Deactivate virtual environment
deactivate

echo "YOLO training completed successfully!"