#!/bin/bash
#SBATCH --job-name=inf22086_yolo12n_boxy_train
#SBATCH --gres=gpu:4   # Use GPUS env var or default to 4
#SBATCH --cpus-per-task=12
#SBATCH --mem=40G
#SBATCH --time=12:00:00
#SBATCH --output=yolo12n_train-%j.out
#SBATCH --error=yolo12n_train-%j.out

# Set default values if not provided as environment variables
GPUS=${GPUS:-4}

# Print configuration
echo "=== Training Configuration ==="
echo "Number of GPUs: $GPUS"
echo "=========================="

# Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    python3 -m venv venv
fi

# Activate virtual environment
source venv/bin/activate

# Install requirements
pip install ultralytics torch torchvision torchaudio

# Run training with parameters
srun python3 -u boxy_train.py --num_gpus $GPUS

# Deactivate virtual environment
deactivate