#!/bin/bash
#SBATCH --job-name=inf22086_multimodal_train_all
#SBATCH --gres=gpu:6                        # 6 GPUs for DDP training
#SBATCH --cpus-per-task=24
#SBATCH --mem=120G
#SBATCH --time=48:00:00                     # Long job for all architectures
#SBATCH --output=logs/multimodal_train_all-%j.out
#SBATCH --error=logs/multimodal_train_all-%j.err

# Create logs directory
mkdir -p logs

echo "🚀 Starting training for all 12 architectures..."

# Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    python3 -m venv venv
fi

# Activate virtual environment
source venv/bin/activate

# Install training dependencies
pip install torch torchvision torchaudio numpy h5py wandb tqdm

# Train all architectures sequentially
srun python3 -u training/multimodal/train_all_variants.py \
    --data-dir data/datasets/multimodal \
    --output-dir data/models/multimodal \
    --use-wandb \
    --world-size 6

echo "✅ All architecture training completed!"