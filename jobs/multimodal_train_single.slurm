#!/bin/bash
#SBATCH --job-name=inf22086_multimodal_train_single
#SBATCH --gres=gpu:6                        # 6 GPUs for DDP training
#SBATCH --cpus-per-task=24
#SBATCH --mem=120G
#SBATCH --time=12:00:00
#SBATCH --output=logs/multimodal_train_single-%j.out
#SBATCH --error=logs/multimodal_train_single-%j.out

# Create logs directory
mkdir -p logs

echo "🤖 Starting single architecture training..."

# Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    python3 -m venv venv
fi

# Activate virtual environment
source venv/bin/activate

# Install training dependencies
pip install torch torchvision torchaudio numpy h5py wandb tqdm

# Train single architecture (specify via argument)
# Usage: sbatch --export=ARCHITECTURE=simple_concat_lstm jobs/multimodal_train_single.slurm
ARCHITECTURE=${ARCHITECTURE:-"simple_concat_lstm"}

srun python3 -u training/multimodal/train_all_variants.py \
    --data-dir data/datasets/multimodal \
    --output-dir data/models/multimodal \
    --architecture $ARCHITECTURE \
    --use-wandb \
    --world-size 6

echo "✅ Single architecture training completed: $ARCHITECTURE"