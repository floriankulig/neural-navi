#!/bin/bash
#SBATCH --job-name=inf22086_multimodal_prepare
#SBATCH --gres=gpu:0                        # No GPUs needed
#SBATCH --cpus-per-task=12
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=logs/multimodal_prepare-%j.out
#SBATCH --error=logs/multimodal_prepare-%j.out

# Create logs directory
mkdir -p logs

echo "📦 Starting dataset preparation pipeline..."

# Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    python3 -m venv venv
fi

# Activate virtual environment
source venv/bin/activate

# Install required packages for dataset preparation
pip install h5py pandas numpy scikit-learn tqdm

# Prepare dataset with sliding windows
srun python3 -u training/multimodal/prepare_dataset.py \
    --recordings-dir data/recordings \
    --output-dir data/datasets/multimodal \
    --sequence-length 20 \
    --sequence-stride 5 \
    --max-detections 12 \
    --random-seed 42

echo "✅ Dataset preparation completed!"