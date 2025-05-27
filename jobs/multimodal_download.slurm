#!/bin/bash
#SBATCH --job-name=inf22086_multimodal_download
#SBATCH --gres=gpu:0                        # No GPUs needed
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --output=logs/multimodal_download-%j.out
#SBATCH --error=logs/multimodal_download-%j.out

# Create logs directory
mkdir -p logs

echo "🚀 Starting multimodal data download pipeline..."

# Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    python3 -m venv venv
fi

# Activate virtual environment
source venv/bin/activate

# Install only necessary packages for this job
pip install requests tqdm

# Download and extract data
# NOTE: Update URLs in the script before running
srun python3 -u training/multimodal/data_download.py \
    --output-dir data/recordings

# Deactivate virtual environment
deactivate

echo "✅ Multimodal data download completed!"