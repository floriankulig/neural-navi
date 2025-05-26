#!/bin/bash
#SBATCH --job-name=inf22086_multimodal_labels
#SBATCH --gres=gpu:0                        # No GPUs needed
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=logs/multimodal_labels-%j.out
#SBATCH --error=logs/multimodal_labels-%j.err

# Create logs directory
mkdir -p logs

echo "🔮 Starting future label generation pipeline..."

# Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    python3 -m venv venv
fi

# Activate virtual environment
source venv/bin/activate

# Install only required packages for label generation
pip install pandas numpy tqdm

# Generate future labels
srun python3 -u training/multimodal/generate_labels.py \
    --recordings-dir data/recordings \
    --coast-threshold 10.0 \
    --force

echo "✅ Future label generation completed!"