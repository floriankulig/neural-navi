#!/bin/bash
#SBATCH --job-name=inf22086_test
#SBATCH --gres=gpu:0                        # No GPUs needed
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=logs/test-%j.out
#SBATCH --error=logs/test-%j.out

# Create logs directory
mkdir -p logs

echo "📦 Starting dataset preparation pipeline..."

# Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    python3 -m venv venv
fi

# Activate virtual environment
source venv/bin/activate

# Install required packages for dataset access
pip install h5py pandas numpy scikit-learn tqdm

# Prepare dataset with sliding windows
srun python3 -u training/datasets/data_loaders.py \
    --use-memory

echo "✅ Test completed!"