#!/bin/bash
#SBATCH --job-name=inf22086_boxy_prepare
#SBATCH --gres=gpu:0                        # No GPUs needed for CPU task
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=06:00:00
#SBATCH --output=logs/boxy_prepare-%j.out
#SBATCH --error=logs/boxy_prepare-%j.out

# Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    python3 -m venv venv
fi

# Activate virtual environment
source venv/bin/activate

# Run the dataset preparation script (updated path)
srun python3 -u training/datasets/boxy_preparation.py

# Deactivate virtual environment
deactivate

echo "Boxy dataset preparation completed successfully!"