#!/bin/bash
#SBATCH --job-name=inf22086_multimodal_evaluate
#SBATCH --gres=gpu:1                        # 1 GPU for evaluation
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=logs/multimodal_evaluate-%j.out
#SBATCH --error=logs/multimodal_evaluate-%j.err

# Create logs directory
mkdir -p logs

echo "📊 Starting model evaluation pipeline..."

# Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    python3 -m venv venv
fi

# Activate virtual environment
source venv/bin/activate

# Install evaluation dependencies
pip install torch numpy matplotlib seaborn scikit-learn pandas h5py tqdm

# Evaluate all trained models
srun python3 -u training/multimodal/evaluate_models.py \
    --models-dir data/models/multimodal \
    --data-dir data/datasets/multimodal \
    --output-dir data/evaluation/multimodal

echo "✅ Model evaluation completed!"