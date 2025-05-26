#!/bin/bash
#SBATCH --job-name=inf22086_multimodal_full_pipeline
#SBATCH --gres=gpu:6                        # 6 GPUs for training steps
#SBATCH --cpus-per-task=24
#SBATCH --mem=120G
#SBATCH --time=72:00:00                     # Full pipeline time
#SBATCH --output=logs/multimodal_full_pipeline-%j.out
#SBATCH --error=logs/multimodal_full_pipeline-%j.err

# Create logs directory
mkdir -p logs

echo "🌟 Starting complete multimodal pipeline..."

# Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    python3 -m venv venv
fi

# Activate virtual environment
source venv/bin/activate

# Install all dependencies needed for the full pipeline
pip install torch torchvision torchaudio \
    ultralytics opencv-python \
    pandas numpy h5py scikit-learn \
    matplotlib seaborn \
    wandb tqdm requests

# Step 1: Download data (if URLs are configured)
echo "📥 Step 1: Data download validation..."
srun python3 training/multimodal/data_download.py --validate-only

# Step 2: Auto-annotation
echo "🎯 Step 2: Auto-annotation..."
srun python3 training/multimodal/auto_annotate.py \
    --recordings-dir data/recordings \
    --model yolo_best.pt \
    --confidence 0.3

# Step 3: Generate future labels
echo "🔮 Step 3: Future label generation..."
srun python3 training/multimodal/generate_labels.py \
    --recordings-dir data/recordings \
    --coast-threshold 10.0

# Step 4: Prepare dataset
echo "📦 Step 4: Dataset preparation..."
srun python3 training/multimodal/prepare_dataset.py \
    --recordings-dir data/recordings \
    --output-dir data/datasets/multimodal \
    --sequence-length 20 \
    --sequence-stride 5 \
    --max-detections 12

# Step 5: Train all models
echo "🤖 Step 5: Training all architectures..."
srun srun python3 training/multimodal/train_all_variants.py \
    --data-dir data/datasets/multimodal \
    --output-dir data/models/multimodal \
    --use-wandb \
    --world-size 6

# Step 6: Evaluate models
# echo "📊 Step 6: Model evaluation..."
# srun python3 training/multimodal/evaluate_models.py \
#     --models-dir data/models/multimodal \
#     --data-dir data/datasets/multimodal \
#     --output-dir data/evaluation/multimodal

echo "🎉 Complete multimodal pipeline finished!"
echo "📁 Results available in:"
echo "   - Models: data/models/multimodal/"
# echo "   - Evaluation: data/evaluation/multimodal/"

# Deactivate virtual environment
deactivate