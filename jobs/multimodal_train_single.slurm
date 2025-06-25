#!/bin/bash
#SBATCH --job-name=inf22086_single_arch_train
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=logs/single_train-%j.out
#SBATCH --error=logs/single_train-%j.out

# Create logs directory
mkdir -p logs

echo "🚀 Starting single architecture training..."
echo "SLURM Job ID: $SLURM_JOB_ID"
echo "SLURM Node: $SLURMD_NODENAME"

# ====================================
# ARCHITECTURE CONFIGURATION
# ====================================

# Architecture to train (only thing to change)
ARCHITECTURE=${ARCHITECTURE:-"simple_concat_lstm"}

# Data paths
DATA_DIR=${DATA_DIR:-"data/datasets/multimodal/s2"}
OUTPUT_DIR=${OUTPUT_DIR:-"data/models/multimodal/coast"}

echo ""
echo "📋 Training Configuration:"
echo "   🏗️ Architecture: $ARCHITECTURE"
echo "   📁 Data Dir: $DATA_DIR"
echo "   📁 Output Dir: $OUTPUT_DIR"

# Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    echo "📦 Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
source venv/bin/activate

# Install dependencies
echo "📦 Installing dependencies..."
pip install torch torchvision torchaudio numpy h5py scikit-learn tqdm

# Check GPU availability
echo ""
echo "🔍 GPU Check:"
srun python3 -c "
import torch
print(f'CUDA available: {torch.cuda.is_available()}')
print(f'Device count: {torch.cuda.device_count()}')
if torch.cuda.is_available():
    print(f'Current device: {torch.cuda.current_device()}')
    print(f'Device name: {torch.cuda.get_device_name()}')
    print(f'Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB')
"

# Start training
echo ""
echo "🚀 Starting training..."
srun python3 -u training/multimodal/train_single.py \
    --data-dir "$DATA_DIR" \
    --output-dir "$OUTPUT_DIR" \
    --architecture "$ARCHITECTURE"

# Check if training succeeded
if [ $? -eq 0 ]; then
    echo ""
    echo "🎉 Training completed successfully!"
    echo "📁 Model saved to: $OUTPUT_DIR/$ARCHITECTURE/"
    
    # Show saved files
    if [ -d "$OUTPUT_DIR/$ARCHITECTURE" ]; then
        echo ""
        echo "📋 Saved Files:"
        ls -lh "$OUTPUT_DIR/$ARCHITECTURE"
    fi
else
    echo ""
    echo "❌ Training failed!"
    exit 1
fi

# Deactivate virtual environment
deactivate

echo ""
echo "✅ Job completed!"