#!/bin/bash
#SBATCH --job-name=inf22086_single_arch_train
#SBATCH --gres=gpu:1                        # Single GPU training
#SBATCH --cpus-per-task=12
#SBATCH --mem=64G
#SBATCH --time=12:00:00                     # 12 hours for full training
#SBATCH --output=logs/single_train-%j.out
#SBATCH --error=logs/single_train-%j.out

# Create logs directory
mkdir -p logs

echo "🚀 Starting single architecture training..."
echo "SLURM Job ID: $SLURM_JOB_ID"
echo "SLURM Node: $SLURMD_NODENAME"
echo "GPU: $CUDA_VISIBLE_DEVICES"

# Configuration
ARCHITECTURE=${ARCHITECTURE:-"simple_concat_lstm"}
EPOCHS=${EPOCHS:-10}
BATCH_SIZE=${BATCH_SIZE:-128}
LEARNING_RATE=${LEARNING_RATE:-1e-5}

echo "📋 Training Configuration:"
echo "   🏗️ Architecture: $ARCHITECTURE"
echo "   🔄 Epochs: $EPOCHS"
echo "   📦 Batch Size: $BATCH_SIZE"
echo "   📈 Learning Rate: $LEARNING_RATE"

# Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    echo "📦 Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
source venv/bin/activate

# Install dependencies
echo "📦 Installing dependencies..."
pip install torch torchvision torchaudio numpy h5py scikit-learn

# Check GPU availability
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
    --data-dir data/datasets/multimodal \
    --output-dir data/models/multimodal \
    --architecture "$ARCHITECTURE" \
    --epochs "$EPOCHS" \
    --batch-size "$BATCH_SIZE" \
    --learning-rate "$LEARNING_RATE"

# Check if training succeeded
if [ $? -eq 0 ]; then
    echo ""
    echo "🎉 Training completed successfully!"
    echo "📁 Model saved to: data/models/multimodal/$ARCHITECTURE/"
else
    echo ""
    echo "❌ Training failed!"
    exit 1
fi

# Deactivate virtual environment
deactivate

echo ""
echo "✅ Job completed!"