#!/bin/bash

# List of all 12 architecture combinations
architectures=(
    "simple_concat_lstm"
    "simple_concat_transformer"
    "simple_cross_lstm"
    "simple_cross_transformer"
    "simple_query_lstm"
    "simple_query_transformer"
    # "attention_concat_lstm"
    # "attention_concat_transformer"
    # "attention_cross_lstm"
    # "attention_cross_transformer"
    # "attention_query_lstm"
    # "attention_query_transformer"
)

echo "🚀 Starting training for ${#architectures[@]} architectures..."

# Submit jobs for each architecture
for arch in "${architectures[@]}"; do
    echo "📤 Submitting job for: $arch"
    sbatch --export=ARCHITECTURE="$arch" jobs/multimodal_train_single.slurm
    sleep 1  # Brief pause to avoid overwhelming scheduler
done

echo "✅ All jobs submitted!"
echo "📊 Monitor with: squeue -u \$USER"