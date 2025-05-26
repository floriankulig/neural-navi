#!/bin/bash
#SBATCH --job-name=inf22086_boxy_visualizer
#SBATCH --gres=gpu:0
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=logs/boxy_visualizer-%j.out
#SBATCH --error=logs/boxy_visualizer-%j.out

# Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    python3 -m venv venv
fi

# Activate virtual environment
source venv/bin/activate

# Install visualization dependencies
pip install numpy matplotlib seaborn opencv-python tqdm pandas

# Create output directories
rm -rf evaluation/boxy_visualizations
mkdir -p evaluation/boxy_visualizations

# Run visualization script (updated path and arguments)
srun python3 -u evaluation/visualization.py \
    --data_dir data/processed/boxy_yolo_n3 \
    --output_dir evaluation/boxy_visualizations \
    --sample_size 50000 \
    --heatmap_resolution 800 \
    --img_width 1232 \
    --img_height 1028 \
    --show_examples 5

# Deactivate virtual environment
deactivate

echo "Boxy visualization completed successfully!"
echo "Results saved to: evaluation/boxy_visualizations"