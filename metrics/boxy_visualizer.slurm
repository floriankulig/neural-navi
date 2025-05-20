#!/bin/bash
#SBATCH --job-name=inf22086_boxy_visualizer
#SBATCH --gres=gpu:0
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00

cd ..

# Erstelle virtuelle Umgebung falls nicht vorhanden
if [ ! -d "venv" ]; then
    python3 -m venv venv
fi

# Aktiviere virtuelle Umgebung
source venv/bin/activate

# Sicherstellen, dass Abhängigkeiten installiert sind
pip install numpy matplotlib seaborn opencv-python tqdm

# Verzeichnisse anlegen
rm -rf metrics/boxy_visualizations
mkdir -p metrics/boxy_visualizations

# Skript ausführen
srun python3 -u metrics/boxy_visualizer.py \
    --data_dir deep/data/boxy_yolo_n3 \
    --output_dir metrics/boxy_visualizations \
    --sample_size 50000 \
    --heatmap_resolution 800 \
    --img_width 1232 \
    --img_height 1028 \
    --show_examples 5