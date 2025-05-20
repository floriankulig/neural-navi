#!/bin/bash
#SBATCH --job-name=inf22086_val_yolo  # Ersetze #USERNAME mit deinem LDAP-Namen
#SBATCH --gres=gpu:1                        # 1 GPU anfordern
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=01:00:00
#SBATCH --output=val_yolo-%j.out
#SBATCH --error=val_yolo-%j.out

# Erstelle virtuelle Umgebung falls nicht vorhanden
if [ ! -d "venv" ]; then
    python3 -m venv venv
fi

# Aktiviere virtuelle Umgebung
source venv/bin/activate

pip install ultralytics torch torchvision torchaudio

yolo val model=runs/detect/yolo12n3_skip/weights/last.pt data=deep/data/dataset_nc3_skip.yaml imgsz=704 batch=48 device=-1

# Deaktiviere virtuelle Umgebung
deactivate

echo "YOLO Val completed successfully!"