import os

os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
import torch
from pathlib import Path
from ultralytics import YOLO
import matplotlib.pyplot as plt
import time

yolo_model_path = "yolo11n.pt"  # Pfad zum vortrainierten YOLOv11n-Modell
output_path = (
    "data/datasets/processed/nuimages"  # Pfad zum Ausgabeordner für das Dataset
)

# Überprüfen der verfügbaren Hardware
print(f"PyTorch version: {torch.__version__}")
if torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Sicherstellen, dass der Ordner für die Trainingsergebnisse existiert
# results_dir = Path("results") / f"finetune_{time.strftime('%Y%m%d_%H%M%S')}"
# os.makedirs(results_dir, exist_ok=True)

# 1. Modell laden (vortrainiertes YOLOv11n)
print(f"Lade vortrainiertes Modell: {yolo_model_path}")
model = YOLO(yolo_model_path)

# 2. Training konfigurieren und starten
print(f"Starte Training mit Gerät: {device}")

# Parameter für das Training
results = model.train(
    data=str(Path(output_path) / "dataset.yaml"),
    # Grundlegende Trainingsparameter
    epochs=60,  # Anzahl der Trainingszyklen
    imgsz=640,  # Bildgröße (Standard für YOLOv11n)
    device=device,  # Verwendetes Gerät (mps, cuda, cpu)
    batch=-1,  # Batch-Größe,
    # patience=5,
    resume=True,
)
