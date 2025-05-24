# Makefile for neural-navi project
# All scripts located in scripts/ directory for clean organization

.PHONY: help install record detect prepare-boxy train-yolo-boxy train-multimodal evaluate clean

# Default target
help:
	@echo "🚗 Neural-Navi Development Commands"
	@echo ""
	@echo "🔧 Setup:"
	@echo "  install          Install package in development mode"
	@echo ""
	@echo "📹 Recording:"
	@echo "  record           Start drive recording (with live preview)"
	@echo "  record-quiet     Start drive recording (no preview)"
	@echo ""
	@echo "🔍 Detection:"
	@echo "  detect           Run vehicle detection on recordings"
	@echo "  detect-conf      Run detection with custom confidence (0.3)"
	@echo ""
	@echo "📊 Dataset Preparation:"
	@echo "  prepare-boxy     Prepare Boxy dataset for YOLO training"
	@echo "  annotate         Run annotation script on recordings"
	@echo ""
	@echo "🧠 Training:"
	@echo "  train-yolo-boxy     Train YOLO on Boxy dataset"
	@echo "  train-yolo-nuimages Train YOLO on NuImages dataset (when ready)"
	@echo "  train-multimodal    Train multimodal model (when ready)"
	@echo ""
	@echo "📈 Evaluation:"
	@echo "  evaluate         Run evaluation and visualization"
	@echo ""
	@echo "🧹 Cleanup:"
	@echo "  clean            Clean cache and temporary files"

# Setup and installation
install:
	pip install -e .

# Recording commands - now using scripts/
record:
	python scripts/record_drive.py --show-live --with-logs

record-quiet:
	python scripts/record_drive.py --with-logs

record-debug:
	python scripts/record_drive.py --show-live --with-logs --interval 1.0

# Detection commands  
detect:
	python scripts/detect_vehicles.py --recordings data/recordings

detect-conf:
	python scripts/detect_vehicles.py --recordings data/recordings --conf 0.3

detect-model:
	python scripts/detect_vehicles.py --recordings data/recordings --model yolo_best.pt

# Dataset preparation - now using scripts/
prepare-boxy:
	python scripts/prepare_boxy.py

annotate:
	python scripts/annotate_recordings.py

# Training commands - now using scripts/
train-yolo-boxy:
	python scripts/train_yolo_boxy.py

train-yolo-nuimages:
	python scripts/train_yolo_nuimages.py

train-multimodal:
	python scripts/train_multimodal.py

train-multimodal-debug:
	python scripts/train_multimodal.py --debug

# Evaluation - now using scripts/
evaluate:
	python scripts/evaluate_model.py

visualize:
	python scripts/visualize_dataset.py

# Development commands
test:
	python -m pytest tests/ -v

test-fast:
	python -m pytest tests/ -x -v --tb=short

format:
	black src/ training/ evaluation/ scripts/ *.py

lint:
	flake8 src/ training/ evaluation/ scripts/ *.py

type-check:
	mypy src/

# Utility commands
clean:
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete
	find . -type d -name "*.egg-info" -exec rm -rf {} +
	rm -rf .pytest_cache/
	rm -rf .mypy_cache/

# Hardware-specific commands for Raspberry Pi
pi-setup:
	@echo "🥧 Setting up for Raspberry Pi..."
	sudo apt-get update
	sudo apt-get install -y python3-opencv
	pip install -r requirements.txt

# Data management
backup-recordings:
	@echo "📦 Creating backup of recordings..."
	tar -czf data/recordings_backup_$(shell date +%Y%m%d_%H%M%S).tar.gz data/recordings/

backup-models:
	@echo "📦 Creating backup of models..."
	tar -czf data/models_backup_$(shell date +%Y%m%d_%H%M%S).tar.gz data/models/

# Show project status
status:
	@echo "📊 Neural-Navi Project Status:"
	@echo "Recordings: $(shell find data/recordings -name "*.jpg" 2>/dev/null | wc -l) images"
	@echo "YOLO Models: $(shell find data/models -name "*.pt" 2>/dev/null | wc -l) checkpoints"
	@echo "Datasets: $(shell ls -d data/datasets/*/ 2>/dev/null | wc -l) prepared"
	@echo "Scripts: $(shell ls scripts/*.py 2>/dev/null | wc -l) available"

# Quick aliases for common tasks
r: record
d: detect  
t: train-yolo-boxy
h: help

# Advanced workflow commands
full-pipeline-boxy:
	@echo "🚀 Running full Boxy pipeline..."
	make prepare-boxy
	make train-yolo-boxy
	make evaluate

experiment-quick:
	@echo "🧪 Quick experiment run..."
	make record-debug
	make detect
	make annotate