# Makefile for neural-navi project
# Scripts located in root directory, SLURM jobs in /jobs

.PHONY: help install record detect prepare-boxy train-yolo-boxy evaluate clean

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
	@echo "🧠 Training (SLURM):"
	@echo "  train-yolo-boxy     Submit YOLO training job to SLURM"
	@echo "  val-yolo            Submit YOLO validation job"
	@echo "  visualize-boxy      Submit Boxy visualization job"
	@echo ""
	@echo "🧹 Cleanup:"
	@echo "  clean            Clean cache and temporary files"

# Setup and installation
install:
	pip install -e .

# Recording commands
record:
	python record_drive.py --show-live --with-logs

record-quiet:
	python record_drive.py --with-logs

record-debug:
	python record_drive.py --show-live --with-logs --interval 1.0

# Detection commands  
detect:
	python detect_vehicles.py --recordings data/recordings

detect-conf:
	python detect_vehicles.py --recordings data/recordings --conf 0.3

detect-model:
	python detect_vehicles.py --recordings data/recordings --model yolo_best.pt

# Dataset preparation
prepare-boxy:
	python training/datasets/boxy_preparation.py

annotate:
	python training/datasets/annotation.py

# SLURM training jobs
train-yolo-boxy:
	sbatch jobs/boxy_train.slurm

val-yolo:
	sbatch jobs/val_yolo.slurm

visualize-boxy:
	sbatch jobs/boxy_visualizer.slurm

prepare-boxy-slurm:
	sbatch jobs/boxy_prepare.slurm

# Evaluation
evaluate:
	python evaluation/boxy_visualization.py

# Development commands
format:
	black src/ training/ evaluation/ *.py

lint:
	flake8 src/ training/ evaluation/ *.py

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
	@echo "SLURM Jobs: $(shell ls jobs/*.slurm 2>/dev/null | wc -l) available"
	@echo "Training Scripts: $(shell ls training/*/*.py 2>/dev/null | wc -l) available"

# Quick aliases for common tasks
r: record
d: detect  
t: train-yolo-boxy
h: help

# Advanced workflow commands
full-pipeline-boxy:
	@echo "🚀 Running full Boxy pipeline..."
	make prepare-boxy-slurm
	make train-yolo-boxy
	make val-yolo

# Cluster monitoring
monitor-jobs:
	@echo "📊 SLURM Job Status:"
	squeue -u $$USER