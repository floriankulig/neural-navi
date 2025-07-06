# Neural-Navi

**Multimodal Machine Learning Approach for Real-Time Critical Driving Situation Recognition in Preventive Driver Assistance Systems**

Neural-Navi is a research project that combines camera data and vehicle telemetry to detect critical driving situations in real-time. The system predicts braking and coasting events 1-5 seconds in advance while considering realistic hardware constraints for automotive deployment.

## 🚀 Quick Start

### Installation

1. **Clone repository:**
   ```bash
   git clone https://github.com/floriankulig/neural-navi.git
   cd neural-navi
   ```

2. **Create virtual environment:**
   ```bash
   python -m venv venv --system-site-packages  # Raspberry Pi OS (keeps Picamera2)
   # or
   python -m venv venv   # Windows/Mac/Linux
   ```

3. **Activate virtual environment:**
   ```bash
   source venv/bin/activate  # Linux/Mac
   # or
   .\venv\Scripts\activate   # Windows
   ```

4. **Install dependencies:**
   ```bash
   pip install -e .
   ```

### Basic Usage

#### Record driving data:
```bash
# Using Makefile (recommended)
make record

# Direct execution
python record_drive.py
```

#### Detect vehicles in recordings:
```bash
# Interactive viewer
make detect
# or: python detect_vehicles.py --recordings data/recordings
```

#### Train multimodal models:
```bash

# Train single architecture
make train-single-arch ARCH=simple_concat_transformer

# Evaluate trained models
make evaluate-multimodal
```

## 📁 Project Structure

```
neural-navi/
├── record_drive.py          # 🎯 Driving data recording
├── detect_vehicles.py       # 🔍 Vehicle detection in recordings
│
├── src/                     # 🔧 Core Application Code
│   ├── recording/           # Data acquisition (camera, telemetry hardware access)
│   ├── processing/          # Data preprocessing & feature engineering
│   │   └── features/        # Derived features (gear, brake force)
│   ├── model/               # Neural network architectures
│   │   ├── encoder.py       # Input encoders (Simple, Attention)
│   │   ├── fusion.py        # Fusion modules (Concat, Cross-Attention, Query)
│   │   ├── decoder.py       # Output decoders (LSTM, Transformer)
│   │   ├── factory.py       # Model factory for different configurations
│   │   └── loss.py          # Unified focal loss system
│   └── utils/               # Utilities (config, device setup, helpers)
│
├── training/                # 🧠 Training Pipeline & Experiments
│   ├── datasets/            # Dataset preparation (Boxy, multimodal)
│   │   ├── data_loaders.py  # HDF5-based efficient data loading
│   │   └── boxy_prep...     # Boxy dataset preprocessing
│   ├── yolo/                # YOLO training for vehicle detection
│   └── multimodal/          # Multimodal model training
│       ├── prepare_dataset.py    # H5 dataset preparation
│       ├── train_single.py       # Single architecture training
│       └── auto_annotate.py      # YOLO-based annotation
│
├── evaluation/              # 📊 Metrics, visualization & analysis
├── tests/                   # 📊 Development test scripts (access to ECU, sensor sync, etc.)
├── jobs/                    # 🖥️ SLURM job scripts for cluster computing
└── data/                    # 📁 Datasets, models, recordings
    └── ...                  # see below
```

### 📁 Data Directory Structure
```
data/                           # 💾 Data storage (gitignored)
├── cache/                      # Temporary processing cache
├── datasets/                   # Dataset storage
│   ├── processed/              # Processed datasets ready for training
│   │   ├── annotations/        # Annotated recording data
│   │   ├── boxy_yolo_n1/       # Boxy dataset in YOLO format (1 class)
│   │   └── nuimages_yolo/      # NuImages dataset in YOLO format
│   ├── raw/                    # Raw dataset files (Boxy, NuImages)
│   ├── boxy_labels.json        # Boxy dataset labels
│   └── dataset.yaml            # YOLO dataset configuration
├── models/                     # Trained model checkpoints
│   ├── yolo/                   # Best YOLO model checkpoints
│   └── multimodal/             # Multimodal model checkpoints
└── recordings/                 # Raw driving recordings
    └── YYYY-MM-DD_HH-MM-SS/    # Recording sessions (timestamped)
        ├── telemetry.csv       # OBD-II data with derived features
        ├── future_labels.csv   # Ground truth future labels for multimodal training (once generated)
        ├── annotations.csv     # YOLO detection results
        └── *.jpg               # Camera frames
```


## 🏗️ Model Architecture

### Modular Design for Systematic Evaluation

The system implements a modular architecture allowing systematic comparison of different component combinations:

**Input Encoders**
   - `SimpleInputEncoder`: Baseline with independent processing
   - `AttentionInputEncoder`: Advanced with self-attention mechanisms

**Fusion Modules**
   - `SimpleConcatenationFusion`: Efficient concatenation-based fusion
   - `CrossModalAttentionFusion`: Advanced cross-modal attention
   - `ObjectQueryFusion`: DETR-inspired learnable queries

**Output Decoders**
   - `LSTMOutputDecoder`: Sequential processing for temporal patterns
   - `TransformerOutputDecoder`: Parallel processing with attention

### Systematic Architecture Evaluation

**Modular Combinatorics for Ablation Studies:**
```python
# Systematic evaluation of all architecture combinations
architectures = {
    "encoders": ["simple", "attention"],
    "fusion": ["concat", "cross_attention", "query"],
    "decoders": ["lstm", "transformer"]
}
# → 2 × 3 × 2 = 12 architecture variants for comparison

# Example configuration
config = {
    "encoder_type": "attention",        # simple | attention
    "fusion_type": "cross_attention",   # concat | cross_attention | query  
    "decoder_type": "lstm",            # lstm | transformer
    "prediction_tasks": ["brake_1s", "brake_2s", "coast_1s", "coast_2s"],
    "embedding_dim": 64,
    "max_detections": 12,
    "max_seq_length": 20
}

# Create model via Factory Pattern
from src.model.factory import create_model_variant
model = create_model_variant(config)
```

## 🔬 Research Context

This project develops a **multimodal Machine Learning approach for real-time critical driving situation detection** as part of automotive AI research.

### Central Research Questions:
- How can camera and vehicle telemetry data be combined for improved predictions?
- What are the computational constraints of real-time processing on embedded hardware?
- How do different neural architectures perform under latency constraints?

### Technical Innovation:
- **Hardware-Aware Design**: Optimized for Raspberry Pi deployment (~80ms YOLO inference)
- **Modular Architecture**: Easily comparable encoder/fusion/decoder combinations
- **Real-World Data**: Custom dataset from German Autobahn driving
- **Synchronized Multimodal Capture**: Camera + OBD-II with <5ms synchronization

### Scientific Contributions:

#### **OBD-II Reverse Engineering - Proprietary Data Extraction**
- **Mode 22 Command Discovery**: Systematic exploration of proprietary diagnostic modes
- **Custom Command Implementation**: `BRAKE_SIGNAL = OBDCommand("BRAKE_SIGNAL", ..., b"223F9F", ...)`
- **Binary Signal Extraction**: Successful extraction of binary brake states from ECU data streams
- **Cross-Validation**: Physical brake pedal actuations as ground truth

#### **Systematic YOLO Hardware Performance Evaluation**
- **Model Variant Benchmarking**: YOLOv12n/s/m/l/x performance matrix under real constraints
- **Raspberry Pi 5 Baseline**: 80ms inference (YOLOv12n + OpenVINO) as hardware reality reference
- **Memory-Latency Trade-offs**: Systematic analysis of parameter count vs. inference speed
- **Production-Ready Metrics**: Realistic performance expectations for automotive deployment

#### **Multimodal Architecture Innovation**
- **Modular Design Philosophy**: Encoder/Fusion/Decoder combinatorics for systematic ablation studies
- **Cross-Modal Attention**: Telemetry-guided object detection relevance scoring
- **Temporal Sequence Modeling**: 1-2s prediction horizons with hardware-aware latency optimization

## 📊 Dataset & Training Status

### Dataset Statistics
- **Total Sequences**: 10,679 (from 12 recordings)
- **Brake Events**: 306 sequences (2.9% - extremely imbalanced)
- **Coast Events**: ~3,050 sequences (7.1% - moderately imbalanced)
- **Dataset Splits**: 70.4% train / 19.5% val / 10.1% test
- **Sequence Length**: 20 frames (10 seconds at 2Hz)

### Prediction Tasks
```python
PREDICTION_TASKS = [
    "brake_1s",   # Primary safety task (1.6% positive rate)
    "brake_2s",   # Secondary safety task  
    "coast_1s",   # Primary efficiency task (7.1% positive rate)
    "coast_2s"    # Secondary efficiency task
]
```

### Advanced Loss System
- **Focal Loss**: Addresses extreme class imbalance (1:63 ratio for brake events)
- **Task Weighting**: Safety-critical tasks prioritized over efficiency tasks
- **Multi-Task Learning**: Simultaneous prediction across multiple horizons

## 🛠️ Development & Cluster Computing

### Running SLURM Jobs
```bash
# Prepare Boxy dataset
sbatch jobs/boxy_prepare.slurm

# YOLO training
sbatch jobs/boxy_train.slurm

# YOLO validation
sbatch jobs/val_yolo.slurm

# Multimodal pipeline
sbatch jobs/multimodal_pipeline_full.slurm

# Train specific architecture
sbatch --export=ARCHITECTURE=simple_concat_transformer jobs/multimodal_train_single.slurm

# Evaluate models
sbatch jobs/multimodal_evaluate.slurm
```

### Adding New Features

1. **New Model Architecture**: Add to `src/model/`
2. **New Data Processing**: Add to `src/processing/`
3. **New Training Script**: Add to `training/` with corresponding SLURM job
4. **New Evaluation Metrics**: Add to `evaluation/`

## 📊 Performance

### Hardware Requirements
- **Recommended**: Raspberry Pi 5 (8GB RAM)
- **Development**: Modern laptop/desktop with GPU for training

### Latency Breakdown (Raspberry Pi 5 8GB + OpenVINO)
```
Component                   Latency       Memory      Details
─────────────────────────────────────────────────────────────────
Camera Capture             ~5ms          ~10MB       PiCamera2 1080p
YOLO Inference (YOLOv12n)  ~80ms        ~50MB        OpenVINO FP16/FP32
YOLO Inference (YOLOv12s)  ~200ms       ~150MB       (extrapolated)
Telemetry Processing       ~2ms          ~1MB        OBD-II + Features
Multimodal Model           ~5-15ms       ~10MB       Depending on architecture
Decision Making            ~1ms          ~1MB        Logic Layer
─────────────────────────────────────────────────────────────────
Total Pipeline (YOLOv12n)  ~95-110ms     ~72MB       Real-Time Capable
```

### Training Performance
- **GPU Memory**: 4-16GB recommended for full dataset
- **Batch Size**: 256 (optimized for Nvidia A100 GPUs)
- **Mixed Precision**: Enabled for faster training

## 🎯 Project Status

- ✅ **Data Recording System** (Camera + OBD-II)
- ✅ **YOLO Vehicle Detection** (Finetuned on Boxy dataset)
- ✅ **Feature Engineering** (Gear detection, brake force estimation)
- ✅ **Modular Model Architecture** (Encoder/Fusion/Decoder)
- ✅ **OBD-II Reverse Engineering** (Proprietary brake signal extraction)
- ✅ **Multimodal Training Pipeline** (Full H5-based pipeline with focal loss)
- ✅ **Dataset Preparation** (42,686 sequences from 12 recordings)
- ✅ **Systematic Architecture Evaluation** (12 architecture variants)
- ✅ **Model Evaluation & Selection** (In Progress)
- ⏳ **Sampling Strategy (Training)** (Planned)

**Note**: This is a research project focused on automotive AI safety systems. It is not intended for production use in vehicles without proper safety validation and certification.