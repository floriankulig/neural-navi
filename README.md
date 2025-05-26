# Neural-Navi

**Multimodal Machine Learning Approach for Real-Time Critical Driving Situation Recognition in Preventive Driver Assistance Systems**

Neural-Navi is a research project that combines camera data and vehicle telemetry to detect critical driving situations in real-time. The system aims to predict braking events 1-5 seconds in advance while considering realistic hardware constraints.

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
python record_drive.py --show-live --with-logs
```

#### Detect vehicles in recordings:
```bash
# Interactive viewer
make detect
# or: python detect_vehicles.py --recordings data/recordings
```

#### Train models:
```bash
# YOLO on Boxy dataset (via SLURM)
sbatch jobs/boxy_train.slurm

# Multimodal model (planned)
python training/multimodal/train_multimodal.py
```

## 📁 Project Structure

```
neural-navi/
├── record_drive.py          # 🎯 Driving data recording
├── detect_vehicles.py       # 🔍 Vehicle detection in recordings
│
├── src/                     # 🔧 Core Application Code
│   ├── recording/           # Data acquisition (camera, telemetry, DriveRecorder)
│   ├── processing/          # Data preprocessing & feature engineering
│   │   ├── features/        # Derived features (gear, brake force)
│   │   └── detection/       # YOLO-based object detection
│   ├── model/               # Neural network architectures
│   │   ├── encoder.py       # Input encoders (Simple, Attention)
│   │   ├── fusion.py        # Fusion modules (Concat, Cross-Attention, Query)
│   │   ├── decoder.py       # Output decoders (LSTM, Transformer)
│   │   └── factory.py       # Model factory for different configurations
│   └── utils/               # Utilities (config, device setup, helpers)
│
├── training/                # 🧠 Training Pipeline & Experiments
│   ├── datasets/            # Dataset preparation (Boxy, NuImages)
│   ├── yolo/                # YOLO training for vehicle detection
│   └── multimodal/          # Multimodal model training
│
├── evaluation/              # 📊 Metrics, visualization & analysis
├── jobs/                    # ⚡ SLURM scripts for cluster training
├── configs/                 # ⚙️ Global configuration files
├── data/                    # 💾 Data storage (gitignored)
│   └── ... see below
└── Makefile                 # 🛠️ Development commands
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
│   ├── raw/                    # Raw dataset files
│   ├── boxy_labels.json        # Boxy dataset labels
│   ├── boxy_labels_val.json    # Boxy validation labels
│   └── dataset.yaml            # YOLO dataset configuration
├── models/                     # Trained model checkpoints
│   ├── yolo_best.pt            # Best YOLO model checkpoint
│   └── multimodal_*.pt         # Multimodal model checkpoints
└── recordings/                 # Raw driving recordings
    └── YYYY-MM-DD_HH-MM-SS/    # Recording sessions (timestamped)
        ├── telemetry.csv       # OBD-II data with derived features
        ├── annotations.csv     # YOLO detection results
        └── *.jpg               # Camera frames
```

## 🎮 Usage Guide

### Recording Driving Data

The DriveRecorder captures synchronized video and telemetry data:

```bash
# Basic recording
make record
# or: python record_drive.py

# With live preview and logging (default via make)
python record_drive.py --show-live --with-logs

# Custom capture interval (default: 0.5s = 2Hz)
python record_drive.py --interval 0.25
```

**Features:**
- Simultaneous camera and OBD-II data capture
- Automatic hardware detection (Raspberry Pi Camera vs USB webcam)
- Synchronized timestamps for all data
- Real-time feature calculation (gear detection, brake force estimation)

### Training Models

#### YOLO Vehicle Detection
```bash
# Prepare Boxy dataset
python training/datasets/boxy_preparation.py

# YOLO training (SLURM)
sbatch jobs/boxy_train.slurm

# Local training (development)
python training/yolo/train_boxy.py
```

#### Multimodal Brake Prediction
```bash
# Training with default config (planned)
python training/multimodal/train_multimodal.py

# With custom config (planned)
python training/multimodal/train_multimodal.py --config configs/experiment1.yaml
```

### Vehicle Detection & Analysis

```bash
# Interactive vehicle detection viewer
make detect
# or: python detect_vehicles.py --recordings data/recordings

# Adjust confidence threshold
python detect_vehicles.py --recordings data/recordings --conf 0.3

# Use custom model
python detect_vehicles.py --model yolo_best.pt
```

## 🔧 Configuration

### Global Settings
Edit `src/utils/config.py` for global settings:
- Recording parameters (resolution, ROI, intervals)
- OBD-II settings and calibration values
- Vision model settings
- Hardware optimizations

### Training Configurations
Training configs organized by purpose:
- `training/yolo/configs/` - YOLO-specific configs for different datasets
- `training/multimodal/configs/` - Multimodal model configurations
- `jobs/` - SLURM scripts for cluster training

## 🧠 Model Architecture

### Multimodal Brake Prediction

The core innovation is a modular architecture combining the following components:

1. **Input Encoders**
   - `SimpleInputEncoder`: Baseline with independent processing
   - `AttentionInputEncoder`: Advanced with self-attention mechanisms

2. **Fusion Modules**
   - `SimpleConcatenationFusion`: Efficient concatenation-based fusion
   - `CrossModalAttentionFusion`: Advanced cross-modal attention
   - `ObjectQueryFusion`: DETR-inspired learnable queries

3. **Output Decoders**
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
    "prediction_tasks": ["brake_1s", "brake_2s"],
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
- **Temporal Sequence Modeling**: 1-5s prediction horizons with hardware-aware latency optimization

## 🛠️ Development & Cluster Computing

### Running SLURM Jobs
```bash
# Prepare Boxy dataset
sbatch jobs/boxy_prepare.slurm

# YOLO training
sbatch jobs/boxy_train.slurm

# YOLO validation
sbatch jobs/val_yolo.slurm

# Create visualizations
sbatch jobs/boxy_visualizer.slurm
```

### Adding New Features

1. **New Model Architecture**: Add to `src/model/`
2. **New Data Processing**: Add to `src/processing/`
3. **New Training Script**: Add to `training/` with corresponding SLURM job
4. **New Evaluation Metrics**: Add to `evaluation/`

## 📊 Performance

### Hardware Requirements
- **Recommended**: Raspberry Pi 5 (8GB RAM)
- **Development**: Modern laptop/desktop (scripts optimized for 🍎 Apple Silicon Chips)

### Latency Breakdown (Raspberry Pi 5 8GB + OpenVINO)
```
Component                   Latency       Memory      Details
─────────────────────────────────────────────────────────────────
Camera Capture             ~5ms          ~10MB       PiCamera2 1080p
YOLO Inference (YOLOv12n)  ~80ms        ~50MB        OpenVINO FP16/FP32
YOLO Inference (YOLOv12s)  ~200ms       ~150MB       (extrapolated)
Telemetry Processing       ~2ms          ~1MB        OBD-II + Features
Multimodal Model           ~TBD          ~TBD        In Development
Decision Making            ~1ms          ~1MB        Logic Layer
─────────────────────────────────────────────────────────────────
Total Pipeline (YOLOv12n)  ~90ms+        ~62MB       Real-Time Capable
```

## 🎯 Project Status

- ✅ **Data Recording System** (Camera + OBD-II)
- ✅ **YOLO Vehicle Detection** (Finetuned on Boxy dataset)
- ✅ **Feature Engineering** (Gear detection, brake force estimation)
- ✅ **Modular Model Architecture** (Encoder/Fusion/Decoder)
- ✅ **OBD-II Reverse Engineering** (Proprietary brake signal extraction)
- 🔄 **Multimodal Training Pipeline** (In Progress)
- ⏳ **Real-time Inference Pipeline** (Planned)
- ⏳ **Hardware Optimization** (Planned)

## 📄 License

This project is licensed under the MIT License - see [LICENSE](LICENSE) file for details.

---

**Note**: This is a research project focused on automotive AI safety systems. It is not intended for production use in vehicles without proper safety validation and certification.