# Neural-Navi

**Multimodal Machine Learning Approach for Real-Time Critical Driving Situation Recognition in Preventive Driver Assistance Systems**

Neural-Navi is a research project that combines camera data and vehicle telemetry to detect critical driving situations in real-time. The system predicts braking and coasting events 1-5 seconds in advance while considering realistic hardware constraints for automotive deployment.

## 🏆 Key Research Findings

### Scientific Contributions

**🔬 OBD-II Reverse Engineering**
- Successfully extracted proprietary brake signals via Mode 22 command discovery (`b"223F9F"`)
- Binary brake state extraction with physical pedal validation as ground truth

**📊 Extreme Imbalance Handling**  
- Focal loss configuration addresses 1:36 brake event ratio (2.8% positive rate)
- brake_1s: PR-AUC = 0.203 (7.25× improvement over random baseline 0.028)
- coast_1s: PR-AUC = 0.740 with 1:14 ratio (7.1% positive rate)

**🏗️ Modular Architecture Evaluation**
- Systematic comparison of 12 Encoder/Fusion/Decoder combinations
- **Transformer > LSTM**: Superior performance, stability, and generalization
- Simple concatenation fusion outperforms complex attention mechanisms in training stability

**⚡ Hardware-Aware Performance**
- **Pipeline Latency** (Raspberry Pi 5): YOLO 92.3% (123ms) + Multimodal 5.4% (7ms) = 133ms total
- **Mixed Precision**: 29% speedup for Transformer, degradation for LSTM
- **Memory**: <1.4GB total (within Raspberry Pi 5 8GB constraints)

**🧠 Temporal Attention Insights**
- **Recency Bias**: 0.36-0.44 attention weight on last 5 positions vs 0.14-0.18 on first 5
- **Position 19 Convergence**: All Transformer layers focus on final timestep
- **Local Context Priority**: Short-term dependencies more relevant than long-range for brake prediction

**📈 Prediction Horizon Analysis**
- **Dramatic Performance Drop**: 88% degradation from 1s to 5s prediction horizons
- **Optimal Horizons**: 1s and 2s provide best accuracy/utility trade-off for preventive systems

---

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

## 🔬 Research Context

This project develops a **multimodal Machine Learning approach for real-time critical driving situation detection** as part of automotive AI research.

### Central Research Questions:
- How can camera and vehicle telemetry data be combined for improved predictions?
- What are the computational constraints of real-time processing on embedded hardware?
- How do different neural architectures perform under latency constraints?

### Technical Innovation:
- **Hardware-Aware Design**: Optimized for Raspberry Pi deployment (~133ms total pipeline)
- **Modular Architecture**: Systematically comparable encoder/fusion/decoder combinations
- **Real-World Data**: Custom dataset from German Autobahn driving
- **Synchronized Multimodal Capture**: Camera + OBD-II with <5ms synchronization

For detailed information about specific components, see:
- **[Source Code Documentation](src/README.md)** - Model architectures, configuration system
- **[Training Pipeline](training/README.md)** - Dataset preparation, training workflows
- **[Evaluation Framework](evaluation/README.md)** - Scientific metrics, benchmarks, analysis tools

## 🏗️ Model Architecture

### Proven Working Configurations

**⚠️ Training Status**: Only `simple_concat_*` architectures are stable during training. Custom attention implementations have convergence issues.

```python
# Recommended stable configuration
BEST_CONFIG = {
    "encoder_type": "simple",        # Baseline with independent processing
    "fusion_type": "concat",         # Efficient concatenation-based fusion  
    "decoder_type": "transformer",   # Superior performance vs LSTM
    "embedding_dim": 64,
    "hidden_dim": 128,
    "dropout_prob": 0.15
}
```

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

# Create model via Factory Pattern
from src.model.factory import create_model_variant
model = create_model_variant(config)
```

## 📊 Dataset & Training Status

### Dataset Statistics
- **Total Sequences**: 42,686 (from 12 recordings, 43,104 frames)
- **Brake Events**: 1,200 sequences (2.8% - extreme imbalance, ratio 1:36)
- **Coast Events**: ~3,050 sequences (7.1% - moderate imbalance, ratio 1:14)
- **Dataset Splits**: 70.4% train / 19.5% val / 10.1% test (recording-based)
- **Sequence Length**: 20 frames (10 seconds at 2Hz)

### Prediction Tasks
```python
PREDICTION_TASKS = [
    "brake_1s",   # Primary safety task (2.8% positive rate)
    "brake_2s",   # Secondary safety task  
    "coast_1s",   # Primary efficiency task (7.1% positive rate)
    "coast_2s"    # Secondary efficiency task
]
```

### Advanced Loss System
- **Focal Loss**: Addresses extreme class imbalance with task-specific α/γ parameters
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

1. **New Model Architecture**: Add to `src/model/` (see [Source Documentation](src/README.md))
2. **New Data Processing**: Add to `src/processing/`
3. **New Training Script**: Add to `training/` with corresponding SLURM job (see [Training Documentation](training/README.md))
4. **New Evaluation Metrics**: Add to `evaluation/` (see [Evaluation Documentation](evaluation/README.md))

## 📊 Performance

### Hardware Requirements
- **Recommended**: Raspberry Pi 5 (8GB RAM)
- **Development**: Modern laptop/desktop with GPU for training

### Latency Breakdown (Raspberry Pi 5 8GB)
```
Component                   Latency       Memory      Details
─────────────────────────────────────────────────────────────────
Camera Capture             ~2.5ms        ~10MB       PiCamera2 1080p
YOLO Inference (YOLOv12n)  ~123ms        ~960MB      Hardware bottleneck
Telemetry Processing       ~2ms          ~1MB        OBD-II + Features
Multimodal Model           ~7ms          ~430MB      Transformer + Mixed Precision
Decision Making            ~1ms          ~1MB        Logic Layer
─────────────────────────────────────────────────────────────────
Total Pipeline             ~133ms        ~1.4GB      Real-Time Capable
```

**Key Insights:**
- YOLO dominates pipeline latency (92.3%) - primary optimization target
- Multimodal model contributes <6% to total latency
- Memory requirements well within Raspberry Pi 5 constraints

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
- ✅ **Systematic Architecture Evaluation** (Transformer > LSTM validated)
- ✅ **Scientific Evaluation Framework** (PR-AUC focus, hardware benchmarks)
- ⏳ **Sampling Strategy (Training)** (Planned)

---

**Note**: This is a research project focused on automotive AI safety systems. It is not intended for production use in vehicles without proper safety validation and certification.