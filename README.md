### 🎯 Script Organization

All entry points are organized in root folder for clean execution:
```bash
make record          # → python record_drive.py --show-live 
make detect          # → python detect_vehicles.py
```

**Benefits:**
- Clean project structure
- Simple execution via Makefile
- Professional command-line tools after installation
- Easy to find and modify entry points# neural-navi

Neural Navi is an Intelligent Decision Support System (IDSS) that analyzes driving and camera data to provide insights and recommendations to drivers. The system is designed to help drivers make better decisions on the road and improve their driving skills.

## 🚀 Quick Start

### Installation

1. **Clone and navigate to the project:**
   ```bash
   git clone <repository-url>
   cd neural-navi
   ```

2. **Create virtual environment:**
   ```bash
   python -m venv venv --system-site-packages  # Raspberry Pi OS (keeps preinstalled Picamera2)
   # or
   python -m venv venv   # Windows/Mac
   ```

3. **Activate virtual environment:**
   ```bash
   source venv/bin/activate  # Linux/Mac
   # or
   .\venv\Scripts\activate   # Windows
   ```

4. **Install the package:**
   ```bash
   pip install -e .  # Development installation
   ```

### Quick Usage

#### Record driving data:
```bash
# Using make (recommended)
make record

# Using script directly  
python scripts/record_drive.py --show-live --with-logs

```

#### Train a model:
```bash
# Train YOLO model on Boxy dataset
make train-yolo-boxy
# or: python scripts/train_yolo_boxy.py

# Train multimodal model (when ready)
make train-multimodal
# or: python scripts/train_multimodal.py --config training/multimodal/configs/base_config.yaml
```

#### Detect vehicles in recordings:
```bash
# Using make
make detect

# Using script directly
python scripts/detect_vehicles.py --recordings data/recordings
```

## 📁 Project Structure

```
neural-navi/
├── scripts/                 # 🎯 Entry points - run everything from here
│   ├── record_drive.py      # Record driving sessions
│   ├── detect_vehicles.py   # Vehicle detection in recordings
│   ├── train_yolo_boxy.py   # Train YOLO on Boxy dataset
│   └── train_multimodal.py  # Train multimodal models
│
├── src/                     # 🔧 Core application code
│   ├── recording/           # Data capture (camera, telemetry, drive recorder)
│   ├── processing/          # Data preprocessing & feature engineering
│   │   ├── features/        # Derived features (gear, brake force)
│   │   └── detection/       # YOLO-based object detection utilities
│   ├── model/              # Neural network architectures
│   └── utils/               # Utilities (config, device setup, helpers)
│
├── training/                # 🧠 Training pipeline & experiments
│   ├── datasets/            # Dataset preparation scripts
│   ├── yolo/                # YOLO training (Boxy, NuImages)
│   ├── multimodal/          # Multimodal model training
│   └── slurm/               # SLURM scripts for cluster training
│
├── evaluation/              # 📊 Metrics, visualization & analysis
├── configs/                 # ⚙️ Global configuration files  
├── data/                    # 💾 Data storage (gitignored)
│   ├── recordings/          # Raw driving recordings
│   ├── datasets/            # Processed datasets
│   └── models/              # Trained model checkpoints
└── tests/                   # 🧪 Unit tests
```

## 🎮 Usage Guide

### Recording Driving Data

The DriveRecorder captures synchronized video and telemetry data:

```bash
# Basic recording
make record
# or: python scripts/record_drive.py

# With live preview and logging (default via make)
python scripts/record_drive.py --show-live --with-logs

# Custom capture interval (default: 0.5s = 2Hz)
python scripts/record_drive.py --interval 0.25
```

**Features:**
- Simultaneous camera and OBD-II data capture
- Automatic hardware detection (Raspberry Pi Camera vs USB webcam)
- Synchronized timestamps for all data
- Real-time feature calculation (gear estimation, brake force)

### Training Models

#### YOLO Vehicle Detection
```bash
# Prepare Boxy dataset
make prepare-boxy
# or: python scripts/prepare_boxy.py

# Train YOLO model
make train-yolo-boxy
# or: python scripts/train_yolo_boxy.py
```

#### Multimodal Braking Prediction
```bash
# Train with default config (when implemented)
make train-multimodal
# or: python scripts/train_multimodal.py

# With custom config
python scripts/train_multimodal.py --config training/multimodal/configs/experiment1.yaml

# Debug mode (fast training)
python scripts/train_multimodal.py --debug
```

### Vehicle Detection & Analysis

```bash
# Interactive vehicle detection viewer
make detect
# or: python scripts/detect_vehicles.py --recordings data/recordings

# Adjust confidence threshold
python scripts/detect_vehicles.py --recordings data/recordings --conf 0.3

# Use custom model
python scripts/detect_vehicles.py --model yolo_best.pt
```

## 🔧 Configuration

### Global Settings
Edit `configs/default.yaml` for global settings:
- Recording parameters (resolution, ROI, intervals)
- OBD-II settings and calibration values
- Vision model settings
- Hardware optimization

### Training Configurations
Training configs organized by purpose:
- `training/yolo/configs/` - YOLO-specific configs for different datasets
- `training/multimodal/configs/` - Multimodal model configurations
- `training/slurm/` - SLURM scripts for cluster training

### Hardware-Specific Settings
Configure for your hardware in `configs/hardware/`:
- `raspberry_pi.yaml` - Optimized for Raspberry Pi
- `desktop.yaml` - Desktop/laptop development

## 🧠 Model Architecture

### Multimodal Braking Prediction

The core innovation is a modular architecture combining:

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

### Configuration Example
```yaml
model:
  encoder_type: "simple"           # simple | attention
  fusion_type: "cross_attention"   # concat | cross_attention | query  
  decoder_type: "lstm"            # lstm | transformer
  prediction_horizons: [1, 3, 5]  # Predict braking 1s, 3s, 5s ahead
```

## 🔬 Research Context

This project develops a **multimodal Machine Learning approach for real-time critical driving situation detection** as part of automotive AI research.

### Key Research Questions:
- How can we combine camera and vehicle telemetry data for improved prediction?
- What are the computational constraints of real-time processing on embedded hardware?
- How do different neural architectures perform under latency constraints?

### Technical Innovation:
- **Hardware-Aware Design**: Optimized for Raspberry Pi deployment (~80ms YOLO inference)
- **Modular Architecture**: Easily comparable encoder/fusion/decoder combinations
- **Real-World Data**: Custom dataset from German Autobahn driving
- **Synchronized Multimodal Capture**: Camera + OBD-II with <5ms synchronization

## 🛠️ Development

### Running Tests
```bash
# Run all tests
python -m pytest tests/

# Run specific test category
python -m pytest tests/test_recording/
```

### Code Quality
```bash
# Format code
black src/ training/ scripts/

# Lint code  
flake8 src/ training/ scripts/

# Type checking
mypy src/
```

### Adding New Features

1. **New Model Architecture**: Add to `src/models/`
2. **New Data Processing**: Add to `src/processing/`
3. **New Training Script**: Add to `scripts/` with entry point and `training/` for logic
4. **New Evaluation Metrics**: Add to `evaluation/`

## 📊 Performance

### Hardware Requirements
- **Minimum**: Raspberry Pi 4 (4GB RAM)
- **Recommended**: Raspberry Pi 5 (8GB RAM)
- **Development**: Any modern laptop/desktop

### Latency Breakdown (Raspberry Pi 5)
```
Component               Latency
Camera Capture          ~5ms
YOLO Inference         ~80ms (YOLOv12n + OpenVINO)
Telemetry Processing    ~2ms
Multimodal Model       ~TBD
Decision Making         ~1ms
Total Pipeline         ~90ms+
```

### Memory Usage
- YOLO Model: ~50MB RAM
- Multimodal Model: ~TBD
- Recording Buffer: ~100MB

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Make your changes following the project structure
4. Add tests for new functionality
5. Commit your changes (`git commit -m 'Add amazing feature'`)
6. Push to the branch (`git push origin feature/amazing-feature`)
7. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🎯 Project Status

- ✅ Data Recording System (Camera + OBD-II)
- ✅ YOLO Vehicle Detection (Fine-tuned on Boxy dataset)
- ✅ Feature Engineering (Gear detection, Brake force estimation)
- ✅ Modular Model Architecture Design
- 🔄 Multimodal Training Pipeline (In Progress)
- ⏳ Real-time Inference Pipeline (Planned)
- ⏳ Hardware Optimization (Planned)

## 📧 Contact

For questions about this research project, please open an issue or contact the development team.

---

**Note**: This is a research project focused on automotive AI safety systems. It is not intended for production use in vehicles without proper safety validation and certification.