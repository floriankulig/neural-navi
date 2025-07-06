# Training Pipeline Overview

## 🏗️ Architecture Overview

The training pipeline is organized into three main phases:

1. **Dataset Preparation** - YOLO finetuning and multimodal dataset creation
2. **Multimodal Training** - Modular architecture training with systematic evaluation
3. **Evaluation & Analysis** - Performance metrics and model comparison

## 📁 Directory Structure

```
training/
├── datasets/           # Dataset preparation and data loading
│   ├── data_loaders.py         # ⭐ PyTorch DataLoader with normalization
│   ├── annotation.py           # YOLO-based vehicle annotation
│   ├── boxy_preparation.py     # Boxy dataset → YOLO format
│   ├── nuimages_preparation.py # NuImages dataset → YOLO format  
│   └── nuimages_download.py    # NuImages dataset download
├── yolo/              # YOLO model training
│   ├── train_nuimages.py       # YOLO finetuning on NuImages
│   └── train_boxy.py           # ⭐ YOLO training on Boxy dataset
└── multimodal/        # Multimodal model training
    ├── prepare_dataset.py      # ⭐ H5 dataset preparation
    ├── train_single.py         # ⭐ Single architecture training
    ├── auto_annotate.py        # ⭐ YOLO-based auto-annotation
    ├── generate_labels.py      # ⭐ Future label generation
    └── data_download.py        # ⭐ SharePoint data download pipeline
```

## 🔄 Complete Training Pipeline

### Phase 1: Dataset Preparation

#### 1.1 YOLO Finetuning (`yolo/train_boxy.py`)
**Purpose**: Train YOLO model specialized for highway vehicle detection

```bash
# Download and prepare Boxy dataset
python training/datasets/boxy_preparation.py

# Train YOLO model (3 GPUs, 110 epochs)
python training/yolo/train_boxy.py
```

**Key Features**:
- Multi-GPU training (3x GPUs, batch_size=192)
- Highway-optimized augmentations (minimal perspective distortion)
- Single vehicle class for highway scenarios
- AdamW optimizer with cosine learning rate scheduling

**Outputs**: `data/models/yolo/yolo12n1.pt` (best checkpoint)

#### 1.2 Recording Auto-Annotation (`multimodal/auto_annotate.py`)
**Purpose**: Annotate recorded driving data with finetuned YOLO model

```bash
python training/multimodal/auto_annotate.py \
    --model yolo12n1.pt \
    --recordings-dir data/recordings \
    --force
```

**Key Features**:
- Recording-specific ROI cropping (camera angle compensation)
- Class-specific confidence thresholds
- Temporal synchronization with telemetry data
- Progress saving every 100 images

**Outputs**: `annotations.csv` per recording directory

#### 1.3 Future Label Generation (`multimodal/generate_labels.py`)
**Purpose**: Generate ground truth labels for brake/coast prediction

```bash
python training/multimodal/generate_labels.py \
    --coast-threshold 7.5 \
    --force
```

**Key Features**:
- Multi-horizon prediction (1s, 2s, 3s, 4s, 5s)
- Brake signal extraction from OBD-II data
- Coast detection via accelerator position threshold
- Temporal sequence validation

**Outputs**: `future_labels.csv` per recording directory

### Phase 2: Multimodal Dataset Creation

#### 2.1 Dataset Preparation (`multimodal/prepare_dataset.py`)
**Purpose**: Create HDF5 datasets for efficient multimodal training

```bash
python training/multimodal/prepare_dataset.py \
    --recordings-dir data/recordings \
    --output-dir data/datasets/multimodal \
    --sequence-length 20 \
    --sequence-stride 5
```

**Key Features**:
- Sliding window sequence extraction (20 frames, 5 frame stride)
- Temporal continuity validation (max 601ms gaps)
- Stratified recording-level splits (train/val/test)
- HDF5 compression for efficient storage

**Data Processing**:
- **Telemetry**: 10 features (4 continuous + 6 one-hot gear encoding)
- **Detections**: Max 12 detections per frame, 6 features per detection
- **Labels**: Binary brake/coast predictions for 4 horizons
- **Masks**: Valid detection indicators

**Outputs**: `train.h5`, `val.h5`, `test.h5`, `dataset_config.json`

### Phase 3: Multimodal Training

#### 3.1 Single Architecture Training (`multimodal/train_single.py`)
**Purpose**: Train individual architecture variants for systematic comparison

```bash
python training/multimodal/train_single.py \
    --architecture simple_concat_lstm \
    --data-dir data/datasets/multimodal \
    --output-dir data/models/multimodal
```

**Architecture Variants**:
```
Encoders:   simple | attention
Fusion:     concat | cross_attention | query  
Decoders:   lstm | transformer
→ 2 × 3 × 2 = 12 total combinations
```

**Training Configuration**:
- **Batch Size**: 256 (optimized for A100 GPUs)
- **Learning Rate**: 5e-5 with AdamW optimizer
- **Loss Function**: Unified focal loss with task weighting
- **Mixed Precision**: Enabled for memory efficiency
- **Early Stopping**: Patience=20 epochs

**Key Components**:
- **Warmup Training**: 8 epochs with reduced learning rate
- **Gradient Clipping**: Norm=1 for training stability  
- **Task Weighting**: Safety-critical tasks prioritized
- **Focal Loss**: Addresses extreme class imbalance (1:63 brake ratio)

## 📊 Data Loading & Normalization

### DataLoader Architecture (`datasets/data_loaders.py`)

**MultimodalDataset Features**:
- **Automatic Normalization**: Telemetry and detection features → [0,1]
- **Memory Loading**: Optional full dataset loading for speed
- **Gear Encoding**: Configurable one-hot vs continuous
- **Class Features**: Optional inclusion of detection class_id

**Normalization Strategy**:
```python
# Telemetry Normalization (speed, rpm, accel_pos, engine_load, gear_onehot)
telemetry_normalized = normalize_telemetry_features(
    telemetry, use_gear_onehot=True
)

# Detection Normalization (confidence, bbox coords, area)
detections_normalized = normalize_detection_features(
    detections, masks, img_width=1920, img_height=575
)
```

**Tensor Shapes**:
```
Input:
  telemetry_seq:  (batch_size, 20, 10)      # 20 frames, 10 features
  detection_seq:  (batch_size, 20, 12, 6)   # 20 frames, max 12 detections, 6 features
  detection_mask: (batch_size, 20, 12)      # Valid detection indicators

Output:
  targets: {
    "brake_1s": (batch_size,),    # Binary brake prediction
    "brake_2s": (batch_size,),
    "coast_1s": (batch_size,),    # Binary coast prediction  
    "coast_2s": (batch_size,)
  }
```

## ⚖️ Loss Function & Class Imbalance

### Unified Focal Loss System
**Purpose**: Handle extreme class imbalance (2.8% brake events)

```python
# Task-specific focal loss configuration
FOCAL_CONFIG = {
    "brake_1s": {"alpha": 0.2, "gamma": 3.0},    # Extreme focus on positives
    "brake_2s": {"alpha": 0.2, "gamma": 2.75},   # High focus  
    "coast_1s": {"alpha": 0.3, "gamma": 2.0},    # Moderate focus
    "coast_2s": {"alpha": 0.3, "gamma": 1.75}    # Moderate focus
}

# Task importance weighting
TASK_WEIGHTS = {
    "brake_1s": 1.0,    # Highest priority (safety-critical)
    "brake_2s": 0.95,   # High priority
    "coast_1s": 0.85,   # Medium priority (efficiency)
    "coast_2s": 0.8     # Lower priority
}
```

## 🎯 Model Architecture Factory

### Modular Design Pattern
```python
# Model configuration
config = {
    "encoder_type": "simple",           # simple | attention
    "fusion_type": "concat",            # concat | cross_attention | query
    "decoder_type": "lstm",             # lstm | transformer
    "telemetry_input_dim": 10,          # After one-hot gear encoding
    "detection_input_dim_per_box": 6,   # Without class_id feature  
    "embedding_dim": 64,
    "hidden_dim": 128,
    "attention_num_heads": 8,
    "decoder_num_layers": 4,
    "dropout_prob": 0.15,
    "prediction_tasks": ["brake_1s", "brake_2s", "coast_1s", "coast_2s"]
}

# Create model via factory
from model.factory import create_model_variant
model = create_model_variant(config)
```

### Feature Dimensions
```
Telemetry Features (10):
├── Continuous (4): SPEED, RPM, ACCELERATOR_POS_D, ENGINE_LOAD  
├── Gear One-Hot (6): gear_0, gear_1, gear_2, gear_3, gear_4, gear_5
└── Normalization: Min-max scaling to [0,1]

Detection Features (6 per box):
├── confidence: [0,1]
├── bbox: x1,y1,x2,y2 normalized by image dimensions
└── area: normalized by total image area
```

## 🔧 Training Best Practices

### Performance Optimization
- **DataLoader Workers**: 8 processes for I/O parallelization
- **Pin Memory**: Enabled for faster GPU transfer
- **Mixed Precision**: 16-bit floating point for memory efficiency
- **Gradient Accumulation**: Effective batch size scaling

### Debugging & Monitoring
- **NaN Detection**: Automatic gradient health monitoring
- **Progress Logging**: Detailed loss tracking per component
- **Checkpoint Saving**: Best model + latest checkpoint preservation
- **Training History**: JSON export for analysis

### Memory Management
```python
# Memory-efficient configurations
BATCH_SIZE = 256                    # A100 GPU optimized
NUM_WORKERS = 8                     # I/O parallelization
PIN_MEMORY = True                   # GPU transfer acceleration
LOAD_INTO_MEMORY = True             # Full dataset caching (if sufficient RAM)
```

## 🎛️ Configuration Management

### Key Feature Flags
```python
# Model behavior configuration
USE_GEAR_ONEHOT = True              # One-hot vs continuous gear encoding
INCLUDE_CLASS_FEATURES_IN_MODEL = False  # Exclude class_id from detections
MIXED_PRECISION = True              # 16-bit training
NORMALIZE_FEATURES = True           # Automatic [0,1] scaling

# Training hyperparameters  
SEQUENCE_LENGTH = 20                # 10 seconds at 2Hz
MAX_DETECTIONS_PER_FRAME = 12       # Hardware memory constraint
PREDICTION_TASKS = ["brake_1s", "brake_2s", "coast_1s", "coast_2s"]
```

## 🚀 Execution Examples

### Complete Pipeline Execution
```bash
# 1. Prepare Boxy dataset and train YOLO
python training/datasets/boxy_preparation.py
python training/yolo/train_boxy.py

# 2. Annotate recordings with trained YOLO
python training/multimodal/auto_annotate.py --model yolo12n1.pt --force

# 3. Generate future labels  
python training/multimodal/generate_labels.py --force

# 4. Create multimodal dataset
python training/multimodal/prepare_dataset.py

# 5. Train architecture variants
for arch in simple_concat_lstm simple_concat_transformer attention_cross_attention_lstm; do
    python training/multimodal/train_single.py --architecture $arch
done
```