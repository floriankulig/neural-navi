# Neural-Navi Source Code Documentation
## Model Architecture

### Core Design Pattern

```python
# Base architecture combining three modular components
class BrakingPredictionBaseModel(nn.Module):
    def __init__(self, input_encoder, fusion_module, output_decoder):
        self.input_encoder = input_encoder    # 1. Process telemetry + detections
        self.fusion_module = fusion_module    # 2. Combine modalities
        self.output_decoder = output_decoder  # 3. Generate predictions
```

### Working Architecture Combinations

**⚠️ Training Status**: Only `simple_concat_*` architectures are stable during training. Custom attention implementations have convergence issues.

```python
# Proven working configurations
WORKING_CONFIGS = {
    "simple_concat_lstm": {
        "encoder_type": "simple",
        "fusion_type": "concat", 
        "decoder_type": "lstm"
    },
    "simple_concat_transformer": {
        "encoder_type": "simple",
        "fusion_type": "concat",
        "decoder_type": "transformer"  
    }
}
```

### Input Processing

**Telemetry Features** (10 dimensions after normalization):
```python
# Continuous features (4D) + One-hot gear encoding (6D)
CONTINUOUS_FEATURES = ["SPEED", "RPM", "ACCELERATOR_POS_D", "ENGINE_LOAD"]
GEAR_ONEHOT = [0, 1, 2, 3, 4, 5]  # Neutral + Gears 1-5
```

**Detection Features** (6 dimensions per object):
```python
# Base detection format (no class information for highway scenarios)
DETECTION_FEATURES = ["confidence", "x1", "y1", "x2", "y2", "area"]
MAX_DETECTIONS_PER_FRAME = 12
```

### Component Implementations

#### Input Encoders
- **`SimpleInputEncoder`**: Basic embedding with optional positional encoding ✅
- **`AttentionInputEncoder`**: Self-attention for temporal/spatial patterns ❌ (training issues)

#### Fusion Modules  
- **`SimpleConcatenationFusion`**: Mean-pooled detections + telemetry concatenation ✅
- **`CrossModalAttentionFusion`**: Telemetry-guided detection attention ❌ (training issues)
- **`ObjectQueryFusion`**: DETR-inspired learnable queries ❌ (training issues)

#### Output Decoders
- **`LSTMOutputDecoder`**: Sequential processing with task-specific heads ✅
- **`TransformerOutputDecoder`**: Parallel processing with attention ✅

### Model Factory Usage

```python
from src.model.factory import create_model_variant
from src.utils.feature_config import get_telemetry_input_dim, get_detection_input_dim_per_box

config = {
    "encoder_type": "simple",
    "fusion_type": "concat", 
    "decoder_type": "transformer",
    "telemetry_input_dim": get_telemetry_input_dim(),      # 10
    "detection_input_dim_per_box": get_detection_input_dim_per_box(),  # 6
    "embedding_dim": 64,
    "hidden_dim": 128,
    "prediction_tasks": ["brake_1s", "brake_2s", "coast_1s", "coast_2s"]
}

model = create_model_variant(config)
```

## Configuration System

### Central Feature Configuration (`src/utils/feature_config.py`)

**Single source of truth** for all model dimensions and dataset parameters:

```python
# Automatically calculated dimensions
TELEMETRY_INPUT_DIM = 10        # 4 continuous + 6 gear_onehot  
DETECTION_INPUT_DIM_PER_BOX = 6 # confidence + bbox + area
MAX_DETECTIONS_PER_FRAME = 12
SEQUENCE_LENGTH = 20            # 10 seconds at 2Hz

# Prediction tasks
PREDICTION_TASKS = ["brake_1s", "brake_2s", "coast_1s", "coast_2s"]

# YOLO configuration  
DEFAULT_VISION_MODEL = "boxyn1hard.pt"
USE_MULTICLASS_DETECTION = False  # Single "vehicle" class for highway
```

### Hardware Configuration (`src/utils/device.py`)

```python
from src.utils.device import setup_device

# Automatic device selection with Apple Silicon MPS support
device = setup_device()  # Returns optimal device (MPS/CUDA/CPU)
```

## Key Components

### Loss System (`src/model/loss.py`)

Unified focal loss addressing extreme class imbalance:

```python
# Brake events: 1:63 ratio → aggressive focal parameters
# Coast events: 1:13 ratio → moderate focal parameters
FOCAL_CONFIG = {
    "brake_1s": {"alpha": 0.2, "gamma": 3.0},    # Extreme imbalance
    "brake_2s": {"alpha": 0.2, "gamma": 2.75},   
    "coast_1s": {"alpha": 0.3, "gamma": 2.0},    # Moderate imbalance  
    "coast_2s": {"alpha": 0.3, "gamma": 1.75}
}
```

### Feature Engineering (`src/processing/features/`)

**Gear Calculator** (`gear.py`):
```python
# Physics-based gear detection from speed/RPM ratio
gear = GearCalculator()(vehicle_speed, rpm, accelerator_pos, engine_load)
```

**Brake Force Calculator** (`brake_force.py`):
```python
# Deceleration-based brake force estimation
brake_force = BrakeForceCalculator(obd_connection)()
```

**OBD-II Custom Commands** (`custom_commands.py`):
```python
# Reverse-engineered brake signal extraction
BRAKE_SIGNAL = OBDCommand("BRAKE_SIGNAL", "...", b"223F9F", 0, decode_brake_signal)
```

### Data Processing

**Normalization** (`src/processing/normalization.py`):
```python
# Unified normalization for both modalities
normalized_telemetry = normalize_telemetry_features(telemetry_seq, use_gear_onehot=True)
normalized_detections = normalize_detection_features(detection_seq, detection_mask, 
                                                   img_width=1920, img_height=575)
```

### Recording System (`src/recording/`)

**Synchronized capture** at 2Hz with <5ms synchronization:
- **Camera** (`camera.py`): Raspberry Pi Camera2 + fallback to webcam
- **Telemetry** (`telemetry.py`): OBD-II with custom brake signal extraction

## Known Issues & Limitations

### Training Stability
- **Attention modules**: NaN/gradient explosion during training
- **Cross-modal attention**: Masked attention causing softmax issues  
- **Object queries**: Convergence problems with learnable parameters

### Recommended Architecture
```python
# Only proven stable configuration for training
RECOMMENDED_CONFIG = {
    "encoder_type": "simple",        # Stable embedding approach
    "fusion_type": "concat",         # Reliable fusion method
    "decoder_type": "transformer",   # Best performance in tests
    "embedding_dim": 64,
    "dropout_prob": 0.15
}
```

### Debug Tools
```python
# NaN detection during training
from src.utils.debug import NaNDebugger
debugger = NaNDebugger(model)
debugger.register_hooks()
```

## Quick Start Examples

### Model Creation & Training Setup
```python
from src.model.factory import create_model_variant
from src.model.loss import create_unified_focal_loss
from src.utils.feature_config import PREDICTION_TASKS

# Create working model
model = create_model_variant({
    "encoder_type": "simple",
    "fusion_type": "concat", 
    "decoder_type": "transformer",
    "telemetry_input_dim": 10,
    "detection_input_dim_per_box": 6,
    "prediction_tasks": PREDICTION_TASKS
})

# Setup loss function
loss_fn = create_unified_focal_loss()
```

### Data Pipeline
```python
from src.processing.normalization import normalize_telemetry_features, normalize_detection_features

# Normalize inputs
telemetry_norm = normalize_telemetry_features(telemetry_seq, use_gear_onehot=True)
detection_norm = normalize_detection_features(detection_seq, detection_mask)

# Forward pass
predictions = model(telemetry_norm, detection_norm, detection_mask)
losses = loss_fn(predictions, targets)
```

This modular design enables systematic architecture evaluation, but currently only simple concatenation-based models train reliably.