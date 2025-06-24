"""
Central Feature Configuration
All feature dimensions and model settings in one place.
"""

# ==============================================
# YOLO VISION MODEL CONFIGURATION
# ==============================================

# Model paths
DEFAULT_VISION_MODEL = "boxyn1.pt"

# Class configuration
USE_MULTICLASS_DETECTION = (
    "boxyn3" in DEFAULT_VISION_MODEL
)  # Use multi-class model if ends with 'n.pt'

# Single-class configuration (highway scenario - recommended)
SINGLE_CLASS_CONFIG = {0: {"name": "vehicle", "confidence": 0.25}}

# Multi-class configuration (spatial awareness)
MULTI_CLASS_CONFIG = {
    0: {"name": "vehicle.left", "confidence": 0.25},
    1: {"name": "vehicle.front", "confidence": 0.25},
    2: {"name": "vehicle.right", "confidence": 0.25},
}

# YOLO processing settings
YOLO_IMG_SIZE = 704
YOLO_BASE_CONFIDENCE = 0.25
USE_CLASS_ONEHOT_ENCODING = (
    True  # Whether to one-hot encode YOLO classes in detection features
)


# ==============================================
# IMAGE AND ROI CONFIGURATION
# ==============================================

# Base image dimensions
IMAGE_WIDTH = 1920
IMAGE_HEIGHT = 1080

# Region of Interest for highway driving (x, y, width, height)
DEFAULT_IMAGE_ROI = (0, 320, 1920, 575)

# Autofocus region (x, y, width, height)
DEFAULT_IMAGE_FOCUS = (680, 393, 560, 424)

# Derived ROI dimensions
ROI_WIDTH = DEFAULT_IMAGE_ROI[2]
ROI_HEIGHT = DEFAULT_IMAGE_ROI[3]

# Image compression
IMAGE_COMPRESSION_QUALITY = 85

# Map to old config.py variable names
DEFAULT_RESOLUTION = (IMAGE_WIDTH, IMAGE_HEIGHT)
IDLE_RPM = 920  # For gear calculation
GEAR_RATIOS_PLOTTED = [9.5, 16.5, 27.5, 40.5, 53.5]  # For gear calculation
ACCELERATOR_POS_MIN = 14.12  # OBD calibration
ACCELERATOR_POS_MAX = 81.56  # OBD calibration

# ==============================================
# TELEMETRY FEATURE CONFIGURATION
# ==============================================

# Telemetry features (order matters!)
TELEMETRY_FEATURES = [
    "SPEED",
    "RPM",
    "ACCELERATOR_POS_D",
    "ENGINE_LOAD",
    # "BRAKE_SIGNAL",
    "GEAR",  # Always last, special handling
]

# Feature ranges for normalization (German highway driving)
TELEMETRY_RANGES = {
    "SPEED": (0.0, 150.0),  # km/h - German Autobahn max
    "RPM": (650.0, 4500.0),  # RPM - typical passenger car range
    "ACCELERATOR_POS_D": (0.0, 100.0),  # % - already normalized
    "ENGINE_LOAD": (0.0, 100.0),  # % - already normalized
}

# GEAR encoding configuration
USE_GEAR_ONEHOT = True  # Recommended for categorical data
GEAR_CLASSES = len([0] + GEAR_RATIOS_PLOTTED)  # 0=neutral, 1-5=gears

# Derived telemetry features (calculated from telemetry)
DERIVED_TELEMETRY_FEATURES = ["BRAKE_SIGNAL", "BRAKE_FORCE"]

# ==============================================
# DETECTION FEATURE CONFIGURATION
# ==============================================

# Base detection features (without class information)
BASE_DETECTION_FEATURES = [
    "confidence",
    "x1",
    "y1",
    "x2",
    "y2",  # Bounding box coordinates
    "area",
]

# Detection processing settings
INCLUDE_CLASS_FEATURES_IN_MODEL = USE_MULTICLASS_DETECTION and USE_CLASS_ONEHOT_ENCODING

# ==============================================
# MODEL ARCHITECTURE CONFIGURATION
# ==============================================


# Prediction tasks and horizons
PREDICTION_TASKS = ["brake_1s", "brake_2s", "coast_1s", "coast_2s"]

# ==============================================
# DATASET CONFIGURATION
# ==============================================

# Sequence processing
MAX_DETECTIONS_PER_FRAME = 12
SEQUENCE_LENGTH = 20  # frames (10 seconds at 2Hz)
SEQUENCE_STRIDE = 4  # frames (2 seconds overlap)

# Future label generation
SAMPLING_RATE_HZ = 2.0  # 2Hz sampling rate
SAMPLING_INTERVAL_S = 1.0 / SAMPLING_RATE_HZ  # 0.5 seconds per frame
COAST_THRESHOLD_PERCENT = 7.5  # Accelerator threshold for coasting detection
FUTURE_HORIZONS = [1, 2, 3, 4, 5]  # seconds

# Dataset splits
TRAIN_SPLIT = 0.75
VAL_SPLIT = 0.15
TEST_SPLIT = 0.10

# ==============================================
# CALCULATED DIMENSIONS (AUTO-COMPUTED)
# ==============================================

# Calculate continuous telemetry features (exclude GEAR)
CONTINUOUS_TELEMETRY_FEATURES = [f for f in TELEMETRY_FEATURES if f != "GEAR"]

# Calculate telemetry input dimension
CONTINUOUS_TELEMETRY_DIM = len(CONTINUOUS_TELEMETRY_FEATURES)
GEAR_DIM = GEAR_CLASSES if USE_GEAR_ONEHOT else 1
TELEMETRY_INPUT_DIM = CONTINUOUS_TELEMETRY_DIM + GEAR_DIM

# Calculate detection class dimensions
if USE_MULTICLASS_DETECTION:
    NUM_DETECTION_CLASSES = len(MULTI_CLASS_CONFIG)
    DETECTION_CLASS_CONFIG = MULTI_CLASS_CONFIG
else:
    NUM_DETECTION_CLASSES = len(SINGLE_CLASS_CONFIG)
    DETECTION_CLASS_CONFIG = SINGLE_CLASS_CONFIG

# Calculate detection input dimension per box
BASE_DETECTION_DIM = len(BASE_DETECTION_FEATURES)

if USE_CLASS_ONEHOT_ENCODING and INCLUDE_CLASS_FEATURES_IN_MODEL:
    # One-hot encoded classes + base features
    DETECTION_INPUT_DIM_PER_BOX = BASE_DETECTION_DIM + NUM_DETECTION_CLASSES
elif INCLUDE_CLASS_FEATURES_IN_MODEL:
    # Class ID + base features
    DETECTION_INPUT_DIM_PER_BOX = BASE_DETECTION_DIM + 1
else:
    # Only base features (no class information) - recommended for highway
    DETECTION_INPUT_DIM_PER_BOX = BASE_DETECTION_DIM


# ==============================================
# HELPER FUNCTIONS
# ==============================================


def get_telemetry_input_dim(use_gear_onehot=None):
    """Get telemetry input dimension with optional override."""
    if use_gear_onehot is None:
        use_gear_onehot = USE_GEAR_ONEHOT

    continuous_dim = len(CONTINUOUS_TELEMETRY_FEATURES)
    gear_dim = GEAR_CLASSES if use_gear_onehot else 1
    return continuous_dim + gear_dim


def get_detection_input_dim_per_box():
    """Get detection input dimension per box."""
    return DETECTION_INPUT_DIM_PER_BOX


def get_yolo_classes_config():
    """Get current YOLO classes configuration."""
    return DETECTION_CLASS_CONFIG


def get_roi_dimensions():
    """Get ROI dimensions (width, height)."""
    return ROI_WIDTH, ROI_HEIGHT


def get_image_dimensions():
    """Get full image dimensions (width, height)."""
    return IMAGE_WIDTH, IMAGE_HEIGHT


def get_roi_config():
    """Get ROI as (x, y, width, height) tuple."""
    return DEFAULT_IMAGE_ROI


def print_config_summary():
    """Print configuration summary for debugging."""
    print("🔧 Neural-Navi Feature Configuration Summary")
    print("=" * 60)

    print(f"📊 Calculated Dimensions:")
    print(
        f"   • Telemetry input: {TELEMETRY_INPUT_DIM} (continuous: {CONTINUOUS_TELEMETRY_DIM}, gear: {GEAR_DIM})"
    )
    print(f"   • Detection input per box: {DETECTION_INPUT_DIM_PER_BOX}")
    print(f"   • Detection classes: {NUM_DETECTION_CLASSES}")
    print(f"   • ROI: {ROI_WIDTH}x{ROI_HEIGHT}")

    print(f"\n🎯 Vision Model:")
    model_type = "Multi-class" if USE_MULTICLASS_DETECTION else "Single-class"
    print(f"   • Type: {model_type}")
    print(f"   • Classes: {NUM_DETECTION_CLASSES}")
    print(f"   • Default model: {DEFAULT_VISION_MODEL}")
    print(f"   • Include class features: {INCLUDE_CLASS_FEATURES_IN_MODEL}")

    print(f"\n📈 Telemetry Features:")
    print(f"   • Features: {TELEMETRY_FEATURES}")
    print(f"   • Continuous: {CONTINUOUS_TELEMETRY_FEATURES}")
    gear_encoding = "One-hot" if USE_GEAR_ONEHOT else "Continuous"
    print(f"   • GEAR encoding: {gear_encoding} ({GEAR_DIM} dims)")

    print(f"\n🎯 Detection Features:")
    print(f"   • Base features: {BASE_DETECTION_FEATURES}")
    print(f"   • Max detections: {MAX_DETECTIONS_PER_FRAME}")
    class_info = (
        "One-hot"
        if USE_CLASS_ONEHOT_ENCODING
        else "Class ID" if INCLUDE_CLASS_FEATURES_IN_MODEL else "None"
    )
    print(f"   • Class information: {class_info}")

    print(f"\n🏗️ Model Architecture:")
    print(f"   • Sequence length: {SEQUENCE_LENGTH}")
    print(f"   • Prediction tasks: {PREDICTION_TASKS}")

    print(f"\n📏 Image Configuration:")
    print(f"   • Full image: {IMAGE_WIDTH}x{IMAGE_HEIGHT}")
    print(f"   • ROI: {DEFAULT_IMAGE_ROI}")
    print(f"   • ROI size: {ROI_WIDTH}x{ROI_HEIGHT}")


if __name__ == "__main__":
    print("🧪 Testing Feature Configuration")
    print("=" * 50)

    print_config_summary()

    print(f"\n📊 Key Dimensions for Reference:")
    print(f"   • Telemetry input dim: {get_telemetry_input_dim()}")
    print(f"   • Detection input dim per box: {get_detection_input_dim_per_box()}")
    print(f"   • ROI dimensions: {get_roi_dimensions()}")
