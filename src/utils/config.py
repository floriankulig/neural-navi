"""
Configuration constants for the Drive Recorder project.
"""

# Paths
RECORDING_OUTPUT_PATH = "data/recordings"

# Time formats
TIME_FORMAT_FILES = "%Y-%m-%d_%H-%M-%S"
TIME_FORMAT_LOG = "%Y-%m-%d %H-%M-%S-%f"

# Capture settings
CAPTURE_INTERVAL = 0.5  # 2 Hz
DEFAULT_RESOLUTION = (1920, 1080)  # Full HD

# Region of interest for image processing
# THIS SHOULD ONLY BE USED FOR FHD IMAGES (1920x1080)
DEFAULT_IMAGE_ROI = (0, 320, 1920, 580)

# Region of interest for Autofocus
# THIS SHOULD ONLY BE USED FOR FHD IMAGES (1920x1080)
DEFAULT_IMAGE_FOCUS = (680, 393, 560, 424)

# Image compression settings
IMAGE_COMPRESSION_QUALITY = 85  # JPEG-Qualität (0-100)

# OBD settings
# Capped values as ECU doesn't deliver this value from 0 to 100
ACCERLERATOR_POS_MIN = 14.12
ACCERLERATOR_POS_MAX = 81.56

# Gear calculation
IDLE_RPM = 920
GEAR_RATIOS_PLOTTED = [9.5, 16.5, 27.5, 40.5, 53.5]

# Feature names for logging
DERIVED_VALUES = ["GEAR", "BRAKE_FORCE", "PRE_BRAKING", "WHILE_BRAKING"]
