import os
import platform
import sys

# Set environment variables for Apple Silicon before torch import
is_mac = platform.system() == "Darwin"
is_apple_silicon = "arm" in platform.processor().lower()

if is_mac and is_apple_silicon:
    # Set MPS environment variables before torch import
    os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
    os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.0"  # Reduces memory usage

    # Check if torch was already imported and needs reloading
    if "torch" in sys.modules:
        print("🔄 Reloading PyTorch modules to apply MPS settings...")
        # Remove all torch-related modules to force reimport
        for module_name in list(sys.modules.keys()):
            if module_name.startswith("torch"):
                sys.modules.pop(module_name, None)

# Now import torch with environment variables properly set
import torch


def setup_device():
    """
    Configure the device for optimal performance on Apple Silicon (M3) and Raspberry Pi 5 8GB.
    Returns the device to use for inference or training.
    """
    # Check if running on macOS with Apple Silicon
    is_mac = platform.system() == "Darwin"
    is_apple_silicon = "arm" in platform.processor().lower()

    if is_mac and is_apple_silicon:
        print("🍎 Running on Apple Silicon. Enabling MPS acceleration.")
        # Enable Metal Performance Shaders if available
        if torch.backends.mps.is_available():
            device = torch.device("mps")

            # Enable TensorFloat32 equivalent for MPS
            # This provides better performance with slightly reduced precision
            torch.set_float32_matmul_precision("high")

            # Verify MPS fallback is enabled
            if os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK") == "1":
                print("✅ MPS fallback is enabled for unsupported operations.")

            print("🚀 Using MPS device for acceleration.")
        else:
            device = torch.device("cpu")
            print("❌ MPS not available, using CPU instead.")
    else:
        # Use CUDA if available, otherwise CPU
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {device}")

    return device
