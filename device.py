import os
import platform
import torch


def setup_device():
    """
    Configure the device for optimal performance on Apple Silicon (M3).
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

            # Optimizations for Apple Silicon
            # Enable MPS fallback for operations not supported by MPS
            os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

            # Enable memory optimization for Apple Silicon
            os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = (
                "0.0"  # Reduces memory usage
            )

            # Enable TensorFloat32 equivalent for MPS
            # This provides better performance with slightly reduced precision
            torch.set_float32_matmul_precision("high")

            print("🚀 Using MPS device for acceleration.")
        else:
            device = torch.device("cpu")
            print("❌ MPS not available, using CPU instead.")
    else:
        # Use CUDA if available, otherwise CPU
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {device}")

    return device
