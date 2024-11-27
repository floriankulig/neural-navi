import numpy as np
import os
import zipfile
from pathlib import Path
from typing import Union, List


def numeric_or_none(response):
    """Unwraps the value from the OBD response or returns None if the response is None."""
    value = (
        response.value.magnitude
        if response is not None and response.value is not None
        else None
    )
    return round2(value)


def normalize(response, input_range, output_range):
    """Unwraps the value from the OBD response and maps it to the output range."""
    value = (
        np.interp(response.value.magnitude, input_range, output_range)
        if response is not None and response.value is not None
        else None
    )
    return round2(value)


def round2(value):
    return round(value, 2) if value is not None else None


def zip_folder(
    folder_path: Union[str, Path],
    output_path: Union[str, Path] = None,
    ignore_patterns: List[str] = None,
    compression: int = zipfile.ZIP_DEFLATED,
) -> Path:
    """
    Compresses a folder into a ZIP file.

    Args:
        folder_path: Path to the folder to be zipped
        output_path: Output path for the ZIP file (optional)
        ignore_patterns: List of file patterns to be ignored (optional)
        compression: Compression method (default: ZIP_DEFLATED)

    Returns:
        Path: Path to the created ZIP file
    """
    folder_path = Path(folder_path)

    # If no output path is specified, create ZIP file next to the folder
    if output_path is None:
        output_path = folder_path.parent / f"{folder_path.name}.zip"
    else:
        output_path = Path(output_path)

    # Default patterns for files to ignore
    if ignore_patterns is None:
        ignore_patterns = [".DS_Store", "__pycache__", "*.pyc"]

    with zipfile.ZipFile(output_path, "w", compression=compression) as zip_file:
        # Iterate through all files and subfolders
        for root, dirs, files in os.walk(folder_path):
            # Create relative paths for the ZIP file
            rel_path = os.path.relpath(root, folder_path)

            for file in files:
                # Check if file should be ignored
                skip_file = any(
                    file.endswith(pattern.strip("*")) or file == pattern
                    for pattern in ignore_patterns
                )

                if not skip_file:
                    file_path = os.path.join(root, file)
                    arc_path = os.path.join(rel_path, file)

                    # Add file to ZIP
                    zip_file.write(file_path, arc_path)

    return output_path
