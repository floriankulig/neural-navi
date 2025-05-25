#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Data Download Pipeline for Multimodal Training
Downloads and extracts recording data from SharePoint for training pipeline.
"""

import os
import sys
import shutil
import subprocess
import zipfile
import time
import stat
from pathlib import Path
from tqdm import tqdm
import argparse
import logging
from typing import List, Optional

# Add project root to path for imports
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from src.utils.config import RECORDING_OUTPUT_PATH


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler("data_download.log"), logging.StreamHandler()],
)
logger = logging.getLogger(__name__)

# Default download configuration
DEFAULT_SHAREPOINT_URLS = [
    # TODO: Add actual SharePoint URLs here
    "https://dhbwstg-my.sharepoint.com/:u:/g/personal/inf22086_lehre_dhbw-stuttgart_de/ESnssdo9T8NGvVzpqhYbh7oBRox71m13Z2COR4YGwTDo0Q?e=RSBLKW&download=1",
    "https://dhbwstg-my.sharepoint.com/:u:/g/personal/inf22086_lehre_dhbw-stuttgart_de/EWI-HE0feuJMl43B6rvrIEgBaewIodBQl0vEOoyCDxCZHw?e=ytDH2KW&download=1",
]

DEFAULT_OUTPUT_DIR = RECORDING_OUTPUT_PATH
CHUNK_SIZE = 8192  # 8KB chunks for download progress


class DataDownloader:
    """
    Handles downloading and extraction of training data from SharePoint.
    """

    def __init__(self, output_dir: str = DEFAULT_OUTPUT_DIR):
        self.output_dir = Path(output_dir)
        self.temp_dir = self.output_dir / "temp_downloads"
        self.setup_directories()

    def setup_directories(self):
        """Create necessary directories for download and extraction."""
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.temp_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"📁 Output directory: {self.output_dir}")
        logger.info(f"📁 Temporary directory: {self.temp_dir}")

    def download_file(
        self, url: str, filename: Optional[str] = None, max_retries: int = 3
    ) -> Optional[Path]:
        """
        Download a file from URL with progress bar and retry logic.

        Args:
            url: URL to download from
            filename: Local filename (optional, derived from URL if not provided)
            max_retries: Maximum number of retry attempts

        Returns:
            Path to downloaded file or None if failed
        """
        if filename is None:
            filename = url.split("/")[-1]
            if not filename.endswith(".zip"):
                filename += ".zip"

        file_path = self.temp_dir / filename

        # Check if file already exists and is valid
        if file_path.exists() and self._is_valid_zip(file_path):
            logger.info(f"✅ File already exists and is valid: {filename}")
            return file_path

        # Remove invalid file if exists
        if file_path.exists():
            file_path.unlink()
            logger.warning(f"🗑️ Removed invalid file: {filename}")

        # Download with retries
        for attempt in range(max_retries):
            try:
                logger.info(
                    f"📥 Downloading {filename} (attempt {attempt + 1}/{max_retries})"
                )

                # Use wget for robust downloading
                cmd = [
                    "wget",
                    "--progress=bar:force",
                    "--tries=3",
                    "--continue",  # Resume partial downloads
                    "--output-document",
                    str(file_path),
                    url,
                ]

                result = subprocess.run(cmd, check=True, capture_output=True, text=True)

                # Validate downloaded file
                if self._is_valid_zip(file_path):
                    logger.info(f"✅ Successfully downloaded: {filename}")
                    return file_path
                else:
                    logger.error(f"❌ Downloaded file is corrupted: {filename}")
                    file_path.unlink()

            except subprocess.CalledProcessError as e:
                logger.error(f"❌ Download failed (attempt {attempt + 1}): {e}")
                if file_path.exists():
                    file_path.unlink()

                if attempt < max_retries - 1:
                    wait_time = 2**attempt  # Exponential backoff
                    logger.info(f"⏳ Waiting {wait_time}s before retry...")
                    time.sleep(wait_time)

        logger.error(f"❌ Failed to download {filename} after {max_retries} attempts")
        return None

    def _is_valid_zip(self, file_path: Path) -> bool:
        """Check if a file is a valid ZIP archive."""
        try:
            with zipfile.ZipFile(file_path, "r") as zip_ref:
                # Test the ZIP file integrity
                zip_ref.testzip()
                # Check if ZIP has content
                return len(zip_ref.namelist()) > 0
        except (zipfile.BadZipFile, zipfile.LargeZipFile, FileNotFoundError):
            return False

    def extract_zip(self, zip_path: Path, extract_to: Optional[Path] = None) -> bool:
        """
        Extract ZIP file with progress bar and error handling.

        Args:
            zip_path: Path to ZIP file
            extract_to: Extraction directory (default: output_dir)

        Returns:
            True if successful, False otherwise
        """
        if extract_to is None:
            extract_to = self.output_dir

        extract_to.mkdir(parents=True, exist_ok=True)

        try:
            with zipfile.ZipFile(zip_path, "r") as zip_ref:
                members = zip_ref.namelist()

                logger.info(f"📦 Extracting {zip_path.name} ({len(members)} files)")

                # Extract with progress bar
                for member in tqdm(members, desc="Extracting"):
                    try:
                        zip_ref.extract(member, extract_to)
                    except (PermissionError, OSError) as e:
                        logger.warning(f"⚠️ Could not extract {member}: {e}")
                        # Try to fix permissions and retry
                        target_path = extract_to / member
                        if target_path.exists():
                            try:
                                target_path.chmod(stat.S_IWUSR | stat.S_IRUSR)
                                zip_ref.extract(member, extract_to)
                                logger.info(
                                    f"✅ Fixed permissions and extracted: {member}"
                                )
                            except Exception as e2:
                                logger.error(f"❌ Still cannot extract {member}: {e2}")

                logger.info(f"✅ Successfully extracted {zip_path.name}")
                return True

        except zipfile.BadZipFile as e:
            logger.error(f"❌ Corrupted ZIP file {zip_path.name}: {e}")
            return False
        except Exception as e:
            logger.error(f"❌ Extraction failed for {zip_path.name}: {e}")
            return False

    def cleanup_temp_files(self):
        """Remove temporary download files."""
        if self.temp_dir.exists():
            shutil.rmtree(self.temp_dir)
            logger.info(f"🧹 Cleaned up temporary files: {self.temp_dir}")

    def validate_extracted_data(self) -> bool:
        """
        Validate that extracted data has expected structure.

        Returns:
            True if data structure is valid
        """
        logger.info("🔍 Validating extracted data structure...")

        # Check for timestamp directories
        timestamp_dirs = [
            d for d in self.output_dir.iterdir() if d.is_dir() and "-" in d.name
        ]

        if not timestamp_dirs:
            logger.error("❌ No timestamp directories found in extracted data")
            return False

        valid_dirs = 0
        total_images = 0

        for timestamp_dir in timestamp_dirs:
            telemetry_file = timestamp_dir / "telemetry.csv"
            image_files = list(timestamp_dir.glob("*.jpg"))

            if telemetry_file.exists() and image_files:
                valid_dirs += 1
                total_images += len(image_files)
                logger.debug(
                    f"✅ Valid: {timestamp_dir.name} ({len(image_files)} images)"
                )
            else:
                logger.warning(f"⚠️ Invalid structure: {timestamp_dir.name}")

        logger.info(f"📊 Validation results:")
        logger.info(f"  - Valid directories: {valid_dirs}/{len(timestamp_dirs)}")
        logger.info(f"  - Total images: {total_images}")

        if valid_dirs == 0:
            logger.error("❌ No valid recording directories found")
            return False

        if valid_dirs < len(timestamp_dirs):
            logger.warning(f"⚠️ Some directories have invalid structure")

        logger.info("✅ Data validation completed")
        return True

    def download_and_extract_all(self, urls: List[str]) -> bool:
        """
        Download and extract all files from given URLs.

        Args:
            urls: List of SharePoint URLs to download

        Returns:
            True if all downloads and extractions were successful
        """
        if not urls:
            logger.error("❌ No URLs provided for download")
            return False

        logger.info(f"🚀 Starting download pipeline for {len(urls)} files")

        downloaded_files = []

        # Download all files
        for i, url in enumerate(urls, 1):
            logger.info(f"📥 Processing file {i}/{len(urls)}")

            file_path = self.download_file(url)
            if file_path:
                downloaded_files.append(file_path)
            else:
                logger.error(f"❌ Failed to download file {i}")
                return False

        logger.info(f"✅ Successfully downloaded {len(downloaded_files)} files")

        # Extract all files
        extraction_success = True
        for file_path in downloaded_files:
            if not self.extract_zip(file_path):
                extraction_success = False

        if not extraction_success:
            logger.error("❌ Some extractions failed")
            return False

        # Validate extracted data
        if not self.validate_extracted_data():
            logger.error("❌ Data validation failed")
            return False

        # Cleanup temporary files
        self.cleanup_temp_files()

        logger.info("🎉 Download pipeline completed successfully!")
        return True


def main():
    """Main entry point for data download pipeline."""
    parser = argparse.ArgumentParser(
        description="Download training data from SharePoint"
    )
    parser.add_argument(
        "--urls",
        nargs="+",
        default=DEFAULT_SHAREPOINT_URLS,
        help="SharePoint URLs to download (space-separated)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=DEFAULT_OUTPUT_DIR,
        help="Output directory for extracted data",
    )
    parser.add_argument(
        "--no-cleanup", action="store_true", help="Keep temporary download files"
    )
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Only validate existing data without downloading",
    )

    args = parser.parse_args()

    # Initialize downloader
    downloader = DataDownloader(args.output_dir)

    if args.validate_only:
        logger.info("🔍 Running validation only...")
        success = downloader.validate_extracted_data()
        sys.exit(0 if success else 1)

    # Check if URLs are provided
    if not args.urls or args.urls == DEFAULT_SHAREPOINT_URLS:
        logger.error("❌ No valid SharePoint URLs provided.")
        logger.error(
            "   Please provide URLs with --urls or update DEFAULT_SHAREPOINT_URLS"
        )
        logger.error(
            "   Example: python data_download.py --urls https://sharepoint.com/file1.zip https://sharepoint.com/file2.zip"
        )
        sys.exit(1)

    # Run download pipeline
    success = downloader.download_and_extract_all(args.urls)

    # Cleanup based on flag
    if not args.no_cleanup:
        downloader.cleanup_temp_files()

    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
