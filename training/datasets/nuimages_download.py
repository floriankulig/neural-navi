import os
import sys
import shutil
import argparse
import time
import stat
from pathlib import Path
import requests
import tarfile
from tqdm import tqdm

def handle_error(func, path, exc_info):
    """
    Error handler for shutil.rmtree to handle permission errors
    """
    print(f"Warning: Permission error for {path}")
    # Make file writeable and try again
    if not os.access(path, os.W_OK):
        os.chmod(path, stat.S_IWUSR)
        func(path)

def clean_directory(directory, verbose=True):
    """
    Safely remove contents of a directory
    
    Args:
        directory: Path to clean
        verbose: Whether to print progress
    """
    if not os.path.exists(directory):
        if verbose:
            print(f"Directory {directory} does not exist, nothing to clean")
        return
    
    if verbose:
        print(f"Cleaning directory: {directory}")
    
    try:
        # Attempt to remove directory and its contents
        shutil.rmtree(directory, onerror=handle_error)
        # Recreate the empty directory
        os.makedirs(directory, exist_ok=True)
        if verbose:
            print(f"✓ Directory {directory} cleaned successfully")
    except Exception as e:
        print(f"Error cleaning directory {directory}: {e}")
        return False
    
    return True

def download_file(url, filename, force_download=False):
    """
    Downloads a file with progress bar
    
    Args:
        url: URL to download from
        filename: Path to save the file to
        force_download: Whether to download even if file exists
    """
    if os.path.exists(filename) and not force_download:
        print(f"✓ {filename} already exists, skipping download")
        return
    
    print(f"Downloading {filename}...")
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()  # Raise exception for HTTP errors
        
        total_size = int(response.headers.get('content-length', 0))
        
        with open(filename, 'wb') as file, tqdm(
            desc=filename,
            total=total_size,
            unit='B',
            unit_scale=True,
            unit_divisor=1024,
        ) as bar:
            for data in response.iter_content(chunk_size=8192):
                size = file.write(data)
                bar.update(size)
        print(f"✓ Downloaded {filename}")
    except requests.exceptions.RequestException as e:
        print(f"Error downloading {url}: {e}")
        if os.path.exists(filename):
            print(f"Removing incomplete download: {filename}")
            os.remove(filename)
        return False
    
    return True

def extract_tarfile(filename, extract_dir, force_extract=False):
    """
    Extracts a tar file with progress bar
    
    Args:
        filename: Path to the tar file
        extract_dir: Directory to extract to
        force_extract: Whether to overwrite existing files
    """
    if not os.path.exists(filename):
        print(f"Error: {filename} does not exist")
        return False
    
    print(f"Extracting {filename} to {extract_dir}...")
    
    # Create extract directory if it doesn't exist
    os.makedirs(extract_dir, exist_ok=True)
    
    try:
        with tarfile.open(filename, 'r:gz') as tar:
            members = tar.getmembers()
            
            # Filter members if not force_extract to skip existing files
            if not force_extract:
                extract_members = []
                for member in members:
                    target_path = os.path.join(extract_dir, member.name)
                    if not os.path.exists(target_path):
                        extract_members.append(member)
                
                if len(extract_members) < len(members):
                    print(f"Skipping {len(members) - len(extract_members)} existing files")
                    members = extract_members
            
            # Extract files with progress bar
            for member in tqdm(members, desc="Extracting"):
                try:
                    tar.extract(member, path=str(extract_dir))
                except (PermissionError, OSError) as e:
                    print(f"Warning: Could not extract {member.name}: {e}")
                    # Try changing permissions and extract again
                    target_path = os.path.join(extract_dir, member.name)
                    if os.path.exists(target_path):
                        try:
                            os.chmod(target_path, stat.S_IWUSR | stat.S_IRUSR)
                            tar.extract(member, path=str(extract_dir))
                            print(f"  Fixed permissions and extracted: {member.name}")
                        except Exception as e2:
                            print(f"  Still cannot extract: {e2}")
        
        print(f"✓ Extracted {filename} to {extract_dir}")
        return True
    except tarfile.ReadError as e:
        print(f"Error extracting {filename}: {e}")
        return False

def setup_nuimages_dataset(dataset_type="mini", clean=False, force=False):
    """
    Sets up the nuImages dataset.
    
    Args:
        dataset_type: Either "mini" or "full"
        clean: Whether to clean the data directory before setup
        force: Whether to force download and extraction
    """
    data_dir = Path("./data/sets/nuimages")
    
    # URLs and filenames
    mini_url = "https://www.nuscenes.org/data/nuimages-v1.0-mini.tgz"
    mini_filename = "nuimages-v1.0-mini.tgz"
    
    full_samples_url = "https://d36yt3mvayqw5m.cloudfront.net/public/nuimages-v1.0/nuimages-v1.0-all-samples.tgz"
    full_samples_filename = "nuimages-v1.0-all-samples.tgz"
    
    full_metadata_url = "https://d36yt3mvayqw5m.cloudfront.net/public/nuimages-v1.0/nuimages-v1.0-all-metadata.tgz"
    full_metadata_filename = "nuimages-v1.0-all-metadata.tgz"
    
    # 1. Clean directory if requested
    if clean:
        if not clean_directory(data_dir):
            print("Could not clean directory. Continuing anyway...")
            # Add a short pause to let things settle
            time.sleep(1)
    
    # 2. Create directory if it doesn't exist
    data_dir.mkdir(parents=True, exist_ok=True)
    print(f"✓ Created directory: {data_dir}")
    
    # 3. Download and extract based on dataset type
    if dataset_type == "mini":
        print("\n=== Setting up nuImages MINI dataset ===")
        if download_file(mini_url, mini_filename, force):
            extract_tarfile(mini_filename, data_dir, force)
    else:  # full dataset
        print("\n=== Setting up nuImages FULL dataset ===")
        
        # Download and extract samples
        print("\n--- Downloading and extracting SAMPLES ---")
        if download_file(full_samples_url, full_samples_filename, force):
            extract_tarfile(full_samples_filename, data_dir, force)
        
        # Download and extract metadata
        print("\n--- Downloading and extracting METADATA ---")
        if download_file(full_metadata_url, full_metadata_filename, force):
            extract_tarfile(full_metadata_filename, data_dir, force)
    
    print("\nSetup complete! The nuImages dataset is ready to use.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download and setup the nuImages dataset")
    parser.add_argument(
        "--type", 
        choices=["mini", "full"], 
        default="mini",
        help="Dataset type to download: 'mini' for the smaller dataset, 'full' for the complete dataset"
    )
    parser.add_argument(
        "--no-clean", 
        action="store_false",
        dest="clean",
        help="Do NOT clean the data directory before setup (default: clean directory)"
    )
    parser.add_argument(
        "--no-force", 
        action="store_false",
        dest="force",
        help="Do NOT force download and extraction (default: force download and overwrite)"
    )
    args = parser.parse_args()
    
    # clean and force are now True by default
    setup_nuimages_dataset(dataset_type=args.type, clean=args.clean, force=args.force)