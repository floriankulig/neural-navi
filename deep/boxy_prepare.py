# prepare_dataset.py
import os
import json
import random
import shutil
import subprocess
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock
import time

USE_SIDE_DETECTION = True
nc = 3 if USE_SIDE_DETECTION else 1
classes = ['vehicle.left', 'vehicle.front', 'vehicle.right'] if USE_SIDE_DETECTION else ['vehicle']

# Define directories
data_dir = "data"
boxy_raw_dir = "boxy_raw"
yolo_dir = os.path.join(data_dir, f"boxy_yolo_n{nc}")
train_images_dir = os.path.join(yolo_dir, "train", "images")
train_labels_dir = os.path.join(yolo_dir, "train", "labels")
val_images_dir = os.path.join(yolo_dir, "val", "images")
val_labels_dir = os.path.join(yolo_dir, "val", "labels")

IMG_WIDTH = 1232
IMG_HEIGHT = 1028
# There are two boxy datasets. We download, down sized images to save space, hence why we must scale coordinates accordingly
BOXY_SCALE_FACTOR = 2
img_width = IMG_WIDTH * BOXY_SCALE_FACTOR
img_height = IMG_HEIGHT * BOXY_SCALE_FACTOR

# Parallel download configuration
MAX_CONCURRENT_DOWNLOADS = 4  # Adjust based on your bandwidth and server limits
DOWNLOAD_TIMEOUT = 5 * 60  # 5 minutes timeout per download

# Create directories (task 0)
for dir_path in [
    data_dir,
    boxy_raw_dir,
    train_images_dir,
    train_labels_dir,
    val_images_dir,
    val_labels_dir,
]:
    if ("val" in dir_path or "train" in dir_path) and os.path.isdir(dir_path):
        shutil.rmtree(dir_path)
    os.makedirs(dir_path, exist_ok=True)

# List of sunny sequences from the screenshot (task 1)
sunny_sequences = [
    "2016-09-30-14-41-23",
    "2016-09-30-15-03-39",
    "2016-09-30-15-19-35",
    "2016-10-04-13-52-40",
    "2016-10-04-14-22-41",
    "2016-10-10-15-17-24",
    "2016-10-10-15-24-37",
    "2016-10-10-15-32-33",
    "2016-10-10-16-00-11",
    "2016-10-10-16-12-20",
    "2016-10-10-16-43-45",
    "2016-10-10-18-25-04",
    "2016-10-26-12-49-56",
    "2016-10-26-13-00-25",
    "2016-10-26-13-04-33",
]
more_sequences = [
    "2016-10-26-17-55-06",
    "2016-10-26-17-57-22",
    "2016-10-26-18-03-11",
    "2016-10-30-10-01-47",
    "2016-10-30-10-04-51",
    "2016-10-30-10-24-32",
    "2016-11-01-10-07-39",
    "2016-11-01-10-20-23",
]
sunny_sequences = sunny_sequences + more_sequences
sunny_sequences = list(set(sunny_sequences))  # filter doubles if any

BOXY_SERVER = "http://5.9.71.146/dqrtq7zmfsr4q59crcya"
# Base URL for downloading batches (replace with updated URL)
base_url = BOXY_SERVER + "/boxy_raw_scaled/bluefox_{sequence}_bag.zip"
json_url = BOXY_SERVER + "/boxy_labels_train.json"  # Replace with updated URL
# json_url = BOXY_SERVER + "/boxy_labels_valid.json"  # Replace with updated URL


# Thread-safe counter for progress tracking
class ProgressCounter:
    def __init__(self):
        self.lock = Lock()
        self.completed = 0
        self.total = 0
        self.skipped = 0
        self.failed = 0

    def increment_completed(self):
        with self.lock:
            self.completed += 1

    def increment_skipped(self):
        with self.lock:
            self.skipped += 1

    def increment_failed(self):
        with self.lock:
            self.failed += 1

    def set_total(self, total):
        with self.lock:
            self.total = total

    def get_status(self):
        with self.lock:
            return {
                "completed": self.completed,
                "skipped": self.skipped,
                "failed": self.failed,
                "total": self.total,
                "downloaded": self.completed - self.skipped,
            }


def is_valid_zip_file(file_path):
    if not os.path.exists(file_path):
        return False

    try:
        with zipfile.ZipFile(file_path, "r") as zip_ref:
            # Test the ZIP file integrity
            zip_ref.testzip()
            # Check if ZIP file has content
            if len(zip_ref.namelist()) == 0:
                return False
            return True
    except (zipfile.BadZipFile, zipfile.LargeZipFile, Exception):
        return False


def download_single_file(sequence, progress_counter):
    url = base_url.format(sequence=sequence)
    zip_file = os.path.join(boxy_raw_dir, f"bluefox_{sequence}_bag.zip")

    result = {
        "sequence": sequence,
        "success": False,
        "skipped": False,
        "error": None,
        "file_path": zip_file,
    }

    try:
        # Check if file already exists and is valid
        if is_valid_zip_file(zip_file):
            print(f"✓ [{sequence}] Already downloaded and valid")
            result["skipped"] = True
            result["success"] = True
            progress_counter.increment_skipped()
            return result

        # If file exists but is invalid, remove it
        if os.path.exists(zip_file):
            os.remove(zip_file)

        # Download the file with timeout
        print(f"⬇ [{sequence}] Starting download...")
        start_time = time.time()

        download_process = subprocess.run(
            [
                "curl",
                "-L",
                "--max-time",
                str(DOWNLOAD_TIMEOUT),
                "--retry",
                "2",
                "--retry-delay",
                "5",
                url,
                "-o",
                zip_file,
            ],
            capture_output=True,
            text=True,
        )

        if download_process.returncode != 0:
            error_msg = f"curl failed: {download_process.stderr}"
            result["error"] = error_msg
            progress_counter.increment_failed()
            return result

        # Validate the downloaded file
        if not is_valid_zip_file(zip_file):
            error_msg = "Downloaded file is corrupted"
            result["error"] = error_msg
            if os.path.exists(zip_file):
                os.remove(zip_file)
            progress_counter.increment_failed()
            return result

        # Success
        download_time = time.time() - start_time
        file_size = os.path.getsize(zip_file) / (1024 * 1024)  # MB
        print(
            f"✓ [{sequence}] Downloaded successfully ({file_size:.1f}MB in {download_time:.1f}s)"
        )

        result["success"] = True
        result["download_time"] = download_time
        result["file_size_mb"] = file_size
        progress_counter.increment_completed()

    except Exception as e:
        result["error"] = str(e)
        if os.path.exists(zip_file):
            os.remove(zip_file)
        progress_counter.increment_failed()

    return result


def download_boxy_files_parallel():
    progress_counter = ProgressCounter()
    progress_counter.set_total(len(sunny_sequences))

    print(f"Starting parallel download of {len(sunny_sequences)} files...")
    print(f"Using {MAX_CONCURRENT_DOWNLOADS} concurrent downloads")
    print("-" * 60)

    start_time = time.time()
    results = []

    # Use ThreadPoolExecutor for parallel downloads
    with ThreadPoolExecutor(max_workers=MAX_CONCURRENT_DOWNLOADS) as executor:
        # Submit all download tasks
        future_to_sequence = {
            executor.submit(download_single_file, sequence, progress_counter): sequence
            for sequence in sunny_sequences
        }

        # Process completed downloads
        for future in as_completed(future_to_sequence):
            sequence = future_to_sequence[future]
            try:
                result = future.result()
                results.append(result)

                # Print progress
                status = progress_counter.get_status()
                progress_pct = (status["completed"] / status["total"]) * 100
                print(
                    f"Progress: {status['completed']}/{status['total']} ({progress_pct:.1f}%) - "
                    f"Downloaded: {status['downloaded']}, Skipped: {status['skipped']}, Failed: {status['failed']}"
                )

                # Print error if download failed
                if not result["success"] and not result["skipped"]:
                    print(f"✗ [{sequence}] Failed: {result['error']}")

            except Exception as e:
                print(f"✗ [{sequence}] Unexpected error: {e}")
                progress_counter.increment_failed()

    total_time = time.time() - start_time

    # Print summary
    print("-" * 60)
    print("Download Summary:")
    final_status = progress_counter.get_status()
    print(f"  Total files: {final_status['total']}")
    print(f"  Downloaded: {final_status['downloaded']}")
    print(f"  Skipped (already valid): {final_status['skipped']}")
    print(f"  Failed: {final_status['failed']}")
    print(f"  Total time: {total_time:.1f}s")

    if final_status["downloaded"] > 0:
        avg_time = (
            sum(r.get("download_time", 0) for r in results if "download_time" in r)
            / final_status["downloaded"]
        )
        total_size = sum(
            r.get("file_size_mb", 0) for r in results if "file_size_mb" in r
        )
        print(f"  Average download time: {avg_time:.1f}s")
        print(f"  Total downloaded: {total_size:.1f}MB")
        print(f"  Average speed: {total_size / total_time:.1f}MB/s")

    return {"results": results, "summary": final_status, "total_time": total_time}


# Download Boxy-Zip Batches in parallel (task 1)
download_results = download_boxy_files_parallel()

# Check if we had critical failures
if download_results["summary"]["failed"] > len(sunny_sequences) // 2:
    print(
        f"\nError: Too many downloads failed ({download_results['summary']['failed']}/{len(sunny_sequences)})"
    )
    print(
        "Consider checking your internet connection or reducing MAX_CONCURRENT_DOWNLOADS"
    )
    exit(1)

# Download Boxy labels JSON (task 2)
json_path = os.path.join(data_dir, "boxy_labels.json")
if not os.path.exists(json_path):
    print("\nDownloading labels JSON...")
    try:
        subprocess.run(
            ["curl", "-L", json_url, "-o", json_path], check=True, timeout=60
        )
        print("✓ Labels JSON downloaded successfully")
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as e:
        print(f"✗ Failed to download labels JSON: {e}")
        exit(1)
else:
    print("✓ Labels JSON already exists")

# Extract zip files into boxy_raw using Python's zipfile
print("\nExtracting ZIP files...")
extracted_count = 0
extraction_errors = []

for zip_file in os.listdir(boxy_raw_dir):
    if zip_file.endswith(".zip"):
        zip_path = os.path.join(boxy_raw_dir, zip_file)
        print(f"Extracting {zip_file}...")

        try:
            with zipfile.ZipFile(zip_path, "r") as zip_ref:
                # Extract with progress indication for large files
                members = zip_ref.namelist()
                print(f"  Found {len(members)} files in archive")
                zip_ref.extractall(boxy_raw_dir)
            print(f"✓ Successfully extracted {zip_file}")
            extracted_count += 1

            # Optional: Remove ZIP file after extraction to save space
            # os.remove(zip_path)
            # print(f"  Removed {zip_file} to save space")

        except zipfile.BadZipFile:
            error_msg = f"Warning: {zip_file} appears to be corrupted"
            print(f"✗ {error_msg}")
            extraction_errors.append(error_msg)
            continue
        except Exception as e:
            error_msg = f"Error extracting {zip_file}: {e}"
            print(f"✗ {error_msg}")
            extraction_errors.append(error_msg)
            continue

print(f"\nExtraction completed: {extracted_count} files successfully extracted")
if extraction_errors:
    print(f"Extraction errors: {len(extraction_errors)}")
    for error in extraction_errors:
        print(f"  - {error}")

# Load the JSON file
with open(json_path, "r") as f:
    labels = json.load(f)

valid_images = list(labels.keys())

print(
    f"{len(valid_images)}/{len(labels)} images without annotation flaws ({(len(valid_images)/len(labels) * 100):.1f}%)"
)

# Shuffle and split into train and val sets (80% train, 20% val)
random.shuffle(valid_images)
num_val = int(0.2 * len(valid_images))
val_images = valid_images[:num_val]
train_images = valid_images[num_val:]

print(f"\nProcessing dataset split:")
print(f"  Training images: {len(train_images)}")
print(f"  Validation images: {len(val_images)}")

# Process each set
for set_type, images in [("train", train_images), ("val", val_images)]:
    dest_image_dir = train_images_dir if set_type == "train" else val_images_dir
    dest_label_dir = train_labels_dir if set_type == "train" else val_labels_dir

    print(f"\nProcessing {set_type} set...")
    processed_count = 0

    for image_path in images:
        full_image_path = os.path.join(boxy_raw_dir, image_path[2:])
        if not os.path.exists(full_image_path):
            print(f"Image not found: {full_image_path}")
            continue

        # Create YOLO annotation file
        annotation = labels[image_path]
        annotation_lines = []
        corrupt_annotations_in_image = False
        for vehicle in annotation["vehicles"]:
            if "rear" in vehicle and vehicle["rear"] is not None:
                bbox = vehicle["rear"]
            elif "AABB" in vehicle and vehicle["AABB"] is not None:
                bbox = vehicle["AABB"]
            else:
                corrupt_annotations_in_image = True
                break
            x1, y1, x2, y2 = bbox["x1"], bbox["y1"], bbox["x2"], bbox["y2"]
            # Skip invalid bounding boxes
            if x1 >= x2 or y1 >= y2:
                corrupt_annotations_in_image = True
                break
            box_width = x2 - x1
            box_height = y2 - y1
            center_x = (x1 + x2) / 2
            center_y = (y1 + y2) / 2
            norm_center_x = center_x / img_width
            norm_center_y = center_y / img_height
            norm_width = box_width / img_width
            norm_height = box_height / img_height

            # Ensure normalized values are within [0,1]
            if (
                0 <= norm_center_x <= 1
                and 0 <= norm_center_y <= 1
                and 0 <= norm_width <= 1
                and 0 <= norm_height <= 1
            ):
                annotation_lines.append(
                    f"0 {norm_center_x} {norm_center_y} {norm_width} {norm_height}"
                )

        # If there is a vehicle where no valid bounding-box was found, process the next image
        if corrupt_annotations_in_image:
            continue

        # Copy image to destination
        dest_image_path = os.path.join(dest_image_dir, os.path.basename(image_path))
        shutil.copy(full_image_path, dest_image_path)

        # Write annotation file
        annotation_file = os.path.splitext(os.path.basename(image_path))[0] + ".txt"
        annotation_path = os.path.join(dest_label_dir, annotation_file)
        with open(annotation_path, "w") as f:
            f.write("\n".join(annotation_lines))

        processed_count += 1

    print(f"  Processed {processed_count} {set_type} images")

# Create dataset.yaml
yaml_content = f"""\
train: {train_images_dir}
val: {val_images_dir}
nc: {str(nc)}
names: {str(classes)}
"""
yaml_path = os.path.join(data_dir, "dataset.yaml")
with open(yaml_path, "w") as f:
    f.write(yaml_content)

print(f"\n✓ Dataset preparation completed successfully!")
print(f"  Dataset configuration saved to: {yaml_path}")
print(f"  Training images: {len(train_images)}")
print(f"  Validation images: {len(val_images)}")

# Final summary
final_summary = download_results["summary"]
print(f"\nFinal Summary:")
print(f"  Downloads completed: {final_summary['downloaded']}")
print(f"  Files skipped: {final_summary['skipped']}")
print(f"  Download failures: {final_summary['failed']}")
print(f"  Extraction errors: {len(extraction_errors)}")
print(f"  Total processing time: {download_results['total_time']:.1f}s")
