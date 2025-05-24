import sys
from pathlib import Path

# Add src and training to path for imports
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "training"))

# prepare_dataset.py
import os
import json
import random
import shutil
import subprocess
import zipfile

USE_SIDE_DETECTION = True
SKIP_SMALL = True
nc = 3 if USE_SIDE_DETECTION else 1
classes = (
    ["vehicle.left", "vehicle.front", "vehicle.right"]
    if USE_SIDE_DETECTION
    else ["vehicle"]
)

# Define directories
data_dir = "data"
boxy_raw_dir = "boxy_raw"
yolo_dir = os.path.join(data_dir, f"boxy_yolo_n{nc}{'_skip' if SKIP_SMALL else ''}")
train_images_dir = os.path.join(yolo_dir, "train", "images")
train_labels_dir = os.path.join(yolo_dir, "train", "labels")
val_images_dir = os.path.join(yolo_dir, "val", "images")
val_labels_dir = os.path.join(yolo_dir, "val", "labels")

IMG_WIDTH = 1232
IMG_HEIGHT = 1028
# There are two boxy datasets. We download down sized images to save space, hence why we must scale coordinates accordingly
BOXY_SCALE_FACTOR = 2
img_width = IMG_WIDTH * BOXY_SCALE_FACTOR
img_height = IMG_HEIGHT * BOXY_SCALE_FACTOR


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
    "2016-10-10-15-35-18",
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
    "2016-10-30-10-01-47",  # rain and traffic
    "2016-10-30-10-04-51",
    "2016-10-30-10-24-32",
    "2016-11-01-10-07-39",  # sunny, different lens
    "2016-11-01-10-20-23",  # sunny, different lens
]
sunny_sequences = sunny_sequences + more_sequences
sunny_sequences = list(set(sunny_sequences))  # filter doubles if any

BOXY_SERVER = "http://5.9.71.146/dqrtq7zmfsr4q59crcya"
# Base URL for downloading batches (replace with updated URL)
base_url = BOXY_SERVER + "/boxy_raw_scaled/bluefox_{sequence}_bag.zip"
json_url = BOXY_SERVER + "/boxy_labels_train.json"  # Replace with updated URL
# json_url = BOXY_SERVER + "/boxy_labels_valid.json"  # Replace with updated URL


def is_valid_zip_file(file_path):
    if not os.path.exists(file_path):
        return False

    try:
        with zipfile.ZipFile(file_path, "r") as zip_ref:
            # Test the ZIP file integrity
            zip_ref.testzip()
            # Check if ZIP file has content
            if len(zip_ref.namelist()) == 0:
                print(f"Warning: {file_path} is empty")
                return False
            return True
    except (zipfile.BadZipFile, zipfile.LargeZipFile):
        print(f"Warning: {file_path} is corrupted")
        return False
    except Exception as e:
        print(f"Error checking {file_path}: {e}")
        return False


def download_with_validation(url, output_path, max_retries=3):
    for attempt in range(max_retries):
        try:
            subprocess.run(["curl", "-L", url, "-o", output_path], check=True)

            # Validate the downloaded file
            if is_valid_zip_file(output_path):
                return True
            else:
                print(
                    f"Downloaded file is invalid (attempt {attempt + 1}/{max_retries})"
                )
                if os.path.exists(output_path):
                    os.remove(output_path)

        except subprocess.CalledProcessError as e:
            print(f"Download failed (attempt {attempt + 1}/{max_retries}): {e}")
            if os.path.exists(output_path):
                os.remove(output_path)

    print(f"Failed to download {url} after {max_retries} attempts")
    return False


def determine_vehicle_class(vehicle, img_width):
    """
    Determine vehicle class based on relationship between rear and AABB bounding boxes.

    Strategy:
    1. If rear == AABB (or very similar): vehicle is directly in front (vehicle.front)
    2. If AABB extends more to the left than right compared to rear: vehicle.left
    3. Otherwise: vehicle.right

    The logic is based on:
    - rear bbox: shows only the visible rear part of the vehicle
    - AABB bbox: shows the complete vehicle bounding box
    - If they're similar, we see the vehicle from behind (front)
    - If AABB is larger, we see the vehicle at an angle, and the direction
      of extension tells us if it's to our left or right

    Returns:
        int: 0 (left), 1 (front), 2 (right), or None if no valid bbox
    """
    rear_bbox = vehicle.get("rear") if "rear" in vehicle else None
    aabb_bbox = vehicle.get("AABB") if "AABB" in vehicle else None

    # Need both bounding boxes for proper classification
    if rear_bbox is None or aabb_bbox is None:
        # Fallback: if only one bbox available, classify based on position in image
        bbox = rear_bbox if rear_bbox is not None else aabb_bbox
        if bbox is None:
            return None

        center_x = (bbox["x1"] + bbox["x2"]) / 2
        norm_center_x = center_x / img_width

        # Simple position-based classification as fallback
        if norm_center_x < 0.33:
            return 0  # vehicle.left
        elif norm_center_x > 0.66:
            return 2  # vehicle.right
        else:
            return 1  # vehicle.front

    # Check if rear and AABB are approximately equal (vehicle directly in front)
    # Allow for small differences due to annotation variations
    x1_diff = abs(aabb_bbox["x1"] - rear_bbox["x1"])
    x2_diff = abs(aabb_bbox["x2"] - rear_bbox["x2"])
    y1_diff = abs(aabb_bbox["y1"] - rear_bbox["y1"])
    y2_diff = abs(aabb_bbox["y2"] - rear_bbox["y2"])

    # Threshold for considering bboxes as "equal" (in pixels)
    equality_threshold = 10  # pixels

    if (
        x1_diff <= equality_threshold
        and x2_diff <= equality_threshold
        and y1_diff <= equality_threshold
        and y2_diff <= equality_threshold
    ):
        return 1  # vehicle.front

    # Calculate how much AABB extends beyond rear on each side
    left_extension = rear_bbox["x1"] - aabb_bbox["x1"]  # positive if AABB extends left
    right_extension = (
        aabb_bbox["x2"] - rear_bbox["x2"]
    )  # positive if AABB extends right
    left_extension = max(0, left_extension)
    right_extension = max(0, right_extension)

    # Classify based on which direction has greater extension
    # We "see" the left side of the vehicle, so its right of us
    if left_extension > right_extension:
        return 2  # vehicle.right
    elif right_extension > left_extension:
        return 0  # vehicle.left
    else:
        return 1  # vehicle.front


def is_in_corrupt_region(x_center, y_center):
    # Bottom center corrupt region (bonnet/motorhaube)
    if 0.35 <= x_center <= 0.65 and 0.90 <= y_center <= 1.0:  # x range  # y range
        return True

    return False


# Download Boxy-Zip Batches with validation (task 1)
print(f"Checking and downloading {len(sunny_sequences)} ZIP files...")
skipped_count = 0
downloaded_count = 0
failed_count = 0

for sequence in sunny_sequences:
    url = base_url.format(sequence=sequence)
    zip_file = os.path.join(boxy_raw_dir, f"bluefox_{sequence}_bag.zip")

    # Check if file already exists and is valid
    if is_valid_zip_file(zip_file):
        print(f"✓ Skipping {sequence} (already downloaded and valid)")
        skipped_count += 1
        continue

    # If file exists but is invalid, remove it
    if os.path.exists(zip_file):
        print(f"Removing corrupted file: {zip_file}")
        os.remove(zip_file)

    # Download the file
    print(f"Downloading {sequence}...")
    if download_with_validation(url, zip_file):
        print(f"✓ Successfully downloaded {sequence}")
        downloaded_count += 1
    else:
        print(f"✗ Failed to download {sequence}")
        failed_count += 1

print(f"\nDownload summary:")
print(f"  Skipped (already valid): {skipped_count}")
print(f"  Downloaded: {downloaded_count}")
print(f"  Failed: {failed_count}")

if failed_count > 0:
    print(
        f"Warning: {failed_count} files failed to download. You may want to retry or check your connection."
    )

# Download Boxy labels JSON (task 2)
json_path = os.path.join(data_dir, "boxy_labels.json")
if not os.path.exists(json_path):
    print("Downloading labels JSON...")
    subprocess.run(["curl", "-L", json_url, "-o", json_path], check=True)
else:
    print("✓ Labels JSON already exists, skipping download")

# Extract zip files into boxy_raw using Python's zipfile
print("Extracting ZIP files...")
extracted_count = 0
for zip_file in os.listdir(boxy_raw_dir):
    if zip_file.endswith(".zip"):
        zip_path = os.path.join(boxy_raw_dir, zip_file)
        print(f"Processing {zip_file}...")

        try:
            with zipfile.ZipFile(zip_path, "r") as zip_ref:
                # Get list of file members (excluding directories)
                file_members = [m for m in zip_ref.infolist() if not m.is_dir()]
                if not file_members:
                    print(f"Warning: {zip_file} contains no files, skipping...")
                    continue

                # Check if all files already exist in boxy_raw_dir
                all_files_exist = all(
                    os.path.exists(os.path.join(boxy_raw_dir, m.filename))
                    for m in file_members
                )
                if all_files_exist:
                    print(
                        f"✓ Skipping extraction of {zip_file} (all files already exist)"
                    )
                    continue

                # If not all files exist, extract the ZIP
                print(f"Extracting {zip_file}...")
                zip_ref.extractall(boxy_raw_dir)
                print(f"Successfully extracted {zip_file}")
                extracted_count += 1

        except zipfile.BadZipFile:
            print(f"Warning: {zip_file} appears to be corrupted, skipping...")
            continue
        except Exception as e:
            print(f"Error processing {zip_file}: {e}")
            continue

print(f"Successfully extracted {extracted_count} ZIP files")


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

# Statistics for class distribution
class_counts = {"train": [0, 0, 0], "val": [0, 0, 0]}  # [left, front, right]

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
            # Determine vehicle class based on position
            if USE_SIDE_DETECTION:
                vehicle_class = determine_vehicle_class(vehicle, img_width)
                if vehicle_class is None:
                    corrupt_annotations_in_image = True
                    break
            else:
                vehicle_class = 0  # Single class for all vehicles

            # Get bounding box for YOLO format (prefer rear, fallback to AABB)
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

            # Skip annotations in corrupt regions
            if is_in_corrupt_region(norm_center_x, norm_center_y):
                continue

            if SKIP_SMALL:
                width_percent = norm_width * 100
                height_percent = norm_height * 100
                min_area_to_cover = 0.015  # 1.5% der Bildfläche
                min_area_to_cover_percent = min_area_to_cover * 100
                # Skip if the bounding box is too small
                if width_percent * height_percent < min_area_to_cover_percent**2:
                    continue

            # Ensure normalized values are within [0,1]
            if (
                0 <= norm_center_x <= 1
                and 0 <= norm_center_y <= 1
                and 0 <= norm_width <= 1
                and 0 <= norm_height <= 1
            ):
                annotation_lines.append(
                    f"{vehicle_class} {norm_center_x} {norm_center_y} {norm_width} {norm_height}"
                )

                # Update class statistics
                if USE_SIDE_DETECTION:
                    class_counts[set_type][vehicle_class] += 1

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

# Print class distribution statistics
if USE_SIDE_DETECTION:
    print(f"\nClass distribution:")
    for set_type in ["train", "val"]:
        total = sum(class_counts[set_type])
        print(f"  {set_type.capitalize()}:")
        for i, class_name in enumerate(classes):
            count = class_counts[set_type][i]
            percentage = (count / total * 100) if total > 0 else 0
            print(f"    {class_name}: {count} ({percentage:.1f}%)")

# Create dataset.yaml
yaml_content = f"""\
train: {os.path.abspath(train_images_dir)}
val: {os.path.abspath(val_images_dir)}
nc: {str(nc)}
names: {str(classes)}
"""
yaml_path = os.path.join(
    data_dir, f"dataset_nc{nc}{'_skip' if SKIP_SMALL else ''}.yaml"
)
with open(yaml_path, "w") as f:
    f.write(yaml_content)

print(f"\n✓ Dataset preparation completed successfully!")
print(f"  Dataset configuration saved to: {yaml_path}")
print(f"  Training images: {len(train_images)}")
print(f"  Validation images: {len(val_images)}")
print(f"  Classes: {classes}")
