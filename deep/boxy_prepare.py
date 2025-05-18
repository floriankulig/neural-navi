# prepare_dataset.py
import os
import json
import random
import shutil
import subprocess

# Define directories
data_dir = "data"
boxy_raw_dir = "boxy_raw"
yolo_dir = os.path.join(data_dir, "boxy_yolo")
train_images_dir = os.path.join(yolo_dir, "train", "images")
train_labels_dir = os.path.join(yolo_dir, "train", "labels")
val_images_dir = os.path.join(yolo_dir, "val", "images")
val_labels_dir = os.path.join(yolo_dir, "val", "labels")

IMG_WIDTH = img_width = 1232
IMG_HEIGHT = img_height = 1028

# Create directories (task 0)
for dir_path in [
    data_dir,
    boxy_raw_dir,
    train_images_dir,
    train_labels_dir,
    val_images_dir,
    val_labels_dir,
]:
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
# sunny_sequences = [
#     "2016-10-10-16-00-11",
# ]

# Base URL for downloading batches (replace with updated URL)
base_url = (
    "http://5.9.71.146/dqrtq7zmfsr4q59crcya/boxy_raw_scaled/bluefox_{sequence}_bag.zip"
)
json_url = "http://5.9.71.146/dqrtq7zmfsr4q59crcya/boxy_labels_train.json"  # Replace with updated URL
# json_url = "http://5.9.71.146/dqrtq7zmfsr4q59crcya/boxy_labels_valid.json"  # Replace with updated URL

# Download Boxy-Zip Batches (task 1)
for sequence in sunny_sequences:
    url = base_url.format(sequence=sequence)
    zip_file = os.path.join(boxy_raw_dir, f"bluefox_{sequence}.zip")
    # subprocess.run(["wget", url, "-O", zip_file], check=True)
    subprocess.run(["curl", url, "-o", zip_file], check=True)

# Download Boxy labels JSON (task 2)
json_path = os.path.join(data_dir, "boxy_labels.json")
# subprocess.run(["wget", json_url, "-O", json_path], check=True)
subprocess.run(["curl", json_url, "-o", json_path], check=True)

# # Extract zip files into boxy_raw
for zip_file in os.listdir(boxy_raw_dir):
    if zip_file.endswith(".zip"):
        zip_path = os.path.join(boxy_raw_dir, zip_file)
        subprocess.run(["unzip", "-o", zip_path, "-d", boxy_raw_dir], check=True)

# Load the JSON file
with open(json_path, "r") as f:
    labels = json.load(f)

# Collect valid images and handle corrupted ones
valid_images = []
for image_path, annotation in labels.items():
    full_image_path = os.path.join(boxy_raw_dir, image_path[2:])  # Remove "./"
    if "no-issues" not in annotation["flaws"]:
        if os.path.exists(full_image_path):
            os.remove(full_image_path)
            print(f"Deleted corrupted image: {full_image_path}")
    else:
        valid_images.append(image_path)

# Shuffle and split into train and val sets (80% train, 20% val)
random.shuffle(valid_images)
num_val = int(0.2 * len(valid_images))
val_images = valid_images[:num_val]
train_images = valid_images[num_val:]

# Process each set
for set_type, images in [("train", train_images), ("val", val_images)]:
    dest_image_dir = train_images_dir if set_type == "train" else val_images_dir
    dest_label_dir = train_labels_dir if set_type == "train" else val_labels_dir

    for image_path in images:
        full_image_path = os.path.join(boxy_raw_dir, image_path[2:])
        if not os.path.exists(full_image_path):
            print(f"Image not found: {full_image_path}")
            continue

        # Copy image to destination
        dest_image_path = os.path.join(dest_image_dir, os.path.basename(image_path))
        shutil.copy(full_image_path, dest_image_path)

        # Create YOLO annotation file
        annotation = labels[image_path]
        annotation_lines = []
        for vehicle in annotation["vehicles"]:
            if "rear" in vehicle and vehicle["rear"] is not None:
                rear = vehicle["rear"]
                x1, y1, x2, y2 = rear["x1"], rear["y1"], rear["x2"], rear["y2"]
                # Skip invalid bounding boxes
                if x1 >= x2 or y1 >= y2:
                    continue
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

        # Write annotation file
        annotation_file = os.path.splitext(os.path.basename(image_path))[0] + ".txt"
        annotation_path = os.path.join(dest_label_dir, annotation_file)
        with open(annotation_path, "w") as f:
            f.write("\n".join(annotation_lines))

# Create dataset.yaml
yaml_content = f"""\
train: {train_images_dir}
val: {val_images_dir}
nc: 1
names: ['vehicle']
"""
with open(os.path.join(data_dir, "dataset.yaml"), "w") as f:
    f.write(yaml_content)

print("Dataset preparation completed successfully!")
