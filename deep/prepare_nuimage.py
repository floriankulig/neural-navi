import os
from nuimages import NuImages
from tqdm import tqdm
from pathlib import Path
import shutil
import yaml
import random
import numpy as np

# Pfad zum NuImages-Datensatz (lokal)
DATA_SET_PATH = "data/sets/nuimages"

# Ausgabepfad für die YOLO-Trainingsdaten
output_path = "data/nuimages_yolo"
os.makedirs(output_path, exist_ok=True)

nuim = NuImages(dataroot=DATA_SET_PATH, version="v1.0-mini", verbose=True, lazy=True)
SAMPLES = nuim.sample
SAMPLES = SAMPLES[:100]

# Front-Kamera-Bilder identifizieren (für Fahrzeuge, die sich wegbewegen)
def filter_front_camera_images(samples):
    """Filtert Bilder aus der Frontkamera."""
    front_camera_samples = []

    for sample in samples: 
        key_camera_token = sample["key_camera_token"]
        sample_data = nuim.get("sample_data", key_camera_token)
        filename = sample_data["filename"]
        is_key_frame = sample_data["is_key_frame"]
        if is_key_frame and ("CAM_FRONT" in filename):
            front_camera_samples.append(sample)

    print(f"Gefunden: {len(front_camera_samples)} Bilder von Frontkameras")
    return front_camera_samples


# filtered_samples = filter_front_camera_images(SAMPLES)
filtered_samples = SAMPLES.copy()

vehicle_mapping = {
    "vehicle.car": 2,  # Original YOLO-Klasse für car
    "vehicle.motorcycle": 3,  # Original YOLO-Klasse für motorcycle
    "vehicle.bus.rigid": 5,  # Original YOLO-Klasse für bus
    "vehicle.truck": 7,  # Original YOLO-Klasse für truck
    "vehicle.trailer": 7,  # Trailer als Truck behandeln (keine eigene Klasse in YOLO)
}
# vehicle_mapping = {
#     "vehicle.car": 0,
#     "vehicle.motorcycle": 1,
#     "vehicle.bus.rigid": 2,
#     "vehicle.truck": 3,
#     "vehicle.trailer": 3,
# }

# NuImages-Kategorien laden und filtern
categories = nuim.category
vehicle_categories = {}

for category in categories:
    # Wenn die Kategorie in unserem Mapping ist, speichern wir ihre Token-ID
    lower_name = category["name"].lower()
    if lower_name in vehicle_mapping:
        vehicle_categories[category["token"]] = {
            "name": lower_name,
            "yolo_id": vehicle_mapping[lower_name],
        }

print(f"Extrahierte Fahrzeugkategorien: {len(vehicle_categories)}")
for token, info in vehicle_categories.items():
    print(f"  - {info['name']}: (Original YOLO-ID: {info['yolo_id']})")

    annotations_by_sample = {}

for sample in tqdm(filtered_samples, desc="Extrahiere Annotationen"):
    sample_token = sample["token"]
    annotations_by_sample[sample_token] = []

    # Alle Objekt-Annotationen für dieses Sample abrufen
    object_tokens, _ = nuim.list_anns(sample_token, verbose=False)

    for object_token in object_tokens:
        ann = nuim.get("object_ann", object_token)
        category_token = ann["category_token"]

        # Prüfen, ob es eine Fahrzeugkategorie ist, die wir behalten wollen
        if category_token in vehicle_categories:
            # Bounding Box im Format [x1, y1, x2, y2] extrahieren
            bbox = ann["bbox"]

            # Kategorie-Information hinzufügen
            annotations_by_sample[sample_token].append(
                {
                    "category_token": category_token,
                    "yolo_id": vehicle_categories[category_token]["yolo_id"],
                    "bbox": bbox,
                }
            )


# 1. Erstelle die notwendigen Verzeichnisse für YOLO-Training
train_path = Path(output_path) / "train"
val_path = Path(output_path) / "val"
train_images_path = train_path / "images"
train_labels_path = train_path / "labels"
val_images_path = val_path / "images"
val_labels_path = val_path / "labels"


def recreate_dirs():
    # Clear train and val directories if they exist
    if train_path.exists():
        shutil.rmtree(train_path)
    if val_path.exists():
        shutil.rmtree(val_path)

    # Verzeichnisse erstellen
    for dir_path in [
        train_images_path,
        train_labels_path,
        val_images_path,
        val_labels_path,
    ]:
        dir_path.mkdir(parents=True, exist_ok=True)


recreate_dirs()


# 2. Die Samples zwischen Training und Validierung aufteilen (80% / 20%)
random.seed(42)  # Für Reproduzierbarkeit
val_split = 0.2

filtered_samples = SAMPLES.copy()
random.shuffle(filtered_samples)
split_idx = int(len(filtered_samples) * (1 - val_split))
train_samples = filtered_samples[:split_idx]
val_samples = filtered_samples[split_idx:]

print(
    f"Training: {len(train_samples)} Samples, Validierung: {len(val_samples)} Samples"
)


# 3. Bilder kopieren und Annotations-Dateien erstellen
processed_samples = {"train": 0, "val": 0}
processed_annotations = {"train": 0, "val": 0}


# YOLO-Format-Funktion: Konvertiert BBox [xmin, ymin, xmax, ymax] zu [class_id, x_center, y_center, width, height]
def convert_to_yolo_format(bbox, img_width, img_height, class_id):
    """
    Konvertiert Bounding Box von [xmin, ymin, xmax, ymax] zu YOLO-Format [x_center, y_center, width, height]
    Alle Werte werden auf Bildgröße normalisiert (0-1)
    """
    xmin, ymin, xmax, ymax = bbox

    # Normalisierte Breite und Höhe
    width = (xmax - xmin) / img_width
    height = (ymax - ymin) / img_height

    # Normalisierte Zentrumskoordinaten
    x_center = (xmin + (xmax - xmin) / 2) / img_width
    y_center = (ymin + (ymax - ymin) / 2) / img_height

    return [class_id, x_center, y_center, width, height]


for dataset_type, samples in [("train", train_samples), ("val", val_samples)]:
    output_img_dir = train_images_path if dataset_type == "train" else val_images_path
    output_label_dir = train_labels_path if dataset_type == "train" else val_labels_path

    for sample in tqdm(samples, desc=f"Erstelle {dataset_type.capitalize()}-Dataset"):
        sample_token = sample["token"]
        key_camera_token = sample["key_camera_token"]

        # Bildpfad ermitteln
        sample_data = nuim.get("sample_data", key_camera_token)
        img_path = Path(DATA_SET_PATH) / sample_data["filename"]

        if not img_path.exists():
            print(f"Warnung: Bild {img_path} nicht gefunden, überspringe...")
            continue

        img_height, img_width = sample_data["height"], sample_data["width"]
        # img_height, img_width = img.shape[:2]

        # Zielbildpfad
        dest_img_name = f"{sample_token}.jpg"
        dest_img_path = output_img_dir / dest_img_name

        # # Bild kopieren
        # shutil.copy(img_path, dest_img_path)

        # Symbolischen Link erstellen anstatt das Bild zu kopieren
        # Absolute Pfade für Quell- und Zieldateien verwenden
        source_absolute = os.path.abspath(img_path)

        if os.path.lexists(dest_img_path):
            os.remove(dest_img_path)  # Entferne existierenden Symlink falls vorhanden

        os.symlink(source_absolute, dest_img_path)
        processed_samples[dataset_type] += 1

        # Annotationen im YOLO-Format erstellen
        annotations = annotations_by_sample.get(sample_token, [])
        if not annotations:
            # Leere Annotationsdatei erstellen, wenn keine Fahrzeuge vorhanden sind
            with open(output_label_dir / f"{sample_token}.txt", "w") as f:
                pass
            continue

        with open(output_label_dir / f"{sample_token}.txt", "w") as f:
            for ann in annotations:
                bbox = ann["bbox"]
                class_id = ann["yolo_id"]

                # Zu YOLO-Format konvertieren
                yolo_bbox = convert_to_yolo_format(
                    bbox, img_width, img_height, class_id
                )

                # In Datei schreiben: class_id x_center y_center width height
                f.write(
                    f"{yolo_bbox[0]} {yolo_bbox[1]:.6f} {yolo_bbox[2]:.6f} {yolo_bbox[3]:.6f} {yolo_bbox[4]:.6f}\n"
                )
                processed_annotations[dataset_type] += 1

# 4. YAML-Konfigurationsdatei für YOLO erstellen
yaml_content = {
    "path": os.path.abspath(output_path),
    "train": "train/images",
    "val": "val/images",
    "names": {
        0: "person",
        1: "bicycle",
        2: "car",
        3: "motorcycle",
        4: "airplane",
        5: "bus",
        6: "train",
        7: "truck",
        8: "boat",
        9: "traffic light",
        10: "fire hydrant",
        11: "stop sign",
        12: "parking meter",
        13: "bench",
        14: "bird",
        15: "cat",
        16: "dog",
        17: "horse",
        18: "sheep",
        19: "cow",
        20: "elephant",
        21: "bear",
        22: "zebra",
        23: "giraffe",
        24: "backpack",
        25: "umbrella",
        26: "handbag",
        27: "tie",
        28: "suitcase",
        29: "frisbee",
        30: "skis",
        31: "snowboard",
        32: "sports ball",
        33: "kite",
        34: "baseball bat",
        35: "baseball glove",
        36: "skateboard",
        37: "surfboard",
        38: "tennis racket",
        39: "bottle",
        40: "wine glass",
        41: "cup",
        42: "fork",
        43: "knife",
        44: "spoon",
        45: "bowl",
        46: "banana",
        47: "apple",
        48: "sandwich",
        49: "orange",
        50: "broccoli",
        51: "carrot",
        52: "hot dog",
        53: "pizza",
        54: "donut",
        55: "cake",
        56: "chair",
        57: "couch",
        58: "potted plant",
        59: "bed",
        60: "dining table",
        61: "toilet",
        62: "tv",
        63: "laptop",
        64: "mouse",
        65: "remote",
        66: "keyboard",
        67: "cell phone",
        68: "microwave",
        69: "oven",
        70: "toaster",
        71: "sink",
        72: "refrigerator",
        73: "book",
        74: "clock",
        75: "vase",
        76: "scissors",
        77: "teddy bear",
        78: "hair drier",
        79: "toothbrush",
    },
}

with open(Path(output_path) / "dataset.yaml", "w") as f:
    yaml.dump(yaml_content, f, sort_keys=False)

print(f"\nDataset vorbereitet:")
print(
    f"- {processed_samples['train']} Trainingsbilder mit {processed_annotations['train']} Annotationen"
)
print(
    f"- {processed_samples['val']} Validierungsbilder mit {processed_annotations['val']} Annotationen"
)
print(f"- Konfiguration gespeichert in {Path(output_path) / 'dataset.yaml'}")