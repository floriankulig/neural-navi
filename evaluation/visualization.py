#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Boxy Dataset Visualization Script
Creates visualizations for the YOLO-annotated Boxy dataset:
1. Heatmap showing vehicle density (similar to Boxy paper)
2. Visualizations to verify left/front/right categorization
3. Examples of suspicious annotations in specific regions
"""

import os
import glob
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import LogNorm
from pathlib import Path
import cv2
import random
from tqdm import tqdm
import argparse
import sys


# Add parent directory to path for imports from main project
script_dir = Path(__file__).parent
project_root = script_dir.parent
sys.path.insert(0, str(project_root))

def parse_args():
    parser = argparse.ArgumentParser(description='Visualize Boxy dataset annotations')
    parser.add_argument('--data_dir', type=str, default='deep/data/boxy_yolo_n3',
                        help='Path to the YOLO-formatted Boxy dataset')
    parser.add_argument('--output_dir', type=str, default='metrics/boxy_visualizations',
                        help='Directory to save visualizations')
    parser.add_argument('--sample_size', type=int, default=1000,
                        help='Number of images to sample for visualizations')
    parser.add_argument('--heatmap_resolution', type=int, default=400,
                        help='Resolution of the heatmap (width and height)')
    parser.add_argument('--img_width', type=int, default=1232,
                        help='Original image width')
    parser.add_argument('--img_height', type=int, default=1028,
                        help='Original image height')
    parser.add_argument('--show_examples', type=int, default=10,
                        help='Number of example images to visualize')
    return parser.parse_args()

def load_annotations(data_dir, subset='train', sample_size=None):
    """Load YOLO format annotations from the dataset directory"""
    labels_dir = os.path.join(data_dir, subset, 'labels')
    images_dir = os.path.join(data_dir, subset, 'images')
    
    print(f"Searching for labels in: {labels_dir}")
    print(f"Searching for images in: {images_dir}")
    
    label_files = glob.glob(os.path.join(labels_dir, '*.txt'))
    print(f"Found {len(label_files)} label files")
    
    if len(label_files) == 0:
        print("ERROR: No label files found! Check your data_dir path.")
        return []
    
    if sample_size and sample_size < len(label_files):
        label_files = random.sample(label_files, sample_size)
    
    annotations = []
    missing_images = 0
    empty_labels = 0
    parsed_annotations = 0
    
    for label_file in tqdm(label_files, desc=f"Loading {subset} annotations"):
        # Get corresponding image file
        base_name = os.path.splitext(os.path.basename(label_file))[0]
        img_path = os.path.join(images_dir, f"{base_name}.png")
        
        if not os.path.exists(img_path):
            missing_images += 1
            if missing_images <= 5:  # Limit the number of error messages
                print(f"Image not found: {img_path}")
            continue
        
        with open(label_file, 'r') as f:
            lines = f.readlines()
        
        if not lines:
            empty_labels += 1
            continue
            
        for line in lines:
            parts = line.strip().split()
            if len(parts) == 5:
                class_id = int(parts[0])
                x_center = float(parts[1])
                y_center = float(parts[2])
                width = float(parts[3])
                height = float(parts[4])
                
                annotations.append({
                    'class_id': class_id,
                    'x_center': x_center,
                    'y_center': y_center,
                    'width': width,
                    'height': height,
                    'img_path': img_path
                })
                parsed_annotations += 1
    
    print(f"Statistics:")
    print(f"  Total label files: {len(label_files)}")
    print(f"  Missing images: {missing_images}")
    print(f"  Empty label files: {empty_labels}")
    print(f"  Successfully parsed annotations: {parsed_annotations}")
    
    if len(annotations) == 0:
        print("WARNING: No valid annotations were loaded!")
    
    return annotations

def create_heatmap(annotations, output_path, img_width, img_height, resolution=400):
    """Create a heatmap showing vehicle density across all images"""
    # Initialize heatmap grid
    heatmap = np.zeros((resolution, resolution))
    
    # For each annotation, increment the corresponding heatmap cell
    for ann in tqdm(annotations, desc="Creating heatmap"):
        # Convert normalized coordinates to heatmap grid coordinates
        x = int(ann['x_center'] * resolution)
        y = int(ann['y_center'] * resolution)
        
        # Ensure coordinates are within bounds
        x = max(0, min(x, resolution-1))
        y = max(0, min(y, resolution-1))
        
        # Increment the heatmap cell
        heatmap[y, x] += 1
    
    # Check if heatmap has any non-zero values
    max_value = heatmap.max()
    if max_value <= 0:
        print("Warning: Heatmap has no data points!")
        max_value = 1  # Set a default value to avoid errors
    
    # Create figure
    plt.figure(figsize=(10, 10))
    
    # Plot heatmap with logarithmic scale - use safe values for vmin and vmax
    min_val = max(1, heatmap[heatmap > 0].min() if np.any(heatmap > 0) else 1)
    plt.imshow(heatmap, cmap='viridis', norm=LogNorm(vmin=min_val, vmax=max(min_val+1, max_value)))
    plt.colorbar(label='Number of vehicles')
    plt.title('Number of vehicles that occupy each pixel')
    plt.axis('off')
    
    # Save figure
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

def visualize_class_distribution(annotations, output_dir, img_width, img_height):
    """Visualize the distribution of left/front/right classes"""
    # Extract data for scatter plot
    class_ids = [ann['class_id'] for ann in annotations]
    x_centers = [ann['x_center'] for ann in annotations]
    y_centers = [ann['y_center'] for ann in annotations]
    
    # Class names mapping
    class_names = ['vehicle.left', 'vehicle.front', 'vehicle.right']
    class_colors = ['blue', 'green', 'red']
    
    # Create scatter plot of object centers by class
    plt.figure(figsize=(12, 10))
    
    for class_id in range(3):
        # Get indices for this class
        indices = [i for i, cid in enumerate(class_ids) if cid == class_id]
        
        # Plot points for this class
        plt.scatter(
            [x_centers[i] for i in indices],
            [y_centers[i] for i in indices],
            c=class_colors[class_id],
            label=class_names[class_id],
            alpha=0.5,
            s=10
        )
    
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.gca().invert_yaxis()  # Invert Y axis to match image coordinates
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.1), ncol=3)
    plt.title('Distribution of Vehicle Annotations by Class')
    plt.xlabel('Normalized X Position')
    plt.ylabel('Normalized Y Position')
    
    # Save figure
    output_path = os.path.join(output_dir, 'class_distribution_scatter.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    # Create horizontal position histogram to show left/front/right distribution
    plt.figure(figsize=(12, 6))
    
    for class_id in range(3):
        # Get x positions for this class
        x_pos = [x for i, x in enumerate(x_centers) if class_ids[i] == class_id]
        
        # Plot histogram for this class
        plt.hist(
            x_pos, 
            bins=50, 
            alpha=0.5, 
            color=class_colors[class_id], 
            label=class_names[class_id]
        )
    
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.title('Horizontal Position Distribution by Vehicle Class')
    plt.xlabel('Normalized X Position')
    plt.ylabel('Count')
    
    # Save figure
    output_path = os.path.join(output_dir, 'x_position_histogram.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

def visualize_example_images(annotations, output_dir, sample_count=10):
    """Visualize a sample of images with bounding boxes colored by class"""
    # Get unique image paths
    unique_images = set(ann['img_path'] for ann in annotations)
    
    # Sample a subset of images
    if sample_count < len(unique_images):
        sampled_images = random.sample(list(unique_images), sample_count)
    else:
        sampled_images = list(unique_images)
    
    # Class colors (BGR for OpenCV)
    colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255)]  # Blue, Green, Red
    class_names = ['vehicle.left', 'vehicle.front', 'vehicle.right']
    
    for img_path in tqdm(sampled_images, desc="Visualizing example images"):
        # Read image
        img = cv2.imread(img_path)
        if img is None:
            continue
        
        # Get annotations for this image
        img_annotations = [ann for ann in annotations if ann['img_path'] == img_path]
        
        # Draw bounding boxes
        for ann in img_annotations:
            # Get normalized box coordinates
            x_center = ann['x_center']
            y_center = ann['y_center']
            width = ann['width']
            height = ann['height']
            class_id = ann['class_id']
            
            # Convert to pixel coordinates
            h, w = img.shape[:2]
            x1 = int((x_center - width/2) * w)
            y1 = int((y_center - height/2) * h)
            x2 = int((x_center + width/2) * w)
            y2 = int((y_center + height/2) * h)
            
            # Draw bounding box
            cv2.rectangle(img, (x1, y1), (x2, y2), colors[class_id], 2)
            
            # Add label
            label = class_names[class_id]
            cv2.putText(img, label, (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, colors[class_id], 2)
        
        # Save annotated image
        output_path = os.path.join(output_dir, f"example_{os.path.basename(img_path)}")
        cv2.imwrite(output_path, img)

def find_suspicious_annotations(annotations, output_dir, region_params=None, sample_count=5):
    """
    Find and visualize images with annotations in suspicious regions of the image
    
    Args:
        annotations: List of annotation dictionaries
        output_dir: Directory to save visualizations
        region_params: Dictionary defining suspicious regions with:
                      {'x_min', 'x_max', 'y_min', 'y_max', 'class_id'}
        sample_count: Maximum number of images to show
    """
    # Default region parameters (bottom center, green cluster)
    if region_params is None:
        region_params = [
            {
                'name': 'bottom_center',
                'x_min': 0.35, 
                'x_max': 0.65, 
                'y_min': 0.90, 
                'y_max': 1.0,
                'class_id': 1  # vehicle.front (green in scatter plot)
            }
        ]
    
    # Create directory for suspicious annotations
    suspicious_dir = os.path.join(output_dir, 'suspicious_regions')
    os.makedirs(suspicious_dir, exist_ok=True)
    
    # Class colors (BGR for OpenCV)
    colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255)]  # Blue, Green, Red
    class_names = ['vehicle.left', 'vehicle.front', 'vehicle.right']
    
    # Process each suspicious region
    for region in region_params:
        region_name = region.get('name', 'unnamed_region')
        region_dir = os.path.join(suspicious_dir, region_name)
        os.makedirs(region_dir, exist_ok=True)
        
        # Filter annotations in this region
        suspicious_anns = [
            ann for ann in annotations 
            if (region.get('class_id', None) is None or ann['class_id'] == region['class_id']) and
               region['x_min'] <= ann['x_center'] <= region['x_max'] and
               region['y_min'] <= ann['y_center'] <= region['y_max']
        ]
        
        print(f"Found {len(suspicious_anns)} annotations in region '{region_name}'")
        
        if not suspicious_anns:
            continue
        
        # Get unique image paths for these annotations
        unique_images = set(ann['img_path'] for ann in suspicious_anns)
        print(f"Found in {len(unique_images)} unique images")
        
        # Sample a subset of images
        if sample_count < len(unique_images):
            sampled_images = random.sample(list(unique_images), sample_count)
        else:
            sampled_images = list(unique_images)
        
        # Create visualization for each sampled image
        for img_path in tqdm(sampled_images, desc=f"Visualizing '{region_name}' region"):
            # Read image
            img = cv2.imread(img_path)
            if img is None:
                continue
            
            # Get all annotations for this image
            all_img_annotations = [ann for ann in annotations if ann['img_path'] == img_path]
            
            # Get suspicious annotations for this image
            sus_img_annotations = [
                ann for ann in all_img_annotations 
                if (region.get('class_id', None) is None or ann['class_id'] == region['class_id']) and
                   region['x_min'] <= ann['x_center'] <= region['x_max'] and
                   region['y_min'] <= ann['y_center'] <= region['y_max']
            ]
            
            # Create a copy of the image for highlighting suspicious boxes
            img_highlight = img.copy()
            
            # Draw all bounding boxes
            for ann in all_img_annotations:
                # Get normalized box coordinates
                x_center = ann['x_center']
                y_center = ann['y_center']
                width = ann['width']
                height = ann['height']
                class_id = ann['class_id']
                
                # Convert to pixel coordinates
                h, w = img.shape[:2]
                x1 = int((x_center - width/2) * w)
                y1 = int((y_center - height/2) * h)
                x2 = int((x_center + width/2) * w)
                y2 = int((y_center + height/2) * h)
                
                # Determine color and line thickness
                is_suspicious = ann in sus_img_annotations
                color = colors[class_id]
                thickness = 4 if is_suspicious else 2
                
                # Draw bounding box
                cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness)
                
                # Add label with position information if suspicious
                label = class_names[class_id]
                if is_suspicious:
                    label += f" x:{x_center:.2f} y:{y_center:.2f}"
                    # Draw crosshair at center of suspicious annotation
                    center_x = int(x_center * w)
                    center_y = int(y_center * h)
                    cv2.drawMarker(img, (center_x, center_y), color, cv2.MARKER_CROSS, 20, 2)
                
                cv2.putText(img, label, (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, thickness//2 + 1)
            
            # Add region boundary visualization
            h, w = img.shape[:2]
            region_x1 = int(region['x_min'] * w)
            region_y1 = int(region['y_min'] * h)
            region_x2 = int(region['x_max'] * w)
            region_y2 = int(region['y_max'] * h)
            cv2.rectangle(img, (region_x1, region_y1), (region_x2, region_y2), (0, 255, 255), 2)  # Yellow boundary
            
            # Save annotated image
            output_filename = f"suspicious_{region_name}_{os.path.basename(img_path)}"
            output_path = os.path.join(region_dir, output_filename)
            cv2.imwrite(output_path, img)
            
            # Create a second visualization with just the suspicious region
            # Extract region of interest and resize for better visibility
            roi = img[region_y1:region_y2, region_x1:region_x2]
            if roi.size > 0:  # Check if ROI is not empty
                scale_factor = 2  # Make ROI larger for better visibility
                roi_resized = cv2.resize(roi, (0, 0), fx=scale_factor, fy=scale_factor)
                roi_output_path = os.path.join(region_dir, f"roi_{output_filename}")
                cv2.imwrite(roi_output_path, roi_resized)

def main():
    args = parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load annotations from train set
    print("Loading training annotations...")
    train_annotations = load_annotations(
        args.data_dir, 
        subset='train', 
        sample_size=args.sample_size
    )
    
    # Create heatmap visualization
    print("Creating heatmap visualization...")
    heatmap_path = os.path.join(args.output_dir, 'vehicle_density_heatmap.png')
    create_heatmap(
        train_annotations, 
        heatmap_path, 
        args.img_width, 
        args.img_height, 
        resolution=args.heatmap_resolution
    )
    
    # Visualize class distribution
    print("Creating class distribution visualizations...")
    visualize_class_distribution(
        train_annotations, 
        args.output_dir, 
        args.img_width, 
        args.img_height
    )
    
    # Visualize example images
    print("Creating example image visualizations...")
    visualize_example_images(
        train_annotations, 
        args.output_dir, 
        sample_count=args.show_examples
    )
    
    # NEW: Find and visualize suspicious annotations
    print("Finding and visualizing suspicious annotations...")
    # Define suspicious regions
    suspicious_regions = [
        {
            'name': 'bottom_center',
            'x_min': 0.35, 
            'x_max': 0.65, 
            'y_min': 0.90, 
            'y_max': 1.0,
            'class_id': 1  # vehicle.front (green in scatter plot)
        }
    ]
    
    find_suspicious_annotations(
        train_annotations,
        args.output_dir,
        sample_count=20  # Show more examples of suspicious annotations
    )
    
    print(f"Visualizations completed and saved to {args.output_dir}")

if __name__ == "__main__":
    main()