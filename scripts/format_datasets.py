#!/usr/bin/env python3
"""
Script to format datasets/ into backup-datasets/ with 80-20 train/validation split.
"""

import os
import shutil
import random
from pathlib import Path

def format_datasets():
    """Format datasets/obj_train_data/ into datasets/ball_yolo/ with 80-20 split."""
    
    # Source and destination paths
    source_dir = Path("/home/spenc/dev/capstone/datasets/obj_train_data")
    backup_dir = Path("/home/spenc/dev/capstone/datasets/ball_yolo")
    
    # Create backup directory structure
    backup_dir.mkdir(parents=True, exist_ok=True)
    
    # Create subdirectories
    for split in ["train", "val"]:
        (backup_dir / "images" / split).mkdir(parents=True, exist_ok=True)
        (backup_dir / "labels" / split).mkdir(parents=True, exist_ok=True)
    
    # Get all image files (assuming .jpg extension)
    image_files = list(source_dir.glob("*.jpg"))
    print(f"Found {len(image_files)} image files")
    
    # Shuffle for random split
    random.seed(42)  # For reproducible splits
    random.shuffle(image_files)
    
    # Calculate split indices
    total_files = len(image_files)
    train_count = int(total_files * 0.8)
    val_count = total_files - train_count
    
    print(f"Split: {train_count} train, {val_count} validation")
    
    # Split files
    train_files = image_files[:train_count]
    val_files = image_files[train_count:]
    
    # Move training files
    print("Moving training files...")
    for img_file in train_files:
        # Move image
        shutil.move(str(img_file), str(backup_dir / "images" / "train" / img_file.name))
        
        # Move corresponding label file
        label_file = img_file.with_suffix('.txt')
        if label_file.exists():
            shutil.move(str(label_file), str(backup_dir / "labels" / "train" / label_file.name))
        else:
            print(f"Warning: No label file found for {img_file.name}")
    
    # Move validation files
    print("Moving validation files...")
    for img_file in val_files:
        # Move image
        shutil.move(str(img_file), str(backup_dir / "images" / "val" / img_file.name))
        
        # Move corresponding label file
        label_file = img_file.with_suffix('.txt')
        if label_file.exists():
            shutil.move(str(label_file), str(backup_dir / "labels" / "val" / label_file.name))
        else:
            print(f"Warning: No label file found for {img_file.name}")
    
    # Create cache files (empty for now)
    (backup_dir / "labels" / "train.cache").touch()
    (backup_dir / "labels" / "val.cache").touch()
    
    # Delete the original obj_train_data directory
    print("Deleting original obj_train_data directory...")
    shutil.rmtree(source_dir)
    
    print(f"Formatting complete!")
    print(f"Training images: {len(list((backup_dir / 'images' / 'train').glob('*.jpg')))}")
    print(f"Validation images: {len(list((backup_dir / 'images' / 'val').glob('*.jpg')))}")
    print(f"Training labels: {len(list((backup_dir / 'labels' / 'train').glob('*.txt')))}")
    print(f"Validation labels: {len(list((backup_dir / 'labels' / 'val').glob('*.txt')))}")

if __name__ == "__main__":
    format_datasets()
