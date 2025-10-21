#!/usr/bin/env python3
"""
Script to format datasets from labels-only workflow.
Takes labels in format player_session_#_serve_#_frame#.txt and automatically
grabs corresponding frames from data/frames/ or extracts them if missing.
"""

import os
import shutil
import random
import subprocess
import re
from pathlib import Path

def parse_label_filename(label_file):
    """Parse label filename to extract player, session, serve, and frame info."""
    # Expected format: player_session_#_serve_#_frame#.txt
    # Example: paarth_session_1_serve_001_frame000123.txt
    pattern = r'^(.+)_session_(\d+)_serve_(\d+)_frame(\d+)\.txt$'
    match = re.match(pattern, label_file.name)
    
    if not match:
        return None
    
    player, session, serve, frame_num = match.groups()
    return {
        'player': player,
        'session': int(session),
        'serve': int(serve),
        'frame_num': int(frame_num)
    }

def find_corresponding_frame(parsed_info, frames_dir):
    """Find the corresponding frame image for a label file."""
    player = parsed_info['player']
    session = parsed_info['session']
    serve = parsed_info['serve']
    frame_num = parsed_info['frame_num']
    
    # Construct expected frame directory path
    serve_dir = frames_dir / f"{player}_session_{session}_serve_{serve:03d}"
    
    if not serve_dir.exists():
        return None, serve_dir
    
    # Look for the specific frame
    frame_file = serve_dir / f"frame{frame_num:06d}.jpg"
    
    if frame_file.exists():
        return frame_file, serve_dir
    else:
        return None, serve_dir

def extract_frames_for_serve(player, session, serve):
    """Extract frames for a specific serve using extract_all_frames.py."""
    print(f"Extracting frames for {player} session {session} serve {serve}...")
    
    # Construct video path
    video_path = f"data/videos/processed/{player}/session_{session}/serve_{serve:03d}.mp4"
    
    if not os.path.exists(video_path):
        print(f"Warning: Video file not found: {video_path}")
        return False
    
    # Run extract_all_frames.py
    cmd = [
        "python", "scripts/extract_all_frames.py",
        "--player", player,
        "--session", str(session),
        "--serve", str(serve)
    ]
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        print(f"Successfully extracted frames for {player}_session_{session}_serve_{serve:03d}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error extracting frames: {e}")
        print(f"stdout: {e.stdout}")
        print(f"stderr: {e.stderr}")
        return False

def format_datasets():
    """Format datasets from labels-only workflow with 80-20 train/validation split."""
    
    # Source and destination paths
    source_dir = Path("datasets/obj_train_data")
    frames_dir = Path("data/frames")
    backup_dir = Path("datasets/ball_yolo")
    
    # Create backup directory structure
    backup_dir.mkdir(parents=True, exist_ok=True)
    
    # Create subdirectories
    for split in ["train", "val"]:
        (backup_dir / "images" / split).mkdir(parents=True, exist_ok=True)
        (backup_dir / "labels" / split).mkdir(parents=True, exist_ok=True)
    
    # Get all label files
    label_files = list(source_dir.glob("*.txt"))
    print(f"Found {len(label_files)} label files")
    
    if len(label_files) == 0:
        print("No label files found in datasets/obj_train_data/")
        return
    
    # Parse label files and find corresponding frames
    valid_pairs = []
    missing_serves = set()
    
    print("Processing label files and finding corresponding frames...")
    for label_file in label_files:
        parsed = parse_label_filename(label_file)
        if not parsed:
            print(f"Warning: Could not parse filename {label_file.name}")
            continue
        
        frame_path, serve_dir = find_corresponding_frame(parsed, frames_dir)
        
        if frame_path and frame_path.exists():
            valid_pairs.append((label_file, frame_path))
        else:
            # Frame not found, need to extract
            serve_key = (parsed['player'], parsed['session'], parsed['serve'])
            missing_serves.add(serve_key)
    
    # Extract frames for missing serves
    for player, session, serve in missing_serves:
        if extract_frames_for_serve(player, session, serve):
            # Re-check for frames after extraction
            for label_file in label_files:
                parsed = parse_label_filename(label_file)
                if parsed and (parsed['player'], parsed['session'], parsed['serve']) == (player, session, serve):
                    frame_path, _ = find_corresponding_frame(parsed, frames_dir)
                    if frame_path and frame_path.exists():
                        valid_pairs.append((label_file, frame_path))
                    else:
                        print(f"Warning: Still no frame found for {label_file.name} after extraction")
    
    print(f"Found {len(valid_pairs)} valid label-image pairs")
    
    if len(valid_pairs) == 0:
        print("No valid label-image pairs found!")
        return
    
    # Shuffle for random split
    random.seed(42)  # For reproducible splits
    random.shuffle(valid_pairs)
    
    # Calculate split indices
    total_files = len(valid_pairs)
    train_count = int(total_files * 0.8)
    val_count = total_files - train_count
    
    print(f"Split: {train_count} train, {val_count} validation")
    
    # Split files
    train_pairs = valid_pairs[:train_count]
    val_pairs = valid_pairs[train_count:]
    
    # Create symlinks for training files
    print("Creating symlinks for training files...")
    for label_file, frame_path in train_pairs:
        # Create symlink for image with proper naming to match label format
        # Extract the base name from label file (e.g., paarth_session_1_serve_001_frame000123)
        label_base = label_file.stem  # Remove .txt extension
        img_filename = f"{label_base}.jpg"  # Add .jpg extension
        train_img_path = backup_dir / "images" / "train" / img_filename
        if not train_img_path.exists():
            train_img_path.symlink_to(frame_path.resolve())
        
        # Create symlink for label file
        train_label_path = backup_dir / "labels" / "train" / label_file.name
        if not train_label_path.exists():
            train_label_path.symlink_to(label_file.resolve())
    
    # Create symlinks for validation files
    print("Creating symlinks for validation files...")
    for label_file, frame_path in val_pairs:
        # Create symlink for image with proper naming to match label format
        # Extract the base name from label file (e.g., paarth_session_1_serve_001_frame000123)
        label_base = label_file.stem  # Remove .txt extension
        img_filename = f"{label_base}.jpg"  # Add .jpg extension
        val_img_path = backup_dir / "images" / "val" / img_filename
        if not val_img_path.exists():
            val_img_path.symlink_to(frame_path.resolve())
        
        # Create symlink for label file
        val_label_path = backup_dir / "labels" / "val" / label_file.name
        if not val_label_path.exists():
            val_label_path.symlink_to(label_file.resolve())
    
    # Create cache files (empty for now)
    (backup_dir / "labels" / "train.cache").touch()
    (backup_dir / "labels" / "val.cache").touch()
    
    print(f"Formatting complete!")
    print(f"Training images: {len(list((backup_dir / 'images' / 'train').glob('*.jpg')))}")
    print(f"Validation images: {len(list((backup_dir / 'images' / 'val').glob('*.jpg')))}")
    print(f"Training labels: {len(list((backup_dir / 'labels' / 'train').glob('*.txt')))}")
    print(f"Validation labels: {len(list((backup_dir / 'labels' / 'val').glob('*.txt')))}")

if __name__ == "__main__":
    format_datasets()
