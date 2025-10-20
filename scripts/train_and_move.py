#!/usr/bin/env python3
"""
Script to clean cache files, run YOLO training, and move the best model.
"""
import os
import shutil
import subprocess
import glob
from pathlib import Path
from datetime import datetime

def clean_cache_files():
    """Remove cache files from datasets/ball_yolo/labels."""
    cache_patterns = [
        "datasets/ball_yolo/labels/train.cache",
        "datasets/ball_yolo/labels/val.cache",
        "datasets/ball_yolo/labels/*.cache"
    ]
    
    removed_count = 0
    for pattern in cache_patterns:
        for file_path in glob.glob(pattern):
            if os.path.exists(file_path):
                os.remove(file_path)
                print(f"Removed: {file_path}")
                removed_count += 1
    
    if removed_count == 0:
        print("No cache files found to remove")
    else:
        print(f"Removed {removed_count} cache files")

def run_yolo_training():
    """Run YOLO training with specified parameters."""
    cmd = [
        "yolo", "detect", "train",
        "data=configs/ball.yaml",
        "model=models/yolov8s.pt",
        "imgsz=1280",
        "batch=8",
        "epochs=50"
    ]
    
    print("Running YOLO training...")
    print(f"Command: {' '.join(cmd)}")
    
    try:
        result = subprocess.run(cmd, check=True, capture_output=False)
        print("Training completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"Training failed with exit code {e.returncode}")
        return False

def find_latest_training_run():
    """Find the most recent training run directory."""
    runs_dir = Path("runs/detect")
    if not runs_dir.exists():
        print("No runs/detect directory found")
        return None
    
    # Look for directories that start with 'train'
    train_dirs = [d for d in runs_dir.iterdir() if d.is_dir() and d.name.startswith('train')]
    
    if not train_dirs:
        print("No training directories found in runs/detect")
        return None
    
    # Sort by modification time, get the most recent
    latest_dir = max(train_dirs, key=lambda d: d.stat().st_mtime)
    print(f"Found latest training run: {latest_dir}")
    return latest_dir

def move_best_model(source_dir):
    """Move best.pt from training run to models/ directory."""
    weights_dir = source_dir / "weights"
    best_model_path = weights_dir / "best.pt"
    
    if not best_model_path.exists():
        print(f"best.pt not found in {best_model_path}")
        return False
    
    # Ensure models directory exists
    models_dir = Path("models")
    models_dir.mkdir(exist_ok=True)
    
    # Backup existing best.pt if it exists
    existing_best = models_dir / "best.pt"
    if existing_best.exists():
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_path = models_dir / f"best.pt.backup_{timestamp}"
        shutil.move(str(existing_best), str(backup_path))
        print(f"Backed up existing model to {backup_path}")
    
    # Move the new best.pt
    destination = models_dir / "best.pt"
    shutil.move(str(best_model_path), str(destination))
    print(f"Moved best.pt to {destination}")
    
    return True

def main():
    """Main execution function."""
    print("Cleaning cache files...")
    clean_cache_files()
    
    print("\nStarting YOLO training...")
    if not run_yolo_training():
        print("Training failed, stopping")
        return
    
    print("\nFinding latest training run...")
    latest_run = find_latest_training_run()
    if not latest_run:
        print("Could not find training run, stopping")
        return
    
    print("\nMoving best model...")
    if move_best_model(latest_run):
        print("\nAll done! Your trained model is now at models/best.pt")
    else:
        print("\nFailed to move best model")

if __name__ == "__main__":
    main()
