# Auto Labeling Workflow

This workflow automates the process of extracting frames from tennis serve videos, running YOLO predictions, and preparing data for CVAT annotation.

## Prerequisites
- YOLO model trained and saved as `models/best.pt`
- CVAT server running (optional, for automatic upload)
- Video files organized in `data/videos/processed/{player}/session_{N}/serve_{NNN}.mp4`

## Steps

### 1. Clear Previous Frames
```bash
rm -rf data/frames/*
```

### 2. Extract Frames from Serve Videos
Use `scripts/extract_all_frames.py` to extract frames from specific serves:

```bash
# Extract frames from a specific serve
python scripts/extract_all_frames.py --player spencer --session 1 --serve 1

# Extract frames from every serve from a player in a session
python scripts/extract_all_frames.py --player spencer --session 1

# Or extract from a specific video file
python scripts/extract_all_frames.py --video path/to/video.mp4 --output data/frames/custom_name
```

### 3. Run Auto Labeling
Execute the auto labeling script to run YOLO predictions and prepare CVAT upload:

```bash
python scripts/auto_label.py
```

This will:
- Run YOLO detection on all serve folders in `data/frames/`
- Merge predictions with original clean images
- Create a CVAT-compatible zip file (`serve_images.zip`)
- Create a YOLO 1.1 annotation package (`serve_yolo_manual.zip`)
- Creates a new task in CVAT with the images

### 4. Upload to CVAT
The script creates two files:
- `serve_images.zip` - Images for CVAT task creation
- `serve_yolo_manual.zip` - YOLO annotations for manual upload

Upload annotations using `serve_yolo_manual.zip` (YOLO 1.1 format)