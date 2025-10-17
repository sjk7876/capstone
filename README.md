## Volleyball Serve Analysis Tool

## Current Software

- **YOLO11n** - Latest YOLO model for ball detection
- **YOLOv8n** - Previous YOLO model for comparison
- **ffmpeg** - Video processing and re-encoding
- **CVAT** - Annotation tool for ball bounding boxes
- **OpenCV** - Frame extraction and video processing 


## Current Workflow

This is the end-to-end process we’re using to build the volleyball serve dataset.

---

### 1) Record Raw Sessions
- Place camera on a 74" (188 cm) tripod, centered on the baseline, slightly tilted down.
- Capture entire toss → baseline → net → landing zone.
- Save raw sessions into: `data/videos/raw/YYYY-MM-DD/session_<num>/filename.mp4`

---

### 2) Split Into Serves
Use the interactive splitter:
```bash
python3 scripts/split_serves.py \
  --video data/videos/raw/YYYY-MM-DD/session_<num>/filename.mp4 \
  --out data/videos/processed \
  --player <player_name>
```

Controls:
- s = mark serve start
- e = mark serve end
- d = delete previous serve
- f = hold to fast-forward
- q = quit

Outputs clips to: `data/videos/processed/<player>/<session>/serve_###.mp4`

Each clip is logged in: `data/metadata/serves.csv` (frame-accurate precision).

**Session auto-detection:** Session ID is automatically detected from the raw video path. Create folders like `data/videos/raw/2025-01-15/session_1/`, `data/videos/raw/2025-01-15/session_2/`, etc.

---

### 3) Label Landing Frames
Run the landing labeler:
```bash
python3 scripts/landing_frame.py
```

Controls:
- f = next frame, d = prev frame
- r = skip +10, e = skip -10
- c = hold to auto-forward, x = hold to auto-backward
- l = label landing (saves to serves.csv)
- q = quit session

Skips already-labeled serves automatically and prints session summary.

---

### 4) Extract Frames for Training
Sample ~30–40 evenly spaced frames per serve:
```bash
python3 scripts/extract_frames.py \
  --input data/videos/processed/spencer/session_1 \
  --output data/frames
```
Or grab every frame for a specific serve:
```bash
python3 scripts/extract_all_frames.py \
  --player spencer \
  --session 1 \
  --serve 1
```

Creates per-serve frame folders for annotation:
```
data/frames/spencer_1_serve_001/
  frame0000.jpg
  frame0030.jpg
  ...
```

---

### 5) Annotate Data
Label landing frame:
```bash
python3 scripts/landing_frame.py
```

Then use CVAT with three projects: ball bounding boxes, court corners, court masks

Annotations required:
- Ball bounding boxes (YOLO format)
- Court corners (once per session)
- Landing frame index is stored in `data/metadata/serves.csv`

---

## Dataset Structure

```
data/
├── videos/
│   ├── raw/YYYY-MM-DD/session_<num>/     # Raw recordings
│   └── processed/<player>/session_<num>/ # Split serves
├── frames/                               # Extracted frames
├── metadata/
│   ├── serves.csv                       # Serve metadata (frame-based)
│   └── players.csv                      # Player information
└── annotations/                         # CVAT annotations
    ├── ball_yolo/                       # YOLO ball detection dataset
    └── court_corners/                   # Court corner annotations
```

---

## YOLO Training Commands

### Training
```bash
yolo detect train data=configs/ball.yaml model=yolov8s.pt imgsz=1280 batch=8 epochs=50
```

### Prediction
```bash
yolo detect predict model=runs/detect/train/weights/best.pt source=data/videos/processed/spencer/session_1/serve_001.mp4
```

---

## Auto Labeling Workflow

This workflow automates the process of extracting frames from tennis serve videos, running YOLO predictions, and preparing data for CVAT annotation.

### Prerequisites
- YOLO model trained and saved as `models/best.pt`
- CVAT server running (optional, for automatic upload)
- Video files organized in `data/videos/processed/{player}/session_{N}/serve_{NNN}.mp4`

### Steps

#### 1. Clear Previous Frames
```bash
rm -rf data/frames/*
```

#### 2. Extract Frames from Serve Videos
Use `scripts/extract_all_frames.py` to extract frames from specific serves:

```bash
# Extract frames from a specific serve
python scripts/extract_all_frames.py --player spencer --session 1 --serve 1

# Extract frames from every serve from a player in a session
python scripts/extract_all_frames.py --player spencer --session 1

# Or extract from a specific video file
python scripts/extract_all_frames.py --video path/to/video.mp4 --output data/frames/custom_name
```

#### 3. Run Auto Labeling
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

#### 4. Upload to CVAT
The script creates two files:
- `serve_images.zip` - Images for CVAT task creation
- `serve_yolo_manual.zip` - YOLO annotations for manual upload

Upload annotations using `serve_yolo_manual.zip` (YOLO 1.1 format)