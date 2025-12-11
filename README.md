# Volleyball Serve Analysis Tool

## Table of Contents

- [How to Use](#how-to-use)
- [User Workflow](#user-workflow)
- [Mapping A Session](#mapping-a-session)
  - [1) Record Raw Session](#1-record-raw-session-1)
  - [2) Split Into Serves](#2-split-into-serves-1)
  - [3) Annotate Court](#3-annotate-court)
  - [4) Compute Homography](#4-compute-homography)
  - [5) Detection → Visualization](#5-detection--visualization)
  - [6) Estimate Landing](#6-estimate-landing)
  - [7) Project Trajectory](#7-project-trajectory)
- [Training Workflow](#training-workflow)
  - [1) Record Raw Sessions](#1-record-raw-sessions)
  - [2) Split Into Serves](#2-split-into-serves)
  - [3) Label Landing Frames](#3-label-landing-frames)
  - [4) Extract Frames for Training](#4-extract-frames-for-training)
  - [5) Annotate Data](#5-annotate-data)
  - [6) Format Dataset into Train/Val Split](#6-format-dataset-into-trainval-split)
  - [7) Train/Tune Model](#7-traintune-model)
- [Dataset Structure](#dataset-structure)
- [Manual YOLO Training Commands](#manual-yolo-training-commands)

---

## How to Use

### Prerequisites

Install Python dependencies:
```bash
pip install -r requirements.txt
```

The project requires the following external dependencies (included in `external/`):
- **SORT** (`external/sort/`) - Simple Online and Realtime Tracking algorithm for ball tracking
- **FilterPy** (`external/filterpy/`) - Kalman filtering library for trajectory smoothing

Clone the external dependencies:
```bash
git clone https://github.com/abewley/sort.git external/sort
git clone https://github.com/rlabbe/filterpy.git external/filterpy
```

---

## User Workflow

### Quick Start

For processing new videos, use the automated workflow:

```bash
# Process all unprocessed videos in user/input/
python user/workflow.py

# Process a specific video
python user/workflow.py --video user/input/my_video.mp4 --player spencer

# Skip steps if already done
python user/workflow.py --skip-split --skip-annotate  # Only process serves

# Skip zone analysis (default: analyzes zones and prompts for target)
python user/workflow.py --skip-zones

# Specify target zone directly (skips prompt)
python user/workflow.py --target-zone 3
```

This script will:
1. Find unprocessed videos in `user/input/`
2. Guide you through splitting serves (interactive)
3. Guide you through annotating court corners (interactive)
4. Automatically process all serves (YOLO → SORT → Landing Analysis → Visualizations)

### Manual Workflow

#### Step 1: Split serves from raw video

```bash
python scripts/split_serves.py \
  --video path/to/raw/video.mp4 \
  --player your_name \
  -u
```

The `-u` or `--user-mode` flag will:
- Save videos to `user/data/videos/` instead of `data/videos/processed/`
- Write metadata to `user/data/user_serves.csv` instead of `data/metadata/serves.csv`
- Include additional columns: `hit_frame`, `landing_x`, `landing_y`

#### Step 2: Annotate court corners

```bash
# Annotate court for a session
python scripts/annotate_court.py --player your_name --session 1 --user-mode
```

This will open a video player where you click 6 points:
1. Far left corner
2. Far right corner
3. Close right corner
4. Close left corner
5. Center line left point
6. Center line right point

#### Step 3: Process serves (Full Pipeline)

```bash
# Process all serves in a session (YOLO → SORT → Landing Analysis → Visualizations)
python user/process_serves.py --session 1

# Process a specific serve
python user/process_serves.py --player your_name --session 1 --serve 001

# Force re-processing (overwrite existing outputs)
python user/process_serves.py --session 1 --force

# Skip homography projection (if no homography file available)
python user/process_serves.py --session 1 --no-homography
```

The script will:
1. Run YOLO detection on serve videos
2. Run SORT tracking to get ball trajectories
3. Estimate hit and landing frames from trajectories
4. Project landing locations to court coordinates (if homography available)
5. Update `user/data/user_serves.csv` with hit_frame, landing_frame, landing_x, landing_y
6. Create two visualizations:
   - `user/visualizations/session_X/landing_locations.png` - Landing locations on court (far side)
   - `user/visualizations/session_X/all_trajectories.png` - All serve trajectories with different colors

#### Step 4: Analyze Landing Zones (Optional)

Analyze landing zones to see how many serves landed in each zone (1-6):

```bash
# Analyze all zones for a session
python scripts/analyze_landing_zones.py --player your_name --session 1 --user-mode

# Analyze with target zone focus
python scripts/analyze_landing_zones.py --target-zone 3 --player your_name --session 1 --user-mode

# Analyze all serves (no filters)
python scripts/analyze_landing_zones.py --target-zone 5 --user-mode
```

Zone layout (far side of court):
```
┌─────┬─────┬─────┐
│  1  │  6  │  5  │  Top row
├─────┼─────┼─────┤
│  2  │  3  │  4  │  Bottom row
└─────┴─────┴─────┘
```

The script outputs:
- Total serves analyzed
- Distribution across all 6 zones
- Target zone statistics (if specified)
- List of serves that landed in the target zone
- Visualization image showing:
  - Green markers for serves in the target zone
  - Red markers for serves out of the target zone
  - Yellow highlight on the target zone
  - Saved to `user/visualizations/session_X/zone_N_analysis.png` (or `data/visualizations/...` in non-user mode)

### User Data Structure

User-facing data is stored in `user/data/`:
- `user_serves.csv`: CSV file tracking all user serves with metadata
- `court_corners.csv`: CSV file mapping sessions to court corner annotation files
- `videos/`: Directory containing split serve video clips
- `detections/`: Directory containing YOLO detection JSON files
- `trajectories/`: Directory containing SORT trajectory JSON files
- `annotations/court_corners/`: Directory containing court corner annotation JSON files
- `calibration/homographies/`: Directory containing homography calibration files

#### `user_serves.csv` Structure

Contains the following columns:
- `player`: Player name
- `serve_id`: Serve number (e.g., "001")
- `session_id`: Session number
- `source_video`: Path to original raw video
- `output_clip`: Path to split serve video
- `start_frame`: Start frame in source video
- `end_frame`: End frame in source video
- `landing_frame`: Landing frame (filled later)
- `hit_frame`: Hit frame (filled later via trajectory analysis)
- `landing_x`: Landing X coordinate (filled later)
- `landing_y`: Landing Y coordinate (filled later)

#### `court_corners.csv` Structure

Maps sessions to their court corner annotation files:
- `session_id`: Session identifier (e.g., "session_1")
- `video_path`: Video file used for annotation
- `court_corners_path`: Path to the court corners JSON annotation file

This CSV is automatically updated when you annotate court corners using `annotate_court.py` with `--user-mode`.

---

## Mapping A Session

1. [Record Raw Session](#1-record-raw-session-1)
2. [Split Into Serves](#2-split-into-serves-1)
3. [Annotate Court](#3-annotate-court)
4. [Compute Homography](#4-compute-homography)
5. [Detection → Visualization](#5-detection--visualization)
6. [Estimate Landing](#6-estimate-landing)
7. [Project Trajectory](#7-project-trajectory)

---

### 1) Record Raw Session

Record raw session video and save to `data/videos/raw/YYYY-MM-DD/session_#/filename.mp4`.

---

### 2) Split Into Serves

Split the session into individual serves. Currently manual, automation coming soon:
```bash
python3 scripts/split_serves.py \
  --video data/videos/raw/YYYY-MM-DD/session_<num>/filename.mp4 \
  --out data/videos/processed \
  --player <player_name>
```

For manual splitting controls, see [Training Workflow step 2](#2-split-into-serves).

---

### 3) Annotate Court

Annotate court corners and center line points for homography computation:
```bash
python3 scripts/annotate_court.py \
  --player <player_name> \
  --session <session_number>
```

This interactive tool requires clicking 6 points:
- 4 court corners (top-left, top-right, bottom-right, bottom-left)
- 2 center line points (left, right)

Outputs to: `data/annotations/court_corners/session_<num>.json`

---

### 4) Compute Homography

Compute homography matrix from court annotations:
```bash
python3 scripts/compute_homography.py \
  --session <session_number>
```

This script:
- Reads court corner annotations from `data/annotations/court_corners/session_<num>.json`
- Computes homography matrix mapping pixel coordinates to world coordinates (meters)
- Saves homography to `data/calibration/homographies/session_<num>.json`

---

### 5) Detection → Visualization

Run the full detection and visualization pipeline (YOLO → SORT → Visualization):
```bash
python3 scripts/run_sort_visualize.py \
  --player <player_name> \
  --session <session_number> \
  [--serve <serve_number>]
```

Or process all serves:
```bash
python3 scripts/run_sort_visualize.py --all
```

This script automatically runs three steps:
1. **Detection To JSON**: Runs YOLO detection on serve videos, outputs to `data/annotations/ball_detections/`
2. **SORT From JSON**: Tracks ball detections using SORT algorithm, outputs to `data/trajectories/`
3. **Visualize From Sort**: Creates trajectory visualization videos, outputs to `data/visualized/`

---

### 6) Estimate Landing

Estimate hit and landing frames from trajectory data:
```bash
python3 scripts/estimate_landing.py \
  --player <player_name> \
  --session <session_number> \
  [--serve <serve_number>]
```

This script:
- Reads trajectory JSON files from `data/trajectories/`
- Estimates hit frame (strongest downward motion)
- Estimates landing frame (lowest ball size after hit)
- Saves estimates to `data/metadata/landing_estimates.csv`

---

### 7) Project Trajectory

Project landing location from pixel coordinates to world coordinates:
```bash
python3 scripts/project_trajectory.py \
  --player <player_name> \
  --session <session_number> \
  [--serve <serve_number>]
```

This script:
- Loads trajectory JSON and homography matrix
- Gets landing frame from `data/metadata/landing_estimates.csv`
- Projects landing pixel coordinates to world coordinates (meters)
- Outputs landing location in court coordinate system

---

## Training Workflow

1. [Record Raw Session](#1-record-raw-sessions)
2. [Split Into Serves](#2-split-into-serves)
3. [Label Landing Frames](#3-label-landing-frames)
4. [Extract Frames for Training](#4-extract-frames-for-training)
5. [Annotate Data](#5-annotate-data)
6. [Format Dataset into Train/Val Split](#6-format-dataset-into-trainval-split)
7. [Train/Tune Model](#7-traintune-model)

---

### 1) Record Raw Sessions
- Place camera on a 74" (188 cm) tripod, centered on the baseline, slightly tilted down.
- Capture entire court in frame.
- Save raw sessions into: `data/videos/raw/YYYY-MM-DD/session_#/filename.mp4`

---

### 2) Split Into Serves
Manually split the session into indvidual serves.
```bash
python3 scripts/split_serves.py \
  --video data/videos/raw/YYYY-MM-DD/session_<num>/filename.mp4 \
  --out data/videos/processed \
  --player <player_name>
```

Controls:
- s = start serve
- e = end serve
- d = delete previous
- f = fast-forward
- b = back 10 frames
- q = quit

Outputs clips to: `data/videos/processed/<player>/session_#/serve_###.mp4`

Each clip is logged in: `data/metadata/serves.csv`

---

### 3) Label Landing Frames
Run the landing labeler to establish the ground truth:
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
Grab every frame for a specific serve:
```bash
python3 scripts/extract_all_frames.py \
  --player spencer \
  --session 1 \
  [--serve 1]
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
Use the current model for semi-annotation, labels the given serves and uploads it to CVAT:
```bash
python3 scripts/auto_label_and_upload.py \
  [--no-cvat] \
  [--player spencer] \
  [--session 1] \
  [--serve 1]
```

---

### 6) Format Dataset into Train/Val Split
Format annotated labels into train/validation splits (80/20):
```bash
python3 scripts/format_datasets.py
```

This script:
- Reads labels from `datasets/obj_train_data/`
- Finds corresponding frames in `data/frames/`
- Extracts missing frames automatically if needed
- Creates 80/20 train/val split in `datasets/ball_yolo/`
- Uses reproducible random seed (42) for consistent splits

Output structure:
```
datasets/ball_yolo/
├── images/
│   ├── train/
│   └── val/
└── labels/
    ├── train/
    └── val/
```

---

### 7) Train/Tune Model

#### Training:

Train the YOLO model from scratch using the formatted dataset. Use the automated script:
```bash
python3 scripts/train_and_move.py
```

This script:
- Runs YOLO training with optimized parameters (imgsz=1920, batch=4, epochs=100, patience=50)
- Automatically moves `best.pt` from the latest training run to `models/best.pt`
- Backs up existing `models/best.pt` with timestamp if it exists

#### Tuning:

Tune the YOLO model from previous best:
```bash
python3 scripts/tune_and_move.py
```

This script:
- Runs YOLO tuning with tuning parameters (imgsz=1280, batch=12, epochs=30, patience=15)
- Uses `models/best.pt` as the starting model (fine-tuning)
- Automatically moves the tuned `best.pt` to `models/best.pt`
- Backs up existing `models/best.pt` with timestamp if it exists

---

## Dataset Structure

```
data/
├── videos/
│   ├── raw/YYYY-MM-DD/session_#/        # Raw recordings
│   └── processed/<player>/session_#/     # Split serves
├── frames/                                # Extracted frames (per serve)
├── metadata/
│   ├── serves.csv                         # Serve metadata (frame-based)
│   └── players.csv                        # Player information
├── annotations/                           # Annotations and detection results
│   ├── ball_yolo/                         # YOLO ball detection dataset (legacy)
│   ├── ball_detections/                   # YOLO detection JSON outputs
│   └── court_corners/                     # Court corner annotations
├── trajectories/                          # Ball trajectory JSON files
│   └── <player>/session_#/serve_###.json
└── visualized/                            # Trajectory visualization videos

datasets/
├── obj_train_data/                        # Source labels for formatting
└── ball_yolo/                             # Formatted train/val dataset
    ├── images/
    │   ├── train/
    │   └── val/
    └── labels/
        ├── train/
        └── val/

models/
├── configs/
│   └── ball.yaml                          # YOLO dataset config
├── best.pt                                # Trained model weights
└── yolov8s.pt                             # Pre-trained base model

runs/
└── detect/                                # YOLO training runs
    └── train*/                            # Training outputs and weights
```

---

## Manual YOLO Training Commands

### Training
```bash
yolo detect train data=models/configs/ball.yaml model=models/yolov8s.pt imgsz=1280 batch=8 epochs=50
```

### Prediction
```bash
yolo detect predict model=models/best.pt source=data/videos/processed/spencer/session_1/serve_001.mp4
```

---