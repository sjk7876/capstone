# User Serve Processing

This folder contains user-facing tools for processing tennis serves.

## Workflow

### Quick Start (Recommended)

Use the master workflow script to process new videos automatically:

```bash
python user/workflow.py
```

This script will:
1. Find unprocessed videos in `user/input/`
2. Guide you through splitting serves (interactive)
3. Guide you through annotating court corners (interactive)
4. Automatically process all serves (YOLO → SORT → Landing Analysis → Visualizations)

### Manual Workflow

If you prefer to run steps manually:

1. **Split serves**: Use `split_serves.py` with `--user-mode` flag to split raw videos into individual serve clips
2. **Annotate court**: Use `annotate_court.py` with `--user-mode` flag to mark court corners
3. **Process serves**: Use `process_serves.py` to run the full pipeline (YOLO → SORT → Landing Analysis → Visualizations)

## Files

- `data/user_serves.csv`: CSV file tracking all user serves with metadata
- `data/court_corners.csv`: CSV file mapping sessions to court corner annotation files
- `data/videos/`: Directory containing split serve video clips
- `data/detections/`: Directory containing YOLO detection JSON files
- `data/trajectories/`: Directory containing SORT trajectory JSON files
- `data/annotations/court_corners/`: Directory containing court corner annotation JSON files
- `data/calibration/homographies/`: Directory containing homography calibration files
- `input/`: Directory for raw input videos
- `debug/`: Directory for debug outputs
- `visualizations/`: Directory for visualization outputs

## Usage

### Automated Workflow

The easiest way to process new videos:

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

The script will:
- Automatically find videos in `user/input/` that haven't been processed
- Prompt for player name (or infer from filename)
- Guide you through splitting serves interactively
- Guide you through court annotation interactively
- Automatically run the full processing pipeline

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
  - Saved to `user/visualizations/session_X/zone_N_analysis.png`

## CSV Structure

### `user_serves.csv`

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

### `court_corners.csv`

Maps sessions to their court corner annotation files:
- `session_id`: Session identifier (e.g., "session_1")
- `video_path`: Video file used for annotation
- `court_corners_path`: Path to the court corners JSON annotation file

This CSV is automatically updated when you annotate court corners using `annotate_court.py` with `--user-mode`.

