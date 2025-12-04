# User Serve Processing

This folder contains user-facing tools for processing tennis serves.

## Workflow

1. **Split serves**: Use `split_serves.py` with `--user-mode` flag to split raw videos into individual serve clips
2. **Process serves**: Use `process_serves.py` to run the full pipeline (YOLO → SORT → Landing Analysis → Visualizations)

## Files

- `user_serves.csv`: CSV file tracking all user serves with metadata
- `court_corners.csv`: CSV file mapping sessions to court corner annotation files
- `videos/`: Directory containing split serve video clips
- `detections/`: Directory containing YOLO detection JSON files
- `trajectories/`: Directory containing SORT trajectory JSON files
- `annotations/court_corners/`: Directory containing court corner annotation JSON files

## Usage

### Step 1: Split serves from raw video

```bash
python scripts/split_serves.py \
  --video path/to/raw/video.mp4 \
  --player your_name \
  -u
```

The `-u` or `--user-mode` flag will:
- Save videos to `user/videos/` instead of `data/videos/processed/`
- Write metadata to `user/user_serves.csv` instead of `data/metadata/serves.csv`
- Include additional columns: `hit_frame`, `landing_x`, `landing_y`

### Step 2: Process serves (Full Pipeline)

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
5. Update `user/user_serves.csv` with hit_frame, landing_frame, landing_x, landing_y
6. Create two visualizations:
   - `user/visualizations/session_X/landing_locations.png` - Landing locations on court (far side)
   - `user/visualizations/session_X/all_trajectories.png` - All serve trajectories with different colors

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

