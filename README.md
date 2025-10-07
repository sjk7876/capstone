## Volleyball Serve Analysis Tool

## Current Software

- ffmpeg (re-encode video)
  - `ffmpeg -i input.mp4 -an -c:v libx264 -b:v 5M -preset slow -r 60 output_5mbps_60fps.mp4`
- CVAT (annotating ball bounding boxes) 


## Current Workflow

This is the end-to-end process we’re using to build the volleyball serve dataset.

---

### 1) Record Raw Sessions
- Place camera on a 74" (188 cm) tripod, centered on the baseline, slightly tilted down.
- Capture entire toss → baseline → net → landing zone.
y- Save raw sessions into: `data/videos/raw/YYYY-MM-DD/session_<num>/filename.mp4`

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

Each clip is logged in: `data/metadata/serves.csv` (times with millisecond precision).

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
  --session 1\
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
python3 scripts/landing_frame.py FINISH
```
instructions here

Then use CVAT with three projects: ball bounding boxes, court corners, court masks

Annotations required:
- Ball bounding boxes (YOLO format)
- Court corners (once per session)
- Landing frame index is stored in `data/metadata/serves.csv`