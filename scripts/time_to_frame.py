#!/usr/bin/env python3
import sys
import cv2
import csv
import os

SERVES_CSV = "data/metadata/serves.csv"

def parse_time(ts: str) -> float:
    """Parse timestamp (hh:mm:ss.ms) or raw seconds (float) into total seconds."""
    if ":" in ts:  # timestamp format
        h, m, s = ts.split(":")
        return int(h) * 3600 + int(m) * 60 + float(s)
    else:  # plain seconds
        return float(ts)

def time_to_frame(seconds: float, fps: float) -> int:
    """Convert seconds to frame index."""
    return round(seconds * fps)

def update_csv(video_path: str, landing_frame: int):
    """Update serves.csv with landing frame for the given serve."""
    if not os.path.exists(SERVES_CSV):
        print(f"{SERVES_CSV} not found")
        return

    rows = []
    updated = False

    with open(SERVES_CSV, "r", newline="") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames or []
        if "landing_frame" not in fieldnames:
            fieldnames = fieldnames + ["landing_frame"]

        for row in reader:
            if row.get("output_clip") == video_path:
                row["landing_frame"] = str(landing_frame)
                updated = True
            rows.append(row)

    if not updated:
        print(f"⚠️ No row found for {video_path}")
    else:
        with open(SERVES_CSV, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        print(f"Updated {SERVES_CSV} with landing_frame={landing_frame} for {video_path}")

def main():
    if len(sys.argv) != 3:
        print("Usage: python ts2frame.py <time or seconds> <serve_clip_path>")
        print("Examples:")
        print("  python ts2frame.py 00:00:07.520 data/videos/processed/spencer/session_1/serve_001.mp4")
        print("  python ts2frame.py 2.416 data/videos/processed/spencer/session_1/serve_001.mp4")
        sys.exit(1)

    ts = sys.argv[1]
    video_path = sys.argv[2]

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Could not open {video_path}")
        sys.exit(1)

    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()

    seconds = parse_time(ts)
    frame = time_to_frame(seconds, fps)

    print(f"FPS: {fps:.2f}")
    print(f"Time: {seconds:.3f} s")
    print(f"Frame index: {frame}")

    update_csv(video_path, frame)

if __name__ == "__main__":
    main()
