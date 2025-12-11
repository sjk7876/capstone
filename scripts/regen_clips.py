"""
Regenerate serve clips from serves.csv metadata.

Re-extracts serve video clips using frame ranges stored in serves.csv,
useful for regenerating clips after metadata updates or video processing changes.
"""
import csv
import subprocess
import os
import cv2
import argparse
import sys

# Add scripts directory to path for imports
script_dir = os.path.dirname(__file__)
sys.path.insert(0, script_dir)

from common_args import add_user_mode_arg, get_user_serves_csv_path, normalize_path

SERVES_CSV = "data/metadata/serves.csv"

def regenerate_serves(csv_path):
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        
        # Check if required columns exist (check fieldnames before iterating)
        if not reader.fieldnames:
            print(f"Error: CSV file {csv_path} appears to be empty or malformed")
            return
        
        required_cols = ["player", "serve_id", "source_video", "output_clip", "start_frame", "end_frame"]
        missing_cols = [col for col in required_cols if col not in reader.fieldnames]
        if missing_cols:
            print(f"Error: CSV file {csv_path} is missing required columns: {', '.join(missing_cols)}")
            print(f"Found columns: {', '.join(reader.fieldnames)}")
            return
        
        for row in reader:
            player = row["player"]
            serve_id = row["serve_id"]
            # Normalize paths in case they were stored with Windows backslashes
            source_video = normalize_path(row["source_video"])
            output_clip = normalize_path(row["output_clip"])
            start_frame = int(row["start_frame"])
            end_frame = int(row["end_frame"])

            os.makedirs(os.path.dirname(output_clip), exist_ok=True)

            print(f"▶Re-generating {player} serve {serve_id} → {output_clip}")

            # Detect FPS from source video to match splitter behavior
            fps = None
            try:
                cap = cv2.VideoCapture(source_video)
                if cap.isOpened():
                    fps = cap.get(cv2.CAP_PROP_FPS)
                cap.release()
            except Exception:
                fps = None

            # Convert frames to time for ffmpeg
            if fps and fps > 0:
                start_time = start_frame / fps
                end_time = end_frame / fps
            else:
                print(f"Warning: Could not get FPS for {source_video}, using default 30fps")
                fps = 30.0
                start_time = start_frame / fps
                end_time = end_frame / fps

            cmd = [
                "ffmpeg", "-n",                # overwrite if exists
                "-ss", str(start_time),
                "-to", str(end_time),
                "-i", source_video,
                "-an",                         # no audio
                "-c:v", "libx264",             # reencode (clean split)
                "-preset", "slow",
                "-crf", "18",
                "-r", str(fps),                # maintain original frame rate
                output_clip,
            ]

            result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            if result.returncode != 0:
                print(f"Failed: {output_clip}")
                print(result.stderr.decode())

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Regenerate serve clips from serves.csv metadata",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Regenerate clips from data/metadata/serves.csv
  python scripts/regen_clips.py

  # Regenerate clips from user/data/user_serves.csv
  python scripts/regen_clips.py --user-mode
        """
    )
    add_user_mode_arg(parser)
    args = parser.parse_args()
    
    if args.user_mode:
        csv_path = get_user_serves_csv_path()
        print(f"Using user mode: {csv_path}")
    else:
        csv_path = SERVES_CSV
        print(f"Using data mode: {csv_path}")
    
    if not os.path.exists(csv_path):
        print(f"Error: CSV file not found: {csv_path}")
        sys.exit(1)
    
    regenerate_serves(csv_path)
