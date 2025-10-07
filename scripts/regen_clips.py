#!/usr/bin/env python3
import csv
import subprocess
import os
import cv2

SERVES_CSV = "data/metadata/serves.csv"

def regenerate_serves():
    with open(SERVES_CSV, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            player = row["player"]
            serve_id = row["serve_id"]
            source_video = row["source_video"]
            output_clip = row["output_clip"]
            start_time = row["start_time"]
            end_time = row["end_time"]

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

            cmd = [
                "ffmpeg", "-n",                # overwrite if exists
                "-ss", start_time,
                "-to", end_time,
                "-i", source_video,
                "-an",                         # no audio
                "-c:v", "libx264",             # reencode (clean split)
                "-preset", "slow",
                "-crf", "18",
            ]

            if fps and fps > 0:
                cmd += ["-r", str(fps)]

            cmd += [
                output_clip,
            ]

            result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            if result.returncode != 0:
                print(f"Failed: {output_clip}")
                print(result.stderr.decode())

if __name__ == "__main__":
    regenerate_serves()
