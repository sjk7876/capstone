#!/usr/bin/env python3
import cv2
import os
import argparse

def extract_frames(video_path, output_dir):
    """Extract ~30-40 evenly spaced frames per video based on fps and length."""
    os.makedirs(output_dir, exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Could not open {video_path}")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    if not fps or fps <= 0:
        fps = 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration_s = total_frames / fps if fps > 0 else 0.0

    # Choose a target count based on duration (about 4 samples per second),
    # clamped to [30, 40]
    suggested = int(duration_s * 4)
    target_count = max(30, min(40, suggested if suggested > 0 else 30))
    target_count = min(target_count, total_frames) if total_frames > 0 else 0

    # Evenly spaced frame indices to sample
    sample_indices = set()
    if total_frames > 0 and target_count > 0:
        step = total_frames / float(target_count)
        for i in range(target_count):
            idx = int(i * step)
            if idx >= total_frames:
                idx = total_frames - 1
            sample_indices.add(idx)

    frame_id = 0
    saved = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_id in sample_indices:
            out_path = os.path.join(output_dir, f"frame{frame_id:04d}.jpg")
            cv2.imwrite(out_path, frame)
            saved += 1

        frame_id += 1

    cap.release()
    print(f"Extracted {saved} frames from {video_path} → {output_dir}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input", type=str, required=True,
        help="Folder containing processed serve videos (e.g., data/videos/processed/spencer/)"
    )
    parser.add_argument(
        "--output", type=str, default="data/videos/frames",
        help="Base folder for extracted frames"
    )
    # interval removed; frames are sampled evenly based on duration
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)

    player_name = os.path.basename(os.path.normpath(args.input))

    for fname in sorted(os.listdir(args.input)):
        if not fname.lower().endswith((".mp4", ".mov", ".mkv")):
            continue

        video_path = os.path.join(args.input, fname)
        serve_id = os.path.splitext(fname)[0]
        serve_outdir = os.path.join(args.output, f"{player_name}_{serve_id}")

        extract_frames(video_path, serve_outdir)


if __name__ == "__main__":
    main()
