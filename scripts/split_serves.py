#!/usr/bin/env python3
import cv2
import subprocess
import os
import argparse
import glob
import re
import time
import csv
from collections import deque

SERVES_CSV = os.path.join("data", "metadata", "serves.csv")

def _next_serve_id(output_dir):
    os.makedirs(output_dir, exist_ok=True)
    existing = glob.glob(os.path.join(output_dir, "serve_*.mp4"))
    max_id = 0
    pattern = re.compile(r"serve_(\d{3})\.mp4$")
    for path in existing:
        base = os.path.basename(path)
        m = pattern.match(base)
        if m:
            try:
                max_id = max(max_id, int(m.group(1)))
            except ValueError:
                continue
    return max_id + 1

def _delete_last_clip(output_dir, player):
    existing = glob.glob(os.path.join(output_dir, "serve_*.mp4"))
    if not existing:
        return False, None, None
    pattern = re.compile(r"serve_(\d{3})\.mp4$")
    last_id = 0
    last_path = None
    for path in existing:
        base = os.path.basename(path)
        m = pattern.match(base)
        if m:
            try:
                clip_id = int(m.group(1))
            except ValueError:
                continue
            if clip_id > last_id:
                last_id = clip_id
                last_path = path
    if last_path and os.path.exists(last_path):
        os.remove(last_path)
        _remove_from_csv(player, last_id, last_path)
        return True, last_id, last_path
    return False, None, None

def _append_to_csv(player, serve_id, video_path, out_file, start_frame, end_frame, session_id):
    os.makedirs(os.path.dirname(SERVES_CSV), exist_ok=True)
    new_file = not os.path.exists(SERVES_CSV)
    with open(SERVES_CSV, "a", newline="") as f:
        writer = csv.writer(f)
        if new_file:
            writer.writerow(["player","serve_id","session_id","source_video","output_clip","start_frame","end_frame","landing_frame"])
        writer.writerow([player, f"{serve_id:03d}", session_id, video_path, out_file, str(start_frame), str(end_frame), ""])

def _remove_from_csv(player, serve_id, out_file):
    if not os.path.exists(SERVES_CSV):
        return
    rows = []
    with open(SERVES_CSV, "r", newline="") as f:
        reader = csv.reader(f)
        for row in reader:
            if row and not (row[0] == player and row[1] == f"{serve_id:03d}" and row[4] == out_file):
                rows.append(row)
    with open(SERVES_CSV, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerows(rows)

def split_serves(video_path, output_dir, player, session_id, max_jobs=None):
    # Save serves under a player-specific subfolder inside output_dir
    session_str = f"session_{session_id}"
    output_dir = os.path.join(output_dir, player, session_str)
    os.makedirs(output_dir, exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Could not open {video_path}")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = frame_count / fps

    print(f"Video length: {duration:.2f}s at {fps:.1f} fps")
    print("Controls: [s] = start serve, [e] = end serve, [d] = delete previous, [f] = fast-forward, [b] = back 10 frames, [q] = quit")

    # Setup window for proper display
    cv2.namedWindow("split_serves", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("split_serves", 1280, 720)  # Set reasonable default size

    serve_id = _next_serve_id(output_dir)
    start_frame = None

    fast_active = False
    last_fast_key_time = 0.0

    # Track background encoding jobs
    active_jobs = []  # list of dicts: {proc, player, serve_id, video_path, out_file, start_frame, end_frame}

    def _start_encode_job(start_frame, end_frame, out_path, current_fps):
        # Convert frames to time for ffmpeg
        start_s = start_frame / current_fps
        end_s = end_frame / current_fps
        
        cmd = [
            "ffmpeg", "-n",
            "-ss", str(start_s),
            "-to", str(end_s),
            "-i", video_path,
            "-an",
            "-c:v", "libx264",
            "-preset", "slow",
            "-crf", "18",
            "-r", str(current_fps),
            out_path,
        ]
        proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        active_jobs.append({
            "proc": proc,
            "player": player,
            "serve_id": serve_id,
            "session_id": session_id,
            "video_path": video_path,
            "out_file": out_path,
            "start_frame": start_frame,
            "end_frame": end_frame,
            "launched_at": time.time(),
        })

    def _harvest_finished_jobs():
        remaining = []
        for job in active_jobs:
            ret = job["proc"].poll()
            if ret is None:
                remaining.append(job)
                continue
            if ret == 0 and os.path.exists(job["out_file"]):
                elapsed = time.time() - job.get("launched_at", time.time())
                print(f"Encoding complete in {elapsed:.1f}s: {job['out_file']}")
                _append_to_csv(job["player"], job["serve_id"], job["video_path"], job["out_file"], job["start_frame"], job["end_frame"], job["session_id"])
            else:
                print(f"Encoding failed for serve {job['serve_id']:03d}: {job['out_file']}")
        active_jobs[:] = remaining

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        cv2.imshow("split_serves", frame)
        base_delay_ms = max(1, int(1000.0 / fps)) if fps and fps > 0 else 30

        # Determine playback pacing and poll keys in short intervals to avoid missing presses
        now = time.time()
        speed_multiplier = 4 if fast_active else 1
        effective_delay_ms = max(1, int(base_delay_ms / speed_multiplier))
        poll_chunk_ms = 1
        remaining_ms = effective_delay_ms
        key = -1
        while remaining_ms > 0:
            chunk = poll_chunk_ms if remaining_ms > poll_chunk_ms else remaining_ms
            k = cv2.waitKey(chunk)
            if k != -1:
                key = k & 0xFF
                break
            remaining_ms -= chunk

        now = time.time()
        if key == ord('f'):
            fast_active = True
            last_fast_key_time = now
        else:
            if fast_active and (now - last_fast_key_time) > 0.2:
                fast_active = False

        if key == ord("s"):
            start_frame = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
            start_time = start_frame / fps
            print(f"Serve {serve_id:03d} start at frame {start_frame} ({start_time:.2f}s)")

        elif key == ord("e") and start_frame is not None:
            end_frame = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
            end_time = end_frame / fps
            out_file = os.path.join(output_dir, f"serve_{serve_id:03d}.mp4")

            _start_encode_job(start_frame, end_frame, out_file, fps)

            serve_id += 1
            start_frame = None

        elif key == ord('d') and start_frame is None:
            deleted, last_id, path = _delete_last_clip(output_dir, player)
            if deleted:
                serve_id = last_id
                print(f"Deleted {path}")

        elif key == ord("b"):
            # Go back 30 frames
            current_frame = cap.get(cv2.CAP_PROP_POS_FRAMES)
            new_frame = max(0, current_frame - 30)
            cap.set(cv2.CAP_PROP_POS_FRAMES, new_frame)
            current_time = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0

        elif key == ord("q"):
            break

        # If we just ended a serve, skip fast-forward frame skipping this iteration
        if fast_active and not (key == ord('e') and start_frame is None):
            for _ in range(speed_multiplier - 1):
                if not cap.grab():
                    break

        # Check for completed encodes and pace playback
        _harvest_finished_jobs()
        # no extra sleep; we already waited via cv2.waitKey polling above

    # Finalize: wait for remaining jobs
    while True:
        if not active_jobs:
            break
        _harvest_finished_jobs()
        time.sleep(0.1)

    cap.release()
    cv2.destroyAllWindows()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", type=str, required=True,
                        help="Path to raw input video")
    parser.add_argument("--out", type=str, required=True,
                        help="Folder to save processed serves")
    parser.add_argument("--player", type=str, required=True,
                        help="Player name for serves.csv")
    parser.add_argument("--jobs", type=int, default=None,
                        help="Max parallel encodes (optional)")
    args = parser.parse_args()

    # Strictly require new raw path format for session detection.
    # Expected: data/videos/raw/YYYY-MM-DD/session_<num>/filename.mp4
    session_id = None
    try:
        norm = os.path.normpath(args.video)
        parts = norm.split(os.sep)
        if "raw" in parts:
            i = parts.index("raw")
            if len(parts) > i + 2:
                session_part = parts[i + 2]
                if session_part.startswith("session_") and session_part[8:].isdigit():
                    session_id = int(session_part[8:])
    except Exception:
        session_id = None
    if session_id is None:
        print("Error: Could not detect session from raw path. Expected data/videos/raw/YYYY-MM-DD/session_<num>/filename.mp4")
        return

    split_serves(args.video, args.out, args.player, session_id, args.jobs)

if __name__ == "__main__":
    main()
