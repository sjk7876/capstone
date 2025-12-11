"""
Split raw video session into individual serve clips.

Interactive tool for manually marking serve start/end points in raw session videos.
Outputs individual serve clips and logs metadata to serves.csv.
"""
import cv2
import subprocess
import os
import argparse
import glob
import re
import time
import csv
from collections import deque
from common_args import add_user_mode_arg, add_player_session_serve_args, get_user_serves_csv_path, get_user_videos_dir, normalize_path

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

def _next_session_id(user_mode=False):
    """Get the next session ID. In user mode, read from CSV. Otherwise return None."""
    if not user_mode:
        return None
    
    csv_path = get_user_serves_csv_path()
    if not os.path.exists(csv_path):
        return 1
    
    max_session = 0
    try:
        with open(csv_path, "r", newline="") as f:
            reader = csv.reader(f)
            header = next(reader, None)
            if not header:
                return 1
            
            for row in reader:
                if len(row) >= 3:
                    try:
                        session_id = int(row[2])  # session_id is column 2
                        max_session = max(max_session, session_id)
                    except (ValueError, IndexError):
                        continue
    except Exception:
        return 1
    
    return max_session + 1

def _delete_last_clip(output_dir, player, video_path, session_id, user_mode=False):
    # Find the last serve from this specific video file by checking CSV
    csv_path = get_user_serves_csv_path() if user_mode else SERVES_CSV
    if not os.path.exists(csv_path):
        return False, None, None
    
    last_serve = None
    last_id = 0
    
    with open(csv_path, "r", newline="") as f:
        reader = csv.reader(f)
        header = next(reader, None)
        if not header:
            return False, None, None
        
        for row in reader:
            if len(row) < 5:
                continue
            # Check: player, session_id, and source_video match
            # Normalize paths for comparison
            row_video_path = normalize_path(row[3]) if len(row) > 3 else ""
            video_path_norm = normalize_path(video_path)
            if (row[0] == player and 
                row[2] == str(session_id) and 
                row_video_path == video_path_norm):
                try:
                    serve_id = int(row[1])
                    if serve_id > last_id:
                        last_id = serve_id
                        last_serve = row
                except ValueError:
                    continue
    
    if last_serve is None:
        print("No serves found from this video file to delete")
        return False, None, None
    
    # Delete the file
    out_file = last_serve[4]  # output_clip column
    # Normalize path in case it was stored with Windows backslashes
    out_file = normalize_path(out_file)
    if os.path.exists(out_file):
        os.remove(out_file)
        _remove_from_csv(player, last_id, out_file, user_mode)
        return True, last_id, out_file
    else:
        # File doesn't exist, but remove from CSV anyway
        _remove_from_csv(player, last_id, out_file, user_mode)
        return True, last_id, out_file

def _append_to_csv(player, serve_id, video_path, out_file, start_frame, end_frame, session_id, user_mode=False):
    csv_path = get_user_serves_csv_path() if user_mode else SERVES_CSV
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    new_file = not os.path.exists(csv_path)
    with open(csv_path, "a", newline="") as f:
        writer = csv.writer(f)
        if new_file:
            if user_mode:
                writer.writerow(["player","serve_id","session_id","source_video","output_clip","start_frame","end_frame","landing_frame","hit_frame","landing_x","landing_y"])
            else:
                writer.writerow(["player","serve_id","session_id","source_video","output_clip","start_frame","end_frame","landing_frame"])
        # Normalize paths for cross-platform compatibility
        video_path_norm = normalize_path(video_path)
        out_file_norm = normalize_path(out_file)
        if user_mode:
            writer.writerow([player, f"{serve_id:03d}", session_id, video_path_norm, out_file_norm, str(start_frame), str(end_frame), "", "", "", ""])
        else:
            writer.writerow([player, f"{serve_id:03d}", session_id, video_path_norm, out_file_norm, str(start_frame), str(end_frame), ""])

def _remove_from_csv(player, serve_id, out_file, user_mode=False):
    csv_path = get_user_serves_csv_path() if user_mode else SERVES_CSV
    if not os.path.exists(csv_path):
        return
    rows = []
    with open(csv_path, "r", newline="") as f:
        reader = csv.reader(f)
        for row in reader:
            if row and not (row[0] == player and row[1] == f"{serve_id:03d}" and row[4] == out_file):
                rows.append(row)
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerows(rows)

def split_serves(video_path, output_dir, player, session_id, user_mode=False):
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
            "user_mode": user_mode,
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
                _append_to_csv(job["player"], job["serve_id"], job["video_path"], job["out_file"], job["start_frame"], job["end_frame"], job["session_id"], job.get("user_mode", False))
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
            deleted, last_id, path = _delete_last_clip(output_dir, player, video_path, session_id, user_mode)
            if deleted:
                serve_id = last_id
                print(f"Deleted {path}")
            else:
                print("Cannot delete: no serves from this video file found")

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
    parser = argparse.ArgumentParser(
        description="Split raw video session into individual serve clips",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python3 scripts/split_serves.py \\
    --video data/videos/raw/2025-01-15/session_1/recording.mp4 \\
    --player spencer

  python3 scripts/split_serves.py \\
    --video data/videos/raw/2025-01-15/session_2/recording.mp4 \\
    --player spencer
        """
    )
    parser.add_argument("--video", type=str, required=True,
                        help="Path to raw input video")
    add_player_session_serve_args(parser)
    add_user_mode_arg(parser)
    args = parser.parse_args()
    
    if not args.player:
        parser.error("--player is required")

    # In user mode, auto-increment session number from CSV
    if args.user_mode:
        session_id = _next_session_id(user_mode=True)
        print(f"Using session {session_id} (auto-incremented from user/data/user_serves.csv)")
    else:
        # Auto-detect session from path format: data/videos/raw/YYYY-MM-DD/session_<num>/filename.mp4
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
            parser.error("Could not detect session from video path. Expected format: data/videos/raw/YYYY-MM-DD/session_<num>/filename.mp4")

    if args.user_mode:
        output_dir = get_user_videos_dir()
    else:
        output_dir = os.path.join("data", "videos", "processed")
    
    # Print instructions when run directly from command line
    print("Controls: [s] = start serve, [e] = end serve, [d] = delete previous, [f] = fast-forward, [b] = back 10 frames, [q] = quit")
    print("Note: Use [e] then [d] to delete any serves that hit the net - these should not be included in the dataset.")
    
    split_serves(args.video, output_dir, args.player, session_id, args.user_mode)

if __name__ == "__main__":
    main()
