"""
Process user serves: YOLO → SORT → Landing Analysis → Visualizations.

Master script that runs the full pipeline for user serves:
1. Run YOLO detection on serve videos
2. Run SORT tracking to get ball trajectory
3. Estimate landing frames and locations
4. Update user/data/user_serves.csv
5. Create visualizations (landing locations + trajectories)

Supports filtering by session.
"""
import os
import sys
import json
import argparse
import glob
import csv
import numpy as np
import cv2
import matplotlib.pyplot as plt
from matplotlib.colors import hsv_to_rgb
from pathlib import Path

# Add scripts directory to path
script_dir = os.path.join(os.path.dirname(__file__), "..", "scripts")
sys.path.insert(0, script_dir)

from detection_to_json import run_yolo_cli
from SORT_from_json import run_sort_from_json
from estimate_landing import estimate_hit_and_landing
from compute_homography import compute_homography
from common_args import add_player_session_serve_args, build_user_paths, get_user_serves_csv_path, format_serve_number
from create_landing_image import create_landing_image
from create_trajectory_image import create_trajectory_image

USER_DETECTIONS_DIR = os.path.join("user", "data", "detections")
USER_TRAJECTORIES_DIR = os.path.join("user", "data", "trajectories")
YOLO_MODEL = os.path.join("models", "best.pt")
USER_SERVES_CSV = get_user_serves_csv_path()


def load_user_serves(session=None):
    """Load serves from user/data/user_serves.csv, optionally filtered by session."""
    csv_path = get_user_serves_csv_path()
    if not os.path.exists(csv_path):
        return []
    
    serves = []
    with open(csv_path, "r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if session is not None:
                try:
                    if int(row["session_id"]) != session:
                        continue
                except (ValueError, KeyError):
                    continue
            serves.append(row)
    return serves


def load_homography(session_id):
    """Load homography matrix, checking both user and data locations."""
    user_path = f"user/data/calibration/homographies/session_{session_id}.json"
    data_path = f"data/calibration/homographies/session_{session_id}.json"
    
    for path in [user_path, data_path]:
        if os.path.exists(path):
            with open(path) as f:
                data = json.load(f)
            return np.array(data["H"], dtype=np.float64)
    
    return None


def warp_point(u, v, H):
    """Project pixel coordinates to world coordinates using homography."""
    p = np.array([u, v, 1.0])
    q = H @ p
    return float(q[0] / q[2]), float(q[1] / q[2])


def update_user_csv(player, session_id, serve_id, hit_frame, landing_frame, landing_x, landing_y):
    """Update user/data/user_serves.csv with landing estimates."""
    if not os.path.exists(USER_SERVES_CSV):
        return
    
    rows = []
    with open(USER_SERVES_CSV, "r", newline="") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames
        rows = list(reader)
    
    updated = False
    for row in rows:
        if (row["player"] == player and 
            row["session_id"] == str(session_id) and 
            row["serve_id"] == format_serve_number(serve_id)):
            row["hit_frame"] = str(hit_frame) if hit_frame is not None else ""
            row["landing_frame"] = str(landing_frame) if landing_frame is not None else ""
            row["landing_x"] = f"{landing_x:.3f}" if landing_x is not None else ""
            row["landing_y"] = f"{landing_y:.3f}" if landing_y is not None else ""
            updated = True
            break
    
    if updated:
        with open(USER_SERVES_CSV, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)


def run_yolo_detection(player, session_id, serve_id, video_path, detect_json, force=False):
    """Run YOLO detection on a serve video."""
    # Create directory
    os.makedirs(os.path.dirname(detect_json), exist_ok=True)
    
    # Check if video exists
    if not os.path.exists(video_path):
        print(f"[skip] {player}/session_{session_id}/serve_{serve_id}: video not found: {video_path}")
        return False
    
    # Skip if already exists (unless force)
    if os.path.exists(detect_json) and not force:
        print(f"[skip] Detection already exists: {player}/session_{session_id}/serve_{serve_id}")
        return True
    
    # Run YOLO
    print(f"[yolo] {player}/session_{session_id}/serve_{serve_id}")
    try:
        run_yolo_cli(video_path, YOLO_MODEL, detect_json)
        return True
    except Exception as e:
        print(f"[error] YOLO failed for {player}/session_{session_id}/serve_{serve_id}: {e}")
        return False


def run_sort_tracking(player, session_id, serve_id, detect_json, traj_json, force=False):
    """Run SORT tracking on detection results."""
    # Create directory
    os.makedirs(os.path.dirname(traj_json), exist_ok=True)
    
    # Skip if already exists (unless force)
    if os.path.exists(traj_json) and not force:
        print(f"[skip] Trajectory already exists: {player}/session_{session_id}/serve_{serve_id}")
        return True
    
    # Check if detection exists
    if not os.path.exists(detect_json):
        print(f"[error] Detection file not found: {detect_json}")
        return False
    
    # Run SORT
    print(f"[sort] {player}/session_{session_id}/serve_{serve_id}")
    try:
        run_sort_from_json(detect_json, traj_json, conf_thresh=0.3, debug=False, visualize=False)
        return True
    except Exception as e:
        print(f"[error] SORT failed for {player}/session_{session_id}/serve_{serve_id}: {e}")
        return False


def estimate_landing_from_trajectory(player, session_id, serve_id, traj_json, homography=None):
    """Estimate landing frames and project to court coordinates."""
    if not os.path.exists(traj_json):
        return None
    
    # Load trajectory
    with open(traj_json) as f:
        track = json.load(f)
    
    if not track or len(track) < 3:
        print(f"[skip] {player}/session_{session_id}/serve_{serve_id}: insufficient trajectory data")
        return None
    
    # Estimate hit and landing frames
    hit_frame, landing_frame = estimate_hit_and_landing(track)
    
    if landing_frame is None:
        print(f"[skip] {player}/session_{session_id}/serve_{serve_id}: could not estimate landing")
        return None
    
    # Find landing point in trajectory
    frames = np.array([p["frame"] for p in track], dtype=int)
    idx = (np.abs(frames - landing_frame)).argmin()
    point = track[idx]
    u, v = point["center"]
    
    # Calculate bottom of ball for more accurate landing
    size = point.get("size", [0.0, 0.0])
    if len(size) >= 2:
        height = size[1]
        v_bottom = v + (height / 2)
    else:
        v_bottom = v
    
    # Project to court coordinates if homography available
    landing_x, landing_y = None, None
    if homography is not None:
        try:
            landing_x, landing_y = warp_point(u, v_bottom, homography)
            # Only keep if landing is on far side (Y > 9m, which is past the net)
            if landing_y < 9.0:
                print(f"[warning] {player}/session_{session_id}/serve_{serve_id}: "
                      f"projected landing ({landing_x:.2f}, {landing_y:.2f}) is on near side (y < 9m), discarding")
                landing_x, landing_y = None, None
        except Exception as e:
            print(f"[warning] {player}/session_{session_id}/serve_{serve_id}: "
                  f"homography projection failed: {e}")
            pass
    
    return {
        "serve_id": serve_id,
        "hit_frame": hit_frame,
        "landing_frame": landing_frame,
        "landing_x": landing_x,
        "landing_y": landing_y,
        "trajectory": track
    }


def process_serve(serve_row, homography=None, force=False):
    """Process a single serve: YOLO → SORT → estimate landing → update CSV."""
    player = serve_row["player"]
    serve_id = serve_row["serve_id"]
    session_id = int(serve_row["session_id"])
    video_path = serve_row["output_clip"]
    
    # Build paths using common_args
    _, detect_json, traj_json = build_user_paths(player, session_id, serve_id)
    
    # 1. Run YOLO detection
    if not run_yolo_detection(player, session_id, serve_id, video_path, detect_json, force=force):
        return None
    
    # 2. Run SORT tracking
    if not run_sort_tracking(player, session_id, serve_id, detect_json, traj_json, force=force):
        return None
    
    # 3. Estimate landing frames and locations
    result = estimate_landing_from_trajectory(player, session_id, serve_id, traj_json, homography)
    if result is None:
        return None
    
    # 4. Update CSV
    update_user_csv(player, session_id, serve_id, 
                   result["hit_frame"], result["landing_frame"], 
                   result["landing_x"], result["landing_y"])
    
    # Add player info to result for visualization
    result["player"] = player
    
    return result


def load_corners(session_id):
    """Load court corner annotations for a session from court_corners.csv or fallback to direct path."""
    # First try to load from CSV
    csv_path = "user/data/court_corners.csv"
    if os.path.exists(csv_path):
        with open(csv_path, "r", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row["session_id"] == f"session_{session_id}":
                    corners_path = row["court_corners_path"]
                    if os.path.exists(corners_path):
                        with open(corners_path) as f2:
                            return json.load(f2)
    
    # Fallback to direct path lookup
    user_path = f"user/data/annotations/court_corners/session_{session_id}.json"
    data_path = f"data/annotations/court_corners/{session_id}.json"
    
    for path in [user_path, data_path]:
        if os.path.exists(path):
            with open(path) as f:
                return json.load(f)
    
    return None


def load_frame_from_video(video_path, frame_number=0):
    """Load a specific frame from a video file."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        cap.release()
        return None
    
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
    ret, frame = cap.read()
    cap.release()
    
    if not ret:
        return None
    
    return frame


def create_landing_overlay(serves_data, session_id, output_path, homography=None):
    """Create visualization of landing locations on warped top-down court view (far side only).
    
    Uses create_landing_image from scripts/create_landing_image.py
    """
    # Get valid landings on far side
    valid_landings = [s for s in serves_data 
                     if s["landing_x"] is not None and s["landing_y"] is not None and s["landing_y"] > 9.0]
    
    if not valid_landings:
        print(f"[warning] No valid landings found for session {session_id}")
        return
    
    # Extract coordinates and labels
    coordinates = [{"x": s["landing_x"], "y": s["landing_y"]} for s in valid_landings]
    labels = [f"#{s['serve_id']}" for s in valid_landings]
    
    # Get player from first serve
    player = valid_landings[0].get("player", "unknown")
    
    # Create landing image using the shared function
    # create_landing_image(coordinates, player, session_id, output_path, user_mode=True, labels=labels)
    create_landing_image(coordinates, player, session_id, output_path, user_mode=True)


def create_trajectory_overlay(serves_data, session_id, output_path):
    """Create visualization of all serve trajectories with different colors on actual court image.
    
    Uses create_trajectory_image from scripts/create_trajectory_image.py
    """
    if not serves_data:
        print(f"[warning] No serve data found for session {session_id}")
        return
    
    # Extract trajectories, landing frames, and labels
    trajectories = [s.get("trajectory", []) for s in serves_data]
    landing_frames = [s.get("landing_frame") for s in serves_data]
    labels = [f"#{s['serve_id']}" for s in serves_data]
    
    # Get player from first serve
    player = serves_data[0].get("player", "unknown")
    
    # Create trajectory image using the shared function
    create_trajectory_image(
        trajectories,
        player,
        session_id,
        output_path,
        user_mode=True,
        landing_frames=landing_frames,
        # labels=labels
    )


def main():
    parser = argparse.ArgumentParser(
        description="Process user serves: YOLO → SORT → Landing Analysis → Visualizations",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process all serves
  python user/process_serves.py

  # Process only session 1
  python user/process_serves.py --session 1

  # Process specific serve (requires player and serve)
  python user/process_serves.py --player spencer --session 1 --serve 001
        """
    )
    add_player_session_serve_args(parser)
    parser.add_argument("--force", action="store_true",
                       help="Re-process even if outputs already exist")
    parser.add_argument("--no-homography", action="store_true",
                       help="Skip homography projection (only estimate frames)")
    args = parser.parse_args()
    
    if args.session is None:
        parser.error("--session is required")
    
    session_id = args.session
    
    # Load homography if available, or compute it if missing
    homography = None
    if not args.no_homography:
        homography = load_homography(session_id)
        if homography is not None:
            print(f"Loaded homography for session {session_id}")
        else:
            # Try to compute homography if court corners exist
            corners_data = load_corners(session_id)
            if corners_data and len(corners_data.get("court_corners", [])) >= 6:
                print(f"No homography found for session {session_id}. Computing from court corners...")
                session_id_str = f"session_{session_id}"
                homography = compute_homography(session_id_str, user_mode=True)
                if homography is not None:
                    print(f"Computed homography for session {session_id}")
                else:
                    print(f"Warning: Failed to compute homography for session {session_id}. Continuing without court projection...")
            else:
                print(f"Warning: No homography found and no court corners available for session {session_id}. Continuing without court projection...")
    
    # Load serves from CSV
    serves = load_user_serves(session=session_id)
    
    if not serves:
        print("No serves found in user/data/user_serves.csv")
        if args.session:
            print(f"(filtered by session={args.session})")
        return
    
    # Apply filters
    if args.player:
        serves = [s for s in serves if s["player"] == args.player]
    
    if args.serve:
        serve_str = args.serve.zfill(3) if args.serve.isdigit() else args.serve
        serves = [s for s in serves if s["serve_id"] == serve_str]
    
    if not serves:
        print("No serves match the specified filters")
        return
    
    print(f"Processing {len(serves)} serve(s)...")
    
    # Process each serve
    serves_data = []
    for serve in serves:
        if args.force:
            # Remove existing outputs
            player = serve["player"]
            serve_id = serve["serve_id"]
            session_id_int = int(serve["session_id"])
            _, detect_json, traj_json = build_user_paths(player, session_id_int, serve_id)
            if os.path.exists(detect_json):
                os.remove(detect_json)
            if os.path.exists(traj_json):
                os.remove(traj_json)
        
        result = process_serve(serve, homography, force=args.force)
        if result:
            serves_data.append(result)
            print(f"[done] {serve['player']}/session_{session_id}/serve_{result['serve_id']}: "
                  f"hit={result['hit_frame']}, landing={result['landing_frame']}, "
                  f"court=({result['landing_x']:.2f}, {result['landing_y']:.2f})" 
                  if result['landing_x'] is not None else "court=(N/A)")
    
    if not serves_data:
        print("No valid serves processed")
        return
    
    # Filter out serves with N/A court coordinates
    serves_with_coords = [s for s in serves_data 
                         if s.get("landing_x") is not None and s.get("landing_y") is not None]
    
    if not serves_with_coords:
        print("No serves with valid court coordinates for visualization")
        return
    
    # Create visualizations
    output_dir = os.path.join("user", "visualizations", f"session_{session_id}")
    os.makedirs(output_dir, exist_ok=True)
    
    landing_overlay_path = os.path.join(output_dir, "landing_locations.png")
    trajectory_overlay_path = os.path.join(output_dir, "all_trajectories.png")
    
    create_landing_overlay(serves_with_coords, session_id, landing_overlay_path, homography)
    create_trajectory_overlay(serves_with_coords, session_id, trajectory_overlay_path)
    
    print(f"\nCompleted: {len(serves_data)} serves processed successfully")
    print(f"Visualizations saved to: {output_dir}")


if __name__ == "__main__":
    main()

