"""
Create trajectory overlay visualization from a list of trajectories.

Given a list of trajectories (each with center coordinates), creates an overlay
image showing all trajectories on a court frame, similar to process_serves.py output.
"""
import os
import sys
import json
import argparse
import cv2
import numpy as np
from pathlib import Path
from matplotlib.colors import hsv_to_rgb

# Add scripts directory to path for imports
script_dir = os.path.dirname(__file__)
sys.path.insert(0, script_dir)

from common_args import add_player_session_serve_args, add_user_mode_arg, build_user_paths, build_trajectory_paths, get_user_serves_csv_path, format_serve_number, normalize_path


def load_corners(session_id, user_mode=False):
    """Load court corner annotations for a session."""
    if user_mode:
        # Try CSV first
        csv_path = os.path.join("user", "data", "court_corners.csv")
        if os.path.exists(csv_path):
            import csv
            with open(csv_path, "r", newline="") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    if row["session_id"] == f"session_{session_id}":
                        # Normalize path in case it was stored with Windows backslashes
                        corners_path = normalize_path(row["court_corners_path"])
                        if os.path.exists(corners_path):
                            with open(corners_path) as f2:
                                return json.load(f2)
        
        # Fallback to direct path
        user_path = os.path.join("user", "data", "annotations", "court_corners", f"session_{session_id}.json")
        if os.path.exists(user_path):
            with open(user_path) as f:
                return json.load(f)
    else:
        data_path = os.path.join("data", "annotations", "court_corners", f"{session_id}.json")
        if os.path.exists(data_path):
            with open(data_path) as f:
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


def create_trajectory_image(trajectories, player, session_id, output_path, user_mode=False, 
                           landing_frames=None, labels=None, title=None):
    """
    Create trajectory overlay visualization from trajectories.
    
    Args:
        trajectories: List of trajectories, each is a list of dicts with 'center' and 'frame' keys
        player: Player name
        session_id: Session ID (integer)
        output_path: Path to save output image
        user_mode: Whether to use user/ paths
        landing_frames: Optional list of landing frame numbers (one per trajectory)
        labels: Optional list of labels for each trajectory (e.g., serve IDs)
        title: Optional title text for the image
    """
    if not trajectories:
        print("[error] No trajectories provided")
        return False
    
    # Load court corners to get video path
    corners_data = load_corners(session_id, user_mode)
    frame = None
    
    if corners_data:
        # Get video path from corners annotation
        video_path = corners_data.get("video_file") or corners_data.get("image_file")
        if video_path:
            # Normalize path in case it was stored with Windows backslashes
            video_path = normalize_path(video_path)
        if not video_path or not os.path.exists(video_path):
            # Try to find a serve video from the session
            if user_mode:
                serve_video_dir = Path(os.path.join("user", "data", "videos", player, f"session_{session_id}"))
            else:
                serve_video_dir = Path(os.path.join("data", "videos", "processed", player, f"session_{session_id}"))
            serve_videos = list(serve_video_dir.glob("serve_*.mp4"))
            if serve_videos:
                video_path = str(serve_videos[0])
            else:
                video_path = None
        
        if video_path and os.path.exists(video_path):
            # Load first frame from video
            frame = load_frame_from_video(video_path, 0)
    
    if frame is None:
        print(f"[warning] Could not load court image for session {session_id}, using blank canvas")
        # Create blank canvas with reasonable size (assuming typical video dimensions)
        frame = np.ones((1080, 1920, 3), dtype=np.uint8) * 255
    
    # Generate distinct colors using HSV color space
    n_trajectories = len(trajectories)
    colors = []
    for i in range(n_trajectories):
        hue = i / n_trajectories if n_trajectories > 0 else 0
        rgb = hsv_to_rgb([hue, 0.8, 0.9])
        # Convert to BGR for OpenCV (0-255 range)
        bgr = (int(rgb[2] * 255), int(rgb[1] * 255), int(rgb[0] * 255))
        colors.append(bgr)
    
    # Draw each trajectory on the frame
    for i, traj in enumerate(trajectories):
        if not traj or len(traj) < 2:
            continue
        
        pts = np.array([p["center"] for p in traj], dtype=np.int32)
        if len(pts) == 0:
            continue
        
        # Draw trajectory line
        for j in range(len(pts) - 1):
            cv2.line(frame, tuple(pts[j]), tuple(pts[j + 1]), colors[i], 3, cv2.LINE_AA)
        
        # Mark start point (green)
        cv2.circle(frame, tuple(pts[0]), 8, (0, 255, 0), -1)
        cv2.circle(frame, tuple(pts[0]), 10, (0, 255, 0), 2)
        
        # Mark landing point if available (red)
        if landing_frames and i < len(landing_frames) and landing_frames[i] is not None:
            traj_frames = np.array([p["frame"] for p in traj])
            try:
                landing_idx = np.where(traj_frames == landing_frames[i])[0]
                if len(landing_idx) > 0:
                    idx = landing_idx[0]
                    if idx < len(pts):
                        # Draw landing point
                        pt = tuple(pts[idx])
                        cv2.circle(frame, pt, 8, (0, 0, 255), -1)
                        cv2.circle(frame, pt, 12, (0, 0, 255), 2)
            except (ValueError, IndexError):
                pass
        
        # Add label if provided
        if labels and i < len(labels):
            label = str(labels[i])
            cv2.putText(frame, label, (pts[0][0] + 15, pts[0][1] - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, colors[i], 2)
    
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)
    cv2.imwrite(output_path, frame)
    print(f"Saved trajectory overlay → {output_path}")
    return True


def load_serves_from_csv(player, session_id, user_mode=False):
    """Load serves from CSV file, filtered by player and session."""
    if user_mode:
        csv_path = get_user_serves_csv_path()
    else:
        csv_path = "data/metadata/serves.csv"
    
    if not os.path.exists(csv_path):
        return []
    
    import csv
    serves = []
    with open(csv_path, "r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get("player") == player and int(row.get("session_id", -1)) == session_id:
                # Normalize paths in case they were stored with Windows backslashes
                if "output_clip" in row:
                    row["output_clip"] = normalize_path(row["output_clip"])
                if "source_video" in row:
                    row["source_video"] = normalize_path(row["source_video"])
                serves.append(row)
    return serves


def load_trajectories_from_serves(player, session_id, user_mode=False):
    """Load trajectories from JSON files for a player/session."""
    serves = load_serves_from_csv(player, session_id, user_mode)
    
    trajectories = []
    landing_frames = []
    labels = []
    
    for serve in serves:
        serve_id = serve.get("serve_id", "")
        if not serve_id:
            continue
        
        # Build trajectory path
        if user_mode:
            _, _, traj_json = build_user_paths(player, session_id, serve_id)
        else:
            _, traj_json = build_trajectory_paths(player, session_id, serve_id)
        
        if os.path.exists(traj_json):
            try:
                with open(traj_json) as f:
                    trajectory = json.load(f)
                if trajectory and len(trajectory) >= 2:
                    trajectories.append(trajectory)
                    # Get landing frame (may be string or empty)
                    lf_str = serve.get("landing_frame", "").strip()
                    if lf_str:
                        try:
                            landing_frames.append(int(lf_str))
                        except ValueError:
                            landing_frames.append(None)
                    else:
                        landing_frames.append(None)
                    labels.append(f"#{serve_id}")
            except (json.JSONDecodeError, IOError) as e:
                print(f"[warning] Could not load trajectory from {traj_json}: {e}")
                continue
    
    return trajectories, landing_frames, labels


def main():
    parser = argparse.ArgumentParser(
        description="Create trajectory overlay visualization from trajectories",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # User mode
  python scripts/create_trajectory_image.py --player spencer --session 1 --output trajectories.png --user-mode

  # Data mode
  python scripts/create_trajectory_image.py --player spencer --session 1 --output trajectories.png

  # With custom landing frames and labels
  python scripts/create_trajectory_image.py --player spencer --session 1 --output trajectories.png --landing-frames 135 131 155 --labels serve_001 serve_002 serve_003
        """
    )
    add_player_session_serve_args(parser)
    add_user_mode_arg(parser)
    parser.add_argument("--output", "-o", type=str, required=True,
                       help="Output image path")
    parser.add_argument("--landing-frames", type=int, nargs="+",
                       help="Optional landing frame numbers (one per trajectory). If not provided, uses landing_frame from CSV.")
    parser.add_argument("--labels", type=str, nargs="+",
                       help="Optional labels for each trajectory (e.g., serve IDs). If not provided, uses serve IDs automatically.")
    parser.add_argument("--title", type=str,
                       help="Optional title text for the image")
    
    args = parser.parse_args()
    
    if not args.player or args.session is None:
        parser.error("Must specify --player and --session")
    
    player = args.player
    session_id = int(args.session)
    
    # Load trajectories from JSON files
    print(f"Loading trajectories for {player}/session_{session_id}...")
    trajectories, auto_landing_frames, auto_labels = load_trajectories_from_serves(player, session_id, user_mode=args.user_mode)
    if not trajectories:
        parser.error(f"No trajectory files found for {player}/session_{session_id}")
    
    landing_frames = args.landing_frames if args.landing_frames is not None else auto_landing_frames
    labels = args.labels if args.labels is not None else None
    print(f"Found {len(trajectories)} trajectories")
    
    # Create image
    create_trajectory_image(
        trajectories,
        player,
        session_id,
        args.output,
        user_mode=args.user_mode,
        landing_frames=landing_frames,
        labels=labels,
        title=args.title
    )


if __name__ == "__main__":
    main()

