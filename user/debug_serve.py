"""
Debug serve visualization: show hit/landing frames and create overlay videos.

Shows:
1. Image of hit frame
2. Image of landing frame
3. Video with trajectory overlay
4. Video with detection overlay
"""
import os
import sys
import json
import argparse
import cv2
import numpy as np
import glob
import re
from pathlib import Path

# Add scripts directory to path
script_dir = os.path.join(os.path.dirname(__file__), "..", "scripts")
sys.path.insert(0, script_dir)

from common_args import add_player_session_serve_args, build_user_paths, format_serve_number
from estimate_landing import estimate_hit_and_landing


def load_frame_from_video(video_path, frame_number):
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


def draw_detection(frame, detection, color=(0, 255, 255), thickness=2):
    """Draw a single detection bounding box on frame."""
    center = detection["center"]
    size = detection.get("size", [0.0, 0.0])
    conf = detection.get("conf", 0.0)
    
    x1 = int(center[0] - size[0] / 2)
    y1 = int(center[1] - size[1] / 2)
    x2 = int(center[0] + size[0] / 2)
    y2 = int(center[1] + size[1] / 2)
    
    # Draw bounding box
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
    
    # Draw center point
    cv2.circle(frame, (int(center[0]), int(center[1])), 5, color, -1)
    
    # Draw confidence
    label = f"{conf:.2f}"
    cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
    
    return frame


def save_frame_image(frame, output_path, title=None):
    """Save a frame as an image with optional title overlay."""
    if title:
        cv2.putText(frame, title, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
    
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)
    cv2.imwrite(output_path, frame)
    print(f"Saved frame image → {output_path}")


def create_trajectory_video(video_path, trajectory, output_path, hit_frame=None, landing_frame=None):
    """Create video with trajectory overlay."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"[error] Could not open video: {video_path}")
        return False
    
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    if not out.isOpened():
        print(f"[error] Could not create output video: {output_path}")
        cap.release()
        return False
    
    # Convert trajectory to numpy for easier indexing
    traj_points = np.array([[p["center"][0], p["center"][1]] for p in trajectory])
    traj_frames = np.array([p["frame"] for p in trajectory])
    
    frame_idx = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Draw trajectory up to current frame
        frame_mask = traj_frames <= frame_idx
        if np.any(frame_mask):
            visible_points = traj_points[frame_mask]
            if len(visible_points) > 1:
                # Draw trajectory line
                for j in range(len(visible_points) - 1):
                    pt1 = (int(visible_points[j][0]), int(visible_points[j][1]))
                    pt2 = (int(visible_points[j+1][0]), int(visible_points[j+1][1]))
                    cv2.line(frame, pt1, pt2, (0, 255, 0), 3)
            
            # Draw current position
            if len(visible_points) > 0:
                current_pt = (int(visible_points[-1][0]), int(visible_points[-1][1]))
                cv2.circle(frame, current_pt, 8, (0, 255, 0), -1)
                cv2.circle(frame, current_pt, 12, (0, 255, 0), 2)
        
        # Mark hit frame
        if hit_frame is not None and frame_idx == hit_frame:
            cv2.putText(frame, "HIT", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 255), 3)
            # Draw green circle at hit point
            hit_mask = traj_frames == hit_frame
            if np.any(hit_mask):
                hit_pt = traj_points[hit_mask][0]
                cv2.circle(frame, (int(hit_pt[0]), int(hit_pt[1])), 15, (0, 255, 255), 3)
        
        # Mark landing frame
        if landing_frame is not None and frame_idx == landing_frame:
            cv2.putText(frame, "LANDING", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)
            # Draw red circle at landing point
            landing_mask = traj_frames == landing_frame
            if np.any(landing_mask):
                landing_pt = traj_points[landing_mask][0]
                cv2.circle(frame, (int(landing_pt[0]), int(landing_pt[1])), 15, (0, 0, 255), 3)
        
        # Frame number
        cv2.putText(frame, f"Frame: {frame_idx}", (10, height - 20), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        out.write(frame)
        frame_idx += 1
    
    cap.release()
    out.release()
    print(f"Saved trajectory video → {output_path}")
    return True


def find_trajectory_files(player, session_id):
    """Find all trajectory JSON files in a session directory."""
    traj_dir = os.path.join("user", "data", "trajectories", player, f"session_{session_id}")
    if not os.path.exists(traj_dir):
        return []
    
    json_files = sorted(glob.glob(os.path.join(traj_dir, "serve_*.json")))
    return json_files


def extract_serve_id_from_path(json_path):
    """Extract serve_id from JSON path (e.g., 'serve_001.json' -> '001')."""
    basename = os.path.basename(json_path)
    match = re.search(r"serve_(\d+)\.json$", basename)
    if match:
        return format_serve_number(match.group(1))
    return None


def process_single_serve(player, session_id, serve_id, no_video=False):
    """Process a single serve: load data and create debug visualizations."""
    # Build paths
    video_path, detect_json, traj_json = build_user_paths(player, session_id, serve_id)
    
    # Check if files exist
    if not os.path.exists(video_path):
        print(f"[skip] {player}/session_{session_id}/serve_{serve_id}: video not found: {video_path}")
        return False
    
    if not os.path.exists(traj_json):
        print(f"[skip] {player}/session_{session_id}/serve_{serve_id}: trajectory JSON not found: {traj_json}")
        return False
    
    if not os.path.exists(detect_json):
        print(f"[skip] {player}/session_{session_id}/serve_{serve_id}: detection JSON not found: {detect_json}")
        return False
    
    # Load trajectory
    with open(traj_json) as f:
        trajectory = json.load(f)
    
    if not trajectory or len(trajectory) < 3:
        print(f"[skip] {player}/session_{session_id}/serve_{serve_id}: insufficient trajectory data")
        return False
    
    # Estimate hit and landing frames
    hit_frame, landing_frame = estimate_hit_and_landing(trajectory)
    
    if hit_frame is None or landing_frame is None:
        print(f"[skip] {player}/session_{session_id}/serve_{serve_id}: could not estimate hit or landing frame")
        return False
    
    print(f"\n[{player}/session_{session_id}/serve_{serve_id}] Hit: {hit_frame}, Landing: {landing_frame}")
    
    # Load detections
    with open(detect_json) as f:
        detections = json.load(f)
    
    # Create output directory
    output_dir = os.path.join("user", "debug", player, f"session_{session_id}", f"serve_{serve_id}")
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Save hit frame image
    hit_frame_img = load_frame_from_video(video_path, hit_frame)
    if hit_frame_img is not None:
        hit_img_path = os.path.join(output_dir, "hit_frame.png")
        save_frame_image(hit_frame_img, hit_img_path, f"Hit Frame: {hit_frame}")
    else:
        print(f"  [warning] Could not load hit frame {hit_frame}")
    
    # 2. Save landing frame image
    landing_frame_img = load_frame_from_video(video_path, landing_frame)
    if landing_frame_img is not None:
        landing_img_path = os.path.join(output_dir, "landing_frame.png")
        save_frame_image(landing_frame_img, landing_img_path, f"Landing Frame: {landing_frame}")
    else:
        print(f"  [warning] Could not load landing frame {landing_frame}")
    
    # 3. Create trajectory overlay video (unless disabled)
    if not no_video:
        traj_video_path = os.path.join(output_dir, "trajectory_overlay.mp4")
        create_trajectory_video(video_path, trajectory, traj_video_path, hit_frame, landing_frame)
    
    # 4. Create detection overlay video (unless disabled)
    if not no_video:
        det_video_path = os.path.join(output_dir, "detection_overlay.mp4")
        create_detection_video(video_path, detections, det_video_path, hit_frame, landing_frame)
    
    print(f"  ✓ Debug outputs saved to: {output_dir}")
    return True


def create_detection_video(video_path, detections, output_path, hit_frame=None, landing_frame=None):
    """Create video with detection overlay."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"[error] Could not open video: {video_path}")
        return False
    
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    if not out.isOpened():
        print(f"[error] Could not create output video: {output_path}")
        cap.release()
        return False
    
    frame_idx = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Draw detections for this frame
        frame_str = str(frame_idx)
        if frame_str in detections:
            for det in detections[frame_str]:
                # Color based on confidence
                conf = det.get("conf", 0.0)
                if conf >= 0.5:
                    color = (0, 255, 0)  # Green for high confidence
                elif conf >= 0.3:
                    color = (0, 255, 255)  # Yellow for medium confidence
                else:
                    color = (0, 165, 255)  # Orange for low confidence
                
                draw_detection(frame, det, color=color, thickness=2)
        
        # Mark hit frame
        if hit_frame is not None and frame_idx == hit_frame:
            cv2.putText(frame, "HIT", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 255), 3)
        
        # Mark landing frame
        if landing_frame is not None and frame_idx == landing_frame:
            cv2.putText(frame, "LANDING", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)
        
        # Frame number
        cv2.putText(frame, f"Frame: {frame_idx}", (10, height - 20), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        out.write(frame)
        frame_idx += 1
    
    cap.release()
    out.release()
    print(f"Saved detection video → {output_path}")
    return True


def main():
    parser = argparse.ArgumentParser(
        description="Debug serve visualization: show hit/landing frames and create overlay videos",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Debug serve 001 for spencer session 1
  python user/debug_serve.py --player spencer --session 1 --serve 001

  # Debug all serves in session 1
  python user/debug_serve.py --player spencer --session 1

  # Debug all serves without generating videos (faster)
  python user/debug_serve.py --player spencer --session 1 --no-video
        """
    )
    add_player_session_serve_args(parser)
    parser.add_argument("-n", "--no-video", action="store_true",
                       help="Skip video generation (only create frame images)")
    args = parser.parse_args()
    
    if not args.player or args.session is None:
        parser.error("Must specify --player and --session")
    
    player = args.player
    session_id = int(args.session)
    
    if args.serve:
        # Single serve specified
        serve_id = format_serve_number(args.serve)
        print(f"Processing serve {serve_id}...")
        process_single_serve(player, session_id, serve_id, no_video=args.no_video)
    else:
        # Process all serves in session
        json_files = find_trajectory_files(player, session_id)
        if not json_files:
            print(f"No trajectory files found in user/data/trajectories/{player}/session_{session_id}/")
            print("Run process_serves.py first to generate trajectory data")
            return
        
        print(f"Processing {len(json_files)} serves in session {session_id}:")
        if args.no_video:
            print("(Video generation disabled)")
        successful = 0
        for json_path in json_files:
            serve_id = extract_serve_id_from_path(json_path)
            if serve_id:
                if process_single_serve(player, session_id, serve_id, no_video=args.no_video):
                    successful += 1
        
        print(f"\n✓ Completed: {successful}/{len(json_files)} serves processed successfully")


if __name__ == "__main__":
    main()

