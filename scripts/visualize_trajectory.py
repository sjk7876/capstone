"""
Visualize ball trajectory on video.

Overlays ball trajectory path on serve videos using trajectory JSON data.
Creates visualization videos showing the ball's path through the frame.
"""
import cv2
import json
import numpy as np
import argparse
import os
from common_args import add_player_session_serve_args, build_trajectory_paths, validate_video_exists


def visualize_trajectory(json_path, video_path, color=(0, 255, 0), line_thickness=2):
    """
    Visualize a trajectory JSON file overlaid on a video.
    
    Args:
        json_path: Path to JSON file containing trajectory data
        video_path: Path to video file
        color: BGR color tuple for trajectory line (default: green)
        line_thickness: Thickness of trajectory line
    """
    # Load trajectory data
    with open(json_path, "r") as f:
        trajectory = json.load(f)
    
    if not trajectory:
        print(f"[error] trajectory file {json_path} is empty")
        return
    
    # Convert to numpy array for easier indexing
    points = np.array([[p["center"][0], p["center"][1]] for p in trajectory])
    frames = np.array([p["frame"] for p in trajectory])
    
    # Open video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"[error] could not open video {video_path}")
        return
    
    frame_idx = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            key = cv2.waitKey(0)
            if key == 27 or key == ord('q'):
                break
        
        # Draw trajectory up to current frame
        frame_mask = frames <= frame_idx
        if np.any(frame_mask):
            visible_points = points[frame_mask]
            if len(visible_points) > 1:
                for j in range(len(visible_points) - 1):
                    pt1 = (int(visible_points[j][0]), int(visible_points[j][1]))
                    pt2 = (int(visible_points[j+1][0]), int(visible_points[j+1][1]))
                    cv2.line(frame, pt1, pt2, color, line_thickness)
            
            # Draw current position
            if len(visible_points) > 0:
                current_pt = (int(visible_points[-1][0]), int(visible_points[-1][1]))
                cv2.circle(frame, current_pt, 5, color, -1)
        
        cv2.imshow("Trajectory Visualization", frame)
        if cv2.waitKey(5) == 27:  # ESC to exit
            break
        
        frame_idx += 1
    
    cap.release()
    cv2.destroyAllWindows()


def save_trajectory_video(json_path, video_path, output_path, color=(0, 255, 0), line_thickness=2):
    """
    Save a trajectory visualization to a video file without displaying it.
    
    Args:
        json_path: Path to JSON file containing trajectory data
        video_path: Path to input video file
        output_path: Path to output video file
        color: BGR color tuple for trajectory line (default: green)
        line_thickness: Thickness of trajectory line
    """
    # Load trajectory data
    with open(json_path, "r") as f:
        trajectory = json.load(f)
    
    if not trajectory:
        print(f"[error] trajectory file {json_path} is empty")
        return
    
    # Convert to numpy array for easier indexing
    points = np.array([[p["center"][0], p["center"][1]] for p in trajectory])
    frames = np.array([p["frame"] for p in trajectory])
    
    # Open input video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"[error] could not open video {video_path}")
        return
    
    # Get video properties
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # Create output directory if needed
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)
    
    # Create video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    if not out.isOpened():
        print(f"[error] could not create output video {output_path}")
        cap.release()
        return
    
    frame_idx = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Draw trajectory up to current frame
        frame_mask = frames <= frame_idx
        if np.any(frame_mask):
            visible_points = points[frame_mask]
            if len(visible_points) > 1:
                for j in range(len(visible_points) - 1):
                    pt1 = (int(visible_points[j][0]), int(visible_points[j][1]))
                    pt2 = (int(visible_points[j+1][0]), int(visible_points[j+1][1]))
                    cv2.line(frame, pt1, pt2, color, line_thickness)
            
            # Draw current position
            if len(visible_points) > 0:
                current_pt = (int(visible_points[-1][0]), int(visible_points[-1][1]))
                cv2.circle(frame, current_pt, 5, color, -1)
        
        out.write(frame)
        frame_idx += 1
    
    cap.release()
    out.release()
    print(f"[save] trajectory video saved to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize trajectory JSON on video")
    
    # Common player/session/serve arguments
    add_player_session_serve_args(parser)
    
    # Option 2: Direct paths (if not using player/session/serve)
    parser.add_argument('--json-path', type=str, default=None,
                       help='Direct path to trajectory JSON file')
    parser.add_argument('--video-path', type=str, default=None,
                       help='Direct path to video file')
    
    parser.add_argument("--color", nargs=3, type=int, default=[0, 255, 0],
                        help="BGR color for trajectory (default: 0 255 0 = green)")
    parser.add_argument("--thickness", type=int, default=2,
                        help="Line thickness (default: 2)")
    
    args = parser.parse_args()
    
    # Resolve paths
    if args.player and args.session is not None and args.serve:
        video_path, json_path = build_trajectory_paths(args.player, args.session, args.serve)
    elif args.json_path and args.video_path:
        json_path = args.json_path
        video_path = args.video_path
    else:
        parser.error("Either specify (--player, --session, --serve) OR (--json-path, --video-path)")
    
    # Check if files exist
    if not os.path.exists(json_path):
        raise SystemExit(f"Error: JSON file not found: {json_path}")
    validate_video_exists(video_path)
    
    visualize_trajectory(
        json_path,
        video_path,
        color=tuple(args.color),
        line_thickness=args.thickness
    )

