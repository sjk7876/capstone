"""
Create landing locations visualization from a list of coordinates.

Given a list of (x, y) coordinates in meters, creates a top-down court view
with landing locations marked, similar to process_serves.py output.
"""
import os
import sys
import json
import argparse
import cv2
import numpy as np
from pathlib import Path

# Add scripts directory to path for imports
script_dir = os.path.dirname(__file__)
sys.path.insert(0, script_dir)

from common_args import add_player_session_serve_args, add_user_mode_arg, build_user_paths, get_user_serves_csv_path, format_serve_number, normalize_path


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


def create_landing_image(coordinates, player, session_id, output_path, user_mode=False, labels=None):
    """
    Create landing locations visualization from coordinates, with a blurred
    heatmap plus explicit landing markers.

    Args:
        coordinates: List of (x, y) tuples in meters, or list of dicts with 'x', 'y' keys
        player: Player name
        session_id: Session ID (integer)
        output_path: Path to save output image
        user_mode: Whether to use user/ paths
        labels: Optional list of labels for each coordinate (e.g., serve IDs)
    """
    # Normalize coordinates format
    coords_list = []
    for coord in coordinates:
        if isinstance(coord, dict):
            x, y = coord.get("x"), coord.get("y")
        elif isinstance(coord, (list, tuple)) and len(coord) >= 2:
            x, y = coord[0], coord[1]
        else:
            continue
        if x is not None and y is not None:
            coords_list.append({"x": float(x), "y": float(y)})

    if not coords_list:
        print("[error] No valid coordinates provided")
        return False

    # Filter to far side only (y > 9m)
    valid_coords = [c for c in coords_list if c["y"] > 9.0]

    if not valid_coords:
        print("[warning] No coordinates on far side (y > 9m)")
        return False

    # Load court corners to get video path
    corners_data = load_corners(session_id, user_mode)
    if not corners_data:
        print(f"[warning] No court corners found for session {session_id}, using blank canvas")
        W, Hm = 900, 1800
        warped = np.ones((Hm, W, 3), dtype=np.uint8) * 255
    else:
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
                print(f"[warning] Could not load frame from {video_path}, using blank canvas")
                W, Hm = 900, 1800
                warped = np.ones((Hm, W, 3), dtype=np.uint8) * 255
            else:
                # Warp frame to top-down view
                # Corner order in annotation: [0]=top-left (farL), [1]=top-right (farR), 
                # [2]=bottom-right (closeR), [3]=bottom-left (closeL)
                # Map to world coords: Y=0 is near baseline, Y=18 is far baseline
                corners = corners_data["court_corners"][:4]
                pts = np.float32([
                    corners[3],  # close left → (0, 0)
                    corners[2],  # close right → (9, 0)
                    corners[1],  # far right → (9, 18)
                    corners[0],  # far left → (0, 18)
                ])
                W, Hm = 900, 1800  # pixels corresponding to 9×18 m (0.01 m/px)
                # dst_pts: near-left=(0,Hm), near-right=(W,Hm), far-right=(W,0), far-left=(0,0)
                dst_pts = np.float32([[0, Hm], [W, Hm], [W, 0], [0, 0]])
                M, _ = cv2.findHomography(pts, dst_pts)
                warped = cv2.warpPerspective(frame, M, (W, Hm))
        else:
            print(f"[warning] No video found for session {session_id}, using blank canvas")
            W, Hm = 900, 1800
            warped = np.ones((Hm, W, 3), dtype=np.uint8) * 255

    # ---------- build heatmap over full warped image ----------
    heatmap = np.zeros((Hm, W), dtype=np.float32)

    for coord in valid_coords:
        x_px = int(coord["x"] * 100)
        y_px_world = int(coord["y"] * 100)

        if y_px_world <= 900:
            continue

        y_px_warped = Hm - y_px_world

        if 0 <= x_px < W and 0 <= y_px_warped < Hm:
            heatmap[y_px_warped, x_px] += 1.0

    # ---------- gaussian glow (BRIGHT + LARGE + NO DARKENING) ----------
    max_raw = float(np.max(heatmap))
    if max_raw > 0:
        # big blur
        heat_blur = cv2.GaussianBlur(heatmap, (0, 0), sigmaX=35, sigmaY=35)

        # normalize 0–255 based on *blurred* max
        heat_norm = cv2.normalize(heat_blur, None, 0, 255, cv2.NORM_MINMAX)

        # optional extra punch
        heat_norm = np.power(heat_norm / 255.0, 1.0) * 255.0  # gamma < 1.0 = brighter blobs

        heat_uint8 = heat_norm.astype(np.uint8)

        # epsilon threshold: only colorize values above threshold
        epsilon = 0.5  # minimum value to show (out of 255)
        mask = heat_uint8 > epsilon
        
        # colorize only above threshold
        heat_color = cv2.applyColorMap(heat_uint8, cv2.COLORMAP_TURBO)
        
        # apply mask: only add color where above epsilon
        heat_color_masked = np.zeros_like(heat_color)
        heat_color_masked[mask] = heat_color[mask]

        # additive blending with transparency (preserves background brightness)
        heatmap_alpha = 0.7  # 70% opacity
        heat_scaled = (heat_color_masked * heatmap_alpha).astype(np.uint8)
        warped = cv2.add(warped, heat_scaled)
    else:
        print("[warning] Heatmap empty; skipping glow")

    # ---------- draw explicit landing points on top (semi-transparent) ----------
    overlay = warped.copy()
    for i, coord in enumerate(valid_coords):
        x_px = int(coord["x"] * 100)
        y_px_world = int(coord["y"] * 100)

        if y_px_world <= 900:
            continue

        y_px_warped = Hm - y_px_world

        if 0 <= x_px < W and 0 <= y_px_warped < Hm:
            # red circle for landing point (drawn on overlay)
            cv2.circle(overlay, (x_px, y_px_warped), 5, (0, 0, 255), -1)

            # Add label only if provided
            if labels and i < len(labels):
                label = str(labels[i])
                cv2.putText(
                    overlay,
                    label,
                    (x_px + 15, y_px_warped - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (255, 255, 255),
                    2,
                )
    
    # Blend overlay with alpha transparency
    alpha = 0.6  # 60% opacity
    warped = cv2.addWeighted(warped, 1.0 - alpha, overlay, alpha, 0)

    # ---------- draw court zone lines (6 zones: 3 columns x 2 rows) ----------
    # Far side is 9m x 9m (900px x 900px in cropped view)
    # Zones: 1=top-left, 2=bottom-left, 3=bottom-middle, 4=bottom-right, 5=top-right, 6=top-middle
    # Vertical lines at 3m and 6m (x=300px and x=600px)
    # Horizontal line at 13.5m (y=450px in cropped, which is y=450px in full warped for far side)
    
    line_color = (200, 200, 200)  # Light gray
    line_thickness = 1
    dash_length = 10
    gap_length = 5
    
    # Draw vertical lines (3 columns)
    for x_pos in [300, 600]:  # 3m and 6m boundaries
        y_start = 0
        y_end = 900  # Far side region
        y = y_start
        while y < y_end:
            y_end_seg = min(y + dash_length, y_end)
            cv2.line(warped, (x_pos, y), (x_pos, y_end_seg), line_color, line_thickness)
            y += dash_length + gap_length
    
    # Draw horizontal line (2 rows)
    y_pos = 450  # 13.5m boundary (middle of far side)
    x_start = 0
    x_end = 900
    x = x_start
    while x < x_end:
        x_end_seg = min(x + dash_length, x_end)
        cv2.line(warped, (x, y_pos), (x_end_seg, y_pos), line_color, line_thickness)
        x += dash_length + gap_length
    
    # Add zone numbers in corners
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1.2
    font_thickness = 2
    text_color = (150, 150, 150)  # Gray
    text_offset = 20  # Padding from edges
    
    # Zone 1: Top left (0-300px x, 0-450px y)
    cv2.putText(warped, "1", (text_offset, text_offset + 25), font, font_scale, text_color, font_thickness)
    
    # Zone 2: Bottom left (0-300px x, 450-900px y)
    cv2.putText(warped, "2", (text_offset, 900 - text_offset), font, font_scale, text_color, font_thickness)
    
    # Zone 3: Bottom middle (300-600px x, 450-900px y)
    text_size = cv2.getTextSize("3", font, font_scale, font_thickness)[0]
    cv2.putText(warped, "3", (450 - text_size[0] // 2, 900 - text_offset), font, font_scale, text_color, font_thickness)
    
    # Zone 4: Bottom right (600-900px x, 450-900px y)
    text_size = cv2.getTextSize("4", font, font_scale, font_thickness)[0]
    cv2.putText(warped, "4", (900 - text_size[0] - text_offset, 900 - text_offset), font, font_scale, text_color, font_thickness)
    
    # Zone 5: Top right (600-900px x, 0-450px y)
    text_size = cv2.getTextSize("5", font, font_scale, font_thickness)[0]
    cv2.putText(warped, "5", (900 - text_size[0] - text_offset, text_offset + 25), font, font_scale, text_color, font_thickness)
    
    # Zone 6: Top middle (300-600px x, 0-450px y)
    text_size = cv2.getTextSize("6", font, font_scale, font_thickness)[0]
    cv2.putText(warped, "6", (450 - text_size[0] // 2, text_offset + 25), font, font_scale, text_color, font_thickness)

    # Crop to far side only (Y from 0 to 900 in warped image, which is 9m to 18m in world)
    # In warped image: top (Y=0) is far baseline, so far side is Y=0 to Y=900
    warped_far = warped[0:900, :]

    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)
    cv2.imwrite(output_path, warped_far)
    print(f"Saved landing locations image → {output_path}")
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
                serves.append(row)
    return serves


def load_coordinates_from_serves(player, session_id, user_mode=False):
    """Load landing coordinates from serves CSV for a player/session."""
    serves = load_serves_from_csv(player, session_id, user_mode)
    
    coordinates = []
    labels = []
    for serve in serves:
        landing_x = serve.get("landing_x", "").strip()
        landing_y = serve.get("landing_y", "").strip()
        
        if landing_x and landing_y:
            try:
                x = float(landing_x)
                y = float(landing_y)
                coordinates.append({"x": x, "y": y})
                labels.append(f"#{serve.get('serve_id', '')}")
            except ValueError:
                continue
    
    return coordinates, labels


def main():
    parser = argparse.ArgumentParser(
        description="Create landing locations visualization from coordinates",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # User mode
  python scripts/create_landing_image.py --player spencer --session 1 --output landing.png --user-mode

  # Data mode
  python scripts/create_landing_image.py --player spencer --session 1 --output landing.png
        """
    )
    add_player_session_serve_args(parser)
    add_user_mode_arg(parser)
    parser.add_argument("--output", "-o", type=str, required=True,
                       help="Output image path")
    parser.add_argument("--labels", type=str, nargs="+",
                       help="Optional labels for each coordinate (e.g., serve IDs). If not provided, uses serve IDs automatically.")
    
    args = parser.parse_args()
    
    if not args.player or args.session is None:
        parser.error("Must specify --player and --session")
    
    player = args.player
    session_id = int(args.session)
    
    # Load coordinates from CSV
    print(f"Loading coordinates from CSV for {player}/session_{session_id}...")
    coordinates, auto_labels = load_coordinates_from_serves(player, session_id, user_mode=args.user_mode)
    if not coordinates:
        parser.error(f"No landing coordinates found in CSV for {player}/session_{session_id}")
    
    labels = args.labels if args.labels is not None else auto_labels
    print(f"Found {len(coordinates)} landing coordinates")
    
    # Create image
    create_landing_image(
        coordinates,
        player,
        session_id,
        args.output,
        user_mode=args.user_mode,
        labels=labels
    )


if __name__ == "__main__":
    main()

