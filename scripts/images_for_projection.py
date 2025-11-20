"""
Generate visualization images for trajectory projection analysis.

Creates annotated images showing ball trajectory in both pixel and world coordinates,
visualizing the homography transformation and projected landing locations.
"""
import os, json, argparse, numpy as np, csv, re, glob, cv2
from common_args import add_player_session_serve_args, build_trajectory_paths, format_serve_number

def load_homography(session_id):
    path = f"data/calibration/homographies/{session_id}.json"
    if not os.path.exists(path):
        raise FileNotFoundError(f"Homography file not found: {path}")
    with open(path) as f:
        data = json.load(f)
    return np.array(data["H"], dtype=np.float64)

def load_corners(session_id):
    """Load court corner annotations for a session."""
    path = f"data/annotations/court_corners/{session_id}.json"
    if not os.path.exists(path):
        raise FileNotFoundError(f"Corner file not found: {path}")
    with open(path) as f:
        return json.load(f)

def load_frame_from_video(video_path, frame_number):
    """Load a specific frame from a video file."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        cap.release()
        raise RuntimeError(f"Could not open video: {video_path}")
    
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
    ret, frame = cap.read()
    cap.release()
    
    if not ret:
        raise RuntimeError(f"Could not read frame {frame_number} from {video_path}")
    
    return frame

def draw_ball_box(frame, center, size, color=(0, 255, 0), thickness=2):
    """Draw a bounding box around the ball on the frame."""
    u, v = center
    width = size[0] if len(size) >= 1 else 20
    height = size[1] if len(size) >= 2 else 20
    
    x1 = int(u - width / 2)
    y1 = int(v - height / 2)
    x2 = int(u + width / 2)
    y2 = int(v + height / 2)
    
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
    cv2.circle(frame, (int(u), int(v)), 3, color, -1)
    return frame

def warp_point(u, v, H):
    p = np.array([u, v, 1.0])
    q = H @ p
    return float(q[0] / q[2]), float(q[1] / q[2])

def get_landing_frame(player, session, serve):
    """Get landing frame from landing_estimates.csv."""
    csv_path = "data/metadata/landing_estimates.csv"
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Landing estimates CSV not found: {csv_path}")
    
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            if (row["player"] == player and 
                str(row["session"]) == str(session) and 
                row["serve_id"] == format_serve_number(serve)):
                landing_actual = row.get("landing_actual", "").strip()
                if landing_actual:
                    try:
                        return int(landing_actual)
                    except ValueError:
                        pass
    return None

def project_landing(player, session, serve):
    """Project landing frame from pixel to world coordinates and save visualization images."""
    video_path, traj_path = build_trajectory_paths(player, session, serve)
    if not os.path.exists(traj_path):
        print(f"[skip] No trajectory for serve {serve}")
        return None

    with open(traj_path) as f:
        traj = json.load(f)

    if not traj:
        print(f"[skip] Empty trajectory for serve {serve}")
        return None

    try:
        landing_frame = get_landing_frame(player, session, serve)
    except FileNotFoundError as e:
        print(f"[skip] {e}")
        return None
    
    if landing_frame is None:
        print(f"[skip] No landing frame in landing_estimates.csv for serve {serve}")
        return None

    # Find nearest frame
    frames = np.array([p["frame"] for p in traj])
    idx = (np.abs(frames - landing_frame)).argmin()
    point = traj[idx]
    u, v = point["center"]
    
    # Calculate bottom of ball: y - (height / 2)
    size = point.get("size", [0.0, 0.0])
    if len(size) >= 2:
        height = size[1]
        v_bottom = v + (height / 2)
    else:
        # Fallback if size not available
        v_bottom = v

    try:
        H = load_homography(f"session_{session}")
    except FileNotFoundError as e:
        print(f"[skip] {e}")
        return None
    
    # Project both center and bottom of ball to world coordinates
    X_center, Y_center = warp_point(u, v, H)
    X, Y = warp_point(u, v_bottom, H)

    out_dir = f"data/landings_world/{player}/session_{session}/"
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"serve_{format_serve_number(serve)}.json")

    with open(out_path, "w") as f:
        json.dump({
            "player": player,
            "session": session,
            "serve_id": format_serve_number(serve),
            "frame": int(frames[idx]),
            "pixel_center": [u, v],
            "pixel_bottom": [u, v_bottom],
            "world_center_m": [X_center, Y_center],
            "world_bottom_m": [X, Y]
        }, f, indent=2)

    # Save visualization images
    try:
        # Load the landing frame from video
        if not os.path.exists(video_path):
            print(f"[skip] Video not found: {video_path}")
        else:
            frame = load_frame_from_video(video_path, landing_frame)
            
            # Draw bounding box around ball
            frame_with_box = draw_ball_box(frame.copy(), [u, v], size, color=(0, 255, 0), thickness=2)
            
            # Save original frame with box
            img_dir = f"data/visualizations/projections/{player}/session_{session}/"
            os.makedirs(img_dir, exist_ok=True)
            original_img_path = os.path.join(img_dir, f"serve_{format_serve_number(serve)}_original.png")
            cv2.imwrite(original_img_path, frame_with_box)
            print(f"Saved original frame → {original_img_path}")
            
            # Create warped top-down view
            try:
                ann = load_corners(f"session_{session}")
                pts = np.float32(ann["court_corners"][:4])  # use 4 outer corners
                
                # Warp the whole image to top-down view
                W, Hm = 900, 1800  # pixels corresponding to 9×18 m (0.01 m/px)
                dst_pts = np.float32([[0, 0], [W, 0], [W, Hm], [0, Hm]])
                M, _ = cv2.findHomography(pts, dst_pts)
                warped = cv2.warpPerspective(frame_with_box, M, (W, Hm))
                
                # Project ball center to warped coordinates
                ball_center_pt = np.array([[u, v]], dtype=np.float32).reshape(-1, 1, 2)
                warped_ball_center = cv2.perspectiveTransform(ball_center_pt, M)[0][0]
                
                # Project landing point (bottom of ball) to warped coordinates
                landing_pt = np.array([[u, v_bottom]], dtype=np.float32).reshape(-1, 1, 2)
                warped_pt = cv2.perspectiveTransform(landing_pt, M)[0][0]
                
                # Draw green dot at ball center on warped view
                cv2.circle(warped, (int(warped_ball_center[0]), int(warped_ball_center[1])), 6, (0, 255, 0), -1)
                
                # Draw landing point on warped view
                cv2.circle(warped, (int(warped_pt[0]), int(warped_pt[1])), 8, (0, 0, 255), -1)
                cv2.circle(warped, (int(warped_pt[0]), int(warped_pt[1])), 12, (0, 0, 255), 2)
                
                # Save warped frame
                warped_img_path = os.path.join(img_dir, f"serve_{format_serve_number(serve)}_warped.png")
                cv2.imwrite(warped_img_path, warped)
                print(f"Saved warped frame → {warped_img_path}")
            except FileNotFoundError as e:
                print(f"[skip] Could not create warped view: {e}")
    except Exception as e:
        print(f"[skip] Could not save images: {e}")

    print(f"Projected landing {serve} → ({X:.2f}m, {Y:.2f}m)")
    return out_path

def main():
    parser = argparse.ArgumentParser(description="Project only the landing frame to world coords")
    add_player_session_serve_args(parser)
    args = parser.parse_args()

    if not args.player or args.session is None:
        parser.error("must specify --player and --session")

    if args.serve:
        project_landing(args.player, args.session, args.serve)
    else:
        # Project all serves in session
        traj_dir = f"data/trajectories/{args.player}/session_{args.session}/"
        if not os.path.exists(traj_dir):
            print(f"Trajectory directory not found: {traj_dir}")
            return
        
        files = sorted(glob.glob(os.path.join(traj_dir, "serve_*.json")))
        if not files:
            print(f"No trajectory files found in {traj_dir}")
            return
        
        print(f"Projecting {len(files)} serves in session {args.session}:")
        for f in files:
            m = re.search(r"serve_(\d+)\.json$", f)
            if m:
                serve_id = m.group(1)
                project_landing(args.player, args.session, serve_id)

if __name__ == "__main__":
    main()
