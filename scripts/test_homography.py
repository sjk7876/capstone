import os
import json
import argparse
import cv2
import numpy as np
from common_args import add_player_session_serve_args

def load_corners(session_id):
    """Load court corner annotations for a session."""
    path = f"data/annotations/court_corners/{session_id}.json"
    if not os.path.exists(path):
        raise FileNotFoundError(f"Corner file not found: {path}")
    with open(path) as f:
        return json.load(f)

def load_homography(session_id):
    """Load homography matrix for a session."""
    path = f"data/calibration/homographies/{session_id}.json"
    if not os.path.exists(path):
        raise FileNotFoundError(f"Homography file not found: {path}")
    with open(path) as f:
        data = json.load(f)
    return np.array(data["H"], dtype=np.float64)

def warp_point(u, v, H):
    """Project pixel (u,v) → world (X,Y) using homography."""
    p = np.array([u, v, 1.0])
    q = H @ p
    return q[0] / q[2], q[1] / q[2]

def test_homography(session_id, show=True):
    """Test homography by measuring court dimensions and showing warped view."""
    ann = load_corners(session_id)
    H = load_homography(session_id)

    img_path = ann.get("video_file")
    if not img_path:
        raise ValueError(f"No video_file in annotation for {session_id}")
    
    cap = cv2.VideoCapture(img_path)
    if not cap.isOpened():
        cap.release()
        raise RuntimeError(f"Could not open video: {img_path}")
    
    ok, frame = cap.read()
    cap.release()
    if not ok:
        raise RuntimeError(f"Could not read first frame of {img_path}")

    pts = np.float32(ann["court_corners"][:4])  # use 4 outer corners
    names = ["farL", "farR", "closeR", "closeL"]

    # Draw on frame
    for (x, y), name in zip(pts, names):
        cv2.circle(frame, (int(x), int(y)), 6, (0, 255, 0), -1)
        cv2.putText(frame, name, (int(x) + 8, int(y) - 8), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    # Warp to world
    world_pts = np.array([warp_point(x, y, H) for (x, y) in pts])
    w_close = np.linalg.norm(world_pts[2] - world_pts[3])   # closeR-closeL
    w_far = np.linalg.norm(world_pts[1] - world_pts[0])     # farR-farL
    l_left = np.linalg.norm(world_pts[0] - world_pts[3])    # farL-closeL
    l_right = np.linalg.norm(world_pts[1] - world_pts[2])   # farR-closeR

    print(f"Width near baseline: {w_close:.2f} m (expected: 9.0 m)")
    print(f"Width far baseline:  {w_far:.2f} m (expected: 9.0 m)")
    print(f"Length left side:    {l_left:.2f} m (expected: 18.0 m)")
    print(f"Length right side:   {l_right:.2f} m (expected: 18.0 m)")

    # Warp the whole image to top-down view
    W, Hm = 900, 1800  # pixels corresponding to 9×18 m (0.01 m/px)
    dst_pts = np.float32([[0, 0], [W, 0], [W, Hm], [0, Hm]])
    M, _ = cv2.findHomography(pts, dst_pts)
    top = cv2.warpPerspective(frame, M, (W, Hm))

    if show:
        cv2.imshow("original (with points)", frame)
        cv2.imshow("warped top-down", top)
        print("Press any key to close windows...")
        cv2.waitKey(0)
        cv2.destroyAllWindows()

def main():
    parser = argparse.ArgumentParser(description="Visual sanity-check for session homography")
    
    # Common player/session/serve arguments (only session is required)
    add_player_session_serve_args(parser)
    
    parser.add_argument("--no-show", action="store_true", help="Skip GUI display")
    
    args = parser.parse_args()
    
    if args.session is None:
        parser.error("Must specify --session")
    
    session_id = f"session_{args.session}"
    test_homography(session_id, show=not args.no_show)

if __name__ == "__main__":
    main()
