"""
Compute homography matrix from court annotations.

Calculates the transformation matrix that maps pixel coordinates to world coordinates
(meters) based on annotated court corners and center line points.
"""
import os
import json
import csv
import cv2
import numpy as np
import argparse
from common_args import add_player_session_serve_args, add_user_mode_arg, normalize_path

def get_court_corners_path(session_id, user_mode=False):
    """Get court corners path from CSV or fallback to direct path."""
    if user_mode:
        # Try CSV first
        csv_path = os.path.join("user", "data", "court_corners.csv")
        if os.path.exists(csv_path):
            with open(csv_path, "r", newline="") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    if row["session_id"] == session_id:
                        # Normalize path in case it was stored with Windows backslashes
                        corners_path = normalize_path(row["court_corners_path"])
                        if os.path.exists(corners_path):
                            return corners_path
        # Fallback to direct path
        return os.path.join("user", "data", "annotations", "court_corners", f"{session_id}.json")
    else:
        return os.path.join("data", "annotations", "court_corners", f"{session_id}.json")


def compute_homography(session_id, user_mode=False):
    ann_path = get_court_corners_path(session_id, user_mode)
    if not os.path.exists(ann_path):
        print(f"[!] missing corner file: {ann_path}")
        return

    with open(ann_path) as f:
        ann = json.load(f)

    corners = ann["court_corners"]
    if len(corners) < 6:
        print(f"[!] not enough points in {ann_path} (need 6, got {len(corners)})")
        return

    # pixel coords (order: closeL, closeR, farR, farL, centerL, centerR)  # left/right order
    pts_img = np.float32([
        corners[3],  # close left
        corners[2],  # close right
        corners[0],  # far left
        corners[1],  # far right
        corners[4],  # center left
        corners[5],  # center right
    ])

    # world coords in meters (origin = near-baseline left corner)
    pts_world = np.float32([
        [0, 0],    # close left
        [9, 0],    # close right
        [0, 18],   # far left
        [9, 18],   # far right
        [0, 9],    # center left
        [9, 9],    # center right
    ])

    H, mask = cv2.findHomography(pts_img, pts_world, cv2.RANSAC, 2.0)
    if H is None:
        print(f"Homography failed for {session_id}")
        return

    if user_mode:
        out_dir = os.path.join("user", "data", "calibration", "homographies")
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, f"{session_id}.json")
    else:
        out_dir = os.path.join("data", "calibration", "homographies")
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, f"{session_id}.json")

    out = {
        "session_id": session_id,
        "origin": "near_left_corner",
        "axes": {"x": "sideline-sideline", "y": "baseline-baseline"},
        "court_size_m": [9.0, 18.0],
        "H": H.tolist()
    }

    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"Saved homography → {out_path}")

    return H


def warp_point(u, v, H):
    p = np.array([u, v, 1.0])
    q = H @ p
    return (q[0]/q[2], q[1]/q[2])


def main():
    parser = argparse.ArgumentParser(description="Compute court homography for session")
    
    # Common player/session/serve arguments (only session is required)
    add_player_session_serve_args(parser)
    add_user_mode_arg(parser)
    
    args = parser.parse_args()
    
    if args.session is None:
        parser.error("Must specify --session")
    
    session_id = f"session_{args.session}"
    compute_homography(session_id, args.user_mode)


if __name__ == "__main__":
    main()
