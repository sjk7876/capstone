"""
Add court corners to an image with ball bounding boxes.

Takes player/session/serve, loads image and labels from datasets/ball_yolo/,
adds the 6 court corners from session annotations, and saves the result.
"""
import cv2
import json
import os
import argparse
import numpy as np
import glob
from common_args import add_player_session_serve_args, format_serve_number


def load_corners(session_id):
    """Load court corner annotations for a session."""
    path = os.path.join("data", "annotations", "court_corners", f"{session_id}.json")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Corner file not found: {path}")
    with open(path) as f:
        return json.load(f)

def load_homography(session_id):
    """Load homography matrix for a session."""
    path = os.path.join("data", "calibration", "homographies", f"{session_id}.json")
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


def find_image_in_datasets(player, session, serve, split="train", frame=None):
    """Find image file in datasets/ball_yolo/images/."""
    serve_str = format_serve_number(serve)
    
    if frame is not None:
        # Search for specific frame
        frame_str = f"frame{int(frame):06d}"
        pattern = f"{player}_session_{session}_serve_{serve_str}_{frame_str}.jpg"
    else:
        # Search for any frame (first match)
        pattern = f"{player}_session_{session}_serve_{serve_str}_frame*.jpg"
    
    # Try train first, then val
    for dataset_split in [split, "train", "val"]:
        search_path = os.path.join("datasets", "ball_yolo", "images", dataset_split, pattern)
        matches = glob.glob(search_path)
        if matches:
            return matches[0], dataset_split
    
    return None, None

def find_label_in_datasets(img_path, split):
    """Find corresponding label file in datasets/ball_yolo/labels/."""
    if img_path is None:
        return None
    
    img_name = os.path.basename(img_path)
    label_name = os.path.splitext(img_name)[0] + ".txt"
    label_path = os.path.join("datasets", "ball_yolo", "labels", split, label_name)
    
    if os.path.exists(label_path):
        return label_path
    return None

def load_yolo_labels(txt_path, img_width, img_height):
    """Load ball bounding boxes from YOLO format .txt file."""
    boxes = []
    if not txt_path or not os.path.exists(txt_path):
        return boxes
    
    with open(txt_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            
            parts = line.split()
            if len(parts) < 5:
                continue
            
            cls = int(parts[0])
            x_norm = float(parts[1])
            y_norm = float(parts[2])
            w_norm = float(parts[3])
            h_norm = float(parts[4])
            
            # Convert normalized YOLO coordinates to pixel coordinates
            cx = x_norm * img_width
            cy = y_norm * img_height
            w = w_norm * img_width
            h = h_norm * img_height
            
            x1 = int(cx - w / 2)
            y1 = int(cy - h / 2)
            x2 = int(cx + w / 2)
            y2 = int(cy + h / 2)
            
            boxes.append({
                "bbox": [x1, y1, x2, y2],
                "center": [cx, cy],
                "size": [w, h],
                "class": cls
            })
    
    return boxes

def draw_ball_boxes(img, boxes, color=(0, 255, 0), thickness=2):
    """Draw ball bounding boxes on image with labels."""
    for i, box in enumerate(boxes):
        x1, y1, x2, y2 = box["bbox"]
        cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness)
        
        # Draw center point
        cx, cy = int(box["center"][0]), int(box["center"][1])
        cv2.circle(img, (cx, cy), 3, color, -1)
        
        # Draw label
        label = f"Ball {i+1}"
        label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        label_y = y1 - 10 if y1 - 10 > 10 else y1 + label_size[1] + 10
        
        # Draw label background
        cv2.rectangle(img, (x1, label_y - label_size[1] - 5), 
                     (x1 + label_size[0] + 5, label_y + 5), color, -1)
        cv2.putText(img, label, (x1 + 2, label_y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)


def draw_court_corners(img, corners):
    """Draw all 6 court corners on image (4 corners + 2 center line points)."""
    if len(corners) != 6:
        raise ValueError(f"Expected 6 court corners, got {len(corners)}")
    
    pts = np.float32(corners)
    names = ["farL", "farR", "closeR", "closeL", "centerL", "centerR"]
    
    # Draw all 6 points
    for i, ((x, y), name) in enumerate(zip(pts, names)):
        if i < 4:
            # 4 outer corners in green
            color = (0, 255, 0)
        else:
            # 2 center line points in red
            color = (0, 0, 255)
        
        x_int, y_int = int(x), int(y)
        cv2.circle(img, (x_int, y_int), 8, color, -1)
        cv2.circle(img, (x_int, y_int), 12, color, 2)
        cv2.putText(img, f"{i+1}:{name}", (x_int + 8, y_int - 8), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    
    # Draw lines connecting the 4 outer corners
    corner_indices = [0, 1, 2, 3, 0]  # Close the rectangle
    for i in range(len(corner_indices) - 1):
        idx1 = corner_indices[i]
        idx2 = corner_indices[i + 1]
        pt1 = (int(pts[idx1][0]), int(pts[idx1][1]))
        pt2 = (int(pts[idx2][0]), int(pts[idx2][1]))
        cv2.line(img, pt1, pt2, (0, 255, 0), 1)
    
    # Draw center line
    pt1 = (int(pts[4][0]), int(pts[4][1]))
    pt2 = (int(pts[5][0]), int(pts[5][1]))
    cv2.line(img, pt1, pt2, (0, 0, 255), 2)


def add_court_corners_to_image(player, session, serve, output_path=None, split="train", frame=None):
    """
    Add court corners to an image with ball bounding boxes from datasets/.
    
    Args:
        player: Player name
        session: Session number
        serve: Serve number
        output_path: Optional output path (default: saves to data/visualizations/court_corners/)
        split: Dataset split to search ("train" or "val")
        frame: Optional frame number to specify (if None, uses first match)
    """
    # Find image in datasets
    img_path, found_split = find_image_in_datasets(player, session, serve, split, frame)
    if img_path is None:
        raise FileNotFoundError(f"Image not found in datasets/ball_yolo/images/ for {player}/session_{session}/serve_{serve}")
    
    print(f"Found image: {img_path}")
    
    # Load image
    img = cv2.imread(img_path)
    if img is None:
        raise ValueError(f"Could not load image: {img_path}")
    
    img_height, img_width = img.shape[:2]
    
    # Find and load label file
    label_path = find_label_in_datasets(img_path, found_split)
    if label_path:
        print(f"Found label: {label_path}")
        boxes = load_yolo_labels(label_path, img_width, img_height)
        print(f"Loaded {len(boxes)} ball bounding box(es) from label file")
    else:
        print("No label file found, drawing only court corners")
        boxes = []
    
    # Load court corner annotations (all 6 points)
    session_id = f"session_{session}"
    ann = load_corners(session_id)
    corners = ann.get("court_corners", [])
    if len(corners) != 6:
        raise ValueError(f"Expected 6 court corners, got {len(corners)}")
    
    # Draw ball boxes with labels
    if boxes:
        draw_ball_boxes(img, boxes)
    
    # Draw all 6 court corners
    draw_court_corners(img, corners)
    
    # Optionally compute and display homography measurements if available
    try:
        H = load_homography(session_id)
        pts = np.float32(corners[:4])  # use 4 outer corners
        world_pts = np.array([warp_point(x, y, H) for (x, y) in pts])
        w_close = np.linalg.norm(world_pts[2] - world_pts[3])   # closeR-closeL
        w_far = np.linalg.norm(world_pts[1] - world_pts[0])     # farR-farL
        l_left = np.linalg.norm(world_pts[0] - world_pts[3])    # farL-closeL
        l_right = np.linalg.norm(world_pts[1] - world_pts[2])   # farR-closeR
        
        print(f"Court dimensions (from homography):")
        print(f"  Width near baseline: {w_close:.2f} m (expected: 9.0 m)")
        print(f"  Width far baseline:  {w_far:.2f} m (expected: 9.0 m)")
        print(f"  Length left side:    {l_left:.2f} m (expected: 18.0 m)")
        print(f"  Length right side:   {l_right:.2f} m (expected: 18.0 m)")
    except FileNotFoundError:
        print("Homography file not found, skipping dimension validation")
    
    # Determine output path
    if output_path is None:
        img_name = os.path.basename(img_path)
        img_base, img_ext = os.path.splitext(img_name)
        # Save to data/visualizations/court_corners/{player}/session_{session}/
        output_dir = os.path.join("data", "visualizations", "court_corners", player, f"session_{session}")
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, f"{img_base}_with_corners{img_ext}")
    
    # Save result
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    cv2.imwrite(output_path, img)
    print(f"Saved result to: {output_path}")
    
    return output_path


def main():
    parser = argparse.ArgumentParser(
        description="Add court corners to an image with ball bounding boxes from datasets/"
    )
    add_player_session_serve_args(parser)
    parser.add_argument("--split", default="train", choices=["train", "val"],
                       help="Dataset split to search (default: train)")
    parser.add_argument("--frame", "-f", type=int, default=None,
                       help="Frame number to use (e.g., 1 for frame000001, if None uses first match)")
    parser.add_argument("--output", default=None,
                       help="Output image path (default: saves to data/visualizations/court_corners/)")
    
    args = parser.parse_args()
    
    if not (args.player and args.session is not None and args.serve):
        parser.error("Must specify --player, --session, and --serve")
    
    try:
        output_path = add_court_corners_to_image(
            args.player,
            args.session,
            args.serve,
            args.output,
            args.split,
            args.frame
        )
        print(f"\nSuccess! Output saved to: {output_path}")
    except Exception as e:
        print(f"Error: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())

