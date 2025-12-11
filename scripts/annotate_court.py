"""
Annotate court corners and center line points for homography computation.

Interactive tool for clicking 6 points on the court (4 corners + 2 center line points)
to enable pixel-to-world coordinate transformation.
"""
import cv2
import json
import os
import argparse
import csv
import random
from pathlib import Path
from common_args import add_player_session_serve_args, add_user_mode_arg, build_trajectory_paths, build_user_paths, validate_video_exists, get_user_serves_csv_path

def annotate_court(input_path, output_path, is_image=False):
    """Interactive court annotation tool for 6 points: 4 corners + 2 center line points."""
    
    cap = None
    
    def load_input():
        """Load the input (video or image) and return frame properties."""
        nonlocal frame, width, height, total_frames, current_frame_num, cap, is_image, input_path
        
        if is_image:
            # Load image
            new_frame = cv2.imread(input_path)
            if new_frame is None:
                print(f"Could not open image {input_path}")
                return False
            
            frame = new_frame
            height, width = frame.shape[:2]
            total_frames = 1
            current_frame_num = 0
            if cap is not None:
                cap.release()
                cap = None
            return True
        else:
            # Load video
            if cap is not None:
                cap.release()
            cap = cv2.VideoCapture(input_path)
            if not cap.isOpened():
                print(f"Could not open {input_path}")
                return False
            
            # Get video properties
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            
            # Start at first frame
            current_frame_num = 0
            cap.set(cv2.CAP_PROP_POS_FRAMES, current_frame_num)
            ret, new_frame = cap.read()
            if not ret:
                print("Could not read frame from video")
                cap.release()
                cap = None
                return False
            frame = new_frame
            return True
    
    # Initial load
    frame = None
    width = 0
    height = 0
    total_frames = 0
    current_frame_num = 0
    
    if not load_input():
        return False
    
    # Annotation state
    points = []
    current_point = 0
    point_names = [
        "Top-left corner",
        "Top-right corner", 
        "Bottom-right corner",
        "Bottom-left corner",
        "Center line (left)",
        "Center line (right)"
    ]
    
    def load_frame(frame_num):
        """Load a specific frame from the video."""
        nonlocal frame, current_frame_num
        if is_image:
            return False  # Can't change frames for images
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
        ret, new_frame = cap.read()
        if ret:
            frame = new_frame
            current_frame_num = frame_num
            return True
        return False
    
    def draw_frame_with_points():
        """Draw the current frame with all annotation points."""
        display_frame = frame.copy()
        
        # Draw existing points
        for i, (px, py) in enumerate(points):
            color = (0, 255, 0) if i < 4 else (0, 0, 255)  # Green for corners, red for center
            cv2.circle(display_frame, (px, py), 8, color, -1)
            cv2.putText(display_frame, f"{i+1}", (px+10, py-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        
        # Show frame number (only for videos)
        if not is_image:
            frame_info = f"Frame {current_frame_num + 1}/{total_frames}"
            cv2.putText(display_frame, frame_info, (10, height - 20), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        return display_frame
    
    def mouse_callback(event, x, y, flags, param):
        nonlocal current_point, points
        
        if event == cv2.EVENT_LBUTTONDOWN and current_point < 6:
            points.append([x, y])
            print(f"Point {current_point + 1}: {point_names[current_point]} at ({x}, {y})")
            current_point += 1
            
            # Redraw frame with all points
            display_frame = draw_frame_with_points()
            
            # Show current instruction
            if current_point < 6:
                cv2.putText(display_frame, f"Click {point_names[current_point]} ({current_point+1}/6)", 
                           (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            else:
                cv2.putText(display_frame, "All points collected! Press 's' to save, 'r' to reset", 
                           (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
            
            nav_text = "Press 'r' to reset, 'q' to quit" if is_image else "Press 'r' to reset, 'n'/'p' for next/prev frame, 'i' for random frame, 'q' to quit"
            cv2.putText(display_frame, nav_text, 
                       (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            cv2.imshow("Court Annotation", display_frame)
    
    # Create window and set mouse callback
    cv2.namedWindow("Court Annotation", cv2.WINDOW_NORMAL)
    cv2.setMouseCallback("Court Annotation", mouse_callback)
    
    # Display instructions
    display_frame = draw_frame_with_points()
    cv2.putText(display_frame, f"Click {point_names[0]} (1/6)", (10, 30), 
               cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    nav_text = "Press 'r' to reset, 'i' for new image, 'q' to quit" if is_image else "Press 'r' to reset, 'n'/'p' for next/prev frame, 'i' for new image, 'q' to quit"
    cv2.putText(display_frame, nav_text, (10, 70), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.imshow("Court Annotation", display_frame)
    
    print(f"Annotating court for: {input_path}")
    print("Instructions:")
    print("1. Click on the 4 court corners (in order: top-left, top-right, bottom-right, bottom-left)")
    print("2. Click on the 2 center line points (left and right)")
    if not is_image:
        print("3. Press 'n' for next frame, 'p' for previous frame")
        print("4. Press 'i' to jump to a random frame")
        print("5. Press 'r' to reset, 'q' to quit")
    else:
        print("3. Press 'r' to reset, 'q' to quit")
    print()
    
    while True:
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord('q'):
            break
        elif key == ord('r'):
            # Reset annotation
            points = []
            current_point = 0
            display_frame = draw_frame_with_points()
            cv2.putText(display_frame, f"Click {point_names[0]} (1/6)", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            nav_text = "Press 'r' to reset, 'q' to quit" if is_image else "Press 'r' to reset, 'n'/'p' for next/prev frame, 'i' for random frame, 'q' to quit"
            cv2.putText(display_frame, nav_text, (10, 70), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.imshow("Court Annotation", display_frame)
            print("Reset annotation")
            continue
        elif (key == ord('n') or key == 83) and not is_image:  # 'n' or right arrow (only for videos)
            # Next frame
            if current_frame_num < total_frames - 1:
                if load_frame(current_frame_num + 1):
                    # Clear points when changing frame
                    points = []
                    current_point = 0
                    display_frame = draw_frame_with_points()
                    cv2.putText(display_frame, f"Click {point_names[0]} (1/6)", (10, 30), 
                               cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                    cv2.putText(display_frame, "Press 'r' to reset, 'n'/'p' for next/prev frame, 'q' to quit", (10, 70), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                    cv2.imshow("Court Annotation", display_frame)
                    print(f"Loaded frame {current_frame_num + 1}/{total_frames}")
            continue
        elif (key == ord('p') or key == 81) and not is_image:  # 'p' or left arrow (only for videos)
            # Previous frame
            if current_frame_num > 0:
                if load_frame(current_frame_num - 1):
                    # Clear points when changing frame
                    points = []
                    current_point = 0
                    display_frame = draw_frame_with_points()
                    cv2.putText(display_frame, f"Click {point_names[0]} (1/6)", (10, 30), 
                               cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                    cv2.putText(display_frame, "Press 'r' to reset, 'n'/'p' for next/prev frame, 'q' to quit", (10, 70), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                    cv2.imshow("Court Annotation", display_frame)
                    print(f"Loaded frame {current_frame_num + 1}/{total_frames}")
            continue
        elif key == ord('i') and not is_image:
            # Load random frame from current video
            if total_frames > 1:
                random_frame = random.randint(0, total_frames - 1)
                if load_frame(random_frame):
                    # Retain points when changing frame
                    display_frame = draw_frame_with_points()
                    
                    # Show current instruction based on points collected
                    if current_point < 6:
                        cv2.putText(display_frame, f"Click {point_names[current_point]} ({current_point+1}/6)", (10, 30), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                    else:
                        cv2.putText(display_frame, "All points collected! Press 's' to save, 'r' to reset", 
                                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
                    
                    cv2.putText(display_frame, "Press 'r' to reset, 'n'/'p' for next/prev frame, 'i' for random frame, 'q' to quit", (10, 70), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                    cv2.imshow("Court Annotation", display_frame)
                    print(f"Loaded random frame {current_frame_num + 1}/{total_frames}")
            continue
        
        # Update instruction text
        if current_point < 6:
            display_frame = draw_frame_with_points()
            
            # Show current instruction
            cv2.putText(display_frame, f"Click {point_names[current_point]} ({current_point+1}/6)", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            nav_text = "Press 'r' to reset, 'q' to quit" if is_image else "Press 'r' to reset, 'n'/'p' for next/prev frame, 'i' for random frame, 'q' to quit"
            cv2.putText(display_frame, nav_text, (10, 70), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            cv2.imshow("Court Annotation", display_frame)
        else:
            # All points collected
            display_frame = draw_frame_with_points()
            
            cv2.putText(display_frame, "All points collected! Press 's' to save, 'r' to reset", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
            nav_text = "Press 'q' to quit" if is_image else "Press 'n'/'p' for next/prev frame, 'i' for random frame, 'q' to quit"
            cv2.putText(display_frame, nav_text, (10, 70), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.imshow("Court Annotation", display_frame)
            
            if key == ord('s'):
                break
    
    if cap is not None:
        cap.release()
    cv2.destroyAllWindows()
    
    # Save annotation if we have all 6 points
    if len(points) == 6:
        # Extract session number from input path
        input_path_parts = Path(input_path).parts
        session_id = "unknown"
        if "session_" in str(input_path):
            for part in input_path_parts:
                if part.startswith("session_"):
                    session_id = part
                    break
        
        annotation = {
            "session_id": session_id,
            "image_resolution": [width, height],
            "court_corners": points
        }
        if is_image:
            annotation["image_file"] = str(input_path)
        else:
            annotation["video_file"] = str(input_path)
        
        # Ensure output directory exists
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        with open(output_path, 'w') as f:
            json.dump(annotation, f, indent=2)
        
        print(f"Saved annotation to: {output_path}")
        
        # Update court_corners.csv if in user mode
        if "user/" in output_path:
            update_court_corners_csv(session_id, input_path, output_path)
        
        return True
    else:
        print("Annotation incomplete. Need exactly 6 points.")
        return False


def update_court_corners_csv(session_id, video_path, corners_path):
    """Update user/data/court_corners.csv with new annotation entry."""
    csv_path = "user/data/court_corners.csv"
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    
    # Read existing rows
    rows = []
    fieldnames = ["session_id", "video_path", "court_corners_path"]
    file_exists = os.path.exists(csv_path) and os.path.getsize(csv_path) > 0
    
    if file_exists:
        with open(csv_path, "r", newline="") as f:
            reader = csv.DictReader(f)
            rows = list(reader)
    
    # Check if entry exists for this session
    updated = False
    for row in rows:
        if row["session_id"] == session_id:
            row["video_path"] = video_path
            row["court_corners_path"] = corners_path
            updated = True
            break
    
    # Add new entry if not found
    if not updated:
        rows.append({
            "session_id": session_id,
            "video_path": video_path,
            "court_corners_path": corners_path
        })
    
    # Write back
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    
    print(f"Updated {csv_path} with session {session_id}")

def get_court_corners_from_csv(session_id):
    """Get court corners path from user/data/court_corners.csv for a given session."""
    csv_path = "user/data/court_corners.csv"
    if not os.path.exists(csv_path):
        return None
    
    with open(csv_path, "r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row["session_id"] == session_id:
                return row["court_corners_path"]
    return None


def find_sessions_without_annotations(user_mode=False):
    """Find sessions that don't have court corner annotations yet."""
    if user_mode:
        processed_dir = Path("user/data/videos")
        annotations_dir = Path("user/data/annotations/court_corners")
    else:
        processed_dir = Path("data/videos/processed")
        annotations_dir = Path("data/annotations/court_corners")
    
    sessions_needing_annotation = []
    
    if not processed_dir.exists():
        return sessions_needing_annotation
    
    # Find all player/session combinations
    for player_dir in processed_dir.iterdir():
        if not player_dir.is_dir():
            continue
        
        for session_dir in player_dir.iterdir():
            if not session_dir.is_dir() or not session_dir.name.startswith("session_"):
                continue
            
            # Check if this session has serve clips
            serve_clips = list(session_dir.glob("serve_*.mp4"))
            if not serve_clips:
                continue
            
            # Check if annotation already exists
            session_id = session_dir.name
            has_annotation = False
            
            if user_mode:
                # Check CSV first
                corners_path = get_court_corners_from_csv(session_id)
                if corners_path and os.path.exists(corners_path):
                    has_annotation = True
                # Also check direct path as fallback
                annotation_file = annotations_dir / f"{session_id}.json"
                if annotation_file.exists():
                    has_annotation = True
            else:
                annotation_file = annotations_dir / f"{session_id}.json"
                if annotation_file.exists():
                    has_annotation = True
            
            if not has_annotation:
                # Check if this session is already in the list (avoid duplicates)
                if not any(s['session_id'] == session_id for s in sessions_needing_annotation):
                    sessions_needing_annotation.append({
                        "player": player_dir.name,
                        "session": session_dir.name,
                        "session_id": session_id,
                        "serve_clip": serve_clips[0],  # Use first serve clip
                        "annotation_file": annotation_file
                    })
    
    return sessions_needing_annotation

def main():
    parser = argparse.ArgumentParser(description="Annotate court corners and center line points")
    
    # Common player/session/serve arguments
    add_player_session_serve_args(parser)
    add_user_mode_arg(parser)
    
    parser.add_argument("--auto", action="store_true",
                       help="Auto-detect and annotate all sessions without annotations")
    
    args = parser.parse_args()
    
    if args.auto:
        # Auto-detect mode
        sessions = find_sessions_without_annotations(user_mode=args.user_mode)
        
        if not sessions:
            print("All sessions already have court annotations!")
            return
        
        print(f"Found {len(sessions)} sessions needing court annotations:")
        for i, session in enumerate(sessions, 1):
            print(f"  {i}. {session['session_id']} ({session['serve_clip']})")
        
        for session in sessions:
            print(f"\n--- Annotating {session['session_id']} ---")
            success = annotate_court(str(session['serve_clip']), str(session['annotation_file']))
            if success:
                print(f"✓ Completed {session['session_id']}")
            else:
                print(f"✗ Failed {session['session_id']}")
                break  # Stop on first failure
        
        print("\nAnnotation session complete!")
        
    elif args.player and args.session is not None:
        # Use player/session to find a video and generate output path
        if args.user_mode:
            if args.serve:
                video_path, _, _ = build_user_paths(args.player, args.session, args.serve)
            else:
                # Use first serve in session
                session_dir = Path(f"user/data/videos/{args.player}/session_{args.session}")
                serve_clips = sorted(session_dir.glob("serve_*.mp4"))
                if not serve_clips:
                    raise SystemExit(f"No serve videos found for {args.player}/session_{args.session}")
                video_path = str(serve_clips[0])
            
            # Generate output path based on session
            output_path = f"user/data/annotations/court_corners/session_{args.session}.json"
        else:
            if args.serve:
                video_path, _ = build_trajectory_paths(args.player, args.session, args.serve)
            else:
                # Use first serve in session
                session_dir = Path(f"data/videos/processed/{args.player}/session_{args.session}")
                serve_clips = sorted(session_dir.glob("serve_*.mp4"))
                if not serve_clips:
                    raise SystemExit(f"No serve videos found for {args.player}/session_{args.session}")
                video_path = str(serve_clips[0])
            
            # Generate output path based on session
            output_path = f"data/annotations/court_corners/session_{args.session}.json"
        
        validate_video_exists(video_path)
        
        success = annotate_court(video_path, output_path, is_image=False)
        if success:
            print("Annotation completed successfully!")
        else:
            print("Annotation failed or was cancelled.")
    else:
        parser.error("Must specify --auto OR (--player and --session)")

if __name__ == "__main__":
    main()
