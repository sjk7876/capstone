#!/usr/bin/env python3
import cv2
import json
import os
import argparse
import glob
from pathlib import Path

def annotate_court(video_path, output_path, annotator="spencer"):
    """Interactive court annotation tool for 6 points: 4 corners + 2 center line points."""
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Could not open {video_path}")
        return False
    
    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # Read first frame for annotation
    ret, frame = cap.read()
    if not ret:
        print("Could not read frame from video")
        cap.release()
        return False
    
    cap.release()
    
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
    
    def mouse_callback(event, x, y, flags, param):
        nonlocal current_point, points
        
        if event == cv2.EVENT_LBUTTONDOWN and current_point < 6:
            points.append([x, y])
            print(f"Point {current_point + 1}: {point_names[current_point]} at ({x}, {y})")
            current_point += 1
            
            # Redraw frame with all points
            display_frame = frame.copy()
            for i, (px, py) in enumerate(points):
                color = (0, 255, 0) if i < 4 else (0, 0, 255)  # Green for corners, red for center
                cv2.circle(display_frame, (px, py), 8, color, -1)
                cv2.putText(display_frame, f"{i+1}", (px+10, py-10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            
            cv2.imshow("Court Annotation", display_frame)
    
    # Create window and set mouse callback
    cv2.namedWindow("Court Annotation", cv2.WINDOW_NORMAL)
    cv2.setMouseCallback("Court Annotation", mouse_callback)
    
    # Display instructions
    display_frame = frame.copy()
    cv2.putText(display_frame, f"Click {point_names[0]} (1/6)", (10, 30), 
               cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    cv2.putText(display_frame, "Press 'r' to reset, 'q' to quit", (10, 70), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.imshow("Court Annotation", display_frame)
    
    print(f"Annotating court for: {video_path}")
    print("Instructions:")
    print("1. Click on the 4 court corners (in order: top-left, top-right, bottom-right, bottom-left)")
    print("2. Click on the 2 center line points (left and right)")
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
            cv2.imshow("Court Annotation", frame)
            print("Reset annotation")
            continue
        
        # Update instruction text
        if current_point < 6:
            display_frame = frame.copy()
            
            # Draw existing points
            for i, (px, py) in enumerate(points):
                color = (0, 255, 0) if i < 4 else (0, 0, 255)
                cv2.circle(display_frame, (px, py), 8, color, -1)
                cv2.putText(display_frame, f"{i+1}", (px+10, py-10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            
            # Show current instruction
            cv2.putText(display_frame, f"Click {point_names[current_point]} ({current_point+1}/6)", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            cv2.putText(display_frame, "Press 'r' to reset, 'q' to quit", (10, 70), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            cv2.imshow("Court Annotation", display_frame)
        else:
            # All points collected
            display_frame = frame.copy()
            for i, (px, py) in enumerate(points):
                color = (0, 255, 0) if i < 4 else (0, 0, 255)
                cv2.circle(display_frame, (px, py), 8, color, -1)
                cv2.putText(display_frame, f"{i+1}", (px+10, py-10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            
            cv2.putText(display_frame, "All points collected! Press 's' to save, 'r' to reset", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
            cv2.imshow("Court Annotation", display_frame)
            
            if key == ord('s'):
                break
    
    cv2.destroyAllWindows()
    
    # Save annotation if we have all 6 points
    if len(points) == 6:
        # Extract session info from video path
        video_path_parts = Path(video_path).parts
        session_id = "unknown"
        if "raw" in video_path_parts:
            raw_idx = video_path_parts.index("raw")
            if len(video_path_parts) > raw_idx + 2:
                date_part = video_path_parts[raw_idx + 1]
                session_part = video_path_parts[raw_idx + 2]
                if session_part.startswith("session_"):
                    session_id = f"{date_part}_{session_part}"
        
        annotation = {
            "session_id": session_id,
            "video_file": str(video_path),
            "image_resolution": [width, height],
            "court_corners": points,
            "annotator": annotator
        }
        
        # Ensure output directory exists
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        with open(output_path, 'w') as f:
            json.dump(annotation, f, indent=2)
        
        print(f"Saved annotation to: {output_path}")
        return True
    else:
        print("Annotation incomplete. Need exactly 6 points.")
        return False

def find_sessions_without_annotations():
    """Find sessions that don't have court corner annotations yet."""
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
            session_id = f"{session_dir.parent.name}_{session_dir.name}"
            annotation_file = annotations_dir / f"{session_id}.json"
            
            if not annotation_file.exists():
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
    parser.add_argument("--video", type=str, default=None,
                       help="Path to specific video file (optional, auto-detects if not provided)")
    parser.add_argument("--output", type=str, default=None,
                       help="Output JSON file path (auto-generated if not provided)")
    parser.add_argument("--annotator", type=str, default="spencer",
                       help="Name of annotator (default: spencer)")
    parser.add_argument("--auto", action="store_true",
                       help="Auto-detect and annotate all sessions without annotations")
    
    args = parser.parse_args()
    
    if args.auto or (args.video is None and args.output is None):
        # Auto-detect mode
        sessions = find_sessions_without_annotations()
        
        if not sessions:
            print("All sessions already have court annotations!")
            return
        
        print(f"Found {len(sessions)} sessions needing court annotations:")
        for i, session in enumerate(sessions, 1):
            print(f"  {i}. {session['session_id']} ({session['serve_clip']})")
        
        for session in sessions:
            print(f"\n--- Annotating {session['session_id']} ---")
            success = annotate_court(str(session['serve_clip']), str(session['annotation_file']), args.annotator)
            if success:
                print(f"✓ Completed {session['session_id']}")
            else:
                print(f"✗ Failed {session['session_id']}")
                break  # Stop on first failure
        
        print("\nAnnotation session complete!")
        
    else:
        # Manual mode
        if args.video is None or args.output is None:
            print("Error: Both --video and --output are required in manual mode")
            return
        
        success = annotate_court(args.video, args.output, args.annotator)
        if success:
            print("Annotation completed successfully!")
        else:
            print("Annotation failed or was cancelled.")

if __name__ == "__main__":
    main()
