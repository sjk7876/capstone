#!/usr/bin/env python3
import cv2
import os
import argparse
import tempfile
import shutil
from pathlib import Path

def extract_all_frames(video_path, output_dir=None, every_n=1):
    """Extract frames from video to output directory."""
    
    # Create temp directory if none specified
    if output_dir is None:
        output_dir = tempfile.mkdtemp(prefix="frames_")
        is_temp = True
    else:
        os.makedirs(output_dir, exist_ok=True)
        is_temp = False
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Could not open {video_path}")
        return None
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    print(f"Video: {total_frames} frames, {fps:.2f} fps, {width}x{height}")
    print(f"Extracting to: {output_dir}")
    if every_n > 1:
        print(f"Sampling every {every_n} frames")
    
    frame_id = 0
    saved = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Save frame if it matches the sampling interval
        if frame_id % every_n == 0:
            out_path = os.path.join(output_dir, f"frame{frame_id:06d}.jpg")
            cv2.imwrite(out_path, frame)
            saved += 1
        
        # Progress indicator
        if frame_id % 100 == 0:
            print(f"  Processed {frame_id}/{total_frames} frames...")
        
        frame_id += 1
    
    cap.release()
    print(f"Extracted {saved} frames from {video_path} → {output_dir}")
    
    return output_dir, is_temp

def find_serves_in_session(player, session):
    """Find all serve videos in a session directory."""
    session_dir = f"data/videos/processed/{player}/session_{session}"
    if not os.path.exists(session_dir):
        return []
    
    serve_files = []
    for file in os.listdir(session_dir):
        if file.startswith("serve_") and file.endswith(".mp4"):
            serve_id = file.replace("serve_", "").replace(".mp4", "")
            serve_files.append((serve_id, os.path.join(session_dir, file)))
    
    return sorted(serve_files, key=lambda x: int(x[0]))

def main():
    parser = argparse.ArgumentParser(description="Extract all frames from a video")
    parser.add_argument("--video", type=str, default=None,
                       help="Path to input video file (optional if using --player/--session/--serve)")
    parser.add_argument("--player", type=str, default=None,
                       help="Player name (e.g., 'spencer')")
    parser.add_argument("--session", type=int, default=None,
                       help="Session number (e.g., 1)")
    parser.add_argument("--serve", type=str, default=None,
                       help="Serve ID (e.g., '1' or '001'). If not provided, extracts all serves in session")
    parser.add_argument("--output", type=str, default=None,
                       help="Output directory (auto-generated if not specified)")
    parser.add_argument("--list", action="store_true",
                       help="List extracted frames and exit")
    parser.add_argument("--every", type=int, default=1,
                       help="Extract every nth frame (default: 1 = all frames)")
    
    args = parser.parse_args()
    
    # Determine video path(s)
    video_paths = []
    if args.video:
        video_paths = [args.video]
    elif args.player and args.session is not None:
        if args.serve:
            # Single serve
            try:
                serve_num = int(args.serve)
                serve_id = f"{serve_num:03d}"
            except ValueError:
                serve_id = args.serve
            video_path = f"data/videos/processed/{args.player}/session_{args.session}/serve_{serve_id}.mp4"
            video_paths = [video_path]
        else:
            # All serves in session
            serve_files = find_serves_in_session(args.player, args.session)
            if not serve_files:
                print(f"No serve videos found in data/videos/processed/{args.player}/session_{args.session}/")
                return
            print(f"Found {len(serve_files)} serves in session {args.session}:")
            for serve_id, video_path in serve_files:
                print(f"  serve_{serve_id}.mp4")
            video_paths = [video_path for _, video_path in serve_files]
    else:
        print("Error: Either provide --video or --player and --session")
        return
    
    # Process each video
    for video_path in video_paths:
        if not os.path.exists(video_path):
            print(f"Video file not found: {video_path}")
            continue
        
        print(f"\nProcessing: {video_path}")
        
        # Determine output directory
        output_dir = args.output
        if output_dir is None:
            # Auto-generate output path based on video path
            video_parts = Path(video_path).parts
            if "processed" in video_parts:
                processed_idx = video_parts.index("processed")
                if len(video_parts) > processed_idx + 3:
                    player = video_parts[processed_idx + 1]
                    session = video_parts[processed_idx + 2]  # session_1
                    serve = video_parts[processed_idx + 3].replace(".mp4", "")  # serve_001
                    output_dir = f"data/frames/{player}_{session}_{serve}"
                else:
                    print(f"Error: Could not parse video path for auto-output: {video_path}")
                    continue
            else:
                print(f"Error: Could not determine output directory for: {video_path}")
                continue
        
        # Extract frames
        result = extract_all_frames(video_path, output_dir, args.every)
        if result is None:
            continue
        
        output_dir, is_temp = result
        
        if args.list:
            # List extracted frames
            frame_files = sorted(Path(output_dir).glob("frame*.jpg"))
            print(f"\nExtracted {len(frame_files)} frames from {os.path.basename(video_path)}:")
            for i, frame_file in enumerate(frame_files[:10]):  # Show first 10
                print(f"  {frame_file.name}")
            if len(frame_files) > 10:
                print(f"  ... and {len(frame_files) - 10} more")
        
        print(f"Frames saved to: {output_dir}")
    
    if len(video_paths) > 1:
        print(f"\nCompleted processing {len(video_paths)} videos")

if __name__ == "__main__":
    main()


