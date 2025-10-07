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

def main():
    parser = argparse.ArgumentParser(description="Extract all frames from a video")
    parser.add_argument("--video", type=str, default=None,
                       help="Path to input video file (optional if using --player/--session/--serve)")
    parser.add_argument("--player", type=str, default=None,
                       help="Player name (e.g., 'spencer')")
    parser.add_argument("--session", type=int, default=None,
                       help="Session number (e.g., 1)")
    parser.add_argument("--serve", type=str, default=None,
                       help="Serve ID (e.g., '1' or '001')")
    parser.add_argument("--output", type=str, default=None,
                       help="Output directory (auto-generated if not specified)")
    parser.add_argument("--list", action="store_true",
                       help="List extracted frames and exit")
    parser.add_argument("--every", type=int, default=1,
                       help="Extract every nth frame (default: 1 = all frames)")
    
    args = parser.parse_args()
    
    # Determine video path
    video_path = args.video
    if video_path is None:
        if args.player and args.session is not None and args.serve:
            # Normalize serve ID to 3-digit format
            try:
                serve_num = int(args.serve)
                serve_id = f"{serve_num:03d}"
            except ValueError:
                # If it's already a string like "001", use as-is
                serve_id = args.serve
            # Construct path from player/session/serve
            video_path = f"data/videos/processed/{args.player}/session_{args.session}/serve_{serve_id}.mp4"
        else:
            print("Error: Either provide --video or all of --player, --session, --serve")
            return
    
    if not os.path.exists(video_path):
        print(f"Video file not found: {video_path}")
        return
    
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
                print("Error: Could not parse video path for auto-output")
                return
        else:
            print("Error: Could not determine output directory")
            return
    
    # Extract frames
    result = extract_all_frames(video_path, output_dir, args.every)
    if result is None:
        return
    
    output_dir, is_temp = result
    
    if args.list:
        # List extracted frames
        frame_files = sorted(Path(output_dir).glob("frame*.jpg"))
        print(f"\nExtracted {len(frame_files)} frames:")
        for i, frame_file in enumerate(frame_files[:10]):  # Show first 10
            print(f"  {frame_file.name}")
        if len(frame_files) > 10:
            print(f"  ... and {len(frame_files) - 10} more")
    
    print(f"\nFrames saved to: {output_dir}")

if __name__ == "__main__":
    main()


