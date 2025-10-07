#!/usr/bin/env python3
import cv2
import os
import argparse
import tempfile
from pathlib import Path

def extract_frame_range(video_path, start_frame, end_frame, output_dir=None):
    """Extract a range of frames from video to output directory."""
    
    # Create temp directory if none specified
    if output_dir is None:
        output_dir = tempfile.mkdtemp(prefix="frames_", dir=".")
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
    
    # Validate frame range
    if start_frame < 0:
        start_frame = 0
    if end_frame >= total_frames:
        end_frame = total_frames - 1
    if start_frame > end_frame:
        print("Error: start_frame must be <= end_frame")
        return None
    
    frame_count = end_frame - start_frame + 1
    
    print(f"Video: {total_frames} frames, {fps:.2f} fps, {width}x{height}")
    print(f"Extracting frames {start_frame} to {end_frame} ({frame_count} frames)")
    print(f"Output: {output_dir}")
    
    # Seek to start frame
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    
    saved = 0
    current_frame = start_frame
    
    while current_frame <= end_frame:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Save frame
        out_path = os.path.join(output_dir, f"frame{current_frame:06d}.jpg")
        cv2.imwrite(out_path, frame)
        saved += 1
        
        # Progress indicator
        if saved % 50 == 0:
            print(f"  Extracted {saved}/{frame_count} frames...")
        
        current_frame += 1
    
    cap.release()
    print(f"Extracted {saved} frames from {video_path} → {output_dir}")
    
    return output_dir, is_temp

def main():
    parser = argparse.ArgumentParser(description="Extract a range of frames from a video")
    parser.add_argument("--video", type=str, required=True,
                       help="Path to input video file")
    parser.add_argument("--start", type=int, required=True,
                       help="Start frame number (0-based)")
    parser.add_argument("--end", type=int, required=True,
                       help="End frame number (inclusive)")
    parser.add_argument("--output", type=str, default=None,
                       help="Output directory (creates temp dir if not specified)")
    parser.add_argument("--list", action="store_true",
                       help="List extracted frames and exit")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.video):
        print(f"Video file not found: {args.video}")
        return
    
    # Extract frames
    result = extract_frame_range(args.video, args.start, args.end, args.output)
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
    
    if is_temp:
        print(f"\nTemporary directory: {output_dir}")
        print("Use --output to specify a permanent directory")
    else:
        print(f"\nFrames saved to: {output_dir}")

if __name__ == "__main__":
    main()
