"""
Master workflow script for processing new user videos.

Automates the full pipeline:
1. Find unprocessed videos in user/input/
2. Split serves (interactive)
3. Annotate court corners (interactive)
4. Process serves (automatic)
"""
import os
import sys
import subprocess
import csv
import argparse
from pathlib import Path

# Add scripts directory to path
script_dir = os.path.join(os.path.dirname(__file__), "..", "scripts")
sys.path.insert(0, script_dir)

from common_args import get_user_serves_csv_path, normalize_path


def get_processed_videos():
    """Get set of video paths that are already in the CSV."""
    csv_path = get_user_serves_csv_path()
    processed = set()
    
    if os.path.exists(csv_path):
        with open(csv_path, "r", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                source_video = row.get("source_video", "").strip()
                if source_video:
                    processed.add(normalize_path(source_video))
    
    return processed


def find_unprocessed_videos():
    """Find videos in user/input/ that aren't in the CSV."""
    input_dir = Path("user/input")
    if not input_dir.exists():
        return []
    
    processed = get_processed_videos()
    unprocessed = []
    
    # Find all video files
    video_extensions = {".mp4", ".avi", ".mov", ".mkv", ".m4v"}
    for video_file in input_dir.iterdir():
        if video_file.is_file() and video_file.suffix.lower() in video_extensions:
            # Normalize path for comparison with processed set
            video_path = str(video_file)
            video_path_normalized = normalize_path(video_path)
            if video_path_normalized not in processed:
                unprocessed.append(video_path)
    
    return sorted(unprocessed)


def get_player_name():
    """Prompt user for player name."""
    while True:
        player = input("Enter player name: ").strip()
        if player:
            return player
        print("Player name cannot be empty. Please try again.")


def get_target_zone():
    """Prompt user for target zone (1-6) or None for all zones."""
    print("\nZone layout (far side of court):")
    print("  ┌─────┬─────┬─────┐")
    print("  │  1  │  6  │  5  │  Top row")
    print("  ├─────┼─────┼─────┤")
    print("  │  2  │  3  │  4  │  Bottom row")
    print("  └─────┴─────┴─────┘")
    
    while True:
        response = input("\nEnter target zone (1-6) or press Enter to analyze all zones: ").strip()
        if not response:
            return None
        try:
            zone = int(response)
            if 1 <= zone <= 6:
                return zone
            else:
                print("Zone must be between 1 and 6. Please try again.")
        except ValueError:
            print("Invalid input. Please enter a number between 1 and 6, or press Enter for all zones.")


def run_split_serves(video_path, player):
    """Run split_serves.py in user mode."""
    print(f"\n{'='*60}")
    print(f"Step 1: Splitting serves from {os.path.basename(video_path)}")
    print(f"{'='*60}")
    print(f"Player: {player}")
    print("\nInstructions:")
    print("  Start the serve before the toss and end the serve after the ball lands.")
    print("  [s] = mark serve start")
    print("  [e] = mark serve end")
    print("  [d] = delete previous serve")
    print("  [f] = fast-forward")
    print("  [b] = back 30 frames")
    print("  [q] = quit")
    print("  Note: Use [e] then [d] to delete any serves that hit the net - these should not be included in the dataset.")
    print("\nPress Enter to start splitting...")
    input()
    
    cmd = [
        sys.executable,
        "scripts/split_serves.py",
        "--video", video_path,
        "--player", player,
        "--user-mode"
    ]
    
    result = subprocess.run(cmd)
    return result.returncode == 0


def get_latest_session(player):
    """Get the latest session number for a player from CSV."""
    csv_path = get_user_serves_csv_path()
    if not os.path.exists(csv_path):
        return None
    
    max_session = 0
    with open(csv_path, "r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get("player") == player:
                try:
                    session_id = int(row.get("session_id", 0))
                    max_session = max(max_session, session_id)
                except (ValueError, TypeError):
                    continue
    
    return max_session if max_session > 0 else None


def run_annotate_court(player, session_id):
    """Run annotate_court.py in user mode."""
    print(f"\n{'='*60}")
    print(f"Step 2: Annotating court corners for session {session_id}")
    print(f"{'='*60}")
    print("\nInstructions:")
    print("  Click 6 points on the court:")
    print("    1. Far left corner")
    print("    2. Far right corner")
    print("    3. Close right corner")
    print("    4. Close left corner")
    print("    5. Center line left point")
    print("    6. Center line right point")
    print("\nPress Enter to start annotation...")
    input()
    
    cmd = [
        sys.executable,
        "scripts/annotate_court.py",
        "--player", player,
        "--session", str(session_id),
        "--user-mode"
    ]
    
    result = subprocess.run(cmd)
    return result.returncode == 0


def run_process_serves(session_id):
    """Run process_serves.py automatically."""
    print(f"\n{'='*60}")
    print(f"Step 3: Processing serves for session {session_id}")
    print(f"{'='*60}")
    print("This step is automatic. Processing all serves...")
    
    cmd = [
        sys.executable,
        "user/process_serves.py",
        "--session", str(session_id)
    ]
    
    result = subprocess.run(cmd)
    return result.returncode == 0


def run_analyze_zones(player, session_id, target_zone=None):
    """Run analyze_landing_zones.py."""
    print(f"\n{'='*60}")
    print(f"Step 4: Analyzing landing zones for session {session_id}")
    print(f"{'='*60}")
    
    cmd = [
        sys.executable,
        "scripts/analyze_landing_zones.py",
        "--player", player,
        "--session", str(session_id),
        "--user-mode"
    ]
    
    if target_zone:
        cmd.extend(["--target-zone", str(target_zone)])
    
    result = subprocess.run(cmd)
    return result.returncode == 0


def main():
    parser = argparse.ArgumentParser(
        description="Master workflow script for processing new user videos",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
This script automates the full pipeline:
1. Finds unprocessed videos in user/input/
2. Splits serves (interactive - you mark start/end points)
3. Annotates court corners (interactive - you click 6 points)
4. Processes serves (automatic - YOLO → SORT → Landing Analysis → Visualizations)
5. Analyzes landing zones (interactive - prompts for target zone, or use --skip-zones to skip)

Examples:
  # Process all unprocessed videos
  python user/workflow.py

  # Process a specific video
  python user/workflow.py --video user/input/my_video.mp4 --player spencer
        """
    )
    parser.add_argument("--video", type=str, default=None,
                       help="Specific video to process (default: find unprocessed videos)")
    parser.add_argument("--player", type=str, default=None,
                       help="Player name (default: prompt or infer from filename)")
    parser.add_argument("--skip-split", action="store_true",
                       help="Skip splitting step (if already done)")
    parser.add_argument("--skip-annotate", action="store_true",
                       help="Skip annotation step (if already done)")
    parser.add_argument("--skip-process", action="store_true",
                       help="Skip processing step")
    parser.add_argument("--skip-zones", action="store_true",
                       help="Skip zone analysis step (default: analyze zones)")
    parser.add_argument("--target-zone", type=int, choices=[1, 2, 3, 4, 5, 6],
                       help="Target zone (1-6) for zone analysis (default: prompt)")
    
    args = parser.parse_args()
    
    # Find video(s) to process
    if args.video:
        if not os.path.exists(args.video):
            print(f"Error: Video file not found: {args.video}")
            return 1
        videos_to_process = [args.video]
    else:
        videos_to_process = find_unprocessed_videos()
        if not videos_to_process:
            print("No unprocessed videos found in user/input/")
            print("All videos in user/input/ have already been processed.")
            return 0
    
    print(f"Found {len(videos_to_process)} video(s) to process:")
    for i, video in enumerate(videos_to_process, 1):
        print(f"  {i}. {os.path.basename(video)}")
    
    # Process each video
    for video_path in videos_to_process:
        print(f"\n{'='*60}")
        print(f"Processing: {os.path.basename(video_path)}")
        print(f"{'='*60}")
        
        # Get player name
        if args.player:
            player = args.player
        else:
            player = get_player_name()
        
        # Step 1: Split serves
        if not args.skip_split:
            if not run_split_serves(video_path, player):
                print(f"\nError: Failed to split serves from {video_path}")
                print("Skipping remaining steps for this video.")
                continue
        else:
            print("\nSkipping split step (--skip-split)")
        
        # Get session ID (should be the latest one for this player)
        session_id = get_latest_session(player)
        if session_id is None:
            print(f"\nError: Could not determine session ID for player {player}")
            print("Make sure serves were successfully split.")
            continue
        
        print(f"\nDetected session ID: {session_id}")
        
        # Step 2: Annotate court
        if not args.skip_annotate:
            if not run_annotate_court(player, session_id):
                print(f"\nWarning: Court annotation failed or was cancelled for session {session_id}")
                print("Continuing to processing step anyway...")
        else:
            print("\nSkipping annotation step (--skip-annotate)")
        
        # Step 3: Process serves
        if not args.skip_process:
            if not run_process_serves(session_id):
                print(f"\nError: Failed to process serves for session {session_id}")
                continue
        else:
            print("\nSkipping processing step (--skip-process)")
        
        # Step 4: Analyze landing zones (default, unless skipped)
        if not args.skip_zones:
            # Get target zone (from args or prompt)
            target_zone = args.target_zone
            if target_zone is None:
                target_zone = get_target_zone()
            run_analyze_zones(player, session_id, target_zone=target_zone)
        else:
            print("\nSkipping zone analysis step (--skip-zones)")
        
        print(f"\n{'='*60}")
        print(f"Completed processing: {os.path.basename(video_path)}")
        print(f"{'='*60}")
    
    print("\nAll videos processed!")
    return 0


if __name__ == "__main__":
    sys.exit(main())

