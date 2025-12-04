"""
Common argument parsing utilities for scripts that work with player/session/serve data.

Provides shared functions for argument parsing, path building, and validation
across multiple scripts in the project.
"""
import argparse
import os


def format_serve_number(serve: str) -> str:
    """Format serve number with leading zeros (e.g., "1" -> "001")."""
    if serve.isdigit():
        return f"{int(serve):03d}"
    return serve


def build_trajectory_paths(player: str, session: int, serve: str) -> tuple[str, str]:
    """Build video and trajectory JSON paths from player/session/serve."""
    serve_str = format_serve_number(serve)
    
    video_path = f"data/videos/processed/{player}/session_{session}/serve_{serve_str}.mp4"
    json_path = f"data/trajectories/{player}/session_{session}/serve_{serve_str}.json"
    
    return video_path, json_path


def add_player_session_serve_args(parser: argparse.ArgumentParser) -> None:
    """
    Add common player/session/serve arguments to an ArgumentParser.
    
    Args:
        parser: argparse.ArgumentParser instance
    """
    # Option 1: Specify player, session, serve
    parser.add_argument('-p', '--player', type=str, default=None,
                       help='Player name (e.g., "spencer")')
    parser.add_argument('-s', '--session', type=int, default=None,
                       help='Session number (e.g., 1)')
    parser.add_argument('-e', '--serve', type=str, default=None,
                       help='Serve number (e.g., "001" or "1")')


def add_user_mode_arg(parser: argparse.ArgumentParser) -> None:
    """
    Add user mode argument to an ArgumentParser.
    
    Args:
        parser: argparse.ArgumentParser instance
    """
    parser.add_argument('-u', '--user-mode', action='store_true',
                       help='User mode: use user/ paths instead of data/ paths')


def resolve_trajectory_paths(args: argparse.Namespace, require_output: bool = True) -> tuple[str, str]:
    """
    Resolve video and JSON paths from args (either player/session/serve or direct paths).
    
    Args:
        args: Parsed arguments with player/session/serve or direct paths
        require_output: If True, output_json is required
    
    Returns:
        tuple: (video_path, json_path) or (video_path, None) if require_output=False
    
    Raises:
        SystemExit: If arguments are invalid or files don't exist
    """
    # Check if using player/session/serve
    if args.player and args.session is not None and args.serve:
        video_path, json_path = build_trajectory_paths(args.player, args.session, args.serve)
        if require_output and json_path:
            os.makedirs(os.path.dirname(json_path), exist_ok=True)
        return video_path, json_path
    
    # Check if using direct paths
    video_path = getattr(args, 'video_path', None)
    json_path = getattr(args, 'output_json', None) if require_output else getattr(args, 'json_path', None)
    
    if not video_path:
        raise SystemExit("Error: Either specify (--player, --session, --serve) OR (--video-path)")
    
    if require_output and not json_path:
        raise SystemExit("Error: output_json/json_path is required")
    
    if json_path:
        os.makedirs(os.path.dirname(json_path), exist_ok=True)
    
    return video_path, json_path


def validate_video_exists(video_path: str) -> None:
    """Validate that video file exists, exit if not."""
    if not os.path.exists(video_path):
        raise SystemExit(f"Error: Video file not found: {video_path}")


def build_user_paths(player: str, session: int, serve: str) -> tuple[str, str, str]:
    """
    Build user mode paths for video, detection JSON, and trajectory JSON.
    
    Args:
        player: Player name
        session: Session number
        serve: Serve number (will be formatted)
    
    Returns:
        tuple: (video_path, detect_json_path, traj_json_path)
    """
    serve_str = format_serve_number(serve)
    
    video_path = f"user/videos/{player}/session_{session}/serve_{serve_str}.mp4"
    detect_json = f"user/detections/{player}/session_{session}/serve_{serve_str}.json"
    traj_json = f"user/trajectories/{player}/session_{session}/serve_{serve_str}.json"
    
    return video_path, detect_json, traj_json


def get_user_serves_csv_path() -> str:
    """Get path to user serves CSV file."""
    return "user/user_serves.csv"


def get_user_videos_dir() -> str:
    """Get path to user videos directory."""
    return "user/videos"

