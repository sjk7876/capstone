"""
Common argument parsing utilities for scripts that work with player/session/serve data.
"""
import argparse
import os


def format_serve_number(serve):
    """Format serve number with leading zeros (e.g., "1" -> "001")."""
    if serve.isdigit():
        return f"{int(serve):03d}"
    return serve


def build_trajectory_paths(player, session, serve):
    """Build video and trajectory JSON paths from player/session/serve."""
    serve_str = format_serve_number(serve)
    
    video_path = f"data/videos/processed/{player}/session_{session}/serve_{serve_str}.mp4"
    json_path = f"data/trajectories/{player}/session_{session}/serve_{serve_str}.json"
    
    return video_path, json_path


def add_player_session_serve_args(parser):
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
    parser.add_argument('--serve', type=str, default=None,
                       help='Serve number (e.g., "001" or "1")')


def resolve_trajectory_paths(args, require_output=True):
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


def validate_video_exists(video_path):
    """Validate that video file exists, exit if not."""
    if not os.path.exists(video_path):
        raise SystemExit(f"Error: Video file not found: {video_path}")

