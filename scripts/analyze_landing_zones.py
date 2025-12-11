"""
Analyze landing zones for serves.

Takes a target zone (1-6) and analyzes landing coordinates to determine
how many serves landed in the target zone and provides statistics.
"""
import os
import sys
import csv
import argparse
import cv2
import numpy as np
from pathlib import Path

# Add scripts directory to path
script_dir = os.path.dirname(__file__)
sys.path.insert(0, script_dir)

from common_args import add_player_session_serve_args, add_user_mode_arg, get_user_serves_csv_path, normalize_path
from create_landing_image import load_corners, load_frame_from_video


# Zone definitions (in warped image pixel coordinates, 900x900 for far side)
# Based on create_landing_image.py zone definitions
ZONE_BOUNDARIES = {
    1: {"x_min": 0, "x_max": 300, "y_min": 0, "y_max": 450},      # Top left
    2: {"x_min": 0, "x_max": 300, "y_min": 450, "y_max": 900},   # Bottom left
    3: {"x_min": 300, "x_max": 600, "y_min": 450, "y_max": 900}, # Bottom middle
    4: {"x_min": 600, "x_max": 900, "y_min": 450, "y_max": 900}, # Bottom right
    5: {"x_min": 600, "x_max": 900, "y_min": 0, "y_max": 450},   # Top right
    6: {"x_min": 300, "x_max": 600, "y_min": 0, "y_max": 450},   # Top middle
}

# Full court height in pixels (18m = 1800px)
FULL_COURT_HEIGHT = 1800


def world_to_pixel_coords(x_m, y_m):
    """
    Convert world coordinates (meters) to warped image pixel coordinates.
    
    Args:
        x_m: X coordinate in meters (0-9m for court width)
        y_m: Y coordinate in meters (9-18m for far side)
    
    Returns:
        tuple: (x_px, y_px) in warped image coordinates (0-900 for far side)
    """
    # X: meters to pixels (1m = 100px)
    x_px = int(x_m * 100)
    
    # Y: convert to warped image coordinates
    # World Y: 9m (net) to 18m (far baseline)
    # Warped Y: 900 (net) to 0 (far baseline) in full image
    y_px_world = int(y_m * 100)
    y_px_warped = FULL_COURT_HEIGHT - y_px_world
    
    # For far side, we crop to 0-900, so y_px_warped should be 0-900
    # y = 9m → y_px_world = 900 → y_px_warped = 1800 - 900 = 900 (bottom)
    # y = 18m → y_px_world = 1800 → y_px_warped = 1800 - 1800 = 0 (top)
    
    return x_px, y_px_warped


def get_zone_for_coordinate(x_m, y_m):
    """
    Determine which zone (1-6) a coordinate falls into.
    
    Args:
        x_m: X coordinate in meters
        y_m: Y coordinate in meters
    
    Returns:
        int: Zone number (1-6) or None if not in far side or invalid
    """
    # Only analyze far side (y > 9m)
    if y_m <= 9.0:
        return None
    
    # Convert to pixel coordinates
    x_px, y_px = world_to_pixel_coords(x_m, y_m)
    
    # Check bounds (should be 0-900 for far side)
    if x_px < 0 or x_px >= 900 or y_px < 0 or y_px >= 900:
        return None
    
    # Check each zone
    for zone_num, bounds in ZONE_BOUNDARIES.items():
        if (bounds["x_min"] <= x_px < bounds["x_max"] and 
            bounds["y_min"] <= y_px < bounds["y_max"]):
            return zone_num
    
    return None


def load_landing_coordinates(player=None, session_id=None, user_mode=False):
    """
    Load landing coordinates from CSV.
    
    Args:
        player: Optional player name filter
        session_id: Optional session ID filter
        user_mode: Whether to use user/ paths
    
    Returns:
        list: List of dicts with 'player', 'session_id', 'serve_id', 'x', 'y', 'zone'
    """
    if user_mode:
        csv_path = get_user_serves_csv_path()
    else:
        csv_path = "data/metadata/serves.csv"
    
    if not os.path.exists(csv_path):
        return []
    
    coordinates = []
    with open(csv_path, "r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            # Filter by player if specified
            if player and row.get("player") != player:
                continue
            
            # Filter by session if specified
            if session_id is not None:
                try:
                    if int(row.get("session_id", -1)) != session_id:
                        continue
                except (ValueError, TypeError):
                    continue
            
            # Get landing coordinates
            landing_x = row.get("landing_x", "").strip()
            landing_y = row.get("landing_y", "").strip()
            
            if not landing_x or not landing_y:
                continue
            
            try:
                x = float(landing_x)
                y = float(landing_y)
                
                # Determine zone
                zone = get_zone_for_coordinate(x, y)
                
                coordinates.append({
                    "player": row.get("player", ""),
                    "session_id": row.get("session_id", ""),
                    "serve_id": row.get("serve_id", ""),
                    "x": x,
                    "y": y,
                    "zone": zone
                })
            except ValueError:
                continue
    
    return coordinates


def analyze_zones(coordinates, target_zone=None):
    """
    Analyze landing zones and provide statistics.
    
    Args:
        coordinates: List of coordinate dicts
        target_zone: Optional target zone (1-6) to focus on
    
    Returns:
        dict: Statistics about zone distribution
    """
    total = len(coordinates)
    zone_counts = {1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0}
    no_zone = 0
    
    for coord in coordinates:
        zone = coord["zone"]
        if zone is None:
            no_zone += 1
        elif zone in zone_counts:
            zone_counts[zone] += 1
    
    stats = {
        "total": total,
        "zone_counts": zone_counts,
        "no_zone": no_zone,
        "target_zone": target_zone,
        "target_count": zone_counts.get(target_zone, 0) if target_zone else None,
        "target_percentage": None
    }
    
    if target_zone and total > 0:
        stats["target_percentage"] = (stats["target_count"] / total) * 100
    
    return stats


def create_zone_visualization(coordinates, target_zone, player, session_id, output_path, user_mode=False):
    """
    Create visualization showing landing coordinates in target zone vs out of target zone.
    
    Args:
        coordinates: List of coordinate dicts with 'x', 'y', 'zone'
        target_zone: Target zone number (1-6)
        player: Player name
        session_id: Session ID
        output_path: Path to save output image
        user_mode: Whether to use user/ paths
    """
    if not coordinates:
        print("[warning] No coordinates to visualize")
        return False
    
    # Separate coordinates by target zone
    in_zone_coords = [c for c in coordinates if c['zone'] == target_zone]
    out_zone_coords = [c for c in coordinates if c['zone'] is not None and c['zone'] != target_zone]
    
    # Load court background
    corners_data = load_corners(session_id, user_mode)
    if not corners_data:
        print(f"[warning] No court corners found for session {session_id}, using blank canvas")
        W, Hm = 900, 1800
        warped = np.ones((Hm, W, 3), dtype=np.uint8) * 255
    else:
        video_path = corners_data.get("video_file") or corners_data.get("image_file")
        if video_path:
            # Normalize path in case it was stored with Windows backslashes
            video_path = normalize_path(video_path)
        if not video_path or not os.path.exists(video_path):
            if user_mode:
                serve_video_dir = Path(os.path.join("user", "data", "videos", player, f"session_{session_id}"))
            else:
                serve_video_dir = Path(os.path.join("data", "videos", "processed", player, f"session_{session_id}"))
            serve_videos = list(serve_video_dir.glob("serve_*.mp4"))
            if serve_videos:
                video_path = str(serve_videos[0])
            else:
                video_path = None

        if video_path and os.path.exists(video_path):
            frame = load_frame_from_video(video_path, 0)
            if frame is None:
                print(f"[warning] Could not load frame from {video_path}, using blank canvas")
                W, Hm = 900, 1800
                warped = np.ones((Hm, W, 3), dtype=np.uint8) * 255
            else:
                corners = corners_data["court_corners"][:4]
                pts = np.float32([
                    corners[3],  # close left → (0, 0)
                    corners[2],  # close right → (9, 0)
                    corners[1],  # far right → (9, 18)
                    corners[0],  # far left → (0, 18)
                ])
                W, Hm = 900, 1800
                dst_pts = np.float32([[0, Hm], [W, Hm], [W, 0], [0, 0]])
                M, _ = cv2.findHomography(pts, dst_pts)
                warped = cv2.warpPerspective(frame, M, (W, Hm))
        else:
            print(f"[warning] No video found for session {session_id}, using blank canvas")
            W, Hm = 900, 1800
            warped = np.ones((Hm, W, 3), dtype=np.uint8) * 255
    
    # Draw zone boundaries and highlight target zone
    line_color = (200, 200, 200)  # Light gray for boundaries
    line_thickness = 1
    dash_length = 10
    gap_length = 5
    
    # Highlight target zone with semi-transparent overlay
    if target_zone in ZONE_BOUNDARIES:
        bounds = ZONE_BOUNDARIES[target_zone]
        highlight = warped.copy()
        cv2.rectangle(highlight, 
                     (bounds["x_min"], bounds["y_min"]),
                     (bounds["x_max"], bounds["y_max"]),
                     (0, 255, 255), -1)  # Yellow fill
        cv2.addWeighted(warped, 0.7, highlight, 0.3, 0, warped)
    
    # Draw vertical lines (3 columns)
    for x_pos in [300, 600]:
        y_start = 0
        y_end = 900
        y = y_start
        while y < y_end:
            y_end_seg = min(y + dash_length, y_end)
            cv2.line(warped, (x_pos, y), (x_pos, y_end_seg), line_color, line_thickness)
            y += dash_length + gap_length
    
    # Draw horizontal line (2 rows)
    y_pos = 450
    x_start = 0
    x_end = 900
    x = x_start
    while x < x_end:
        x_end_seg = min(x + dash_length, x_end)
        cv2.line(warped, (x, y_pos), (x_end_seg, y_pos), line_color, line_thickness)
        x += dash_length + gap_length
    
    # Add zone numbers
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1.2
    font_thickness = 2
    text_color = (150, 150, 150)
    text_offset = 20
    
    # Zone 1: Top left
    if target_zone == 1:
        cv2.putText(warped, "1", (text_offset, text_offset + 25), font, font_scale, (0, 0, 0), font_thickness + 2)
        cv2.putText(warped, "1", (text_offset, text_offset + 25), font, font_scale, (0, 255, 255), font_thickness)
    else:
        cv2.putText(warped, "1", (text_offset, text_offset + 25), font, font_scale, text_color, font_thickness)
    
    # Zone 2: Bottom left
    if target_zone == 2:
        cv2.putText(warped, "2", (text_offset, 900 - text_offset), font, font_scale, (0, 0, 0), font_thickness + 2)
        cv2.putText(warped, "2", (text_offset, 900 - text_offset), font, font_scale, (0, 255, 255), font_thickness)
    else:
        cv2.putText(warped, "2", (text_offset, 900 - text_offset), font, font_scale, text_color, font_thickness)
    
    # Zone 3: Bottom middle
    text_size = cv2.getTextSize("3", font, font_scale, font_thickness)[0]
    if target_zone == 3:
        cv2.putText(warped, "3", (450 - text_size[0] // 2, 900 - text_offset), font, font_scale, (0, 0, 0), font_thickness + 2)
        cv2.putText(warped, "3", (450 - text_size[0] // 2, 900 - text_offset), font, font_scale, (0, 255, 255), font_thickness)
    else:
        cv2.putText(warped, "3", (450 - text_size[0] // 2, 900 - text_offset), font, font_scale, text_color, font_thickness)
    
    # Zone 4: Bottom right
    text_size = cv2.getTextSize("4", font, font_scale, font_thickness)[0]
    if target_zone == 4:
        cv2.putText(warped, "4", (900 - text_size[0] - text_offset, 900 - text_offset), font, font_scale, (0, 0, 0), font_thickness + 2)
        cv2.putText(warped, "4", (900 - text_size[0] - text_offset, 900 - text_offset), font, font_scale, (0, 255, 255), font_thickness)
    else:
        cv2.putText(warped, "4", (900 - text_size[0] - text_offset, 900 - text_offset), font, font_scale, text_color, font_thickness)
    
    # Zone 5: Top right
    text_size = cv2.getTextSize("5", font, font_scale, font_thickness)[0]
    if target_zone == 5:
        cv2.putText(warped, "5", (900 - text_size[0] - text_offset, text_offset + 25), font, font_scale, (0, 0, 0), font_thickness + 2)
        cv2.putText(warped, "5", (900 - text_size[0] - text_offset, text_offset + 25), font, font_scale, (0, 255, 255), font_thickness)
    else:
        cv2.putText(warped, "5", (900 - text_size[0] - text_offset, text_offset + 25), font, font_scale, text_color, font_thickness)
    
    # Zone 6: Top middle
    text_size = cv2.getTextSize("6", font, font_scale, font_thickness)[0]
    if target_zone == 6:
        cv2.putText(warped, "6", (450 - text_size[0] // 2, text_offset + 25), font, font_scale, (0, 0, 0), font_thickness + 2)
        cv2.putText(warped, "6", (450 - text_size[0] // 2, text_offset + 25), font, font_scale, (0, 255, 255), font_thickness)
    else:
        cv2.putText(warped, "6", (450 - text_size[0] // 2, text_offset + 25), font, font_scale, text_color, font_thickness)
    
    # Draw landing points
    overlay = warped.copy()
    
    # Draw out-of-zone points (red)
    for coord in out_zone_coords:
        x_px = int(coord["x"] * 100)
        y_px_world = int(coord["y"] * 100)
        if y_px_world <= 900:
            continue
        y_px_warped = Hm - y_px_world
        if 0 <= x_px < W and 0 <= y_px_warped < Hm:
            cv2.circle(overlay, (x_px, y_px_warped), 6, (0, 0, 255), -1)  # Red
    
    # Draw in-zone points (green)
    for coord in in_zone_coords:
        x_px = int(coord["x"] * 100)
        y_px_world = int(coord["y"] * 100)
        if y_px_world <= 900:
            continue
        y_px_warped = Hm - y_px_world
        if 0 <= x_px < W and 0 <= y_px_warped < Hm:
            cv2.circle(overlay, (x_px, y_px_warped), 8, (0, 255, 0), -1)  # Green, slightly larger
            cv2.circle(overlay, (x_px, y_px_warped), 8, (0, 0, 0), 2)  # Black border
    
    # Blend overlay
    alpha = 0.8
    warped = cv2.addWeighted(warped, 1.0 - alpha, overlay, alpha, 0)
    
    # Crop to far side only
    warped_far = warped[0:900, :]
    
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)
    cv2.imwrite(output_path, warped_far)
    print(f"Saved zone visualization → {output_path}")
    return True


def print_analysis(stats, coordinates, target_zone=None):
    """Print zone analysis results."""
    print("\n" + "="*60)
    print("LANDING ZONE ANALYSIS")
    print("="*60)
    
    if target_zone:
        print(f"\nTarget Zone: {target_zone}")
    
    print(f"\nTotal serves analyzed: {stats['total']}")
    
    if stats['total'] == 0:
        print("\nNo serves with landing coordinates found.")
        return
    
    print("\nZone Distribution:")
    print("-" * 60)
    for zone_num in sorted(stats['zone_counts'].keys()):
        count = stats['zone_counts'][zone_num]
        percentage = (count / stats['total']) * 100 if stats['total'] > 0 else 0
        marker = " ← TARGET" if target_zone == zone_num else ""
        print(f"  Zone {zone_num}: {count:3d} serves ({percentage:5.1f}%){marker}")
    
    if stats['no_zone'] > 0:
        no_zone_pct = (stats['no_zone'] / stats['total']) * 100
        print(f"  No zone:  {stats['no_zone']:3d} serves ({no_zone_pct:5.1f}%)")
    
    if target_zone:
        print("\n" + "-" * 60)
        print(f"Target Zone {target_zone} Results:")
        print(f"  Serves in zone: {stats['target_count']}")
        print(f"  Percentage: {stats['target_percentage']:.1f}%")
        
        # Show serves that landed in target zone
        target_serves = [c for c in coordinates if c['zone'] == target_zone]
        if target_serves:
            print(f"\n  Serves in Zone {target_zone}:")
            for serve in target_serves:
                print(f"    {serve['player']}/session_{serve['session_id']}/serve_{serve['serve_id']} "
                      f"→ ({serve['x']:.2f}m, {serve['y']:.2f}m)")
    
    print("\n" + "="*60)


def main():
    parser = argparse.ArgumentParser(
        description="Analyze landing zones for serves",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Zone Layout (far side of court):
  ┌─────┬─────┬─────┐
  │  1  │  6  │  5  │  Top row
  ├─────┼─────┼─────┤
  │  2  │  3  │  4  │  Bottom row
  └─────┴─────┴─────┘

Examples:
  # Analyze all serves for target zone 1
  python scripts/analyze_landing_zones.py --target-zone 1

  # Analyze specific player/session
  python scripts/analyze_landing_zones.py --target-zone 3 --player spencer --session 1

  # User mode
  python scripts/analyze_landing_zones.py --target-zone 5 --user-mode
        """
    )
    
    parser.add_argument("--target-zone", type=int, choices=[1, 2, 3, 4, 5, 6],
                       help="Target zone to analyze (1-6)")
    add_player_session_serve_args(parser)
    add_user_mode_arg(parser)
    
    args = parser.parse_args()
    
    # Load coordinates
    coordinates = load_landing_coordinates(
        player=args.player,
        session_id=args.session,
        user_mode=args.user_mode
    )
    
    # Analyze
    stats = analyze_zones(coordinates, target_zone=args.target_zone)
    
    # Print results
    print_analysis(stats, coordinates, target_zone=args.target_zone)
    
    # Create visualization if target zone specified and we have coordinates
    if args.target_zone and coordinates and stats['total'] > 0:
        # Determine player and session from coordinates if not provided
        vis_player = args.player
        vis_session = args.session
        
        if not vis_player or vis_session is None:
            # Get from first coordinate
            if coordinates:
                vis_player = coordinates[0]['player']
                try:
                    vis_session = int(coordinates[0]['session_id'])
                except (ValueError, TypeError):
                    vis_session = None
        
        if vis_player and vis_session is not None:
            # Determine output path
            if args.user_mode:
                output_dir = os.path.join("user", "visualizations", f"session_{vis_session}")
            else:
                output_dir = os.path.join("data", "visualizations", vis_player, f"session_{vis_session}")
            
            os.makedirs(output_dir, exist_ok=True)
            output_path = os.path.join(output_dir, f"zone_{args.target_zone}_analysis.png")
            
            create_zone_visualization(
                coordinates, 
                args.target_zone, 
                vis_player, 
                vis_session, 
                output_path, 
                user_mode=args.user_mode
            )
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
