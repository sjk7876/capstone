"""
Generate visualization images for landing frame analysis.

Creates annotated images showing ball size, position, and estimated landing frames
for analysis and validation of landing detection algorithms.
"""
import json
import numpy as np
import argparse
import os
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
from common_args import add_player_session_serve_args, build_trajectory_paths, format_serve_number
from estimate_landing import (
    estimate_hit_and_landing,
    append_estimate_to_csv,
    load_serves_csv,
    extract_serve_id_from_path,
    find_trajectory_files
)


def compute_smoothed_signals(track):
    """Compute smoothed y, dy, and size signals from trajectory."""
    if not track or len(track) < 3:
        return None, None, None, None
    
    frames = np.array([p["frame"] for p in track], dtype=int)
    ys = np.array([p["center"][1] for p in track], dtype=float)
    
    # Handle size safely
    sizes = []
    for p in track:
        if "size" in p and len(p["size"]) >= 2:
            sizes.append([p["size"][0], p["size"][1]])
        else:
            sizes.append([0.0, 0.0])
    
    ws = np.array([s[0] for s in sizes], dtype=float)
    hs = np.array([s[1] for s in sizes], dtype=float)
    size = np.sqrt(ws * hs)
    
    # Smoothing
    window_length = min(9, len(track) if len(track) % 2 == 1 else len(track) - 1)
    if window_length < 3:
        window_length = 3
    
    size_s = savgol_filter(size, window_length, min(2, window_length - 1), mode='interp')
    y_s = savgol_filter(ys, window_length, min(2, window_length - 1), mode='interp')
    dy = np.gradient(y_s, frames)
    
    return frames, y_s, dy, size_s


def plot_trajectory_signals(track, hit_frame, landing_frame, actual_landing, save_path):
    """Plot y, dy, and size over frames with vertical lines for hit and landing."""
    frames, y_s, dy, size_s = compute_smoothed_signals(track)
    if frames is None:
        return
    
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
    
    # Plot y position
    ax1.plot(frames, y_s, 'b-', linewidth=1.5, label='y position')
    ax1.set_ylabel('y position (px)', color='b')
    ax1.tick_params(axis='y', labelcolor='b')
    ax1.grid(True, alpha=0.3)
    ax1.invert_yaxis()
    
    # Plot dy (velocity)
    ax2.plot(frames, dy, 'g-', linewidth=1.5, label='dy/dt')
    ax2.set_ylabel('dy/dt (px/frame)', color='g')
    ax2.tick_params(axis='y', labelcolor='g')
    ax2.grid(True, alpha=0.3)
    ax2.axhline(y=0, color='k', linestyle='--', linewidth=0.5)
    
    # Plot size
    ax3.plot(frames, size_s, 'r-', linewidth=1.5, label='size')
    ax3.set_ylabel('size (px)', color='r')
    ax3.set_xlabel('frame')
    ax3.tick_params(axis='y', labelcolor='r')
    ax3.grid(True, alpha=0.3)
    
    # Add vertical lines for hit and landing
    if hit_frame is not None:
        ax1.axvline(x=hit_frame, color='orange', linestyle='--', linewidth=2, label='hit')
        ax2.axvline(x=hit_frame, color='orange', linestyle='--', linewidth=2, label='hit')
        ax3.axvline(x=hit_frame, color='orange', linestyle='--', linewidth=2, label='hit')
    
    if landing_frame is not None:
        ax1.axvline(x=landing_frame, color='purple', linestyle='--', linewidth=2, label='landing (est)')
        ax2.axvline(x=landing_frame, color='purple', linestyle='--', linewidth=2, label='landing (est)')
        ax3.axvline(x=landing_frame, color='purple', linestyle='--', linewidth=2, label='landing (est)')
    
    if actual_landing is not None:
        ax1.axvline(x=actual_landing, color='red', linestyle='--', linewidth=2, label='landing (actual)')
        ax2.axvline(x=actual_landing, color='red', linestyle='--', linewidth=2, label='landing (actual)')
        ax3.axvline(x=actual_landing, color='red', linestyle='--', linewidth=2, label='landing (actual)')
    
    ax1.legend(loc='lower left')
    ax2.legend(loc='lower left')
    ax3.legend(loc='lower left')
    
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved plot → {save_path}")
    
    plt.close()


def process_trajectory(json_path, player, session, serves_csv):
    """Process a single trajectory file and print results with actual landing frame."""
    if not os.path.exists(json_path):
        print(f"[skip] {os.path.basename(json_path)}: file not found")
        return
    
    serve_id = extract_serve_id_from_path(json_path)
    if not serve_id:
        print(f"[skip] {os.path.basename(json_path)}: could not extract serve_id")
        return
    
    with open(json_path) as f:
        track = json.load(f)

    hit, land = estimate_hit_and_landing(track)
    if hit is None or land is None:
        print(f"{os.path.basename(json_path)}: insufficient data (need at least 3 points)")
        return
    
    # Get actual landing frame from CSV
    key = (player, session, serve_id)
    actual_landing = serves_csv.get(key)
    
    if actual_landing is not None:
        error = land - actual_landing
        print(f"{os.path.basename(json_path)}: hit={hit}, landing_est={land}, landing_actual={actual_landing}, error={error:+d}")
        append_estimate_to_csv(player, session, serve_id, hit, land, actual_landing)
    else:
        print(f"{os.path.basename(json_path)}: hit={hit}, landing={land} (no CSV entry)")
        append_estimate_to_csv(player, session, serve_id, hit, land)
    
    # Save visualization plot
    serve_str = format_serve_number(serve_id)
    plot_path = os.path.join("data", "visualizations", "landing_analysis", player, f"session_{session}", f"serve_{serve_str}.png")
    plot_trajectory_signals(track, hit, land, actual_landing, plot_path)


def main():
    parser = argparse.ArgumentParser(description="Estimate hit and landing frames from trajectory JSON")
    
    # Common player/session/serve arguments
    add_player_session_serve_args(parser)
    
    args = parser.parse_args()
    
    if not (args.player and args.session is not None):
        parser.error("Must specify --player and --session")
    
    # Load serves.csv
    serves_csv = load_serves_csv()
    
    if args.serve:
        # Single serve specified
        _, json_path = build_trajectory_paths(args.player, args.session, args.serve)
        process_trajectory(json_path, args.player, args.session, serves_csv)
    else:
        # Process all serves in session
        json_files = find_trajectory_files(args.player, args.session)
        if not json_files:
            print(f"No trajectory files found in data/trajectories/{args.player}/session_{args.session}/")
            return
        
        print(f"Processing {len(json_files)} serves in session {args.session}:")
        for json_path in json_files:
            process_trajectory(json_path, args.player, args.session, serves_csv)


if __name__ == "__main__":
    main()
