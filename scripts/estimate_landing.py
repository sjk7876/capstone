import json
import numpy as np
import argparse
import os
import glob
import csv
import re
from scipy.signal import savgol_filter, find_peaks
from common_args import add_player_session_serve_args, build_trajectory_paths, format_serve_number

def estimate_hit_and_landing(track):
    if not track or len(track) < 3:
        return None, None
    
    frames = np.array([p["frame"] for p in track], dtype=int)
    ys = np.array([p["center"][1] for p in track], dtype=float)

    # --- handle size safely ---
    sizes = []
    for p in track:
        if "size" in p and len(p["size"]) >= 2:
            sizes.append([p["size"][0], p["size"][1]])
        else:
            sizes.append([0.0, 0.0])

    ws = np.array([s[0] for s in sizes], dtype=float)
    hs = np.array([s[1] for s in sizes], dtype=float)
    size = np.sqrt(ws * hs)

    # --- smoothing ---
    window_length = min(9, len(track) if len(track) % 2 == 1 else len(track) - 1)
    if window_length < 3:
        window_length = 3

    size_s = savgol_filter(size, window_length, min(2, window_length - 1), mode='interp')
    y_s = savgol_filter(ys, window_length, min(2, window_length - 1), mode='interp')

    dy = np.gradient(y_s, frames)
    ddy = np.gradient(dy)

    # --- hit: strongest downward motion ---
    hit_idx = np.argmax(dy)
    left, right = max(0, hit_idx - 3), min(len(frames), hit_idx + 4)
    local_range = np.arange(left, right)
    if len(local_range) > 0:
        hit_refine = local_range[np.argmax(np.abs(ddy[local_range]))]
        hit_idx = hit_refine
    hit_frame = int(frames[hit_idx])

    # --- landing: lowest rolling-average size after hit ---
    roll_win = 5
    if len(size_s) < roll_win:
        roll_win = max(3, len(size_s) // 2 * 2 + 1)
    rolling = np.convolve(size_s, np.ones(roll_win) / roll_win, mode="valid")
    offset = roll_win // 2
    roll_frames = frames[offset : offset + len(rolling)]

    mask = roll_frames > hit_frame + 20
    if np.any(mask):
        landing_idx = np.argmin(rolling[mask])
        landing_frame = int(roll_frames[mask][landing_idx])
    else:
        landing_frame = int(frames[np.argmin(size_s)])

    # fine-tune landing: local inflection near min(size)
    mins, _ = find_peaks(-size_s, prominence=0.05)
    nearest = None
    for idx in mins:
        if abs(frames[idx] - landing_frame) <= 3:
            nearest = idx
            break
    if nearest is not None:
        left, right = max(0, nearest - 3), min(len(frames), nearest + 4)
        local_range = np.arange(left, right)
        if len(local_range) > 0:
            land_refine = local_range[np.argmax(np.abs(np.gradient(dy[local_range])))]
            landing_frame = int(frames[land_refine])

    # --- sanity check: enforce landing after hit ---
    if landing_frame <= hit_frame:
        # fallback: pick the first frame > hit_frame with min(size)
        after_hit = np.where(frames > hit_frame)[0]
        if len(after_hit):
            landing_frame = int(frames[after_hit[np.argmin(size_s[after_hit])]])
        else:
            landing_frame = hit_frame + 5  # minimal fallback

    # --- skip if ball never rebounds (no landing in clip) ---
    if np.all(np.diff(size_s) < 0) or np.all(np.diff(y_s) < 0):
        # no rebound or track ends while still shrinking
        return hit_frame, None

    return hit_frame, landing_frame


def append_estimate_to_csv(player, session, serve_id, hit, landing, actual=None):
    """Write results to metadata/landing_estimates.csv, updating existing row if present."""
    os.makedirs("data/metadata", exist_ok=True)
    out_path = "data/metadata/landing_estimates.csv"
    
    # Read existing rows if file exists
    rows = []
    fieldnames = ["player", "session", "serve_id", "hit_est", "landing_est", "landing_actual", "error"]
    file_exists = os.path.exists(out_path) and os.path.getsize(out_path) > 0
    
    if file_exists:
        with open(out_path, "r", newline="") as f:
            reader = csv.DictReader(f)
            rows = list(reader)
    
    # Calculate error
    err = "" if actual is None or landing is None else landing - actual
    
    # Check if row exists for this (player, session, serve_id)
    updated = False
    for row in rows:
        if (row.get("player") == str(player) and 
            row.get("session") == str(session) and 
            row.get("serve_id") == str(serve_id)):
            # Update existing row
            row["hit_est"] = str(hit) if hit is not None else ""
            row["landing_est"] = str(landing) if landing is not None else ""
            row["landing_actual"] = str(actual) if actual is not None else ""
            row["error"] = str(err) if err != "" else ""
            updated = True
            break
    
    # If not found, add new row
    if not updated:
        rows.append({
            "player": str(player),
            "session": str(session),
            "serve_id": str(serve_id),
            "hit_est": str(hit) if hit is not None else "",
            "landing_est": str(landing) if landing is not None else "",
            "landing_actual": str(actual) if actual is not None else "",
            "error": str(err) if err != "" else ""
        })
    
    # Write all rows back
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def load_serves_csv():
    """Load serves.csv into a dictionary keyed by (player, session_id, serve_id)."""
    csv_path = "data/metadata/serves.csv"
    if not os.path.exists(csv_path):
        return {}
    
    serves = {}
    with open(csv_path, "r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            player = row["player"]
            session_id = int(row["session_id"])
            serve_id = row["serve_id"]
            landing_frame = row.get("landing_frame", "").strip()
            if landing_frame:
                try:
                    serves[(player, session_id, serve_id)] = int(landing_frame)
                except ValueError:
                    pass
    return serves


def extract_serve_id_from_path(json_path):
    """Extract serve_id from JSON path (e.g., 'serve_001.json' -> '001')."""
    basename = os.path.basename(json_path)
    match = re.search(r"serve_(\d+)\.json$", basename)
    if match:
        return format_serve_number(match.group(1))
    return None


def find_trajectory_files(player, session):
    """Find all trajectory JSON files in a session directory."""
    traj_dir = f"data/trajectories/{player}/session_{session}"
    if not os.path.exists(traj_dir):
        return []
    
    json_files = sorted(glob.glob(os.path.join(traj_dir, "serve_*.json")))
    return json_files


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
