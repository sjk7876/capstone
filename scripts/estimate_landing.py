"""
Estimate hit and landing frames from ball trajectory data.

Analyzes trajectory JSON files to detect hit frame (strongest downward motion)
and landing frame (lowest ball size after hit) using signal processing techniques.
"""
import json
import numpy as np
import argparse
import os
import glob
import csv
import re
from scipy.signal import savgol_filter, find_peaks
from common_args import add_player_session_serve_args, build_trajectory_paths, format_serve_number

# Hit estimation constants
MIN_HIT_FRAME = 20  # Hit frame will not be in the first N frames
MIN_TRACK_LENGTH_FALLBACK = 5  # Minimum track length for fallback logic
SIZE_THRESHOLD_MULTIPLIER = 0.9  # Multiplier for max size to define "near server" region
MIN_REGION_SIZE = 3  # Minimum size for valid region
SHRINK_DETECTION_WINDOW = 15  # Number of consecutive frames to check for shrink
MIN_SIZE_DROP = 0.5  # Minimum total size drop to consider a hit

# Landing estimation constants
ROLLING_WINDOW_MIN = 3  # Minimum rolling window size for size smoothing
LANDING_SEARCH_OFFSET = 10  # Frames after hit to start searching for landing
PEAK_PROMINENCE = 0.05  # Prominence threshold for peak detection
PEAK_MATCH_TOLERANCE = 3  # Frame tolerance for matching peaks to landing estimate
REFINEMENT_WINDOW_LEFT = 3  # Frames to look left for landing refinement
REFINEMENT_WINDOW_RIGHT = 4  # Frames to look right for landing refinement
FALLBACK_LANDING_OFFSET = 1  # Frames to add to hit frame if no landing found
Y_PEAK_PROMINENCE = 5.0  # tweak per your scale
Y_PEAK_MIN_DISTANCE = 2  # min frames between y-peaks
MAX_LANDING_WINDOW = 120  # don't search forever after hit
MAX_LANDING_SIZE_RATIO = 0.5  # Maximum ball size (as fraction of max size) to consider for landing detection

# Signal processing constants
SAVGOL_WINDOW_LENGTH = 9  # Savitzky-Golay filter window length
SAVGOL_POLY_ORDER = 2  # Savitzky-Golay filter polynomial order
MIN_TRACK_LENGTH = 3  # Minimum track length to process


def _estimate_hit_from_toss_and_size(frames: np.array, y_s: np.array, size_s: np.array) -> tuple[int, int]:
    """Estimate hit frame from toss pattern and ball size."""
    n = len(frames)
    
    # Hit frame will not be in the first N frames - skip early frames
    min_frame = MIN_HIT_FRAME
    min_frame_idx = None
    for i, f in enumerate(frames):
        if f >= min_frame:
            min_frame_idx = i
            break
    
    if min_frame_idx is None:
        # If clip is shorter than MIN_HIT_FRAME frames, use all frames
        min_frame_idx = 0
    
    if n < MIN_TRACK_LENGTH_FALLBACK:
        # dumb fallback - skip first 20 frames
        search_size = size_s[min_frame_idx:]
        if len(search_size) > 0:
            hit_idx = int(np.argmin(search_size)) + min_frame_idx
        else:
            hit_idx = int(np.argmin(size_s))
        return int(frames[hit_idx]), hit_idx

    max_size = float(np.max(size_s))

    # 1) "near server" region: ball still big (toss + hit)
    size_thresh = SIZE_THRESHOLD_MULTIPLIER * max_size
    near_mask = size_s >= size_thresh
    
    near_idxs = np.where(near_mask)[0]
    if len(near_idxs) == 0:
        # fallback if size threshold fails - skip first 20 frames
        search_size = size_s[min_frame_idx:]
        if len(search_size) > 0:
            hit_idx = int(np.argmin(search_size)) + min_frame_idx
        else:
            hit_idx = int(np.argmin(size_s))
        return int(frames[hit_idx]), hit_idx

    start_idx = int(near_idxs[0])
    end_idx   = int(near_idxs[-1])
    
    # Adjust start_idx to be at least min_frame_idx
    start_idx = max(start_idx, min_frame_idx)

    # also don't go too far into the clip (avoid landing)
    end_idx = min(end_idx, int(0.7 * n))

    if end_idx - start_idx < MIN_REGION_SIZE:
        hit_idx = int(np.argmin(size_s[start_idx:end_idx+1]) + start_idx)
        return int(frames[hit_idx]), hit_idx

    # 2) toss apex = minimum y in this near region
    local_apex_rel = int(np.argmin(y_s[start_idx:end_idx+1]))
    apex_idx = start_idx + local_apex_rel

    # 3) from apex forward, look for first "sustained shrink" in size
    # Check if size decreases on average over k frames
    k = SHRINK_DETECTION_WINDOW
    hit_idx = None
    for i in range(apex_idx, end_idx - k):
        # Check average shrink over k frames
        seq = size_s[i : i + k + 1]
        avg_shrink = np.mean(np.diff(seq))
        # also require a minimum total drop to avoid noise
        if avg_shrink < 0 and seq[0] - seq[-1] > MIN_SIZE_DROP:
            hit_idx = i
            break

    if hit_idx is None:
        # fallback: choose where shrink becomes strongest after apex
        dsize = np.gradient(size_s, frames)
        region = dsize[apex_idx:end_idx+1]
        local_idx = int(np.argmin(region))
        hit_idx = apex_idx + local_idx

    hit_frame = int(frames[hit_idx])
    return hit_frame, hit_idx


def _estimate_landing_from_size_and_motion(frames, y_s, size_s, dy, hit_frame):
    n = len(frames)
    max_size = float(np.max(size_s))
    size_threshold = max_size * MAX_LANDING_SIZE_RATIO

    # 1) choose a search segment after hit
    start_frame = hit_frame + LANDING_SEARCH_OFFSET
    end_frame   = min(frames[-1], hit_frame + MAX_LANDING_WINDOW)

    search_mask = (frames >= start_frame) & (frames <= end_frame)
    idxs = np.where(search_mask)[0]
    if len(idxs) == 0:
        # fallback: global min size after hit (with size constraint)
        after_hit = np.where(frames > hit_frame)[0]
        if len(after_hit):
            after_hit_sizes = size_s[after_hit]
            after_hit_frames = frames[after_hit]
            # Filter by size threshold
            size_valid = after_hit_sizes <= size_threshold
            if np.any(size_valid):
                valid_idxs = after_hit[size_valid]
                return int(frames[valid_idxs[np.argmin(size_s[valid_idxs])]])
            else:
                # If no valid size found, use minimum anyway
                return int(frames[after_hit[np.argmin(after_hit_sizes)]])
        return hit_frame + FALLBACK_LANDING_OFFSET

    # 2) find first strong *maximum* in y (ball hits floor, y locally largest)
    # Filter by size threshold first
    size_valid_mask = size_s[idxs] <= size_threshold
    valid_idxs = idxs[size_valid_mask]
    
    if len(valid_idxs) == 0:
        # No valid candidates by size, fall through to rolling-size logic
        valid_idxs = idxs
    
    y_seg = y_s[valid_idxs]
    peaks, props = find_peaks(
        y_seg,
        prominence=Y_PEAK_PROMINENCE,
        distance=Y_PEAK_MIN_DISTANCE
    )

    landing_idx = None
    if len(peaks):
        # pick earliest peak that actually looks like a bounce:
        # y before increasing, y after not strictly increasing
        for p in peaks:
            # p is an index into y_seg, which corresponds to valid_idxs
            g_idx = valid_idxs[p]
            if g_idx == 0 or g_idx >= n - 1:
                continue
            # require "coming down" before
            if y_s[g_idx] <= y_s[g_idx-1]:
                continue
            # and "not still screaming down" after
            if y_s[g_idx+1] >= y_s[g_idx]:
                continue
            # also check size threshold (should already be valid, but double-check)
            if size_s[g_idx] <= size_threshold:
                landing_idx = g_idx
                break

    # 3) if y-peak failed, fall back to old rolling-size logic (but only as backup)
    if landing_idx is None:
        roll_win = max(ROLLING_WINDOW_MIN,
                       len(size_s) // 2 * 2 + 1 if len(size_s) < ROLLING_WINDOW_MIN else ROLLING_WINDOW_MIN)
        rolling = np.convolve(size_s, np.ones(roll_win) / roll_win, mode="valid")
        offset = roll_win // 2
        roll_frames = frames[offset : offset + len(rolling)]

        mask2 = roll_frames > hit_frame + LANDING_SEARCH_OFFSET
        if np.any(mask2):
            # Filter rolling average by size threshold
            roll_sizes = rolling[mask2]
            roll_frames_masked = roll_frames[mask2]
            size_valid_roll = roll_sizes <= size_threshold
            
            if np.any(size_valid_roll):
                landing_idx_local = np.argmin(roll_sizes[size_valid_roll])
                coarse = int(roll_frames_masked[size_valid_roll][landing_idx_local])
            else:
                # If no valid size found, use minimum anyway
                landing_idx_local = np.argmin(roll_sizes)
                coarse = int(roll_frames_masked[landing_idx_local])

            # refine on *y* and *size* in a tiny window around coarse
            cand = np.where((frames >= coarse - 2) & (frames <= coarse + 2))[0]
            if len(cand):
                # Filter candidates by size threshold, then prefer largest y (closest to floor); if tie, smallest size
                cand_valid = [i for i in cand if size_s[i] <= size_threshold]
                if len(cand_valid):
                    best = max(cand_valid, key=lambda i: (y_s[i], -size_s[i]))
                    landing_idx = best
                else:
                    # If no valid candidates, use best from all candidates
                    best = max(cand, key=lambda i: (y_s[i], -size_s[i]))
                    landing_idx = best
            else:
                landing_idx = np.argmin(size_s)  # absolute fallback
        else:
            landing_idx = np.argmin(size_s)

    landing_frame = int(frames[landing_idx])

    # 4) sanity: enforce landing after hit
    if landing_frame <= hit_frame:
        after_hit = np.where(frames > hit_frame)[0]
        if len(after_hit):
            # Filter by size threshold
            after_hit_sizes = size_s[after_hit]
            size_valid = after_hit_sizes <= size_threshold
            if np.any(size_valid):
                valid_idxs = after_hit[size_valid]
                landing_frame = int(frames[valid_idxs[np.argmin(size_s[valid_idxs])]])
            else:
                landing_frame = int(frames[after_hit[np.argmin(after_hit_sizes)]])
        else:
            landing_frame = hit_frame + FALLBACK_LANDING_OFFSET

    return landing_frame
    

def estimate_hit_and_landing(track):
    """
    Estimate hit and landing frames from ball trajectory.
    
    Args:
        track: List of trajectory points, each with 'frame', 'center', and optionally 'size'
    
    Returns:
        Tuple of (hit_frame, landing_frame) or (None, None) if insufficient data
    """
    if not track or len(track) < MIN_TRACK_LENGTH:
        return None, None
    
    # ----------------- unpack -----------------
    frames = np.array([p["frame"] for p in track], dtype=int)
    xs     = np.array([p["center"][0] for p in track], dtype=float)
    ys     = np.array([p["center"][1] for p in track], dtype=float)

    # handle size safely
    sizes = []
    for p in track:
        if "size" in p and len(p["size"]) >= 2:
            sizes.append([p["size"][0], p["size"][1]])
        else:
            sizes.append([0.0, 0.0])

    ws = np.array([s[0] for s in sizes], dtype=float)
    hs = np.array([s[1] for s in sizes], dtype=float)
    size = np.sqrt(ws * hs)

    # ----------------- smoothing -----------------
    window_length = min(SAVGOL_WINDOW_LENGTH, len(track) if len(track) % 2 == 1 else len(track) - 1)
    if window_length < MIN_REGION_SIZE:
        window_length = MIN_REGION_SIZE

    x_s    = savgol_filter(xs,   window_length, min(SAVGOL_POLY_ORDER, window_length - 1), mode='interp')
    y_s    = savgol_filter(ys,   window_length, min(SAVGOL_POLY_ORDER, window_length - 1), mode='interp')
    size_s = savgol_filter(size, window_length, min(SAVGOL_POLY_ORDER, window_length - 1), mode='interp')

    dy = np.gradient(y_s, frames)   # keep for landing refinement

    # ----------------- hit from toss pattern + size -----------------
    hit_frame, hit_idx = _estimate_hit_from_toss_and_size(frames, y_s, size_s)

    # ----------------- landing estimate -----------------
    landing_frame = _estimate_landing_from_size_and_motion(frames, y_s, size_s, dy, hit_frame)

    # skip if ball never rebounds (no landing in clip)
    if np.all(np.diff(size_s) < 0) or np.all(np.diff(y_s) < 0):
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
    traj_dir = os.path.join("data", "trajectories", player, f"session_{session}")
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
