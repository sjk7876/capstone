import os, json, argparse, numpy as np, csv, re, glob
from common_args import add_player_session_serve_args, build_trajectory_paths, format_serve_number

def load_homography(session_id):
    path = f"data/calibration/homographies/{session_id}.json"
    if not os.path.exists(path):
        raise FileNotFoundError(f"Homography file not found: {path}")
    with open(path) as f:
        data = json.load(f)
    return np.array(data["H"], dtype=np.float64)

def warp_point(u, v, H):
    p = np.array([u, v, 1.0])
    q = H @ p
    return float(q[0] / q[2]), float(q[1] / q[2])

def get_landing_frame(player, session, serve):
    """Get landing frame from landing_estimates.csv."""
    csv_path = "data/metadata/landing_estimates.csv"
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Landing estimates CSV not found: {csv_path}")
    
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            if (row["player"] == player and 
                str(row["session"]) == str(session) and 
                row["serve_id"] == format_serve_number(serve)):
                landing_actual = row.get("landing_actual", "").strip()
                if landing_actual:
                    try:
                        return int(landing_actual)
                    except ValueError:
                        pass
    return None

def project_landing(player, session, serve):
    """Project landing frame from pixel to world coordinates."""
    _, traj_path = build_trajectory_paths(player, session, serve)
    if not os.path.exists(traj_path):
        print(f"[skip] No trajectory for serve {serve}")
        return None

    with open(traj_path) as f:
        traj = json.load(f)

    if not traj:
        print(f"[skip] Empty trajectory for serve {serve}")
        return None

    try:
        landing_frame = get_landing_frame(player, session, serve)
    except FileNotFoundError as e:
        print(f"[skip] {e}")
        return None
    
    if landing_frame is None:
        print(f"[skip] No landing frame in landing_estimates.csv for serve {serve}")
        return None

    # Find nearest frame
    frames = np.array([p["frame"] for p in traj])
    idx = (np.abs(frames - landing_frame)).argmin()
    point = traj[idx]
    u, v = point["center"]
    
    # Calculate bottom of ball: y - (height / 2)
    size = point.get("size", [0.0, 0.0])
    if len(size) >= 2:
        height = size[1]
        v_bottom = v + (height / 2)
    else:
        # Fallback if size not available
        v_bottom = v

    try:
        H = load_homography(f"session_{session}")
    except FileNotFoundError as e:
        print(f"[skip] {e}")
        return None
    
    # Project both center and bottom of ball to world coordinates
    X_center, Y_center = warp_point(u, v, H)
    X, Y = warp_point(u, v_bottom, H)

    out_dir = f"data/landings_world/{player}/session_{session}/"
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"serve_{format_serve_number(serve)}.json")

    with open(out_path, "w") as f:
        json.dump({
            "player": player,
            "session": session,
            "serve_id": format_serve_number(serve),
            "frame": int(frames[idx]),
            "pixel_center": [u, v],
            "pixel_bottom": [u, v_bottom],
            "world_center_m": [X_center, Y_center],
            "world_bottom_m": [X, Y]
        }, f, indent=2)

    print(f"Projected landing {serve} → ({X:.2f}m, {Y:.2f}m)")
    return out_path

def main():
    parser = argparse.ArgumentParser(description="Project only the landing frame to world coords")
    add_player_session_serve_args(parser)
    args = parser.parse_args()

    if not args.player or args.session is None:
        parser.error("must specify --player and --session")

    if args.serve:
        project_landing(args.player, args.session, args.serve)
    else:
        # Project all serves in session
        traj_dir = f"data/trajectories/{args.player}/session_{args.session}/"
        if not os.path.exists(traj_dir):
            print(f"Trajectory directory not found: {traj_dir}")
            return
        
        files = sorted(glob.glob(os.path.join(traj_dir, "serve_*.json")))
        if not files:
            print(f"No trajectory files found in {traj_dir}")
            return
        
        print(f"Projecting {len(files)} serves in session {args.session}:")
        for f in files:
            m = re.search(r"serve_(\d+)\.json$", f)
            if m:
                serve_id = m.group(1)
                project_landing(args.player, args.session, serve_id)

if __name__ == "__main__":
    main()
