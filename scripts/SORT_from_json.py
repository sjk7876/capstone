import os
import sys
import json
import argparse
import numpy as np
import matplotlib.pyplot as plt

sys.path.append("external")
sys.path.append("external/filterpy")
from sort.sort import Sort
from common_args import add_player_session_serve_args, format_serve_number


def merge_tracks(trajectories,
                 frame_gap_thresh=8,
                 dist_thresh=120,
                 overlap_back=8,          # allow up to 8 frames of overlap (negative gap)
                 alpha_gap=1.0,           # cost weight for temporal gap
                 beta_dist=0.02,          # cost weight for spatial distance
                 debug=False):
    """
    Greedy O(n^2) multi-chain merging of SORT fragments.

    - Build all candidate links i->j when start(j) is within [-overlap_back, +frame_gap_thresh] of end(i) 
    - and tail->head center distance <= dist_thresh.
    - Cost = alpha_gap * max(gap, 0) + beta_dist * dist + 0.5 * max(-gap, 0)  (small penalty for overlap).
    - Greedily accept links by ascending cost while enforcing indegree/outdegree <= 1 and preventing cycles.
    - Return list of merged chains (each chain is a list of points with {frame, center}).
    """

    # 1) normalize & index fragments
    frags = []
    for tid, tr in trajectories.items():
        if not tr:
            continue
        tr = sorted(tr, key=lambda p: p["frame"])
        frags.append({
            "id": tid,
            "pts": tr,
            "start_f": tr[0]["frame"],
            "end_f": tr[-1]["frame"],
            "head": np.array(tr[0]["center"], dtype=float),
            "tail": np.array(tr[-1]["center"], dtype=float),
        })
    if not frags:
        return []

    n = len(frags)

    # 2) all candidate edges i->j (O(n^2))
    edges = []
    for i in range(n):
        fi = frags[i]
        for j in range(n):
            if i == j:
                continue
            fj = frags[j]
            gap = fj["start_f"] - fi["end_f"]               # negative = overlap
            if gap > frame_gap_thresh or gap < -overlap_back:
                continue
            dist = float(np.linalg.norm(fj["head"] - fi["tail"]))
            if dist > dist_thresh:
                continue
            # cheap cost: prefer small positive gaps and short distances; mild penalty for overlap
            cost = alpha_gap * max(gap, 0) + beta_dist * dist + 0.5 * max(-gap, 0)
            edges.append((cost, i, j, gap, dist))

    edges.sort(key=lambda e: e[0])

    # 3) greedy matching with cycle prevention
    prev = {k: None for k in range(n)}
    nxt  = {k: None for k in range(n)}

    def creates_cycle(u, v):
        # would adding u->v create a cycle? check reachability from v forward to u
        cur = v
        seen = set()
        while cur is not None and cur not in seen:
            if cur == u:
                return True
            seen.add(cur)
            cur = nxt[cur]
        return False

    accepted = []
    for cost, i, j, gap, dist in edges:
        if nxt[i] is not None:     # i already points to someone
            continue
        if prev[j] is not None:    # j already has a predecessor
            continue
        if creates_cycle(i, j):
            continue
        nxt[i] = j
        prev[j] = i
        accepted.append((i, j, gap, dist, cost))

    # 4) assemble chains starting from heads (nodes with no prev)
    heads = [k for k in range(n) if prev[k] is None]
    chains = []
    for h in heads:
        # walk forward
        order = []
        cur = h
        while cur is not None:
            order.append(cur)
            cur = nxt[cur]

        # concatenate points; keep temporal order and then dedupe per-frame (keep later fragment’s point)
        pts = []
        for idx in order:
            pts.extend(frags[idx]["pts"])
        # de-dupe by frame (keep last occurrence)
        latest = {}
        for p in pts:
            latest[p["frame"]] = p["center"]
        merged_pts = [{"frame": f, "center": c} for f, c in sorted(latest.items())]
        chains.append(merged_pts)

    # 5) any fragments not in a chain (isolated cycles shouldn't happen, but just in case)
    used = set()
    for h in heads:
        cur = h
        while cur is not None:
            used.add(cur)
            cur = nxt[cur]
    for k in range(n):
        if k not in used and prev[k] is None and nxt[k] is None:
            # isolated single fragment
            tr = frags[k]["pts"]
            latest = {p["frame"]: p["center"] for p in tr}
            chains.append([{"frame": f, "center": c} for f, c in sorted(latest.items())])

    if debug:
        print(f"[merge_tracks] fragments={n}, candidates={len(edges)}, accepted={len(accepted)}, chains={len(chains)}")
        for (i, j, gap, dist, cost) in accepted[:24]:
            print(f"  link {i:03d} -> {j:03d} | gap={gap:+d} dist={dist:6.1f} cost={cost:6.2f}")
        if len(accepted) > 24:
            print(f"  ... {len(accepted)-24} more links")

    return chains


def pick_served_ball(merged_tracks):
    """Pick the merged trajectory with the highest average speed."""
    scores = {}
    for i, pts in enumerate(merged_tracks):
        if len(pts) < 2:
            continue
        centers = np.array([p["center"] for p in pts])
        diffs = np.linalg.norm(np.diff(centers, axis=0), axis=1)
        scores[i] = np.mean(diffs) * np.log(len(pts))
    if not scores:
        return None
    return max(scores, key=scores.get)


def build_sort_paths(player, session, serve):
    serve_str = format_serve_number(serve)
    detect_json = f"data/annotations/ball_detections/{player}/session_{session}/serve_{serve_str}.json"
    output_json = f"data/trajectories/{player}/session_{session}/serve_{serve_str}.json"
    return detect_json, output_json


def plot_tracks(merged_tracks, served_idx=None, title=None):
    plt.figure(figsize=(8, 6))
    for i, tr in enumerate(merged_tracks):
        pts = np.array([p["center"] for p in tr])
        if len(pts) == 0:
            continue
        color = "red" if i == served_idx else "gray"
        lw = 2.5 if i == served_idx else 1.0
        plt.plot(pts[:, 0], pts[:, 1], color=color, linewidth=lw, alpha=0.8)
        if i == served_idx:
            plt.scatter(pts[0, 0], pts[0, 1], color="green", label="start")
            plt.scatter(pts[-1, 0], pts[-1, 1], color="red", label="end")
    plt.gca().invert_yaxis()
    plt.title(title or "SORT Trajectories")
    plt.xlabel("x (px)")
    plt.ylabel("y (px)")
    plt.legend()
    plt.tight_layout()
    plt.show()


def run_sort_from_json(detect_json, output_json, conf_thresh=0.3, debug=True, visualize=True):
    if not os.path.exists(detect_json):
        raise FileNotFoundError(f"Missing detection JSON: {detect_json}")

    with open(detect_json) as f:
        detections = json.load(f)

    frames = sorted(map(int, detections.keys()))
    if not frames:
        # No detections found, save empty trajectory
        os.makedirs(os.path.dirname(output_json), exist_ok=True)
        with open(output_json, "w") as f:
            json.dump([], f)
        print(f"[warn] no detections found in {detect_json}, saved empty trajectory")
        return
    
    min_f, max_f = frames[0], frames[-1]
    for i in range(min_f, max_f + 1):
        if str(i) not in detections:
            detections[str(i)] = []

    tracker = Sort(max_age=30, min_hits=1, iou_threshold=0.3)
    trajectories = {}

    for frame_idx in sorted(map(int, detections.keys())):
        dets = [
            d for d in detections[str(frame_idx)]
            if d["conf"] >= conf_thresh
        ]

        dets_np = np.array([
            [
                d["center"][0] - d["size"][0] / 2,
                d["center"][1] - d["size"][1] / 2,
                d["center"][0] + d["size"][0] / 2,
                d["center"][1] + d["size"][1] / 2,
                d["conf"]
            ]
            for d in dets
        ]) if dets else np.empty((0, 5))

        tracks = tracker.update(dets_np)
        for x1, y1, x2, y2, tid in tracks:
            cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
            trajectories.setdefault(int(tid), []).append({
                "frame": frame_idx,
                "center": [float(cx), float(cy)]
            })

    merged = merge_tracks(trajectories, frame_gap_thresh=15, dist_thresh=250, debug=debug)
    served_idx = pick_served_ball(merged)
    os.makedirs(os.path.dirname(output_json), exist_ok=True)

    if served_idx is not None:
        served_track = merged[served_idx]
        
        # Collapse duplicates (keep last point per frame)
        frame_map = {}
        for p in served_track:
            frame_map[p["frame"]] = p["center"]
        
        # Rebuild sorted unique trajectory
        served_track = [
            {"frame": f, "center": c}
            for f, c in sorted(frame_map.items())
        ]
        
        with open(output_json, "w") as f:
            json.dump(served_track, f, indent=2)
        
        print(f"[save] served trajectory ({len(served_track)} pts) → {output_json}")
    else:
        with open(output_json, "w") as f:
            json.dump([], f)
        print("[warn] no active ball identified, saved empty trajectory")

    if visualize:
        plot_tracks(merged, served_idx, title=os.path.basename(output_json))


def main():
    ap = argparse.ArgumentParser(description="Run SORT tracking from YOLO detection JSON (overlap-aware, saves all tracks)")
    add_player_session_serve_args(ap)
    ap.add_argument("--conf", type=float, default=0.3)
    ap.add_argument("--no-viz", action="store_true", help="Disable matplotlib visualization")
    args = ap.parse_args()

    if not (args.player and args.session is not None and args.serve):
        ap.error("Must specify --player, --session, and --serve")

    detect_json, output_json = build_sort_paths(args.player, args.session, args.serve)
    run_sort_from_json(detect_json, output_json, conf_thresh=args.conf, visualize=not args.no_viz)


if __name__ == "__main__":
    main()
