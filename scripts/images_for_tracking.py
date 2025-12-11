"""
Generate visualization images for ball tracking analysis.

Creates annotated images showing ball detections and tracking results
for analysis and debugging of the tracking pipeline.
"""
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
                 overlap_back=8,
                 alpha_gap=1.0,
                 beta_dist=0.02,
                 debug=False):
    """Greedy O(n^2) multi-chain merging of SORT fragments."""
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
    edges = []
    for i in range(n):
        fi = frags[i]
        for j in range(n):
            if i == j:
                continue
            fj = frags[j]
            gap = fj["start_f"] - fi["end_f"]
            if gap > frame_gap_thresh or gap < -overlap_back:
                continue
            dist = float(np.linalg.norm(fj["head"] - fi["tail"]))
            if dist > dist_thresh:
                continue
            cost = alpha_gap * max(gap, 0) + beta_dist * dist + 0.5 * max(-gap, 0)
            edges.append((cost, i, j, gap, dist))

    edges.sort(key=lambda e: e[0])
    prev = {k: None for k in range(n)}
    nxt = {k: None for k in range(n)}

    def creates_cycle(u, v):
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
        if nxt[i] is not None:
            continue
        if prev[j] is not None:
            continue
        if creates_cycle(i, j):
            continue
        nxt[i] = j
        prev[j] = i
        accepted.append((i, j, gap, dist, cost))

    heads = [k for k in range(n) if prev[k] is None]
    chains = []
    for h in heads:
        order = []
        cur = h
        while cur is not None:
            order.append(cur)
            cur = nxt[cur]
        pts = []
        for idx in order:
            pts.extend(frags[idx]["pts"])
        latest = {}
        for p in pts:
            latest[p["frame"]] = {
                "center": p["center"],
                "size": p.get("size", [0.0, 0.0])
            }
        merged_pts = [
            {"frame": f, "center": data["center"], "size": data["size"]}
            for f, data in sorted(latest.items())
        ]
        chains.append(merged_pts)

    used = set()
    for h in heads:
        cur = h
        while cur is not None:
            used.add(cur)
            cur = nxt[cur]
    for k in range(n):
        if k not in used and prev[k] is None and nxt[k] is None:
            tr = frags[k]["pts"]
            latest = {}
            for p in tr:
                latest[p["frame"]] = {
                    "center": p["center"],
                    "size": p.get("size", [0.0, 0.0])
                }
            chains.append([
                {"frame": f, "center": data["center"], "size": data["size"]}
                for f, data in sorted(latest.items())
            ])

    if debug:
        print(f"[merge_tracks] fragments={n}, candidates={len(edges)}, accepted={len(accepted)}, chains={len(chains)}")
        for (i, j, gap, dist, cost) in accepted[:24]:
            print(f"  link {i:03d} -> {j:03d} | gap={gap:+d} dist={dist:6.1f} cost={cost:6.2f}")
        if len(accepted) > 24:
            print(f"  ... {len(accepted)-24} more links")

    return chains


def plot_raw_tracks(trajectories, save_path=None, title="Raw SORT Fragments"):
    plt.figure(figsize=(8, 6))
    for tid, tr in trajectories.items():
        pts = np.array([p["center"] for p in tr])
        if len(pts) == 0:
            continue
        plt.plot(pts[:, 0], pts[:, 1], linewidth=1.2, alpha=0.8)
        plt.scatter(pts[0, 0], pts[0, 1], color="green", s=15)
        plt.scatter(pts[-1, 0], pts[-1, 1], color="red", s=15)
        plt.text(pts[0, 0], pts[0, 1], str(tid), fontsize=8, color="blue")
    plt.gca().invert_yaxis()
    plt.title(title)
    plt.xlabel("x (px)")
    plt.ylabel("y (px)")
    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved visualization → {save_path}")
    else:
        plt.show()
    plt.close()


def pick_served_ball(merged_tracks):
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
    detect_json = os.path.join("data", "annotations", "ball_detections", player, f"session_{session}", f"serve_{serve_str}.json")
    output_json = os.path.join("data", "trajectories", player, f"session_{session}", f"serve_{serve_str}.json")
    return detect_json, output_json


def plot_tracks(merged_tracks, served_idx=None, save_path=None, title=None):
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
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved visualization → {save_path}")
    else:
        plt.show()
    plt.close()


def run_sort_from_json(detect_json, output_json, conf_thresh=0.3, debug=True, visualize=True, player=None, session=None, serve=None):
    if not os.path.exists(detect_json):
        raise FileNotFoundError(f"Missing detection JSON: {detect_json}")

    with open(detect_json) as f:
        detections = json.load(f)

    frames = sorted(map(int, detections.keys()))
    if not frames:
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
            width = x2 - x1
            height = y2 - y1
            trajectories.setdefault(int(tid), []).append({
                "frame": frame_idx,
                "center": [float(cx), float(cy)],
                "size": [float(width), float(height)]
            })

    # Generate visualization paths
    viz_paths = {}
    if visualize and player and session is not None and serve:
        serve_str = format_serve_number(serve)
        viz_dir = os.path.join("data", "visualizations", "sort", player, f"session_{session}")
        viz_paths = {
            "before": os.path.join(viz_dir, f"serve_{serve_str}_before_merge.png"),
            "after": os.path.join(viz_dir, f"serve_{serve_str}_after_merge.png"),
            "final": os.path.join(viz_dir, f"serve_{serve_str}_final.png")
        }

    # Visualize all fragments before merge
    if visualize:
        plot_raw_tracks(trajectories, save_path=viz_paths.get("before"), title="Before Merge: Raw SORT Tracks")

    merged = merge_tracks(trajectories, frame_gap_thresh=15, dist_thresh=250, debug=debug)

    # Visualize merged chains
    if visualize:
        plot_tracks(merged, save_path=viz_paths.get("after"), title="After Merge: Merged Tracks")

    served_idx = pick_served_ball(merged)
    os.makedirs(os.path.dirname(output_json), exist_ok=True)

    if served_idx is not None:
        served_track = merged[served_idx]
        frame_map = {}
        for p in served_track:
            frame_map[p["frame"]] = {
                "center": p["center"],
                "size": p.get("size", [0.0, 0.0])
            }
        served_track = [
            {"frame": f, "center": data["center"], "size": data["size"]}
            for f, data in sorted(frame_map.items())
        ]
        with open(output_json, "w") as f:
            json.dump(served_track, f, indent=2)
        print(f"[save] served trajectory ({len(served_track)} pts) → {output_json}")
    else:
        with open(output_json, "w") as f:
            json.dump([], f)
        print("[warn] no active ball identified, saved empty trajectory")

    # Visualize final selected serve
    if visualize:
        plot_tracks(merged, served_idx, save_path=viz_paths.get("final"), title="Final: Served Trajectory")


def main():
    ap = argparse.ArgumentParser(description="Run SORT tracking from YOLO detection JSON (with pre/post merge visualization)")
    add_player_session_serve_args(ap)
    ap.add_argument("--conf", type=float, default=0.3)
    ap.add_argument("--no-viz", action="store_true", help="Disable matplotlib visualization")
    args = ap.parse_args()

    if not (args.player and args.session is not None and args.serve):
        ap.error("Must specify --player, --session, and --serve")

    detect_json, output_json = build_sort_paths(args.player, args.session, args.serve)
    run_sort_from_json(detect_json, output_json, conf_thresh=args.conf, visualize=not args.no_viz,
                      player=args.player, session=args.session, serve=args.serve)


if __name__ == "__main__":
    main()
