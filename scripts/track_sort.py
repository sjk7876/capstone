"""
Track ball using SORT algorithm with real-time visualization. (Deprecated)

Runs YOLO detection and SORT tracking in real-time, displaying results
for debugging and validation of the tracking pipeline.
"""
import sys
import cv2
import json
import numpy as np
import argparse
import os
import matplotlib.pyplot as plt
from ultralytics import YOLO

sys.path.append("external")
sys.path.append("external/filterpy")
from sort.sort import Sort
from common_args import add_player_session_serve_args, resolve_trajectory_paths, validate_video_exists


def merge_tracks(trajectories, frame_gap_thresh=8, dist_thresh=120, debug=False):
    """Merge multiple SORT tracks if they appear to be the same object, with full gap logs."""
    # sort frames inside each trajectory
    for tid, tr in trajectories.items():
        trajectories[tid] = sorted(tr, key=lambda p: p["frame"])
    # sort all trajectories by first frame
    tracks = sorted(trajectories.values(), key=lambda tr: tr[0]["frame"] if tr else 0)

    merged = []
    merge_info = []

    for i, track in enumerate(tracks):
        if not merged:
            merged.append(track)
            continue

        prev = merged[-1]
        last_frame = prev[-1]["frame"]
        last_pos = np.array(prev[-1]["center"])
        first_frame = track[0]["frame"]
        first_pos = np.array(track[0]["center"])

        frame_gap = first_frame - last_frame
        dist = np.linalg.norm(first_pos - last_pos)

        if frame_gap <= frame_gap_thresh and dist <= dist_thresh:
            merged[-1].extend(track)
            merge_info.append((i - 1, i, frame_gap, dist))
        else:
            merged.append(track)

    if debug:
        print(f"[merge_tracks] started with {len(tracks)} fragments → merged into {len(merged)}")
        print(f"  (gap ≤ {frame_gap_thresh}, dist ≤ {dist_thresh}) = merged\n")
        for i in range(1, len(tracks)):
            prev = tracks[i - 1]
            track = tracks[i]
            last_frame = prev[-1]["frame"] if i > 1 else 0
            first_frame = track[0]["frame"]
            frame_gap = first_frame - last_frame
            last_pos = np.array(prev[-1]["center"])
            first_pos = np.array(track[0]["center"])
            dist = np.linalg.norm(first_pos - last_pos)
            merged_flag = (frame_gap <= frame_gap_thresh and dist <= dist_thresh)
            status = "MERGED" if merged_flag else "NO MERGE"
            print(f"  {i-1:02d} → {i:02d}: gap={frame_gap:3d}, dist={dist:7.1f}  [{status}]")

    return merged


def pick_served_ball(merged_tracks):
    """Pick the merged trajectory with the highest average speed."""
    scores = {}
    for i, pts in enumerate(merged_tracks):
        if len(pts) < 2:
            continue
        centers = np.array([p["center"] for p in pts])
        diffs = np.linalg.norm(np.diff(centers, axis=0), axis=1)
        scores[i] = np.mean(diffs) * np.log(len(pts))  # prefer long + fast
    if not scores:
        return None
    return max(scores, key=scores.get)


def plot_tracks(merged_tracks, served_idx=None, title=None):
    """Visualize merged SORT tracks using matplotlib."""
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


def track_video(video_path, model_path, output_json=None, conf_thresh=0.3, debug=True, visualize=True):
    model = YOLO(model_path)
    tracker = Sort(max_age=30, min_hits=1, iou_threshold=0.3)

    cap = cv2.VideoCapture(video_path)
    trajectories = {}
    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        results = model(frame, verbose=False)[0]
        detections = [
            [x1, y1, x2, y2, conf]
            for x1, y1, x2, y2, conf, cls in results.boxes.data.cpu().numpy()
            if conf >= conf_thresh
        ]

        detections = np.array(detections) if len(detections) else np.empty((0, 5))
        tracks = tracker.update(detections)

        for x1, y1, x2, y2, tid in tracks:
            cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
            trajectories.setdefault(int(tid), []).append({
                "frame": frame_idx,
                "center": [float(cx), float(cy)]
            })

        # draw all tracked boxes
        for x1, y1, x2, y2, tid in tracks:
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
            cv2.putText(frame, f"ID {int(tid)}", (int(x1), int(y1) - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        cv2.imshow("SORT tracking", frame)
        if cv2.waitKey(1) == 27:  # ESC
            break

        frame_idx += 1

    cap.release()
    cv2.destroyAllWindows()

    merged_tracks = merge_tracks(trajectories, frame_gap_thresh=15, dist_thresh=250, debug=debug)
    served_idx = pick_served_ball(merged_tracks)

    if output_json:
        os.makedirs(os.path.dirname(output_json), exist_ok=True)
        if served_idx is not None:
            served_track = merged_tracks[served_idx]
            with open(output_json, "w") as f:
                json.dump(served_track, f, indent=2)
            print(f"[save] served trajectory #{served_idx} — {len(served_track)} frames saved")
        else:
            with open(output_json, "w") as f:
                json.dump([], f)
            print("[save] no active ball found")

    if visualize:
        plot_tracks(merged_tracks, served_idx, title=os.path.basename(video_path))

    return trajectories


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Track ball using YOLO and SORT with visualization + gap logging")
    
    # Common player/session/serve arguments
    add_player_session_serve_args(parser)
    
    # Option 2: Direct paths (if not using player/session/serve)
    parser.add_argument("--video-path", type=str, default=None, help="Direct path to video file")
    parser.add_argument("--output-json", type=str, default=None, help="Direct path to output JSON file")
    
    parser.add_argument("--model", type=str, default="models/best.pt", help="Path to YOLO model")
    parser.add_argument("--conf", type=float, default=0.3, help="Confidence threshold")
    parser.add_argument("--no-viz", action="store_true", help="Disable matplotlib visualization")

    args = parser.parse_args()

    # Resolve paths
    video_path, output_json = resolve_trajectory_paths(args, require_output=False)
    validate_video_exists(video_path)

    track_video(
        video_path=video_path,
        model_path=args.model,
        output_json=output_json,
        conf_thresh=args.conf,
        debug=True,
        visualize=not args.no_viz
    )
