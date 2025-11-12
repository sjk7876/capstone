import os
import glob
import json
from tqdm import tqdm
from SORT_from_json import run_sort_from_json
from detection_to_json import run_yolo_cli
from visualize_trajectory import save_trajectory_video

# adjust as needed
YOLO_MODEL = "models/best.pt"
RAW_VIDEOS = "data/videos/processed"
DETECT_DIR = "data/annotations/ball_detections"
TRAJ_DIR = "data/trajectories"
VIS_DIR = "data/visualized"

def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def process_serve(video_path, player, session, serve_str):
    """run YOLO → SORT → visualize for one serve"""
    detect_json = os.path.join(DETECT_DIR, player, f"session_{session}", f"serve_{serve_str}.json")
    traj_json = os.path.join(TRAJ_DIR, player, f"session_{session}", f"serve_{serve_str}.json")
    vis_outdir = os.path.join(VIS_DIR, player, f"session_{session}")
    ensure_dir(os.path.dirname(detect_json))
    ensure_dir(os.path.dirname(traj_json))
    ensure_dir(vis_outdir)

    # 1️⃣ YOLO detect
    if not os.path.exists(detect_json):
        run_yolo_cli(video_path, YOLO_MODEL, detect_json)

    # 2️⃣ SORT + merge
    run_sort_from_json(detect_json, traj_json, conf_thresh=0.3, debug=False, visualize=False)

    # 3️⃣ visualize
    vis_out = os.path.join(vis_outdir, f"serve_{serve_str}.mp4")
    save_trajectory_video(traj_json, video_path, vis_out, color=(0, 255, 0), line_thickness=2)
    print(f"[done] {player}/session_{session}/serve_{serve_str}")


def main(player=None, session=None):
    """
    Run full pipeline for all serves in dataset (or filtered by player/session)
    """
    players = [player] if player else sorted(os.listdir(RAW_VIDEOS))
    for p in players:
        player_dir = os.path.join(RAW_VIDEOS, p)
        sessions = [session] if session else sorted(os.listdir(player_dir))

        for s in sessions:
            session_dir = os.path.join(player_dir, s)
            serve_videos = sorted(glob.glob(os.path.join(session_dir, "serve_*.mp4")))

            for vp in tqdm(serve_videos, desc=f"{p}/session_{s}"):
                serve_str = os.path.basename(vp).split("_")[-1].split(".")[0]
                process_serve(vp, p, s, serve_str)


if __name__ == "__main__":
    main()
