import os
import glob
import json
import argparse
from tqdm import tqdm
from SORT_from_json import run_sort_from_json
from detection_to_json import run_yolo_cli
from visualize_trajectory import save_trajectory_video
from common_args import add_player_session_serve_args, format_serve_number

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


def main(player=None, session=None, serve=None):
    """
    Run full pipeline for all serves in dataset (or filtered by player/session/serve)
    """
    players = [player] if player else sorted(os.listdir(RAW_VIDEOS))
    for p in players:
        player_dir = os.path.join(RAW_VIDEOS, p)
        if session is not None:
            sessions = [f"session_{session}"]
        else:
            # Get all session directories
            sessions = [d for d in sorted(os.listdir(player_dir)) if d.startswith("session_")]

        for s in sessions:
            session_dir = os.path.join(player_dir, s)
            serve_videos = sorted(glob.glob(os.path.join(session_dir, "serve_*.mp4")))

            if serve is not None:
                # Filter to specific serve
                serve_str_formatted = format_serve_number(str(serve))
                serve_videos = [vp for vp in serve_videos 
                               if format_serve_number(os.path.basename(vp).split("_")[-1].split(".")[0]) == serve_str_formatted]
                if not serve_videos:
                    print(f"[warning] serve_{serve_str_formatted} not found for {p}/session_{s}")
                    continue

            for vp in tqdm(serve_videos, desc=f"{p}/{s}"):
                serve_str = os.path.basename(vp).split("_")[-1].split(".")[0]
                # Extract session number from directory name (e.g., "session_4" -> "4")
                session_num = s.replace("session_", "") if s.startswith("session_") else s
                process_serve(vp, p, session_num, serve_str)


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Run YOLO → SORT → visualize pipeline")
    
    # Common player/session/serve arguments
    add_player_session_serve_args(parser)
    
    parser.add_argument("--all", action="store_true", 
                       help="Process all serves (alternative to specifying player/session/serve)")
    
    args = parser.parse_args()
    
    # Require either all three (player, session, serve) OR --all flag
    has_all_three = args.player and args.session is not None and args.serve
    if not (has_all_three or args.all):
        parser.error("Must specify either (--player, --session, --serve) OR --all")
    
    return args.player, args.session, args.serve, args.all


if __name__ == "__main__":
    player, session, serve, process_all = parse_args()
    if process_all:
        main(player=None, session=None, serve=None)
    else:
        main(player=player, session=session, serve=serve)
