"""
Run YOLO detection on serve videos and output detections as JSON.

Runs YOLO ball detection on serve videos and saves bounding box detections
in JSON format for further processing by tracking algorithms.
"""
import subprocess, os, json, argparse, glob, shutil
from common_args import add_player_session_serve_args, format_serve_number, validate_video_exists


def build_detection_paths(player, session, serve):
    serve_str = format_serve_number(serve)
    video_path = f"data/videos/processed/{player}/session_{session}/serve_{serve_str}.mp4"
    json_path = f"data/annotations/ball_detections/{player}/session_{session}/serve_{serve_str}.json"
    return video_path, json_path


def run_yolo_cli(video_path, model_path, output_json):
    """Run YOLO CLI, parse labels → JSON, then clean up."""
    img_width, img_height = 1920, 1080
    project = os.path.join("runs", "detect_temp")
    name = os.path.splitext(os.path.basename(video_path))[0]

    cmd = [
        "yolo", "detect", "predict",
        f"model={model_path}",
        f"source={video_path}",
        "imgsz=1920",
        "save_txt=True",
        "save_conf=True",
        "save=False",
        f"project={project}",
        f"name={name}",
        "exist_ok=True",
        "verbose=False",
        "conf=0.3",
        "iou=0.65",
        "nms=True"
    ]
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    label_dir = os.path.join(project, name, "labels")
    frames = {}
    for file in sorted(glob.glob(os.path.join(label_dir, "*.txt"))):
        base = os.path.splitext(os.path.basename(file))[0]
        frame_idx = int(base.split("_")[-1]) if "_" in base else int(base[-6:])
        dets = []
        with open(file) as f:
            for line in f:
                vals = line.strip().split()
                if len(vals) < 6:
                    continue
                cls, x_norm, y_norm, w_norm, h_norm, conf = vals

                # convert normalized YOLO → pixel coordinates
                cx = float(x_norm) * img_width
                cy = float(y_norm) * img_height
                w = float(w_norm) * img_width
                h = float(h_norm) * img_height

                dets.append({
                    "class": int(float(cls)),
                    "center": [cx, cy],
                    "size": [w, h],
                    "conf": float(conf)
                })
        frames[frame_idx] = dets

    os.makedirs(os.path.dirname(output_json), exist_ok=True)
    with open(output_json, "w") as f:
        json.dump(frames, f, indent=2)
    print(f"[save] {len(frames)} frames → {output_json}")

    # clean temp directory
    if os.path.exists(project):
        shutil.rmtree(project, ignore_errors=True)


def main():
    ap = argparse.ArgumentParser(description="Run YOLO detection and save per-frame boxes to JSON")
    add_player_session_serve_args(ap)
    args = ap.parse_args()

    if not (args.player and args.session is not None and args.serve):
        ap.error("Must specify --player, --session, and --serve")

    video_path, output_json = build_detection_paths(args.player, args.session, args.serve)
    validate_video_exists(video_path)
    run_yolo_cli(video_path, "models/best.pt", output_json)


if __name__ == "__main__":
    main()
