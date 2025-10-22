import os
import shutil
import subprocess
import zipfile
import argparse
from datetime import datetime

# ===============================
# CONFIGURATION
# ===============================

# YOLO model + dataset paths
YOLO_MODEL = "models/best.pt"
SERVE_DIR = "data/frames"
YOLO_RUNS_DIR = "runs/detect"
OUTPUT_DIR = "cvat_upload"

# CVAT settings
CVAT_ENABLED = True
CVAT_URL = "http://192.168.1.33:8080"
CVAT_USERNAME = "admin"
CVAT_PASSWORD = "admin"
CVAT_PROJECT_ID = 1        # optional, or set to None
CVAT_LABELS = '[{"name": "Ball"}]'  # only needed if not inside a project
CVAT_TASK_NAME = None       # auto-generated if None

# ===============================


def ensure_dirs():
    """Create needed directories for YOLO outputs and merged dataset."""
    for d in [
        YOLO_RUNS_DIR,
        OUTPUT_DIR,
        os.path.join(OUTPUT_DIR, "images"),
        os.path.join(OUTPUT_DIR, "labels"),
    ]:
        os.makedirs(d, exist_ok=True)


def run_yolo_on_serves(player=None, session=None, serve=None):
    """Run YOLO prediction on serve folders, optionally filtered by player/session/serve."""
    all_serve_folders = sorted(
        [
            os.path.join(SERVE_DIR, d)
            for d in os.listdir(SERVE_DIR)
            if os.path.isdir(os.path.join(SERVE_DIR, d)) and "serve" in d
        ]
    )

    if not all_serve_folders:
        print("No serve folders found in data/frames/")
        return []

    # Filter serve folders based on arguments
    serve_folders = []
    for folder in all_serve_folders:
        folder_name = os.path.basename(folder)
        
        # Parse folder name: player_session_X_serve_XXX
        parts = folder_name.split('_')
        if len(parts) >= 4 and parts[1] == 'session' and parts[3] == 'serve':
            folder_player = parts[0]
            folder_session = int(parts[2])
            folder_serve = int(parts[4])
            
            # Apply filters
            if player and folder_player != player:
                continue
            if session is not None and folder_session != session:
                continue
            if serve is not None:
                serve_num = int(serve) if serve.isdigit() else int(serve)
                if folder_serve != serve_num:
                    continue
            
            serve_folders.append(folder)

    if not serve_folders:
        print("No serve folders match the specified criteria")
        return []

    # Show what we're processing
    filter_desc = []
    if player:
        filter_desc.append(f"player={player}")
    if session is not None:
        filter_desc.append(f"session={session}")
    if serve is not None:
        filter_desc.append(f"serve={serve}")
    
    filter_str = f" (filtered: {', '.join(filter_desc)})" if filter_desc else ""
    print(f"Running YOLO on {len(serve_folders)} serve folders{filter_str}...")
    
    for d in serve_folders:
        name = os.path.basename(d)
        print(f"  → {name}")
        subprocess.run([
            "yolo", "detect", "predict",
            f"model={YOLO_MODEL}",
            f"source={d}",
            f"project={YOLO_RUNS_DIR}",
            f"name={name}",
            'save=False',
            "imgsz=1920",
            "save_txt=True",
            "exist_ok=True",
            "verbose=false",
            "patience=10",
            "conf=0.4",
            "iou=0.5",
            "nms=True"
        ])
    return serve_folders


def generate_task_name(serve_folders):
    """Auto-generate a CVAT task name based on serve folders."""
    if not serve_folders:
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M")
        return f"serve_batch_empty_{timestamp}"

    names = [os.path.basename(f) for f in serve_folders]
    prefix = os.path.commonprefix(names).strip("_-")
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M")

    if len(names) == 1:
        return f"{names[0]}_{timestamp}"
    elif prefix:
        return f"serve_batch_{prefix}_{len(names)}folders_{timestamp}"
    else:
        return f"serve_batch_{len(names)}folders_{timestamp}"


def collect_prediction_folders():
    """Find YOLO prediction folders that match serve names (not training runs)."""
    folders = []
    for sub in os.listdir(YOLO_RUNS_DIR):
        path = os.path.join(YOLO_RUNS_DIR, sub)
        if not os.path.isdir(path):
            continue
        name = sub.lower()
        if ("serve" in name or "session" in name) and not name.startswith(("train", "val")):
            folders.append(path)
    return sorted(folders)


def merge_predictions(folders):
    """Merge YOLO outputs (labels) with original clean images from SERVE_DIR."""
    images_dir = os.path.join(OUTPUT_DIR, "images")
    labels_dir = os.path.join(OUTPUT_DIR, "labels")
    total_files = 0

    for folder in folders:
        serve_name = os.path.basename(folder)
        serve_src_dir = os.path.join(SERVE_DIR, serve_name)
        pred_labels_dir = os.path.join(folder, "labels")

        if not os.path.exists(serve_src_dir):
            print(f"Warning: No matching original serve folder found for {serve_name}")
            continue

        serve_images = sorted([
            f for f in os.listdir(serve_src_dir)
            if f.lower().endswith((".jpg", ".png"))
        ])

        for img_name in serve_images:
            # Extract frame number from original filename (e.g., frame000000.jpg -> 000000)
            frame_match = img_name.replace('frame', '').replace('.jpg', '')
            
            # Create new filename: serve_name_frame000000.jpg
            base = f"{serve_name}_{img_name}"
            src_img = os.path.join(serve_src_dir, img_name)
            src_txt = os.path.join(pred_labels_dir, os.path.splitext(img_name)[0] + ".txt")

            dst_img = os.path.join(images_dir, base)
            dst_txt = os.path.join(labels_dir, os.path.splitext(base)[0] + ".txt")

            shutil.copy(src_img, dst_img)
            if os.path.exists(src_txt):
                shutil.copy(src_txt, dst_txt)
            total_files += 1

    print(f"Merged {total_files} clean frames from {len(folders)} serve folders.")


def make_yolo_zip():
    """Create a CVAT-compatible YOLO 1.1 zip (labels only, with train.txt)."""
    yolo_dir = os.path.join(OUTPUT_DIR, "yolo_manual")
    obj_dir = os.path.join(yolo_dir, "obj_train_data")
    labels_dir = os.path.join(OUTPUT_DIR, "labels")

    # rebuild structure
    if os.path.exists(yolo_dir):
        shutil.rmtree(yolo_dir)
    os.makedirs(obj_dir, exist_ok=True)

    label_files = [
        f for f in os.listdir(labels_dir)
        if f.endswith(".txt")
    ]

    for f in label_files:
        shutil.copy(os.path.join(labels_dir, f), os.path.join(obj_dir, f))

    # write train.txt listing all label paths
    train_txt_path = os.path.join(yolo_dir, "train.txt")
    with open(train_txt_path, "w", encoding="utf-8") as f:
        for f_name in label_files:
            f.write(f"obj_train_data/{f_name}\n")

    # write obj.data
    with open(os.path.join(yolo_dir, "obj.data"), "w", encoding="utf-8") as f:
        f.write("classes=1\n")
        f.write("train=train.txt\n")
        f.write("names=obj.names\n")
        f.write("backup=backup/\n")

    # write obj.names
    with open(os.path.join(yolo_dir, "obj.names"), "w", encoding="utf-8") as f:
        f.write("Ball\n")

    # zip up the dataset (correct root layout)
    yolo_zip = os.path.abspath("serve_yolo_manual.zip").replace("\\", "/")
    cwd = os.getcwd()
    try:
        os.chdir(yolo_dir)
        with zipfile.ZipFile(yolo_zip, "w", zipfile.ZIP_DEFLATED) as zipf:
            for root, _, files in os.walk("."):
                for file in files:
                    full = os.path.join(root, file)
                    zipf.write(full, full)
    finally:
        os.chdir(cwd)

    print(f"Created YOLO 1.1 annotation package (labels only): {yolo_zip}")
    print(f" • {len(label_files)} label files listed in train.txt")
    print(" • Ready for manual upload in CVAT → Upload Annotations → YOLO 1.1\n")


def cleanup_temp_files(keep_images_zip=False):
    """Clean up temporary files and directories after processing."""
    import shutil
    
    # Remove the cvat_upload directory
    if os.path.exists(OUTPUT_DIR):
        print(f"Cleaning up temporary directory: {OUTPUT_DIR}")
        shutil.rmtree(OUTPUT_DIR)
        print("Temporary files cleaned up.")
    
    # Remove the images zip file (unless keeping it)
    images_zip = os.path.abspath("serve_images.zip")
    if os.path.exists(images_zip):
        if keep_images_zip:
            print(f"Keeping images zip: {images_zip}")
        else:
            os.remove(images_zip)
            print(f"Removed temporary zip: {images_zip}")
    
    # Remove serve folders from runs directory
    if os.path.exists(YOLO_RUNS_DIR):
        serve_folders = []
        for item in os.listdir(YOLO_RUNS_DIR):
            item_path = os.path.join(YOLO_RUNS_DIR, item)
            if os.path.isdir(item_path):
                # Check if it's a serve folder (contains "serve" or "session" in name)
                name_lower = item.lower()
                if ("serve" in name_lower or "session" in name_lower) and not name_lower.startswith(("train", "val")):
                    serve_folders.append(item_path)
        
        for folder in serve_folders:
            print(f"Removing serve folder: {folder}")
            shutil.rmtree(folder)
        
        if serve_folders:
            print(f"Cleaned up {len(serve_folders)} serve folders from runs directory")



def upload_to_cvat(task_name):
    """Upload images to CVAT, then make YOLO zip for manual annotation upload."""
    if not CVAT_ENABLED:
        print("CVAT upload disabled. Skipping.")
        return

    cli_path = shutil.which("cvat-cli") or shutil.which("cvat") or shutil.which("cvat.exe")
    if not cli_path:
        print("cvat-cli not found in PATH.")
        return

    images_dir = os.path.abspath(os.path.join(OUTPUT_DIR, "images")).replace("\\", "/")

    print(f"Uploading to CVAT task: {task_name}")

    images_zip = os.path.abspath("serve_images.zip").replace("\\", "/")
    with zipfile.ZipFile(images_zip, "w", zipfile.ZIP_DEFLATED) as zipf:
        for root, _, files in os.walk(images_dir):
            for f in files:
                full = os.path.join(root, f)
                rel = os.path.relpath(full, images_dir)
                zipf.write(full, rel)
    print(f"Zipped {images_dir} -> {images_zip}")

    # Legacy CLI syntax (2.47)
    create_cmd = [
        cli_path,
        "--server-host", CVAT_URL,
        "--auth", f"{CVAT_USERNAME}:{CVAT_PASSWORD}",
        "task", "create",
        task_name,
        "local",
        images_zip,
    ]

    if CVAT_PROJECT_ID:
        create_cmd += ["--project_id", str(CVAT_PROJECT_ID)]
    else:
        create_cmd += ["--labels", CVAT_LABELS]

    print("Running:", " ".join(create_cmd))

    try:
        subprocess.run(create_cmd, check=True)
        print("Uploaded images to CVAT.")
    except subprocess.CalledProcessError as e:
        print("CVAT image upload failed:")
        print(e)
        return

    # after uploading, make the manual YOLO zip
    make_yolo_zip()
    
    # Clean up temporary files
    cleanup_temp_files()


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Auto-label serves with YOLO and upload to CVAT")
    parser.add_argument('-n', '--no-cvat', action='store_true', 
                       help='Disable CVAT upload (only create YOLO zip)')
    parser.add_argument('-p', '--player', type=str, default=None,
                       help='Process only serves from specific player (e.g., "spencer")')
    parser.add_argument('-s', '--session', type=int, default=None,
                       help='Process only serves from specific session (e.g., 1)')
    parser.add_argument('--serve', type=str, default=None,
                       help='Process only specific serve (e.g., "001" or "1")')
    return parser.parse_args()


def main():
    args = parse_args()
    
    # Override CVAT_ENABLED if -n flag is used
    global CVAT_ENABLED
    if args.no_cvat:
        CVAT_ENABLED = False
        print("CVAT upload disabled via -n flag")
    
    ensure_dirs()
    serve_folders = run_yolo_on_serves(args.player, args.session, args.serve)
    folders = collect_prediction_folders()
    if not folders:
        print("No YOLO prediction folders found.")
        return
    merge_predictions(folders)
    task_name = CVAT_TASK_NAME or generate_task_name(serve_folders)
    upload_to_cvat(task_name)
    
    # Clean up even if CVAT upload was disabled
    if not CVAT_ENABLED:
        make_yolo_zip()
        cleanup_temp_files(keep_images_zip=True)


if __name__ == "__main__":
    main()
