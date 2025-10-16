#!/usr/bin/env python3
import cv2
import csv
import os

SERVES_CSV = "data/metadata/serves.csv"

def update_csv(clip_path, landing_frame):
    rows = []
    updated = False
    with open(SERVES_CSV, "r", newline="") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames or []
        if "landing_frame" not in fieldnames:
            fieldnames = fieldnames + ["landing_frame"]
        for row in reader:
            if row.get("output_clip") == clip_path:
                row["landing_frame"] = str(landing_frame)
                updated = True
            rows.append(row)
    if updated:
        with open(SERVES_CSV, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        print(f"Saved landing_frame={landing_frame} for {clip_path}")
    else:
        print(f"No row found in serves.csv for {clip_path}")

def label_clip(clip_path):
    cap = cv2.VideoCapture(clip_path)
    if not cap.isOpened():
        print(f"Could not open {clip_path}")
        return None

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    current = 0

    print(f"\nOpened {clip_path} ({total_frames} frames, {fps:.2f} fps)")

    auto_forward = False
    auto_backward = False
    labeled = False
    quit_all = False

    while True:
        cap.set(cv2.CAP_PROP_POS_FRAMES, current)
        ret, frame = cap.read()
        if not ret:
            break

        display = frame.copy()
        cv2.putText(display, f"Frame {current}/{total_frames-1}", (10,30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
        cv2.imshow("Landing Labeler", display)

        key = cv2.waitKey(30) & 0xFF

        if key == ord("f"):
            current = min(current+1, total_frames-1)
        elif key == ord("d"):
            current = max(current-1, 0)
        elif key == ord("r"):
            current = min(current+10, total_frames-1)
        elif key == ord("e"):
            current = max(current-10, 0)
        elif key == ord("c"):
            auto_forward = True
        elif key == ord("x"):
            auto_backward = True
        elif key == 255:  # no key pressed
            if auto_forward:
                current = min(current+3, total_frames-1)
            if auto_backward:
                current = max(current-3, 0)
        else:
            auto_forward = False
            auto_backward = False

        if key == ord("l"):
            update_csv(clip_path, current)
            labeled = True
            break
        elif key == ord("q"):
            quit_all = True
            break

    cap.release()
    cv2.destroyAllWindows()
    return labeled, quit_all

def main():
    with open(SERVES_CSV, newline="") as f:
        reader = csv.DictReader(f)
        clips = [(row["output_clip"], row.get("landing_frame", "") or "") for row in reader]

    # stats before labeling
    total = len(clips)
    already_labeled = sum(1 for _, landing in clips if landing and landing.strip() != "")
    remaining = total - already_labeled

    # find first unlabeled clip
    start_index = 0
    for i, (_, landing) in enumerate(clips):
        if not landing or landing.strip() == "":
            start_index = i
            break

    print("Controls:")
    print("  [f] = next frame, [d] = prev frame")
    print("  [r] = skip +10 frames, [e] = skip -10 frames")
    print("  [c] = hold to auto-forward, [x] = hold to auto-backward")
    print("  [l] = label landing, [q] = quit session")
    print(f"\n▶Starting from serve {start_index+1} of {total} (first unlabeled)")
    print(f"  ℹ{already_labeled} already labeled, {remaining} remaining\n")

    labeled_this_run = 0

    for clip, landing in clips[start_index:]:
        if not os.path.exists(clip):
            print(f"Skipping missing {clip}")
            continue
        if landing and landing.strip() != "":
            print(f"Skipping {clip} (already labeled)")
            continue
        result, quit_all = label_clip(clip)
        if quit_all:
            break
        if result:
            labeled_this_run += 1

    # reload csv for final stats
    with open(SERVES_CSV, newline="") as f:
        reader = csv.DictReader(f)
        final_labeled = sum(1 for row in reader if row.get("landing_frame", "").strip() != "")

    print("\nLabeling session summary:")
    print(f"  Labeled this run: {labeled_this_run}")
    print(f"  Total labeled:    {final_labeled}/{total}")
    print(f"  Remaining:        {total - final_labeled}")

if __name__ == "__main__":
    main()
