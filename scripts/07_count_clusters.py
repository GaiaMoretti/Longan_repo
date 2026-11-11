
#!/usr/bin/env python3
"""
Cluster counting with tracking-by-detection: YOLO26s (boxes) vs YOLO26s-seg (masks).

Both pipelines use IDENTICAL settings: same sequences, input size, confidence, class filter,
ByteTrack, size/shape filters, persistence rule and duplicate suppression. The only differences
are those that come from the output representation:
  - segmentation: centroid of the mask + extra compactness filter (mask fill ratio)
  - detection:    centroid of the box

Timing: only the processing of each frame is timed (inference + tracking + filtering + counting),
not the reading of the .bag file nor the display. The first WARMUP_FRAMES are excluded.

Outputs (in OUT_DIR):
  per_frame_<mode>_<sequence>.csv   detections, cumulative count and processing time per frame
  table3_per_sequence.csv           count and FPS per sequence for each pipeline (Table 3)
  table4_summary.csv                MAE, RMSE, MAPE, exact accuracy, mean FPS (Table 4)

Requirements: pip install ultralytics pyrealsense2 opencv-python pandas
"""

# =============================== CONFIGURATION ===============================
MODELS = {
    "seg": r"E:\Dataset CVAT\runs_longan\longan_yolov26s-seg_1280_aug\weights\best.pt",
    "det": r"E:\Dataset CVAT\runs_longan\longan_yolo26s_1280_aug-2\weights\best.pt",
}
BAG_DIR = r"D:\Gaia\LONGAN\17-01-26"
SEQUENCES = {   # bag file name : ground-truth number of clusters
    r"D:\Gaia\LONGAN\17-01-26\20260117_035819.bag": 3,
    r"D:\Gaia\LONGAN\17-01-26\20260117_035836.bag": 3,
    r"D:\Gaia\LONGAN\17-01-26\20260117_035905.bag": 5,
    r"D:\Gaia\LONGAN\17-01-26\20260117_040018.bag": 2,
    r"D:\Gaia\LONGAN\17-01-26\20260117_040131.bag": 2,
    r"D:\Gaia\LONGAN\17-01-26\20260117_040919.bag": 2,
    r"D:\Gaia\LONGAN\17-01-26\20260117_040947.bag": 4,
    r"D:\Gaia\LONGAN\17-01-26\20260117_041012.bag": 2,
    r"D:\Gaia\LONGAN\17-01-26\20260117_041137.bag": 3,
    r"D:\Gaia\LONGAN\17-01-26\20260117_041243.bag": 5,
    r"D:\Gaia\LONGAN\17-01-26\20260117_041319.bag": 3,
}
OUT_DIR = r"counting_results"

CLASS_NAME = "Cluster_fruit"   # only this class is tracked and counted
IMGSZ = 960                    # same input size used for training
CONF = 0.2
DEVICE = "auto"                # "auto" = GPU if available, otherwise CPU; or 0 / "cpu"
TRACKER = "bytetrack.yaml"

# filters applied to BOTH pipelines (on the bounding box)
MIN_BOX_AREA = 5000            # px^2
MIN_REL_AREA = 0.005           # fraction of the frame
MIN_SIDE = 50                  # px, minimum width and height
MIN_ASPECT = 0.5               # width / height
MAX_ELONGATION = 2.5           # height <= 2.5 * width
# extra filter available only with masks
MIN_FILL_RATIO = 0.3           # mask area / box area

MIN_FRAMES = 5                 # a track is counted once it has persisted for this many frames
DUP_SUPPRESSION = True         # centroid-based duplicate suppression, applied to BOTH pipelines
MERGE_DIST = 200               # px

WARMUP_FRAMES = 5              # excluded from the FPS
SHOW = True                    # show the output video while processing (not included in the timing)
SAVE_VIDEO = True              # also save it as <mode>_<sequence>.mp4 in OUT_DIR
DISPLAY_DELAY_MS = 1           # pause between frames in the window (q = quit, space = pause)
# ==============================================================================

import csv
import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from ultralytics import YOLO


def bag_frames(path):
    """Yields the colour frames of a RealSense .bag as BGR images (depth is not used for counting)."""
    import pyrealsense2 as rs
    pipeline, config = rs.pipeline(), rs.config()
    config.enable_device_from_file(str(path), repeat_playback=False)
    config.enable_stream(rs.stream.color)
    profile = pipeline.start(config)
    profile.get_device().as_playback().set_real_time(False)   # no dropped frames
    try:
        while True:
            try:
                frames = pipeline.wait_for_frames(timeout_ms=2000)
            except RuntimeError:
                break   # end of file
            cf = frames.get_color_frame()
            if not cf:
                continue
            img = np.asanyarray(cf.get_data())
            if cf.get_profile().format() == rs.format.rgb8:
                img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            yield img
    finally:
        pipeline.stop()


def box_ok(x1, y1, x2, y2, img_w, img_h):
    w, h = x2 - x1, y2 - y1
    area = w * h
    return (area >= MIN_BOX_AREA and area >= MIN_REL_AREA * img_w * img_h
            and w >= MIN_SIDE and h >= MIN_SIDE
            and w / (h + 1e-6) >= MIN_ASPECT and h <= MAX_ELONGATION * w)


def track_color(tid):
    rng = np.random.default_rng(tid)          # stable colour for each track ID
    return tuple(int(c) for c in rng.integers(60, 255, 3))


def draw_frame(frame, drawn, mode, seq_name, frame_id, count, gt,
               counted, suppressed, track_life, times_ms):
    """Draws masks/boxes, IDs and counting status. Green = counted, yellow = waiting for
    MIN_FRAMES, red = suppressed as duplicate."""
    vis = frame.copy()
    for tid, (x1, y1, x2, y2), (cx, cy), mask in drawn:
        if tid in counted:
            status, col = "counted", (0, 200, 0)
        elif tid in suppressed:
            status, col = "duplicate", (0, 0, 255)
        else:
            status, col = f"{track_life.get(tid, 0)}/{MIN_FRAMES}", (0, 215, 255)
        if mask is not None:
            overlay = vis.copy()
            overlay[mask] = track_color(tid)
            vis = cv2.addWeighted(overlay, 0.45, vis, 0.55, 0)
        cv2.rectangle(vis, (int(x1), int(y1)), (int(x2), int(y2)), col, 2)
        cv2.circle(vis, (int(cx), int(cy)), 5, col, -1)
        ty = int(y1) - 6 if int(y1) - 6 > 105 else int(y1) + 20   # keep labels out of the header
        cv2.putText(vis, f"ID {tid} {status}", (int(x1) + 3, ty),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, col, 2)
    fps = 1000 / np.mean(times_ms[-30:]) if times_ms else 0.0
    lines = [f"{'Segmentation' if mode == 'seg' else 'Detection'} | {Path(seq_name).stem} | frame {frame_id}",
             f"Count: {count}  (GT {gt})",
             f"Processing: {fps:.1f} FPS"]
    cv2.rectangle(vis, (0, 0), (vis.shape[1], 30 + 28 * len(lines) - 20), (0, 0, 0), -1)
    for k, t in enumerate(lines):
        cv2.putText(vis, t, (10, 25 + 28 * k), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    return vis


def run_sequence(mode, model_path, bag_path, seq_name, out_dir, gt="?"):
    model = YOLO(model_path)   # new model = new tracker state for every sequence
    cls_id = [k for k, v in model.names.items() if v == CLASS_NAME]
    if not cls_id:
        raise SystemExit(f"Class {CLASS_NAME} not found in {model_path}: {model.names}")

    track_life, counted_tracks, counted_centroids, suppressed = {}, set(), [], set()
    times_ms, rows = [], []
    writer, quit_all = None, False

    for frame_id, frame in enumerate(bag_frames(bag_path)):
        img_h, img_w = frame.shape[:2]
        drawn = []
        t0 = time.perf_counter()

        r = model.track(frame, persist=True, tracker=TRACKER, conf=CONF, imgsz=IMGSZ,
                        classes=cls_id, device=DEVICE, retina_masks=True, verbose=False)[0]
        n_valid = 0
        if r.boxes is not None and r.boxes.id is not None:
            ids = r.boxes.id.int().cpu().tolist()
            boxes = r.boxes.xyxy.cpu().numpy()
            masks = r.masks.data.cpu().numpy() if (mode == "seg" and r.masks is not None) else None

            for i, tid in enumerate(ids):
                x1, y1, x2, y2 = boxes[i]
                if not box_ok(x1, y1, x2, y2, img_w, img_h):
                    continue
                if mode == "seg":
                    if masks is None:
                        continue
                    m = masks[i] > 0.5          # retina_masks=True -> already at frame resolution
                    area = int(m.sum())
                    if area == 0 or area / max((x2 - x1) * (y2 - y1), 1) < MIN_FILL_RATIO:
                        continue
                    ys, xs = np.nonzero(m)
                    cx, cy = xs.mean(), ys.mean()
                else:
                    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2

                n_valid += 1
                track_life[tid] = track_life.get(tid, 0) + 1
                if track_life[tid] == MIN_FRAMES and tid not in counted_tracks:
                    is_new = True
                    if DUP_SUPPRESSION:
                        is_new = all(np.hypot(cx - mx, cy - my) >= MERGE_DIST
                                     for mx, my in counted_centroids)
                    if is_new:
                        counted_tracks.add(tid)
                        counted_centroids.append((cx, cy))
                    else:
                        suppressed.add(tid)
                if SHOW or SAVE_VIDEO:   # only stored here, drawn after the timing
                    drawn.append((tid, (x1, y1, x2, y2), (cx, cy), m if mode == "seg" else None))

        dt = (time.perf_counter() - t0) * 1000   # processing time only
        if frame_id >= WARMUP_FRAMES:
            times_ms.append(dt)
        rows.append([frame_id, n_valid, len(counted_tracks), round(dt, 2)])

        if SHOW or SAVE_VIDEO:
            vis = draw_frame(frame, drawn, mode, seq_name, frame_id, len(counted_tracks), gt,
                             counted_tracks, suppressed, track_life, times_ms)
            if SAVE_VIDEO:
                if writer is None:
                    writer = cv2.VideoWriter(str(out_dir / f"{mode}_{Path(seq_name).stem}.mp4"),
                                             cv2.VideoWriter_fourcc(*"mp4v"), 15, (img_w, img_h))
                writer.write(vis)
            if SHOW:
                cv2.imshow("Cluster counting", vis)
                key = cv2.waitKey(DISPLAY_DELAY_MS) & 0xFF
                if key == ord(" "):
                    key = cv2.waitKey(50) & 0xFF   # pause until another key is pressed
                if key == ord("q"):
                    quit_all = True
                    break

    if writer is not None:
        writer.release()
    if quit_all:
        cv2.destroyAllWindows()
        raise SystemExit("Stopped by the user (q).")

    with open(out_dir / f"per_frame_{mode}_{Path(seq_name).stem}.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["frame", "valid_detections", "cumulative_count", "processing_ms"])
        w.writerows(rows)

    fps = 1000 / np.mean(times_ms) if times_ms else float("nan")
    return len(counted_tracks), fps, len(rows)


def resolve_device():
    """Chooses the device and returns a description of the hardware, saved with the results."""
    global DEVICE
    import platform
    import torch
    if DEVICE == "auto":
        DEVICE = 0 if torch.cuda.is_available() else "cpu"
    if DEVICE != "cpu" and torch.cuda.is_available():
        return f"GPU: {torch.cuda.get_device_name(int(DEVICE))}"
    return f"CPU: {platform.processor() or platform.machine()}"


def main():
    hardware = resolve_device()
    print(f"Device: {hardware}")
    if hardware.startswith("CPU"):
        print("[!] Running on CPU: counts are valid, but FPS will be much lower than on the GPU "
              "and must be reported as CPU values.")
    out = Path(OUT_DIR)
    out.mkdir(parents=True, exist_ok=True)
    results = []
    for seq, gt in SEQUENCES.items():
        bag = Path(BAG_DIR) / seq
        if not bag.is_file():
            print(f"[!] missing: {bag}")
            continue
        row = {"sequence": Path(seq).stem.split("_")[-1], "GT": gt}
        for mode, model_path in MODELS.items():
            count, fps, n = run_sequence(mode, model_path, bag, seq, out, gt)
            row[f"{mode}_count"], row[f"{mode}_fps"] = count, round(fps, 1)
            print(f"{seq}  {mode}: count {count} (GT {gt}), {fps:.1f} FPS, {n} frames")
        results.append(row)
    cv2.destroyAllWindows()

    t3 = pd.DataFrame(results)
    t3.to_csv(out / "table3_per_sequence.csv", index=False)

    summary = []
    for mode in MODELS:
        err = t3[f"{mode}_count"] - t3["GT"]
        summary.append({
            "pipeline": mode,
            "MAE": round(err.abs().mean(), 3),
            "RMSE": round(float(np.sqrt((err ** 2).mean())), 3),
            "MAPE_%": round(100 * (err.abs() / t3["GT"]).mean(), 1),
            "exact_accuracy_%": round(100 * (err == 0).mean(), 1),
            "mean_FPS": round(t3[f"{mode}_fps"].mean(), 1),
            "hardware": hardware,
        })
    t4 = pd.DataFrame(summary)
    t4.to_csv(out / "table4_summary.csv", index=False)

    pd.set_option("display.width", 200)
    print("\n=== Table 3 ===\n" + t3.to_string(index=False))
    print("\n=== Table 4 ===\n" + t4.to_string(index=False))
    print(f"\nResults in: {out.resolve()}")


if __name__ == "__main__":
    main()