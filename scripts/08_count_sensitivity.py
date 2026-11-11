#!/usr/bin/env python3
"""
Sensitivity of the cluster counts to the duplicate-suppression radius and to the persistence
threshold (Reviewer 1, point 8).

Step 1 (slow, once): runs tracking on every sequence with both pipelines, exactly as
count_clusters.py, and saves every filtered detection (frame, track ID, centroid) to tracks.csv.
Step 2 (fast): re-applies the counting rules offline for every combination of
radius and persistence, and writes sensitivity_counts.csv and sensitivity_summary.csv.

Set COUNT_SCRIPT to your counting script (e.g. "count clusters.py"): all settings (models, sequences,
ground truth, filters) are taken from there, so the two analyses are identical.
"""

COUNT_SCRIPT = "07_count_clusters.py"         # path of your counting script (any file name)
RADII = [100, 150, 200, 250, 300, None]     # px; None = no duplicate suppression
PERSISTENCE = [3, 5, 7, 10]                  # frames
TRACKS_CSV = "tracks.csv"

import os
from pathlib import Path
import numpy as np
import pandas as pd
from ultralytics import YOLO
import importlib.util
_spec = importlib.util.spec_from_file_location("cc", COUNT_SCRIPT)
cc = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(cc)


def dump_tracks():
    cc.resolve_device()
    rows = []
    for seq, gt in cc.SEQUENCES.items():
        bag = Path(cc.BAG_DIR) / seq
        if not bag.is_file():
            print("[!] missing:", bag); continue
        for mode, model_path in cc.MODELS.items():
            model = YOLO(model_path)
            cls_id = [k for k, v in model.names.items() if v == cc.CLASS_NAME]
            for f, frame in enumerate(cc.bag_frames(bag)):
                h, w = frame.shape[:2]
                r = model.track(frame, persist=True, tracker=cc.TRACKER, conf=cc.CONF, imgsz=cc.IMGSZ,
                                classes=cls_id, device=cc.DEVICE, retina_masks=True, verbose=False)[0]
                if r.boxes is None or r.boxes.id is None:
                    continue
                ids = r.boxes.id.int().cpu().tolist()
                boxes = r.boxes.xyxy.cpu().numpy()
                masks = r.masks.data.cpu().numpy() if (mode == "seg" and r.masks is not None) else None
                for i, tid in enumerate(ids):
                    x1, y1, x2, y2 = boxes[i]
                    if not cc.box_ok(x1, y1, x2, y2, w, h):
                        continue
                    if mode == "seg":
                        if masks is None:
                            continue
                        m = masks[i] > 0.5
                        area = m.sum()
                        if area == 0 or area / max((x2 - x1) * (y2 - y1), 1) < cc.MIN_FILL_RATIO:
                            continue
                        ys, xs = np.nonzero(m); cx, cy = xs.mean(), ys.mean()
                    else:
                        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
                    rows.append([os.path.basename(str(seq)), gt, mode, f, tid, cx, cy])
            n = sum(1 for r_ in rows if r_[0] == os.path.basename(str(seq)) and r_[2] == mode)
            print(f"{seq} {mode}: {n} filtered detections")
    if not rows:
        raise SystemExit("[!] No detections were saved: check BAG_DIR, SEQUENCES and the model paths.")
    pd.DataFrame(rows, columns=["sequence", "GT", "mode", "frame", "track", "cx", "cy"]).to_csv(TRACKS_CSV, index=False)


def count(df, min_frames, radius):
    life, counted, cents = {}, set(), []
    for _, r in df.sort_values("frame").iterrows():
        life[r.track] = life.get(r.track, 0) + 1
        if life[r.track] == min_frames and r.track not in counted:
            if radius is None or all(np.hypot(r.cx - x, r.cy - y) >= radius for x, y in cents):
                counted.add(r.track); cents.append((r.cx, r.cy))
    return len(counted)


def sensitivity():
    tr = pd.read_csv(TRACKS_CSV)
    tr["sequence"] = tr["sequence"].astype(str).map(lambda p: os.path.basename(p.replace("\\", "/")))
    gts = {os.path.basename(str(k).replace("\\", "/")): v for k, v in cc.SEQUENCES.items()}
    print("Rows in tracks.csv:", len(tr))
    print(tr.groupby(["sequence", "mode"]).size().unstack(fill_value=0))
    missing = [k for k in gts if k not in set(tr["sequence"])]
    if missing:
        print("[!] Sequences without any detection in tracks.csv:", missing)
    rows = []
    for rad in RADII:
        for pers in PERSISTENCE:
            for mode in ["seg", "det"]:
                for seq, gt in gts.items():
                    sub = tr[(tr["sequence"] == seq) & (tr["mode"] == mode)]
                    rows.append([("none" if rad is None else rad), pers, mode, seq, gt, count(sub, pers, rad)])
    res = pd.DataFrame(rows, columns=["radius", "persistence", "mode", "sequence", "GT", "count"])
    res["error"] = res["count"] - res["GT"]
    res.to_csv("sensitivity_counts.csv", index=False)
    summ = (res.assign(abs_err=res.error.abs(), exact=res.error.eq(0), ape=100 * res.error.abs() / res.GT)
               .groupby(["radius", "persistence", "mode"])
               .agg(MAE=("abs_err", "mean"), MAPE=("ape", "mean"), exact=("exact", "sum"),
                    total=("count", "sum")).round(2).reset_index())
    wide = summ.pivot_table(index=["radius", "persistence"], columns="mode",
                            values=["MAE", "MAPE", "exact", "total"])
    wide.columns = [f"{a}_{b}" for a, b in wide.columns]
    wide.to_csv("sensitivity_summary.csv")
    pd.set_option("display.width", 200)
    print(wide.to_string())
    print("\nGround-truth total:", sum(gts.values()))


if __name__ == "__main__":
    if not os.path.isfile(TRACKS_CSV):
        dump_tracks()
    sensitivity()
