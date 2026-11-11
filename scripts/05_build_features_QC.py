#!/usr/bin/env python3
"""
Feature extraction with PRE-SPECIFIED view-level quality control (Reviewer 1, point 1).

Same descriptors and processing as 4_build_corr_dataset.py, plus:
  - per-view QC metrics, computed for EVERY view before any calibration:
      dt_ms           time offset between the paired RGB and depth frames (epoch ms in the file names)
      valid_cov       fraction of mask pixels with a valid depth value (0 < z <= MAX_DEPTH)
  - two QC rules, fixed here BEFORE looking at the calibration results:
      rule 1: dt_ms <= MAX_DT_MS
      rule 2: valid_cov >= MIN_VALID_COVERAGE
  - extra statistics requested by the reviewer:
      n_masks         number of cluster masks detected in the view (R1-9a)
      sel_conf        confidence of the selected (largest) mask
      n_valid_pts     number of valid depth points inside the mask (R1-m6a)
  - fixes the column mislabelling of the old image-level export.

Outputs:
  image_level_QC.xlsx    one row per view, all metrics and pass/fail flags (all views kept)
  cluster_level_QC.xlsx  one row per cluster, descriptors averaged over QC-passed views only
  qc_summary.xlsx        views and clusters removed by each rule (Supplementary table)
"""

# =============================== CONFIGURATION ===============================
SAMPLES_JSON = "samples.json"
MODEL_PATH = r"E:\Dataset CVAT\runs_longan\longan_yolov26s-seg_1280_aug\weights\best.pt"

MAX_DEPTH = 2.0               # m
MAX_DT_MS = 100               # rule 1: max RGB-depth offset (about 3 frames at 30 fps)
MIN_VALID_COVERAGE = 0.50     # rule 2: min fraction of mask pixels with valid depth

CONF = 0.01                   # as in the original script
INTRINSICS = (608.0819702148438, 608.1353759765625, 317.5625305175781, 245.44386291503906)
# ==============================================================================

import json, os
import cv2
import numpy as np
import pandas as pd
from scipy.spatial import ConvexHull
from ultralytics import YOLO

model = YOLO(MODEL_PATH)
cluster_cls = [k for k, v in model.names.items() if v.lower().startswith("cluster")]


def epoch_ms(path):
    """'_Color_1770708197709.67724609375000.png' -> 1770708197709.677 (epoch milliseconds)."""
    try:
        return float(os.path.basename(path).split("_")[-1].rsplit(".", 1)[0])
    except (ValueError, IndexError):
        return None


def load_depth(path, shape):
    d = np.loadtxt(path, delimiter=",").astype(np.float32)
    d = cv2.resize(d, (shape[1], shape[0]), interpolation=cv2.INTER_NEAREST)  # as in the original pipeline
    d[(d <= 0) | (d > MAX_DEPTH)] = 0
    return d


def masks_of(rgb):
    r = model(rgb, conf=CONF, classes=cluster_cls, verbose=False)[0]
    if r.masks is None:
        return [], []
    ms = [cv2.resize(m, (rgb.shape[1], rgb.shape[0]), interpolation=cv2.INTER_NEAREST) > 0.5
          for m in r.masks.data.cpu().numpy()]
    return ms, r.boxes.conf.cpu().numpy().tolist()


def point_cloud(mask, depth):
    fx, fy, cx, cy = INTRINSICS
    ys, xs = np.nonzero(mask)
    z = depth[ys, xs]
    ok = z > 0
    xs, ys, z = xs[ok], ys[ok], z[ok]
    if len(z) < 20:
        return None
    pts = np.column_stack([(xs - cx) * z / fx, (ys - cy) * z / fy, z])
    lo, hi = np.percentile(pts[:, 2], [5, 95])
    pts = pts[(pts[:, 2] >= lo) & (pts[:, 2] <= hi)]
    return pts if len(pts) >= 20 else None


def voxel_volume(pts, s=0.01):
    idx = np.floor((pts - pts.min(0)) / s).astype(int)
    return len(np.unique(idx, axis=0)) * s ** 3


def view_features(rgb, depth, mask):
    area = int(mask.sum())
    if area < 50:
        return None
    vals = depth[mask]
    vals = vals[vals > 0]
    pts = point_cloud(mask, depth)
    if len(vals) < 20 or pts is None:
        return None
    try:
        hull = ConvexHull(pts).volume
    except Exception:
        return None
    if not (1e-4 <= hull <= 0.05):
        return None
    ys, xs = np.nonzero(mask)
    w, h = xs.max() - xs.min() + 1, ys.max() - ys.min() + 1
    zm = float(vals.mean())
    return dict(Area=area, Width=w, Height=h, Aspect_Ratio=w / h, Density=area / (w * h),
                Depth_Mean=zm, Depth_Std=float(vals.std()), Voxel_Volume=voxel_volume(pts),
                Convex_Hull_Volume=hull, Proxy_Volume=area * zm ** 2)


def main():
    samples = json.load(open(SAMPLES_JSON))
    rows = []
    for ci, s in enumerate(samples):
        cid = ci + 1
        print(f"Cluster {cid}: {len(s['rgb'])} views")
        for rgb_path, depth_path in zip(s["rgb"], s["depth"]):
            row = dict(Cluster_ID=cid, Weight=s["weight"], RGB=os.path.basename(rgb_path),
                       Depth=os.path.basename(depth_path))
            t1, t2 = epoch_ms(rgb_path), epoch_ms(depth_path)
            row["dt_ms"] = abs(t1 - t2) if (t1 is not None and t2 is not None) else np.nan
            rgb = cv2.imread(rgb_path)
            if rgb is None or not os.path.isfile(depth_path):
                row["status"] = "missing file"; rows.append(row); continue
            depth = load_depth(depth_path, rgb.shape)
            ms, confs = masks_of(rgb)
            row["n_masks"] = len(ms)
            if not ms:
                row["status"] = "no mask"; rows.append(row); continue
            k = int(np.argmax([m.sum() for m in ms]))            # largest mask, as before
            mask = ms[k]
            row["sel_conf"] = confs[k]
            valid = (depth[mask] > 0)
            row["n_valid_pts"] = int(valid.sum())
            row["valid_cov"] = float(valid.mean())
            f = view_features(rgb, depth, mask)
            if f is None:
                row["status"] = "failed geometry filters"; rows.append(row); continue
            row.update(f)
            row["status"] = "ok"
            rows.append(row)

    df = pd.DataFrame(rows)
    df["pass_dt"] = df["dt_ms"] <= MAX_DT_MS
    df["pass_cov"] = df["valid_cov"] >= MIN_VALID_COVERAGE
    df["pass_qc"] = (df["status"] == "ok") & df["pass_dt"] & df["pass_cov"]
    df.to_excel("image_level_QC.xlsx", index=False)

    desc = ["Area", "Voxel_Volume", "Convex_Hull_Volume", "Proxy_Volume", "Width", "Height",
            "Aspect_Ratio", "Density", "Depth_Mean", "Depth_Std"]
    ok = df[df["pass_qc"]]
    cl = ok.groupby("Cluster_ID").agg(Weight=("Weight", "first"), n_views=("RGB", "size"),
                                      **{f"Mean_{d}": (d, "mean") for d in desc}).reset_index()
    cl.to_excel("cluster_level_QC.xlsx", index=False)

    geo = df[df["status"] == "ok"]
    all_ids = set(df["Cluster_ID"])
    def lost(mask_ok):
        kept = set(geo.loc[mask_ok, "Cluster_ID"])
        return len(all_ids - kept)
    summ = pd.DataFrame([
        ["Views extracted", len(df), "", len(all_ids)],
        ["Valid after geometric filters", len(geo), "", geo["Cluster_ID"].nunique()],
        [f"Rule 1: RGB-depth offset <= {MAX_DT_MS} ms", int(geo["pass_dt"].sum()),
         int((~geo["pass_dt"]).sum()), lost(geo["pass_dt"])],
        [f"Rule 2: valid-depth coverage >= {MIN_VALID_COVERAGE:.0%}", int(geo["pass_cov"].sum()),
         int((~geo["pass_cov"]).sum()), lost(geo["pass_cov"])],
        ["Both rules (QC-filtered dataset)", len(ok), len(geo) - len(ok), len(all_ids) - ok["Cluster_ID"].nunique()],
    ], columns=["Step", "Views kept", "Views removed", "Clusters with no remaining view"])
    summ.to_excel("qc_summary.xlsx", index=False)
    print(summ.to_string(index=False))
    print("\nViews with more than one cluster mask:",
          f"{(df['n_masks'] > 1).sum()} of {df['n_masks'].notna().sum()}")
    print("Valid depth points per mask (median):", geo["n_valid_pts"].median())
    print("\nSaved: image_level_QC.xlsx, cluster_level_QC.xlsx, qc_summary.xlsx")


if __name__ == "__main__":
    main()
