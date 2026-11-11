#!/usr/bin/env python3
"""
How often does the largest-mask rule select the annotated cluster? (Reviewer 1, point 9)

For every view in samples.json, the script runs YOLO26s-seg as in the feature extraction,
selects the largest cluster mask, and compares it with the cluster polygons annotated in CVAT
(the annotations XML found in each task folder). For each view it records:
  n_pred         number of predicted cluster masks
  n_annot        number of annotated clusters
  best_iou       IoU between the selected mask and the best-matching annotated cluster
  iou_largest    IoU between the selected mask and the largest annotated cluster
Outputs: mask_selection_views.csv and a printed summary.
"""

SAMPLES_JSON = "samples.json"
MODEL_PATH = r"E:\Dataset CVAT\runs_longan\longan_yolov26s-seg_1280_aug\weights\best.pt"
CLUSTER_LABEL = "Cluster_fruit"
IOU_MATCH = 0.5
CONF = 0.01

import glob, json, os
import xml.etree.ElementTree as ET
import cv2
import numpy as np
import pandas as pd
from ultralytics import YOLO

model = YOLO(MODEL_PATH)
cls = [k for k, v in model.names.items() if v == CLUSTER_LABEL]


def rle_mask(el, h, w):
    """Decodes a CVAT <mask> element (rle, left, top, width, height) into a full-image mask."""
    counts = [int(v) for v in el.get("rle").split(",")]
    left, top = int(float(el.get("left"))), int(float(el.get("top")))
    mw, mh = int(float(el.get("width"))), int(float(el.get("height")))
    flat = np.zeros(mw * mh, np.uint8); pos, val = 0, 0
    for c in counts:
        flat[pos:pos + c] = val; pos += c; val = 1 - val
    full = np.zeros((h, w), bool)
    full[top:top + mh, left:left + mw] = flat.reshape(mh, mw).astype(bool)
    return full


def load_annotations(task_dir):
    ann = {}
    for xml in glob.glob(os.path.join(task_dir, "**", "*.xml"), recursive=True):
        root = ET.parse(xml).getroot()
        for img in root.iter("image"):
            h, w = int(img.get("height")), int(img.get("width"))
            ms = []
            for el in img:
                if el.get("label") != CLUSTER_LABEL:
                    continue
                m = np.zeros((h, w), np.uint8)
                if el.tag == "polygon":
                    pts = np.array([[float(v) for v in p.split(",")] for p in el.get("points").split(";")])
                    cv2.fillPoly(m, [pts.round().astype(np.int32)], 1); ms.append(m.astype(bool))
                elif el.tag == "box":
                    x1, y1, x2, y2 = (int(float(el.get(k))) for k in ("xtl", "ytl", "xbr", "ybr"))
                    m[y1:y2, x1:x2] = 1; ms.append(m.astype(bool))
                elif el.tag == "mask":
                    ms.append(rle_mask(el, h, w))
            ann[os.path.basename(img.get("name"))] = ms
    return ann


def iou(a, b):
    u = np.logical_or(a, b).sum()
    return np.logical_and(a, b).sum() / u if u else 0.0


rows = []
for ci, s in enumerate(json.load(open(SAMPLES_JSON))):
    task_dir = os.path.dirname(os.path.dirname(os.path.dirname(s["rgb"][0])))
    ann = load_annotations(task_dir)
    for p in s["rgb"]:
        img = cv2.imread(p)
        if img is None:
            continue
        r = model(img, conf=CONF, classes=cls, verbose=False)[0]
        preds = [] if r.masks is None else [cv2.resize(m, (img.shape[1], img.shape[0]),
                 interpolation=cv2.INTER_NEAREST) > 0.5 for m in r.masks.data.cpu().numpy()]
        gts = ann.get(os.path.basename(p), [])
        row = dict(Cluster_ID=ci + 1, image=os.path.basename(p), n_pred=len(preds), n_annot=len(gts))
        if preds and gts:
            sel = preds[int(np.argmax([m.sum() for m in preds]))]
            ious = [iou(sel, g) for g in gts]
            row["best_iou"] = max(ious)
            row["iou_largest"] = ious[int(np.argmax([g.sum() for g in gts]))]
        rows.append(row)
    print(f"Cluster {ci + 1}: {len(s['rgb'])} views")

df = pd.DataFrame(rows)
df.to_csv("mask_selection_views.csv", index=False)
v = df.dropna(subset=["best_iou"])
print("\nViews with predictions and annotations:", len(v), "of", len(df))
print(f"Views with > 1 predicted cluster mask: {(df.n_pred > 1).mean():.1%}")
print(f"Views with > 1 annotated cluster:      {(df.n_annot > 1).mean():.1%}")
print(f"Selected mask matches an annotated cluster (IoU >= {IOU_MATCH}): {(v.best_iou >= IOU_MATCH).mean():.1%}")
print(f"Selected mask matches the largest annotated cluster:          {(v.iou_largest >= IOU_MATCH).mean():.1%}")
m = v[v.n_annot > 1]
if len(m):
    print(f"  in views with > 1 annotated cluster (n = {len(m)}): {(m.iou_largest >= IOU_MATCH).mean():.1%}")
print(f"Median IoU of the selected mask: {v.best_iou.median():.2f}")
