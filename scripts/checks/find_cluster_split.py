#!/usr/bin/env python3
"""
Reconstructs, for each of the 51 harvested clusters, whether its frames were used in the
YOLO training set, in the validation set, or in neither (Reviewer 1, point 4a).

It works without split.txt: every frame name contains a unique timestamp, so the frames of each
cluster (listed in samples.json) are matched with the image names in the YOLO dataset folders.

Output: cluster_split.csv  (Cluster_ID, task folder, frames, frames in train, frames in val, split)
"""

# =============================== CONFIGURATION ===============================
SAMPLES_JSON = "samples.json"
YOLO_DATASET = r"E:\Dataset CVAT\new_dataset"     # folder with images/train and images/val
# ==============================================================================

import json, os, re
import pandas as pd

TS = re.compile(r"(\d{13}\.\d+)")   # e.g. 1770708197709.67724609375000


def stamps_in(folder):
    out = set()
    for root, _, files in os.walk(folder):
        for f in files:
            m = TS.search(f)
            if m:
                out.add(m.group(1))
    return out


train = stamps_in(os.path.join(YOLO_DATASET, "images", "train"))
val = stamps_in(os.path.join(YOLO_DATASET, "images", "val"))
print(f"Timestamps found: train {len(train)}, val {len(val)}")

rows = []
for i, s in enumerate(json.load(open(SAMPLES_JSON))):
    task = os.path.basename(os.path.dirname(os.path.dirname(os.path.dirname(s["rgb"][0]))))
    st = [m.group(1) for m in (TS.search(os.path.basename(p)) for p in s["rgb"]) if m]
    n_tr = sum(t in train for t in st)
    n_va = sum(t in val for t in st)
    split = ("train" if n_tr and not n_va else "val" if n_va and not n_tr
             else "both" if n_tr and n_va else "not in YOLO dataset")
    rows.append(dict(Cluster_ID=i + 1, Task=task, Frames=len(st),
                     In_train=n_tr, In_val=n_va, Split=split))

df = pd.DataFrame(rows)
df.to_csv("cluster_split.csv", index=False)
print(df.to_string(index=False))
print("\nSummary:\n", df["Split"].value_counts().to_string())
print("\nSaved cluster_split.csv")
