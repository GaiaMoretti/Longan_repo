#!/usr/bin/env python3
"""
Valida i modelli sul validation set e stampa P, R, mAP@0.5, mAP@0.5:0.95 per la classe scelta,
a livello box e maschera, nel formato della Tab. 2 del paper.
Le metriche di accuratezza sono identiche su CPU e su GPU: si puo' lanciare anche senza GPU.
Uso: modifica i percorsi e lancia  python val_per_class.py
"""

# =============================== CONFIGURAZIONE ===============================
MODELS = {
    "YOLOv8s-seg": r"E:\Dataset CVAT\runs_longan\NOME_RUN_V8\weights\best.pt",
    "YOLO11s-seg": r"E:\Dataset CVAT\runs_longan\longan_yolo11s-seg_1280_aug-5\weights\best.pt",
    "YOLO26s-seg": r"E:\Dataset CVAT\runs_longan\longan_yolov26s-seg_1280_aug\weights\best.pt",
    "YOLO26s":     r"E:\Dataset CVAT\runs_longan\longan_yolo26s_1280_aug-2\weights\best.pt",
}
DATA_YAML = r"E:\Dataset CVAT\new_dataset\data.yaml"
CLASS_NAME = "Cluster_fruit"
IMGSZ = 960
DEVICE = "cpu"          # 0 se c'e' una GPU CUDA
OUT_XLSX = "table2_cluster.xlsx"
# ==============================================================================

from pathlib import Path
import pandas as pd
import yaml
from ultralytics import YOLO


def data_yaml_ok(path):
    """Corregge il campo 'path' di data.yaml se punta a un disco/cartella che non esiste piu'."""
    cfg = yaml.safe_load(Path(path).read_text())
    root = Path(cfg.get("path") or Path(path).parent)
    if (root / cfg["val"]).exists():
        return path
    cfg["path"] = Path(path).parent.resolve().as_posix()
    fixed = Path("data_fixed.yaml")
    fixed.write_text(yaml.safe_dump(cfg, allow_unicode=True))
    print(f"[!] data.yaml puntava a {root}: uso {cfg['path']}")
    return str(fixed)


def main():
    data = data_yaml_ok(DATA_YAML)
    rows = []
    for name, w in MODELS.items():
        print(f"\n>>> {name}")
        model = YOLO(w)
        m = model.val(data=data, imgsz=IMGSZ, batch=4, device=DEVICE, split="val",
                      plots=False, verbose=False)
        idx = [model.names[int(c)] for c in m.ap_class_index].index(CLASS_NAME)
        row = {"Model": name}
        p, r, ap50, ap = m.box.class_result(idx)
        row.update({"P(B)": p, "R(B)": r, "mAP@0.5(B)": ap50, "mAP@0.5:0.95(B)": ap})
        if hasattr(m, "seg"):
            p, r, ap50, ap = m.seg.class_result(idx)
            row.update({"P(M)": p, "R(M)": r, "mAP@0.5(M)": ap50, "mAP@0.5:0.95(M)": ap})
        rows.append(row)

    df = pd.DataFrame(rows).round(3)
    df.to_excel(OUT_XLSX, index=False)
    pd.set_option("display.width", 200)
    print(f"\n=== Tab. 2 - classe {CLASS_NAME} ===")
    print(df.to_string(index=False, na_rep="—"))
    print(f"\nSalvato: {OUT_XLSX}")


if __name__ == "__main__":
    main()
