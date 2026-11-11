#!/usr/bin/env python3
"""
Benchmark di modelli YOLO GIA' ALLENATI (detect o seg) sullo stesso validation set.

Per ogni modello:
  1. controllo che sia stato allenato in modo confrontabile (dataset, imgsz, epoche da args.yaml)
  2. metriche su val: box e, se seg, maschere (generali e per classe) + confusion matrix
  3. velocita' reale a batch 1 (pre + inferenza + post/NMS), parametri e GFLOPs
  4. ROBUSTEZZA: stesse metriche su copie del val con luce/qualita' alterate
     (scuro, sovraesposto, ombre forti, foschia/controluce, mosso)

Output in OUTPUT: benchmark.csv/png, robustness.csv/png, pr_f1_curves.png, training_curves.png
Requisiti: ultralytics, opencv-python, matplotlib
Uso: modifica la CONFIGURAZIONE e lancia  python benchmark_trained.py
"""

# =============================== CONFIGURAZIONE ===============================
MODELS = {  # nome da mostrare : percorso del best.pt
    "YOLOv8s-seg": r"D:\Dataset CVAT\runs_longan\NOME_RUN_V8\weights\best.pt",
    "YOLO11s-seg": r"D:\Dataset CVAT\runs_longan\NOME_RUN_11\weights\best.pt",
    "YOLO26s-seg": r"D:\Dataset CVAT\runs_longan\NOME_RUN_26\weights\best.pt",
    "YOLO26s":     r"D:\Dataset CVAT\runs_longan\longan_yolo26s_1280_aug-2\weights\best.pt",  # detection
}
DATA_YAML = r"D:\Dataset CVAT\new_dataset\data.yaml"   # validation set comune a tutti
OUTPUT    = r"D:\Dataset CVAT\benchmark_results"        # dove salvare i risultati

IMGSZ   = 960        # usa la risoluzione con cui hai allenato i modelli
BATCH   = 8
DEVICE  = 0          # 0 = GPU, "cpu" per CPU
HALF    = False      # True = FP16 per la misura di velocita' (piu' realistico su GPU)
N_TIMING_IMAGES = 100

ROBUSTNESS = True    # test con condizioni di luce alterate (crea copie del val in OUTPUT)
# ==============================================================================

import csv
import gc
import shutil
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import yaml
from ultralytics import YOLO

IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


# ---------------------------------------------------------------- utilita'
def imread(p):
    return cv2.imdecode(np.fromfile(str(p), np.uint8), cv2.IMREAD_COLOR)


def imwrite(p, img):
    ok, buf = cv2.imencode(Path(p).suffix, img)
    if ok:
        buf.tofile(str(p))


def free_gpu():
    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass


def fix_data_yaml():
    """Se il campo 'path' di data.yaml punta a una cartella che non esiste piu'
    (es. disco passato da D: a E:), usa la cartella in cui si trova data.yaml."""
    global DATA_YAML
    cfg = yaml.safe_load(Path(DATA_YAML).read_text())
    root = Path(cfg.get("path") or Path(DATA_YAML).parent)
    if (root / cfg["val"]).exists():
        return
    alt = Path(DATA_YAML).parent
    if not (alt / cfg["val"]).exists():
        raise SystemExit(f"Cartella di validation non trovata ne' in {root / cfg['val']} "
                         f"ne' in {alt / cfg['val']}: controlla DATA_YAML.")
    print(f"[!] data.yaml punta a {root}, che non esiste: uso {alt}")
    cfg["path"] = alt.resolve().as_posix()
    Path(OUTPUT).mkdir(parents=True, exist_ok=True)
    fixed = Path(OUTPUT) / "data_fixed.yaml"
    fixed.write_text(yaml.safe_dump(cfg, allow_unicode=True))
    DATA_YAML = str(fixed)


def dataset_info():
    cfg = yaml.safe_load(Path(DATA_YAML).read_text())
    root = Path(cfg.get("path") or Path(DATA_YAML).parent)
    val_img = root / cfg["val"]
    parts = list(val_img.parts)                      # .../images/val -> .../labels/val
    i = len(parts) - 1 - parts[::-1].index("images")
    val_lbl = Path(*parts[:i], "labels", *parts[i + 1:])
    names = cfg["names"]
    return val_img, val_lbl, names


# ---------------------------------------------------------------- 1) controllo confrontabilita'
def check_training_args():
    print("=== Come sono stati allenati i modelli (da args.yaml) ===")
    print(f"{'modello':15s} {'imgsz':>6s} {'epochs':>7s} {'batch':>6s}  dataset")
    datas = set()
    for name, w in MODELS.items():
        a = Path(w).parents[1] / "args.yaml"
        if not a.exists():
            print(f"{name:15s}  args.yaml non trovato")
            continue
        args = yaml.safe_load(a.read_text())
        datas.add(str(args.get("data")))
        print(f"{name:15s} {str(args.get('imgsz')):>6s} {str(args.get('epochs')):>7s} "
              f"{str(args.get('batch')):>6s}  {args.get('data')}")
        if str(args.get("imgsz")) != str(IMGSZ):
            print(f"   [!] allenato a imgsz={args.get('imgsz')}, ma il benchmark usa IMGSZ={IMGSZ}")
    if len(datas) > 1:
        print("   [!] I modelli sono stati allenati su dataset DIVERSI: il confronto potrebbe non essere equo")
        print("       (es. immagini di val finite nel train di un modello se il dataset e' stato rigenerato).")
    print()


# ---------------------------------------------------------------- 2) metriche
def metrics_row(m, names_map, prefix=""):
    row = {f"{prefix}box_P": m.box.mp, f"{prefix}box_R": m.box.mr,
           f"{prefix}box_mAP50": m.box.map50, f"{prefix}box_mAP50_95": m.box.map}
    for i, c in enumerate(m.ap_class_index):
        p, r, ap50, ap = m.box.class_result(i)
        n = names_map[int(c)]
        row[f"{prefix}box_{n}_P"], row[f"{prefix}box_{n}_R"] = p, r
        row[f"{prefix}box_{n}_mAP50"], row[f"{prefix}box_{n}_mAP50_95"] = ap50, ap
    if hasattr(m, "seg"):
        row.update({f"{prefix}mask_P": m.seg.mp, f"{prefix}mask_R": m.seg.mr,
                    f"{prefix}mask_mAP50": m.seg.map50, f"{prefix}mask_mAP50_95": m.seg.map})
        for i, c in enumerate(m.ap_class_index):
            p, r, ap50, ap = m.seg.class_result(i)
            n = names_map[int(c)]
            row[f"{prefix}mask_{n}_P"], row[f"{prefix}mask_{n}_R"] = p, r
            row[f"{prefix}mask_{n}_mAP50"], row[f"{prefix}mask_{n}_mAP50_95"] = ap50, ap
    return row


def timing(model, imgs):
    kw = {"half": True} if HALF else {}
    for im in imgs[:5]:
        model.predict(im, imgsz=IMGSZ, device=DEVICE, verbose=False, **kw)
    tot = np.zeros(3)
    for im in imgs:
        s = model.predict(im, imgsz=IMGSZ, device=DEVICE, verbose=False, **kw)[0].speed
        tot += [s["preprocess"], s["inference"], s["postprocess"]]
    pre, inf, post = tot / max(1, len(imgs))
    return {"inference_ms": round(inf, 1), "total_ms": round(pre + inf + post, 1),
            "FPS": round(1000 / (pre + inf + post), 1)}


# ---------------------------------------------------------------- 4) robustezza
def add_shadows(img, rng):
    h, w = img.shape[:2]
    mask = np.zeros((h, w), np.float32)
    for _ in range(3):
        pts = rng.integers([0, 0], [w, h], size=(rng.integers(4, 7), 2)).astype(np.int32)
        cv2.fillPoly(mask, [cv2.convexHull(pts)], 1.0)
    mask = cv2.GaussianBlur(mask, (0, 0), max(3, min(h, w) / 60))[..., None]
    return (img * (1 - 0.6 * mask)).astype(np.uint8)


def motion_blur(img, k=11):
    kernel = np.zeros((k, k), np.float32)
    kernel[k // 2, :] = 1.0 / k
    return cv2.filter2D(img, -1, kernel)


CONDITIONS = {
    "scuro":        lambda im, rng: (np.power(im / 255.0, 2.0) * 255).astype(np.uint8),
    "sovraesposto": lambda im, rng: (np.power(im / 255.0, 0.45) * 255).astype(np.uint8),
    "ombre":        add_shadows,
    "controluce":   lambda im, rng: cv2.addWeighted(im, 0.55, np.full_like(im, 235), 0.45, 0),
    "mosso":        lambda im, rng: motion_blur(im),
}


def build_robustness_sets(val_img, val_lbl, names):
    root = Path(OUTPUT) / "robustness_data"
    yamls = {}
    imgs = sorted(p for p in val_img.iterdir() if p.suffix.lower() in IMG_EXT)
    for cond, fn in CONDITIONS.items():
        d = root / cond
        yml = d / "data.yaml"
        n_done = len(list((d / "images" / "val").glob("*"))) if (d / "images" / "val").exists() else 0
        if n_done != len(imgs):  # (ri)creo solo se manca o e' incompleto
            print(f"  creo val '{cond}' ({len(imgs)} immagini)...")
            shutil.rmtree(d, ignore_errors=True)
            (d / "images" / "val").mkdir(parents=True)
            shutil.copytree(val_lbl, d / "labels" / "val")
            rng = np.random.default_rng(0)  # stesse alterazioni per tutti i modelli
            for p in imgs:
                im = imread(p)
                if im is not None:
                    imwrite(d / "images" / "val" / p.name, fn(im, rng))
        yml.write_text(yaml.safe_dump({"path": d.resolve().as_posix(), "train": "images/val",
                                       "val": "images/val", "names": names}, allow_unicode=True))
        yamls[cond] = str(yml)
    return yamls


# ---------------------------------------------------------------- report
def write_csv(rows, path):
    keys = list(dict.fromkeys(k for r in rows for k in r))
    with open(path, "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=keys)
        wr.writeheader()
        wr.writerows(rows)


def fmt(v):
    return "-" if v is None else (f"{v:.3f}" if isinstance(v, float) else str(v))


def print_table(rows, cols, title):
    print(f"\n{title}")
    widths = [max(len(c), 9) for c in cols]
    print("  ".join(c.rjust(w) for c, w in zip(cols, widths)))
    for r in rows:
        print("  ".join(fmt(r.get(c)).rjust(w) for c, w in zip(cols, widths)))


def plots(rows, rob_rows, classes, kind):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    names = [r["model"] for r in rows]
    metrics = [(f"{kind}_mAP50", "mAP50"), (f"{kind}_mAP50_95", "mAP50-95")]
    for c in classes:
        metrics += [(f"{kind}_{c}_mAP50", f"{c}\nmAP50"), (f"{kind}_{c}_mAP50_95", f"{c}\nmAP50-95")]

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(16, 5.5), gridspec_kw={"width_ratios": [3, 1.2]})
    x = np.arange(len(metrics))
    w = 0.8 / len(rows)
    for i, r in enumerate(rows):
        bars = a1.bar(x + i * w - 0.4 + w / 2, [r.get(k) or 0 for k, _ in metrics], w, label=r["model"])
        a1.bar_label(bars, fmt="%.2f", fontsize=7)
    a1.set_xticks(x, [l for _, l in metrics], fontsize=8)
    a1.set_ylim(0, 1.08)
    a1.set_title(f"Accuratezza su val ({'maschere' if kind == 'mask' else 'box'})")
    a1.legend()
    for r in rows:
        a2.scatter(r["total_ms"], r.get(f"{kind}_mAP50_95") or 0, s=80)
        a2.annotate(r["model"], (r["total_ms"], r.get(f"{kind}_mAP50_95") or 0),
                    textcoords="offset points", xytext=(5, 5), fontsize=8)
    a2.set_xlabel(f"ms per immagine (batch 1, {IMGSZ}px)")
    a2.set_ylabel(f"{kind} mAP50-95")
    a2.set_title("Velocita' vs accuratezza")
    a2.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(Path(OUTPUT) / "benchmark.png", dpi=150)
    plt.close(fig)

    if rob_rows:
        conds = ["originale"] + list(CONDITIONS)
        fig, axes = plt.subplots(1, len(classes), figsize=(7 * len(classes), 4.5), squeeze=False)
        for ax, c in zip(axes[0], classes):
            x = np.arange(len(conds))
            for i, r in enumerate(rob_rows):
                vals = [r.get(f"{cd}_{kind}_{c}_mAP50") or 0 for cd in conds]
                ax.bar(x + i * w - 0.4 + w / 2, vals, w, label=r["model"])
            ax.set_xticks(x, conds, rotation=15)
            ax.set_ylim(0, 1.05)
            ax.set_title(f"Robustezza - {c} ({kind} mAP50)")
            ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(Path(OUTPUT) / "robustness.png", dpi=150)
        plt.close(fig)


def plot_curves(curves, classes, kind):
    """Riga 1: curve Precision-Recall per classe. Riga 2: F1 vs confidenza (soglia migliore marcata)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, len(classes), figsize=(7 * len(classes), 10), squeeze=False)
    for j, c in enumerate(classes):
        ax_pr, ax_f1 = axes[0][j], axes[1][j]
        for name, cv in curves.items():
            if c not in cv["cls"]:
                continue
            i = cv["cls"].index(c)
            prec = cv["prec"][i]
            integ = getattr(np, "trapezoid", None) or np.trapz
            ap50 = float(integ(prec, cv["px"]))
            ax_pr.plot(cv["px"], prec, label=f"{name} (AP50~{ap50:.2f})")
            f1 = cv["f1"][i]
            k = int(f1.argmax())
            line, = ax_f1.plot(cv["px"], f1, label=f"{name}: F1 max {f1[k]:.2f} @ conf {cv['px'][k]:.2f}")
            ax_f1.scatter(cv["px"][k], f1[k], color=line.get_color(), zorder=3)
        ax_pr.set(xlim=(0, 1), ylim=(0, 1.02), xlabel="Recall", ylabel="Precision",
                  title=f"Precision-Recall - {c} ({kind}, IoU 0.5)")
        ax_f1.set(xlim=(0, 1), ylim=(0, 1.02), xlabel="Soglia di confidenza", ylabel="F1",
                  title=f"F1 vs confidenza - {c}")
        for ax in (ax_pr, ax_f1):
            ax.grid(alpha=0.3)
            ax.legend(fontsize=8, loc="lower left")
    fig.tight_layout()
    fig.savefig(Path(OUTPUT) / "pr_f1_curves.png", dpi=150)
    plt.close(fig)


def plot_training(curves, kind):
    """Confronta l'andamento dei training leggendo results.csv di ogni run."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    t = "M" if kind == "mask" else "B"
    loss = "seg_loss" if kind == "mask" else "box_loss"
    panels = [(f"metrics/mAP50({t})", "mAP50 su val"), (f"metrics/mAP50-95({t})", "mAP50-95 su val"),
              (f"train/{loss}", f"train {loss}"), (f"val/{loss}", f"val {loss}")]
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    found = False
    for name, cv in curves.items():
        f = cv["run_dir"] / "results.csv"
        if not f.exists():
            continue
        with open(f, newline="") as fh:
            data = [{k.strip(): v.strip() for k, v in r.items() if k} for r in csv.DictReader(fh)]
        if not data:
            continue
        found = True
        ep = [float(r["epoch"]) for r in data]
        for ax, (col, title) in zip(axes.flat, panels):
            if col in data[0]:
                y = [float(r[col]) if r[col] not in ("", "nan") else np.nan for r in data]
                ax.plot(ep, y, label=name)
                ax.set_title(title)
    if not found:
        plt.close(fig)
        return
    for ax in axes.flat:
        ax.set_xlabel("epoca")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(Path(OUTPUT) / "training_curves.png", dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------- main
def main():
    out = Path(OUTPUT)
    out.mkdir(parents=True, exist_ok=True)
    for name, w in MODELS.items():
        if not Path(w).exists():
            raise SystemExit(f"Pesi non trovati per {name}: {w}")

    check_training_args()
    fix_data_yaml()
    val_img, val_lbl, names = dataset_info()
    names_map = names if isinstance(names, dict) else dict(enumerate(names))
    classes = [names_map[k] for k in sorted(names_map)]
    all_imgs = sorted(str(p) for p in val_img.iterdir() if p.suffix.lower() in IMG_EXT)
    step = max(1, len(all_imgs) // N_TIMING_IMAGES)
    timing_imgs = all_imgs[::step][:N_TIMING_IMAGES]

    rob_yamls = {}
    if ROBUSTNESS:
        print("Preparo i validation set alterati per il test di robustezza:")
        rob_yamls = build_robustness_sets(val_img, val_lbl, names)

    rows, rob_rows, curves = [], [], {}
    for name, w in MODELS.items():
        print(f"\n>>> {name}")
        model = YOLO(w)
        row = {"model": name,
               "params_M": round(sum(p.numel() for p in model.model.parameters()) / 1e6, 2)}
        try:
            from ultralytics.utils.torch_utils import get_flops
            row["GFLOPs"] = round(get_flops(model.model, IMGSZ), 1)
        except Exception:
            row["GFLOPs"] = None

        m = model.val(data=DATA_YAML, imgsz=IMGSZ, batch=BATCH, device=DEVICE, split="val",
                      plots=True, verbose=False, project=str(out), name=f"val_{name}", exist_ok=True)
        row.update(metrics_row(m, names_map))
        row.update(timing(model, timing_imgs))
        o = m.seg if hasattr(m, "seg") else m.box
        curves[name] = {"px": np.array(o.px), "prec": np.array(o.prec_values), "f1": np.array(o.f1_curve),
                        "cls": [names_map[int(c)] for c in m.ap_class_index], "run_dir": Path(w).parents[1]}
        rows.append(row)

        if rob_yamls:
            rr = {"model": name}
            rr.update(metrics_row(m, names_map, prefix="originale_"))
            for cond, yml in rob_yamls.items():
                print(f"    robustezza: {cond}")
                mc = model.val(data=yml, imgsz=IMGSZ, batch=BATCH, device=DEVICE, split="val",
                               plots=False, verbose=False, project=str(out / "robustness_runs"),
                               name=f"{name}_{cond}", exist_ok=True)
                rr.update(metrics_row(mc, names_map, prefix=f"{cond}_"))
            rob_rows.append(rr)

        write_csv(rows, out / "benchmark.csv")
        if rob_rows:
            write_csv(rob_rows, out / "robustness.csv")
        del model
        free_gpu()

    kind = "mask" if any("mask_mAP50" in r for r in rows) else "box"
    base = ["model", "params_M", "GFLOPs", "total_ms", "FPS"]
    for k in (["mask", "box"] if kind == "mask" else ["box"]):
        cols = base + [f"{k}_P", f"{k}_R", f"{k}_mAP50", f"{k}_mAP50_95"] + \
               [f"{k}_{c}_{m}" for c in classes for m in ("mAP50", "mAP50_95")]
        print_table(rows, cols, f"=== ACCURATEZZA ({'maschere' if k == 'mask' else 'box'}) ===")

    if rob_rows:
        conds = ["originale"] + list(CONDITIONS)
        for c in classes:
            print_table(rob_rows, ["model"] + [f"{cd}_{kind}_{c}_mAP50" for cd in conds],
                        f"=== ROBUSTEZZA {c} ({kind} mAP50) ===")
        for r in rob_rows:  # calo medio rispetto all'originale
            o = r.get(f"originale_{kind}_mAP50") or 0
            drops = [o - (r.get(f"{cd}_{kind}_mAP50") or 0) for cd in CONDITIONS]
            r["calo_medio"] = round(float(np.mean(drops)), 3)
        print("\nCalo medio di mAP50 nelle condizioni difficili (piu' basso = piu' robusto):")
        for r in rob_rows:
            print(f"  {r['model']:15s} {r['calo_medio']:.3f}")
        write_csv(rob_rows, out / "robustness.csv")

    # Tab. 2 (formato del paper): classe cluster, box e maschera, piu' il tempo
    main_cls = classes[0]
    cols = ["model"] + [f"box_{main_cls}_{k}" for k in ("P", "R", "mAP50", "mAP50_95")] + \
           [f"mask_{main_cls}_{k}" for k in ("P", "R", "mAP50", "mAP50_95")] + ["total_ms"]
    print_table(rows, cols, f"=== TAB. 2 - classe {main_cls} (box | maschera) ===")
    pd.DataFrame([{c: r.get(c) for c in cols} for r in rows]).to_excel(out / "table2_cluster.xlsx", index=False)

    plots(rows, rob_rows, classes, kind)
    plot_curves(curves, classes, kind)
    plot_training(curves, kind)
    print(f"\nRisultati in: {out.resolve()}")
    print("  benchmark.csv / benchmark.png   -> accuratezza, velocita'")
    if rob_rows:
        print("  robustness.csv / robustness.png -> condizioni di luce difficili")
    print("  pr_f1_curves.png                -> curve Precision-Recall e F1-confidenza a confronto")
    print("  training_curves.png             -> andamento dei training a confronto")
    print("  val_<modello>/                  -> confusion matrix e curve di ogni modello")


if __name__ == "__main__":
    main()
