#!/usr/bin/env python3
"""
Training YOLO (Ultralytics) per detection di cluster di longan + peduncolo
in ambiente con forte occlusione e variazione di luce.

Requisiti:
  pip install -U ultralytics
  pip install albumentations   # opzionale: aggiunge blur/CLAHE/grayscale leggeri in automatico

Uso: modifica la CONFIGURAZIONE e lancia  python train_yolo_longan.py
"""

# =============================== CONFIGURAZIONE ===============================
DATA_YAML = r"D:\Dataset CVAT\new_dataset\data.yaml"   # creato da make_yolo_dataset.py
PROJECT   = r"D:\Dataset CVAT\runs_longan"           # dove salvare i risultati
NAME      = "longan_yolo26s_1280_aug"              # nome dell'esperimento (aggiornalo se cambi MODEL/IMGSZ)

MODEL   = "yolo26s.pt"   # detection: "yolo26s.pt" / "yolo11s.pt" | segmentazione: "yolo26s-seg.pt" / "yolo11s-seg.pt"
IMGSZ   = 1280           # alto per il peduncolo (largo ~1% dell'immagine)
EPOCHS  = 250
PATIENCE = 60            # early stopping se non migliora per N epoche
BATCH   = 4              # a 1280 con modello "s" servono ~10-12 GB; -1 = automatico
DEVICE  = 0              # 0 = prima GPU, "cpu" per CPU, [0,1] multi-GPU
WORKERS = 4              # su Windows se hai errori prova 0 o 2
SEED    = 42
CACHE   = False          # "ram" o "disk" velocizza se hai memoria a sufficienza

# Per RIPRENDERE un training interrotto: percorso del suo last.pt, es.
# r"D:\Dataset CVAT\runs_longan\longan_yolo11s-seg_1280_aug-4\weights\last.pt"
# Riparte dall'ultima epoca completata con gli stessi parametri. Lascia "" per un training nuovo.
RESUME  = r""

# --------------------------- AUGMENTATION ------------------------------------
AUG = dict(
    # --- luce / colore (variazione luminosa) ---
    hsv_h=0.015,      # tinta: bassa, il colore del longan e' informativo (non confonderlo con foglie)
    hsv_s=0.5,        # saturazione: moderata (la luce e' gia' variata nell'augmentation offline)
    hsv_v=0.4,        # luminosita': moderata (idem)

    # --- geometria ---
    degrees=10.0,     # piccole rotazioni (rami inclinati, camera non perfettamente dritta)
    translate=0.1,
    scale=0.3,        # ridotto: con 0.5 il peduncolo diventava di pochi pixel
    shear=2.0,
    perspective=0.0005,
    fliplr=0.5,       # flip orizzontale ok
    flipud=0.0,       # NO flip verticale: i grappoli pendono, il peduncolo sta SOPRA il cluster

    # --- occlusione / contesto ---
    mosaic=1.0,       # combina 4 immagini: tanti oggetti parziali e tagliati ai bordi
    mixup=0.05,       # basso: la trasparenza "cancella" oggetti sottili come il peduncolo
    close_mosaic=15,  # disattiva il mosaic nelle ultime 15 epoche per rifinire su immagini "reali"
    copy_paste=0.0,   # funziona solo con maschere di segmentazione, inutile con sole box
)

# --------------------------- OTTIMIZZAZIONE -----------------------------------
OPTIM = dict(
    optimizer="auto",   # sceglie lui (per YOLO26 usa MuSGD/SGD a seconda della durata)
    cos_lr=True,        # learning rate cosine: piu' stabile su training lunghi
    warmup_epochs=3,
    amp=True,           # mixed precision: meno VRAM, piu' veloce
)
# ==============================================================================

from pathlib import Path
from ultralytics import YOLO


def train():
    if RESUME:
        print(f"Riprendo il training da: {RESUME}")
        model = YOLO(RESUME)
        model.train(resume=True)
        return Path(model.trainer.save_dir)

    print(f"Modello: {MODEL} | imgsz: {IMGSZ} | batch: {BATCH} | run: {NAME}")
    model = YOLO(MODEL)
    model.train(
        data=DATA_YAML,
        epochs=EPOCHS,
        patience=PATIENCE,
        imgsz=IMGSZ,
        batch=BATCH,
        device=DEVICE,
        workers=WORKERS,
        seed=SEED,
        cache=CACHE,
        project=PROJECT,
        name=NAME,
        exist_ok=False,
        plots=True,
        **AUG,
        **OPTIM,
    )
    return Path(model.trainer.save_dir)


def evaluate(save_dir: Path):
    best = save_dir / "weights" / "best.pt"
    model = YOLO(best)
    m = model.val(data=DATA_YAML, imgsz=IMGSZ, batch=BATCH if BATCH > 0 else 8,
                  device=DEVICE, split="val", plots=True,
                  project=str(save_dir), name="val_best")

    print("\n================ RISULTATI (best.pt su val) ================")
    print(f"Precision: {m.box.mp:.3f}   Recall: {m.box.mr:.3f}   "
          f"mAP50: {m.box.map50:.3f}   mAP50-95: {m.box.map:.3f}")
    print(f"\n{'classe':15s} {'P':>7s} {'R':>7s} {'mAP50':>7s} {'mAP50-95':>9s}")
    for i, c in enumerate(m.ap_class_index):
        p, r, ap50, ap = m.box.class_result(i)
        print(f"{model.names[int(c)]:15s} {p:7.3f} {r:7.3f} {ap50:7.3f} {ap:9.3f}")
    if hasattr(m, "seg"):   # modello -seg: metriche anche sulle maschere
        print(f"\nMASCHERE  Precision: {m.seg.mp:.3f}   Recall: {m.seg.mr:.3f}   "
              f"mAP50: {m.seg.map50:.3f}   mAP50-95: {m.seg.map:.3f}")
        for i, c in enumerate(m.ap_class_index):
            p, r, ap50, ap = m.seg.class_result(i)
            print(f"{model.names[int(c)]:15s} {p:7.3f} {r:7.3f} {ap50:7.3f} {ap:9.3f}")
    print(f"\nPesi migliori: {best}")
    print(f"Grafici e confusion matrix in: {save_dir}")


if __name__ == "__main__":   # necessario su Windows per i worker del dataloader
    run_dir = train()
    evaluate(run_dir)
