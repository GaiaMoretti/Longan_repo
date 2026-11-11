#!/usr/bin/env python3
"""
Crea un dataset YOLO da cartelle "task #XXXX" (immagini + annotation XML CVAT / Pascal VOC).

Novita':
  - campiona solo una parte delle immagini di ogni task (frame equidistanti -> meno duplicati)
  - split train/val PER TASK
  - augmentation OFFLINE solo sul train: luce (ombre, controluce, gamma, CLAHE),
    blur/rumore e occlusione con "foglie" sintetiche che NON coprono mai il peduncolo
  - cartella preview/ con esempi augmentati e box disegnati, per controllo visivo

Requisiti:  pip install albumentations opencv-python
Uso: modifica la CONFIGURAZIONE e lancia  python make_yolo_dataset.py
"""

# =============================== CONFIGURAZIONE ===============================
SOURCE = r"D:\Dataset CVAT"                 # cartella che contiene le cartelle "task #..."
OUTPUT = r"D:\Dataset CVAT\new_dataset"     # dove creare il dataset YOLO
VAL_RATIO = 0.2                             # frazione di TASK in validation (se VAL_TASKS e' vuoto)
VAL_TASKS = []                              # scegli tu i task di val, es. ["2026983", "2023159"]
                                            # vuoto = scelta casuale con VAL_RATIO e SEED
SEED = 42                                   # cambia per ottenere uno split diverso
CLASSES = []                                # es. ["Cluster_fruit", "Fruit_branch"]; vuoto = tutte
TASK = "segment"                            # "detect" (box) oppure "segment" (poligoni, per YOLO-seg)
BOX_IN_SEGMENT = "skip"                     # in "segment", oggetti annotati SOLO come box:
                                            #   "skip" = ignorati, "rectangle" = maschera rettangolare
OVERWRITE = True                            # ricrea images/ labels/ preview/ data.yaml in OUTPUT

# --- campionamento per task (vale per train E val) ---
IMAGES_PER_TASK = 80        # max immagini prese da ogni task (None = tutte)
SAMPLING = "uniform"        # "uniform" = equidistanti nella sequenza, "random" = casuali
EMPTY_RATIO = 0.10          # quota max di immagini SENZA annotazioni nel campione di ogni task

# --- augmentation offline (solo train) ---
N_AUG = 2                   # copie augmentate per ogni immagine di train (0 = nessuna)
AUG_ONLY_ANNOTATED = True   # augmenta solo immagini che hanno almeno un box
OCCLUSION_PROB = 0.5        # probabilita' di aggiungere foglie sintetiche
MAX_OCCLUSION = 0.40        # frazione massima di un cluster che le foglie possono coprire
PROTECT_CLASSES = ["Fruit_branch"]  # classi che le foglie sintetiche non devono MAI coprire
N_PREVIEW = 30              # quante immagini augmentate salvare in preview/ con i box disegnati
# ==============================================================================

import random
import re
import shutil
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import cv2
import numpy as np

try:
    import albumentations as A
except ImportError:
    A = None

IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
TASK_RE = re.compile(r"task\s*#?\s*(\d+)", re.IGNORECASE)


# ---------------------------------------------------------------- I/O immagini (path con spazi/#/unicode su Windows)
def imread(path: Path):
    data = np.fromfile(str(path), dtype=np.uint8)
    return cv2.imdecode(data, cv2.IMREAD_COLOR)


def imwrite(path: Path, img):
    ok, buf = cv2.imencode(path.suffix, img)
    if ok:
        buf.tofile(str(path))
    return ok


def image_size(path: Path):
    img = imread(path)
    return (float(img.shape[1]), float(img.shape[0])) if img is not None else (0.0, 0.0)


# ---------------------------------------------------------------- ricerca task e parsing XML
def find_tasks(src: Path):
    return [(TASK_RE.fullmatch(p.name.strip()).group(1), p)
            for p in sorted(src.rglob("*"))
            if p.is_dir() and TASK_RE.fullmatch(p.name.strip())]


def cvat_mask_to_polygon(m):
    """Decodifica una <mask rle=... left top width height> di CVAT nel contorno esterno piu' grande."""
    try:
        rle = [int(v) for v in m.get("rle", "").split(",") if v.strip()]
        left, top = int(float(m.get("left"))), int(float(m.get("top")))
        mw, mh = int(float(m.get("width"))), int(float(m.get("height")))
    except (TypeError, ValueError):
        return None
    flat = np.zeros(mw * mh, np.uint8)
    pos, val = 0, 0
    for run in rle:                       # run alternati: sfondo, oggetto, sfondo, ...
        if val:
            flat[pos:pos + run] = 255
        pos += run
        val ^= 1
    mask = flat[:mw * mh].reshape(mh, mw)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    c = cv2.approxPolyDP(max(contours, key=cv2.contourArea), 1.0, True)
    pts = [(float(x + left), float(y + top)) for [[x, y]] in c]
    return pts if len(pts) >= 3 else None


def parse_cvat(root):
    out = {}
    for img in root.iter("image"):
        name = Path(img.get("name", "")).name
        w, h = float(img.get("width", 0)), float(img.get("height", 0))
        objs = []
        for b in img.findall("box"):
            objs.append((b.get("label"), float(b.get("xtl")), float(b.get("ytl")),
                         float(b.get("xbr")), float(b.get("ybr")), None))
        for poly in img.findall("polygon"):
            pts = [tuple(map(float, p.split(","))) for p in poly.get("points", "").split(";") if p]
            if len(pts) >= 3:
                xs, ys = zip(*pts)
                objs.append((poly.get("label"), min(xs), min(ys), max(xs), max(ys), pts))
        for m in img.findall("mask"):                     # maschere a pennello di CVAT (RLE)
            pts = cvat_mask_to_polygon(m)
            if pts:
                xs, ys = zip(*pts)
                objs.append((m.get("label"), min(xs), min(ys), max(xs), max(ys), pts))
        out[name] = (w, h, objs)
    return out


def parse_voc(root, xml_path: Path):
    fn = Path(root.findtext("filename") or xml_path.stem).name
    size = root.find("size")
    w = float(size.findtext("width", 0)) if size is not None else 0.0
    h = float(size.findtext("height", 0)) if size is not None else 0.0
    objs = []
    for obj in root.findall("object"):
        bb = obj.find("bndbox")
        if bb is not None:
            objs.append((obj.findtext("name"), float(bb.findtext("xmin")), float(bb.findtext("ymin")),
                         float(bb.findtext("xmax")), float(bb.findtext("ymax")), None))
    return {fn: (w, h, objs), xml_path.stem: (w, h, objs)}


def load_task_annotations(task_dir: Path):
    ann = {}
    for xml_path in task_dir.rglob("*.xml"):
        try:
            root = ET.parse(xml_path).getroot()
        except ET.ParseError as e:
            print(f"  [!] XML non valido: {xml_path} ({e})")
            continue
        if root.tag == "annotations":
            parsed = parse_cvat(root)
        elif root.tag == "annotation":
            parsed = parse_voc(root, xml_path)
        else:
            print(f"  [!] Formato XML sconosciuto ({root.tag}): {xml_path}")
            continue
        for name, data in parsed.items():
            ann[Path(name).stem.lower()] = data
    return ann


# ---------------------------------------------------------------- campionamento
def pick(items, k, rng):
    if k >= len(items):
        return list(items)
    if k <= 0:
        return []
    if SAMPLING == "random":
        return rng.sample(items, k)
    step = len(items) / k                    # frame equidistanti lungo la sequenza
    return [items[int(i * step)] for i in range(k)]


def sample_task(samples, rng):
    """samples: lista di (img_path, w, h, objs) ordinata per nome (= ordine temporale)."""
    if IMAGES_PER_TASK is None:
        return samples
    annotated = [s for s in samples if s[3]]
    empty = [s for s in samples if not s[3]]
    n_total = min(IMAGES_PER_TASK, len(samples))
    n_empty = min(len(empty), int(round(n_total * EMPTY_RATIO)))
    n_ann = min(len(annotated), n_total - n_empty)
    if n_ann < n_total - n_empty:            # pochi annotati: riempi con vuoti
        n_empty = min(len(empty), n_total - n_ann)
    return sorted(pick(annotated, n_ann, rng) + pick(empty, n_empty, rng), key=lambda s: s[0].name)


# ---------------------------------------------------------------- conversione YOLO
def to_yolo_lines(w, h, objs, class_map, unknown, skipped):
    lines = []
    for label, x1, y1, x2, y2, pts in objs:
        if label not in class_map:
            unknown.add(label)
            continue
        if TASK == "segment":
            if pts is None:
                if BOX_IN_SEGMENT != "rectangle":
                    skipped[label] = skipped.get(label, 0) + 1
                    continue
                pts = [(x1, y1), (x2, y1), (x2, y2), (x1, y2)]
            coords = " ".join(f"{min(max(x / w, 0), 1):.6f} {min(max(y / h, 0), 1):.6f}" for x, y in pts)
            lines.append(f"{class_map[label]} {coords}")
            continue
        x1, x2 = sorted((max(0, min(w, x1)), max(0, min(w, x2))))
        y1, y2 = sorted((max(0, min(h, y1)), max(0, min(h, y2))))
        bw, bh = x2 - x1, y2 - y1
        if bw <= 1 or bh <= 1:
            continue
        lines.append(f"{class_map[label]} {(x1 + x2) / 2 / w:.6f} {(y1 + y2) / 2 / h:.6f} "
                     f"{bw / w:.6f} {bh / h:.6f}")
    return lines


# ---------------------------------------------------------------- augmentation
def build_photometric():
    """Solo trasformazioni di colore/luce/qualita': i box restano identici.
    La geometria (scala, rotazione, mosaic) la fa gia' YOLO durante il training."""
    return A.Compose([
        A.OneOf([                                                  # esposizione
            A.RandomBrightnessContrast(brightness_limit=0.35, contrast_limit=0.30, p=1.0),
            A.RandomGamma(gamma_limit=(55, 160), p=1.0),           # controluce / sottoesposto
            A.CLAHE(clip_limit=3.0, p=1.0),                        # contrasto locale (ombre chioma)
            A.RandomToneCurve(scale=0.2, p=1.0),
        ], p=0.9),
        A.OneOf([                                                  # luce non uniforme
            A.RandomShadow(shadow_roi=(0, 0, 1, 1), num_shadows_limit=(1, 3),
                           shadow_dimension=6, p=1.0),
            A.RandomSunFlare(flare_roi=(0, 0, 1, 0.5), src_radius=180,
                             num_flare_circles_range=(3, 6), p=1.0),
        ], p=0.5),
        A.HueSaturationValue(hue_shift_limit=5, sat_shift_limit=30, val_shift_limit=15, p=0.5),
        A.OneOf([                                                  # camera in movimento / sensore
            A.MotionBlur(blur_limit=(3, 7), p=1.0),
            A.GaussianBlur(blur_limit=(3, 5), p=1.0),
            A.GaussNoise(std_range=(0.02, 0.06), p=1.0),
        ], p=0.35),
        A.ImageCompression(quality_range=(60, 95), p=0.3),
    ])


def leaf_occlusion(img, objs, rng):
    """Disegna 1-4 foglie sintetiche (ellissi verdi con nervatura) vicino ai cluster.
    Non tocca mai i box in PROTECT_CLASSES e non copre piu' di MAX_OCCLUSION di un cluster."""
    h, w = img.shape[:2]
    protect = [o for o in objs if o[0] in PROTECT_CLASSES]
    targets = [o for o in objs if o[0] not in PROTECT_CLASSES]
    occl = np.zeros((h, w), np.uint8)
    out = img.copy()

    for _ in range(rng.randint(1, 4)):
        for _try in range(15):
            if targets and rng.random() < 0.75:
                _, x1, y1, x2, y2 = rng.choice(targets)[:5]
                if TASK == "segment":
                    # in segmentazione la foglia va sul BORDO del cluster: il poligono YOLO non puo'
                    # avere buchi, quindi solo un'occlusione sul bordo si puo' togliere dall'etichetta
                    edge = rng.choice("lrtb")
                    cx = {"l": x1, "r": x2}.get(edge, rng.uniform(x1, x2))
                    cy = {"t": y1, "b": y2}.get(edge, rng.uniform(y1, y2))
                else:
                    cx, cy = rng.uniform(x1, x2), rng.uniform(y1, y2)
                base = max(x2 - x1, y2 - y1)
            else:
                cx, cy = rng.uniform(0, w), rng.uniform(0, h)
                base = min(w, h) * 0.3
            ax = max(4, int(base * rng.uniform(0.15, 0.35)))
            ay = max(3, int(ax * rng.uniform(0.25, 0.5)))
            ang = rng.uniform(0, 180)
            m = np.zeros((h, w), np.uint8)
            cv2.ellipse(m, (int(cx), int(cy)), (ax, ay), ang, 0, 360, 255, -1)

            # mai sopra il peduncolo (con margine)
            if any(m[max(0, int(o[2]) - 6):int(o[4]) + 6, max(0, int(o[1]) - 6):int(o[3]) + 6].any()
                   for o in protect):
                continue
            new = cv2.bitwise_or(occl, m)
            too_much = False
            for o in targets:
                roi = new[int(o[2]):int(o[4]), int(o[1]):int(o[3])]
                if roi.size and (roi > 0).mean() > MAX_OCCLUSION:
                    too_much = True
                    break
            if too_much:
                continue

            occl = new
            # colore foglia con variazioni + nervatura centrale
            base_col = np.array([rng.randint(15, 60), rng.randint(55, 125), rng.randint(15, 60)], np.float32)
            leaf = np.empty_like(out, dtype=np.float32)
            leaf[:] = base_col
            leaf += np.random.default_rng(rng.randint(0, 1 << 30)).normal(0, 8, leaf.shape)
            rad = np.deg2rad(ang)
            p1 = (int(cx - ax * np.cos(rad)), int(cy - ax * np.sin(rad)))
            p2 = (int(cx + ax * np.cos(rad)), int(cy + ax * np.sin(rad)))
            cv2.line(leaf, p1, p2, (base_col * 1.4).tolist(), max(1, ay // 8))
            leaf = cv2.GaussianBlur(leaf, (3, 3), 0)
            # bordo morbido
            alpha = cv2.GaussianBlur(m.astype(np.float32) / 255.0, (5, 5), 0)[..., None] * 0.95
            out = (out * (1 - alpha) + leaf * alpha).clip(0, 255).astype(np.uint8)
            break
    return out, occl


def visible_part(objs, occl, min_visible=0.2):
    """Segmentazione: toglie dai poligoni la parte coperta dalle foglie sintetiche,
    cosi' la maschera di training resta la parte VISIBILE del cluster (come nelle annotazioni)
    e il modello non impara a includere il fogliame nella maschera."""
    if TASK != "segment" or not occl.any():
        return objs
    h, w = occl.shape
    out = []
    for o in objs:
        label, x1, y1, x2, y2, pts = o
        if pts is None:
            out.append(o)
            continue
        m = np.zeros((h, w), np.uint8)
        cv2.fillPoly(m, [np.round(np.array(pts)).astype(np.int32)], 255)
        area = int((m > 0).sum())
        vis = cv2.bitwise_and(m, cv2.bitwise_not(occl))
        if area == 0 or (vis > 0).sum() < min_visible * area:
            continue  # quasi tutto coperto: niente istanza
        cnts, _ = cv2.findContours(vis, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        c = cv2.approxPolyDP(max(cnts, key=cv2.contourArea), 1.0, True)
        new = [(float(x), float(y)) for [[x, y]] in c]
        if len(new) < 3:
            continue
        xs, ys = zip(*new)
        out.append((label, min(xs), min(ys), max(xs), max(ys), new))
    return out


def draw_preview(img, objs, class_map):
    out = img.copy()
    colors = [(255, 60, 20), (20, 220, 255), (60, 255, 60), (200, 60, 255)]
    for label, x1, y1, x2, y2, pts in objs:
        c = colors[class_map.get(label, 0) % len(colors)]
        if TASK == "segment" and pts:
            cv2.polylines(out, [np.array(pts, np.int32)], True, c, 2)
        else:
            cv2.rectangle(out, (int(x1), int(y1)), (int(x2), int(y2)), c, 2)
        cv2.putText(out, str(label), (int(x1), max(12, int(y1) - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, c, 1)
    return out


# ---------------------------------------------------------------- main
def main():
    src, out = Path(SOURCE), Path(OUTPUT)
    if not src.is_dir():
        sys.exit(f"La cartella SOURCE non esiste: {src}")
    if N_AUG > 0 and A is None:
        sys.exit("Serve albumentations per l'augmentation:  pip install albumentations")

    if OVERWRITE:
        for sub in ("images", "labels", "preview"):
            shutil.rmtree(out / sub, ignore_errors=True)
        (out / "data.yaml").unlink(missing_ok=True)
        (out / "split.txt").unlink(missing_ok=True)
    elif (out / "images").exists():
        sys.exit(f"{out / 'images'} esiste gia'. Imposta OVERWRITE = True o cambia OUTPUT.")

    tasks = find_tasks(src)
    if not tasks:
        sys.exit(f"Nessuna cartella 'task #...' trovata in {src}")
    print(f"Trovati {len(tasks)} task\n")

    rng = random.Random(SEED)

    # 1) lettura annotazioni + campionamento per task
    task_data, all_labels = [], set()
    for tid, tdir in tasks:
        ann = load_task_annotations(tdir)
        imgs = sorted(p for p in tdir.rglob("*") if p.is_file() and p.suffix.lower() in IMG_EXT)
        samples = []
        for img in imgs:
            w, h, objs = ann.get(img.stem.lower(), (0.0, 0.0, []))
            samples.append((img, w, h, objs))
            all_labels.update(o[0] for o in objs)
        chosen = sample_task(samples, rng)
        n_ann = sum(1 for s in chosen if s[3])
        print(f"  task #{tid}: {len(imgs):5d} immagini -> prese {len(chosen):4d} "
              f"({n_ann} annotate, {len(chosen) - n_ann} vuote)")
        task_data.append((tid, chosen))

    classes = CLASSES or sorted(l for l in all_labels if l)
    class_map = {c: i for i, c in enumerate(classes)}
    print(f"\nClassi: {class_map}   |   formato: {TASK}")
    if TASK == "segment":
        n_poly, n_box = {}, {}
        for _, chosen in task_data:
            for s in chosen:
                for o in s[3]:
                    d = n_poly if o[5] else n_box
                    d[o[0]] = d.get(o[0], 0) + 1
        for c in classes:
            print(f"  {c:15s}: {n_poly.get(c, 0):5d} poligoni/maschere, {n_box.get(c, 0):5d} solo box")

    # 2) split per task
    if VAL_TASKS:
        wanted = {str(t).replace("#", "").strip() for t in VAL_TASKS}
        found = {t[0] for t in task_data}
        if wanted - found:
            print(f"[!] Task in VAL_TASKS non trovati: {sorted(wanted - found)}")
        splits = {"val": [t for t in task_data if t[0] in wanted],
                  "train": [t for t in task_data if t[0] not in wanted]}
        if not splits["val"]:
            sys.exit("Nessun task di VAL_TASKS trovato: controlla i numeri.")
    else:
        rng.shuffle(task_data)
        n_val = max(1, round(len(task_data) * VAL_RATIO)) if len(task_data) > 1 else 0
        splits = {"val": task_data[:n_val], "train": task_data[n_val:]}
    for s in splits:
        (out / "images" / s).mkdir(parents=True, exist_ok=True)
        (out / "labels" / s).mkdir(parents=True, exist_ok=True)
    if N_AUG > 0 and N_PREVIEW > 0:
        (out / "preview").mkdir(parents=True, exist_ok=True)

    photometric = build_photometric() if N_AUG > 0 else None
    np.random.seed(SEED)  # albumentations usa anche numpy

    # 3) scrittura
    unknown, skipped, stats, n_preview = set(), {}, {}, 0
    for split, items in splits.items():
        n_orig = n_aug = 0
        box_count = {c: 0 for c in classes}
        for tid, chosen in items:
            for img_path, w, h, objs in chosen:
                base = f"task{tid}_{img_path.stem}"
                ext = img_path.suffix.lower()
                k = 1
                while (out / "images" / split / f"{base}{ext}").exists():
                    base = f"task{tid}_{img_path.stem}_{k}"
                    k += 1

                if w <= 0 or h <= 0:
                    w, h = image_size(img_path)
                lines = to_yolo_lines(w, h, objs, class_map, unknown, skipped) if w > 0 and h > 0 else []
                for ln in lines:
                    box_count[classes[int(ln.split()[0])]] += 1

                shutil.copy2(img_path, out / "images" / split / f"{base}{ext}")
                (out / "labels" / split / f"{base}.txt").write_text("\n".join(lines) + ("\n" if lines else ""))
                n_orig += 1

                # augmentation offline (solo train)
                if split != "train" or N_AUG <= 0 or (AUG_ONLY_ANNOTATED and not lines):
                    continue
                img = imread(img_path)
                if img is None:
                    print(f"  [!] Immagine illeggibile: {img_path}")
                    continue
                # se le dimensioni XML differiscono da quelle reali, riscalo i box
                sx, sy = img.shape[1] / w, img.shape[0] / h
                objs_px = [(l, x1 * sx, y1 * sy, x2 * sx, y2 * sy,
                            [(x * sx, y * sy) for x, y in pts] if pts else None)
                           for l, x1, y1, x2, y2, pts in objs]
                for a in range(N_AUG):
                    aug = photometric(image=img)["image"]
                    aug_objs, aug_lines = objs_px, lines
                    if rng.random() < OCCLUSION_PROB:
                        aug, occl = leaf_occlusion(aug, objs_px, rng)
                        if TASK == "segment":
                            aug_objs = visible_part(objs_px, occl)
                            aug_lines = to_yolo_lines(img.shape[1], img.shape[0], aug_objs,
                                                      class_map, unknown, {})
                    name = f"{base}_aug{a}"
                    imwrite(out / "images" / split / f"{name}.jpg", aug)
                    (out / "labels" / split / f"{name}.txt").write_text(
                        "\n".join(aug_lines) + ("\n" if aug_lines else ""))
                    for ln in aug_lines:
                        box_count[classes[int(ln.split()[0])]] += 1
                    n_aug += 1
                    if n_preview < N_PREVIEW and rng.random() < 0.3:
                        imwrite(out / "preview" / f"{name}.jpg", draw_preview(aug, aug_objs, class_map))
                        n_preview += 1
        stats[split] = (len(items), n_orig, n_aug, box_count)

    # 4) data.yaml
    yaml = [f"path: {out.resolve().as_posix()}", "train: images/train", "val: images/val", "", "names:"]
    yaml += [f"  {i}: {c}" for c, i in class_map.items()]
    (out / "data.yaml").write_text("\n".join(yaml) + "\n")

    # split.txt: quali task sono in train e quali in val (per riprodurre lo stesso split)
    split_txt = [f"# split per task - VAL_TASKS={VAL_TASKS or 'casuale'} SEED={SEED}"]
    for s in ("train", "val"):
        split_txt.append(f"\n[{s}]")
        split_txt += [f"{tid}\t{len(chosen)} immagini" for tid, chosen in sorted(splits[s])]
    (out / "split.txt").write_text("\n".join(split_txt) + "\n")

    # riepilogo
    print("\n=== RIEPILOGO ===")
    for s, (nt, no, na, bc) in stats.items():
        boxes = ", ".join(f"{c}: {n}" for c, n in bc.items())
        print(f"{s:5s}: {nt} task | {no} originali + {na} augmentate = {no + na} immagini | box -> {boxes}")
        print(f"       task: {', '.join('#' + t[0] for t in splits[s])}")
    if skipped:
        print(f"[!] Oggetti con SOLO box ignorati (BOX_IN_SEGMENT='skip'): {skipped}")
    if unknown:
        print(f"[!] Label ignorate (non in CLASSES): {sorted(unknown)}")
    if n_preview:
        print(f"\nControlla gli esempi augmentati in: {(out / 'preview').resolve()}")
    print(f"Split salvato in:  {(out / 'split.txt').resolve()}")
    print(f"Dataset pronto in: {out.resolve()}")


if __name__ == "__main__":
    main()
