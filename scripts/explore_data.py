"""Scan data/raw datasets and dump per-dataset diagnostics to JSON.

Read-only on data/raw. Outputs:
  data/exploration_results.json  - machine-readable findings
  data/exploration_samples/<dataset>/*.jpg - 5 box-drawn samples per boxed dataset
"""
import csv
import hashlib
import json
import os
import sys
import traceback
from collections import Counter, defaultdict
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
OUT_JSON = ROOT / "data" / "exploration_results.json"
SAMPLES = ROOT / "data" / "exploration_samples"

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
COLORS = ["red", "lime", "cyan", "yellow", "magenta", "orange", "white"]


def md5(path, chunk=1 << 20):
    h = hashlib.md5()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def count_by_ext(files):
    return dict(Counter(f.suffix.lower() for f in files))


def yolo_stats(label_files):
    """Parse YOLO txt files -> (per-class counts, malformed list)."""
    cls_counts = Counter()
    malformed = []
    n_boxes = 0
    for lf in label_files:
        try:
            txt = lf.read_text().strip()
            if not txt:
                continue
            for line in txt.splitlines():
                parts = line.split()
                if len(parts) < 5:
                    malformed.append(f"{lf.name}: '{line[:60]}'")
                    continue
                cls = parts[0]
                try:
                    vals = [float(x) for x in parts[1:5]]
                except ValueError:
                    malformed.append(f"{lf.name}: non-numeric '{line[:60]}'")
                    continue
                if not all(0.0 <= v <= 1.0 for v in vals):
                    malformed.append(f"{lf.name}: out-of-range '{line[:60]}'")
                    continue
                cls_counts[cls] += 1
                n_boxes += 1
        except Exception as e:
            malformed.append(f"{lf.name}: {e}")
    return cls_counts, malformed, n_boxes


def open_classnames(dirpath):
    f = dirpath / "classes.txt"
    if f.exists():
        return [l.strip() for l in f.read_text().splitlines() if l.strip()], str(f)
    for p in sorted(dirpath.rglob("classes.txt")):
        return [l.strip() for l in p.read_text().splitlines() if l.strip()], str(p)
    return None, None


def image_sizes(img_files):
    sizes = []
    corrupt = []
    for p in img_files:
        try:
            with Image.open(p) as im:
                im.verify()
            with Image.open(p) as im2:
                sizes.append((im2.width, im2.height))
        except Exception as e:
            corrupt.append(f"{p.relative_to(RAW)}: {type(e).__name__} {e}")
    return sizes, corrupt


def orphans(img_files, label_files):
    img_bases = defaultdict(list)
    for p in img_files:
        img_bases[p.stem].append(p)
    lbl_bases = defaultdict(list)
    for p in label_files:
        lbl_bases[p.stem].append(p)
    imgs_no_lbl = [str(p.relative_to(RAW)) for p in img_files if p.stem not in lbl_bases]
    lbls_no_img = [str(p.relative_to(RAW)) for p in label_files if p.stem not in img_bases]
    return imgs_no_lbl, lbls_no_img


def draw_samples(name, items, out_dir, n=5):
    """items: list of (img_path, [(x1,y1,x2,y2,label)]) in pixel coords."""
    out_dir.mkdir(parents=True, exist_ok=True)
    made = 0
    for img_path, boxes in items[:n]:
        try:
            im = Image.open(img_path).convert("RGB")
        except Exception:
            continue
        d = ImageDraw.Draw(im)
        try:
            font = ImageFont.load_default(16)
        except TypeError:
            font = ImageFont.load_default()
        for x1, y1, x2, y2, lbl in boxes:
            c = COLORS[hash(lbl) % len(COLORS)]
            d.rectangle([x1, y1, x2, y2], outline=c, width=3)
            d.text((x1 + 2, max(0, y1 - 18)), lbl, fill=c, font=font)
        out_name = f"{img_path.stem}_boxes.jpg"
        im.save(out_dir / out_name, quality=85)
        made += 1
    return made


def scan_generic(ds_dir, name):
    """Full scan for one dataset dir. Returns dict."""
    files = sorted(p for p in ds_dir.rglob("*") if p.is_file())
    res = {"dir": str(ds_dir.relative_to(ROOT)), "file_count": len(files)}
    res["files_by_ext"] = count_by_ext(files)

    img_files = [p for p in files if p.suffix.lower() in IMG_EXTS]
    res["image_count"] = len(img_files)
    res["images_by_ext"] = count_by_ext(img_files)

    # classify non-image files
    txt_lbls = []
    jsons, csvs, xmls, others = [], [], [], []
    for p in files:
        if p in img_files:
            continue
        s = p.suffix.lower()
        if s == ".txt":
            txt_lbls.append(p)
        elif s == ".json":
            jsons.append(p)
        elif s == ".csv":
            csvs.append(p)
        elif s == ".xml":
            xmls.append(p)
        else:
            others.append(p)
    res["label_like"] = {"txt": len(txt_lbls), "json": [str(p.name) for p in jsons],
                         "csv": [str(p.name) for p in csvs], "xml": len(xmls),
                         "other": [str(p.relative_to(ds_dir)) for p in others]}

    res["class_names_file"] = None
    res["classes"] = {}
    res["yolo_malformed"] = []
    res["yolo_total_boxes"] = 0

    # COCO json?
    coco_info = None
    for p in jsons:
        try:
            data = json.loads(p.read_text(encoding="utf-8-sig"))
            if isinstance(data, dict) and {"images", "annotations"} <= set(data.keys()):
                cats = {c["id"]: c.get("name", f"id{c['id']}") for c in data.get("categories", [])}
                cat_counts = Counter(cats.get(a["category_id"], f"?{a['category_id']}")
                                     for a in data["annotations"])
                coco_info = {"file": str(p.relative_to(ds_dir)), "num_images": len(data["images"]),
                             "num_annotations": len(data["annotations"]), "categories": cats,
                             "per_class": dict(cat_counts)}
        except Exception as e:
            coco_info = {"file": str(p.relative_to(ds_dir)), "error": str(e)}
    res["coco"] = coco_info

    # CSV?
    csv_info = None
    for p in csvs:
        try:
            with open(p, newline="", encoding="utf-8-sig") as f:
                r = csv.reader(f)
                header = next(r)
                rows = list(r)
            ci = header.index("class_name") if "class_name" in header else None
            cc = Counter(row[ci] for row in rows if ci is not None and len(row) > ci)
            csv_info = {"file": str(p.relative_to(ds_dir)), "header": header,
                        "rows": len(rows), "per_class": dict(cc)}
        except Exception as e:
            csv_info = {"file": str(p.relative_to(ds_dir)), "error": str(e)}
    res["csv"] = csv_info

    # YOLO txt labels (exclude classes.txt)
    yolo_lbls = [p for p in txt_lbls if p.name.lower() != "classes.txt"]
    if yolo_lbls:
        names, names_file = open_classnames(ds_dir)
        res["class_names_file"] = names_file
        cc, malformed, nbox = yolo_stats(yolo_lbls)
        res["yolo_malformed"] = malformed[:20]
        res["yolo_total_boxes"] = nbox
        res["classes"] = {names[int(k)] if names and k.isdigit() and int(k) < len(names)
                          else f"class_{k}": v for k, v in sorted(cc.items())}
    elif csv_info and csv_info.get("per_class"):
        res["classes"] = csv_info["per_class"]
    elif coco_info and coco_info.get("per_class"):
        res["classes"] = coco_info["per_class"]

    # dims + corrupt
    sizes, corrupt = image_sizes(img_files)
    res["corrupt_files"] = corrupt
    if sizes:
        ws = [w for w, _ in sizes]; hs = [h for _, h in sizes]
        res["dims"] = {"min_w": min(ws), "max_w": max(ws), "min_h": min(hs),
                       "max_h": max(hs), "consistent": len(set(sizes)) == 1,
                       "mode": Counter(sizes).most_common(1)[0]}

    # orphans: label candidates = yolo txts (+ coco/csv matched separately)
    imgs_no, lbls_no = orphans(img_files, yolo_lbls)
    res["orphans"] = {"images_without_label": imgs_no, "labels_without_image": lbls_no}

    # samples with boxes
    items = []
    if yolo_lbls and res["class_names_file"]:
        base_names = names or []
        lbl_by_stem = {p.stem: p for p in yolo_lbls}
        with_lbl = [p for p in img_files if p.stem in lbl_by_stem]
        for p in with_lbl:
            try:
                w, h = Image.open(p).size
            except Exception:
                continue
            boxes = []
            for line in lbl_by_stem[p.stem].read_text().splitlines():
                t = line.split()
                if len(t) >= 5:
                    cx, cy, bw, bh = (float(x) for x in t[1:5])
                    x1 = (cx - bw / 2) * w; y1 = (cy - bh / 2) * h
                    x2 = x1 + bw * w; y2 = y1 + bh * h
                    cls = t[0]
                    lbl = base_names[int(cls)] if cls.isdigit() and int(cls) < len(base_names) else f"class_{cls}"
                    boxes.append((x1, y1, x2, y2, lbl))
            if boxes:
                items.append((p, boxes))
    elif csv_info and csv_info.get("header") and "x_min" in csv_info["header"]:
        with open(csvs[0], newline="", encoding="utf-8-sig") as f:
            r = csv.DictReader(f)
            by_img = defaultdict(list)
            for row in r:
                by_img[row["image_name"]].append((float(row["x_min"]), float(row["y_min"]),
                                                  float(row["x_max"]), float(row["y_max"]),
                                                  row.get("class_name", "?")))
        img_by_name = {p.name: p for p in img_files}
        for nm, boxes in by_img.items():
            if nm in img_by_name:
                items.append((img_by_name[nm], boxes))
    elif coco_info and "annotations" in (coco_info or {}):
        data = json.loads((ds_dir / coco_info["file"]).read_text(encoding="utf-8-sig"))
        cats = {c["id"]: c.get("name", "?") for c in data.get("categories", [])}
        img_by_id = {i["id"]: i for i in data["images"]}
        by_img = defaultdict(list)
        for a in data["annotations"]:
            x, y, w, h = a["bbox"]
            ii = img_by_id.get(a["image_id"])
            if ii:
                by_img[ii["file_name"]].append((x, y, x + w, y + h, cats.get(a["category_id"], "?")))
        img_by_name = {p.name: p for p in img_files}
        for nm, boxes in by_img.items():
            if nm in img_by_name:
                items.append((img_by_name[nm], boxes))

    # pick spread across the list
    step = max(1, len(items) // 5)
    picked = items[::step][:5] if len(items) > 5 else items
    out_dir = SAMPLES / name
    if out_dir.exists():
        for f in out_dir.glob("*.jpg"):
            f.unlink()
    res["samples_made"] = draw_samples(name, picked, out_dir) if picked else 0
    return res


def main():
    datasets = {}
    for d in sorted(RAW.iterdir()):
        if d.is_dir():
            print(f"Scanning {d.name} ...", flush=True)
            try:
                datasets[d.name] = scan_generic(d, d.name)
            except Exception:
                datasets[d.name] = {"error": traceback.format_exc()}
            # dupes (hash every file)
            hashes = defaultdict(list)
            for p in sorted(d.rglob("*")):
                if p.is_file():
                    try:
                        hashes[md5(p)].append(str(p.relative_to(d)))
                    except Exception:
                        pass
            dupes = {h: ps for h, ps in hashes.items() if len(ps) > 1}
            n_extra = sum(len(ps) - 1 for ps in dupes.values())
            datasets[d.name]["duplicates"] = {"groups": len(dupes), "redundant_files": n_extra,
                                              "examples": [ps for ps in list(dupes.values())[:10]]}
            print(f"  done: {datasets[d.name].get('image_count', '?')} imgs", flush=True)

    # cross-dataset dupes by image hash
    print("Cross-dataset duplicate check ...", flush=True)
    cross = defaultdict(list)
    for d in sorted(RAW.iterdir()):
        if not d.is_dir():
            continue
        for p in d.rglob("*"):
            if p.suffix.lower() in IMG_EXTS and p.is_file():
                try:
                    cross[md5(p)].append(str(p.relative_to(RAW)))
                except Exception:
                    pass
    cross_dupes = {h: ps for h, ps in cross.items() if len(ps) > 1}
    result = {"datasets": datasets,
              "cross_dataset_duplicates": {"groups": len(cross_dupes),
                                           "examples": list(cross_dupes.values())[:20]}}
    OUT_JSON.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"Wrote {OUT_JSON}")


if __name__ == "__main__":
    sys.exit(main())
