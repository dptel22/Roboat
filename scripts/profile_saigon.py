"""T2 - Profile the Saigon River dataset (D5) in the exploration-report style.

Read-only on data/processed/saigon_src/extracted. Outputs:
  data/processed/saigon_profile.json
  data/exploration_samples/saigon/*.jpg (5 box-drawn samples)
"""
import hashlib
import json
import random
import re
import zipfile
from collections import Counter, defaultdict
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

SEED = 42
ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "data" / "processed" / "saigon_src"
E = SRC / "extracted"
RAW = ROOT / "data" / "raw"
OUT = ROOT / "data" / "processed" / "saigon_profile.json"
SAMPLES = ROOT / "data" / "exploration_samples" / "saigon"
COLORS = {0: "red", 1: "lime", 2: "cyan"}


def md5(p):
    h = hashlib.md5()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def docx_text(path):
    """Extract readable text from a .docx without python-docx."""
    try:
        with zipfile.ZipFile(path) as z:
            xml = z.read("word/document.xml").decode("utf-8", "ignore")
        return re.sub(r"<[^>]+>", "\n", xml)
    except Exception as e:
        return f"(extraction failed: {e})"


def main():
    random.seed(SEED)
    classes = [l.split(":")[0].strip() for l in
               (SRC / "extracted" / "classes.txt").read_text().splitlines() if l.strip()]
    imgs = sorted((E / "images").iterdir())
    lbls = sorted((E / "labels").iterdir())
    cls_counts = Counter()
    malformed = []
    n_boxes = 0
    for lf in lbls:
        for line in lf.read_text().splitlines():
            t = line.split()
            if len(t) < 5:
                malformed.append(f"{lf.name}: '{line[:60]}'")
                continue
            try:
                c, cx, cy, w, h = int(t[0]), *map(float, t[1:5])
            except ValueError:
                malformed.append(f"{lf.name}: non-numeric")
                continue
            cls_counts[classes[c] if c < len(classes) else str(c)] += 1
            n_boxes += 1
    dims, corrupt = [], []
    for p in imgs:
        try:
            with Image.open(p) as im:
                im.verify()
            with Image.open(p) as im2:
                dims.append(im2.size)
        except Exception as e:
            corrupt.append(f"{p.name}: {e}")
    ws = [w for w, _ in dims] or [0]
    hs = [h for _, h in dims] or [0]
    # orphans
    istems = {p.stem for p in imgs}
    lstems = {p.stem for p in lbls}
    orph = {"images_without_label": sorted(istems - lstems),
            "labels_without_image": sorted(lstems - istems)}
    # dupes: within saigon (by image content), and vs other datasets
    hcount = Counter(md5(p) for p in imgs)
    within = {h: n for h, n in hcount.items() if n > 1}
    hashes = {h: p.name for h, p in zip((md5(p) for p in imgs), imgs)}
    cross = []
    for ds in RAW.iterdir():
        if not ds.is_dir():
            continue
        for p in ds.rglob("*.jp*g"):
            try:
                if md5(p) in hashes:
                    cross.append(f"saigon:{hashes[md5(p)]} == {p.relative_to(RAW)}")
            except OSError:
                pass
    # box pixel sizes at imgsz 640/960, matched per image stem (no index zip)
    dims_by_stem = {p.stem: d for p, d in zip(imgs, dims)}
    px = {}
    for s in (640, 960):
        vals = []
        for lf in lbls:
            d = dims_by_stem.get(lf.stem)
            if d is None:
                continue
            W, H = d
            sc = s / max(W, H)
            for line in lf.read_text().splitlines():
                t = line.split()
                if len(t) >= 5:
                    vals += [float(t[3]) * W * sc, float(t[4]) * H * sc]
        if vals:
            import numpy as np
            a = np.array(vals)
            px[s] = {"median": float(np.median(a)), "p10": float(np.percentile(a, 10)),
                     "share_lt8px": float((a < 8).mean()), "share_lt16px": float((a < 16).mean())}
    # samples
    SAMPLES.mkdir(parents=True, exist_ok=True)
    with_lbl = [p for p in imgs if (E / "labels" / (p.stem + ".txt")).exists()]
    for p in random.sample(with_lbl, min(5, len(with_lbl))):
        im = Image.open(p).convert("RGB")
        W, H = im.size
        im.thumbnail((1600, 1600))
        sc = im.size[0] / W
        d = ImageDraw.Draw(im)
        try:
            font = ImageFont.load_default(20)
        except TypeError:
            font = ImageFont.load_default()
        for line in (E / "labels" / (p.stem + ".txt")).read_text().splitlines():
            t = line.split()
            if len(t) >= 5:
                c, cx, cy, w, h = int(t[0]), *map(float, t[1:5])
                x1, y1 = (cx - w / 2) * W * sc, (cy - h / 2) * H * sc
                d.rectangle([x1, y1, x1 + w * W * sc, y1 + h * H * sc],
                            outline=COLORS.get(c, "white"), width=4)
                d.text((x1, max(0, y1 - 22)), classes[c] if c < len(classes) else str(c),
                       fill=COLORS.get(c, "white"), font=font)
        im.save(SAMPLES / f"{p.stem}_boxes.jpg", quality=85)

    readme = docx_text(SRC / "README.docx")
    prof = {"n_images": len(imgs), "n_labels": len(lbls),
            "classes": classes, "per_class": dict(cls_counts), "n_boxes": n_boxes,
            "malformed_lines": len(malformed), "malformed_examples": malformed[:10],
            "dims": {"min_w": min(ws), "max_w": max(ws), "min_h": min(hs), "max_h": max(hs),
                     "n_sizes": len(set(dims))},
            "corrupt": corrupt, "orphans": orph,
            "duplicates_within": len(within), "duplicates_cross": cross[:20],
            "box_px": px,
            "readme_excerpt": " ".join(readme.split())[:1200]}
    OUT.write_text(json.dumps(prof, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in prof.items() if k != "readme_excerpt"},
                     indent=2)[:2000])


if __name__ == "__main__":
    main()
