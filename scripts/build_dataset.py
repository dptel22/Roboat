"""T3 - Build the merged YOLO dataset (D1, D2, D4, D6, D7, D8, D9, D10).

Read-only on data/raw. Writes under data/processed/:
  merged2/  PRIMARY 2-class tree: images|labels /{train,val,test}   (0 litter, 1 hyacinth)
  merged3/  ABLATION 3-class tree: images (hardlinks) |labels        (0 litter, 1 hyacinth, 2 entangled_plastic)
  lists/    per-source val/test txts + combined; train.txt (RFS-expanded), train_base.txt
  yamls/    <source>[_<config>].yaml + combined_<config>.yaml  (config c2 default names)
  ood_aquatrash/ (D6) + yaml; manifest.csv; build_stats.json; box_size_stats.csv; rfs_table.csv
  contact sheet of Hagenbeek empty-label images (D9) -> data/exploration_samples/processed/

--dry-run: computes everything (stats, manifest, tables) but writes NO dataset files.
"""
import argparse
import csv
import json
import random
import shutil
from collections import Counter, defaultdict
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

SEED = 42
ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
PROC = ROOT / "data" / "processed"
AUDIT = PROC / "audit"
SAMPLES_OUT = ROOT / "data" / "exploration_samples" / "processed"

CLASS2 = {"litter": 0, "hyacinth": 1}
CLASS3 = {"litter": 0, "hyacinth": 1, "entangled_plastic": 2}
HAG_NAMES = ["ff_litter", "hyacinth", "ent_litter"]
TILE, STRIDE, MIN_AREA_FRAC = 640, 512, 0.40
MAX_EMPTY_TILE_FRAC = 0.15
MAX_AERIAL_SHARE = 0.35
MAX_BG_FRAC = 0.10
RFS_TS = [0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.3, 0.5, 0.75, 1.0, 1.5, 2.0]


def yolo_boxes(lab_path):
    out = []
    if lab_path.exists():
        for line in lab_path.read_text().splitlines():
            t = line.split()
            if len(t) >= 5:
                try:
                    out.append((int(t[0]), float(t[1]), float(t[2]), float(t[3]), float(t[4])))
                except ValueError:
                    pass
    return out


def place(img_src, stem, split, b2, b3, dry):
    """Copy image into merged2, hardlink into merged3, write both label variants."""
    for tree, boxes in (("merged2", b2), ("merged3", b3)):
        if not dry:
            (PROC / tree / "images" / split).mkdir(parents=True, exist_ok=True)
            (PROC / tree / "labels" / split).mkdir(parents=True, exist_ok=True)
            dst = PROC / tree / "images" / split / f"{stem}.jpg"
            if dst.exists():
                dst.unlink()
            try:
                dst.hardlink_to(img_src)
            except OSError:
                shutil.copy2(img_src, dst)
            (PROC / tree / "labels" / split / f"{stem}.txt").write_text(
                "".join(f"{c} {x:.6f} {y:.6f} {w:.6f} {h:.6f}\n" for c, x, y, w, h in boxes))


def _fixed_out(root: Path, name: str) -> Path:
    """Output path for a bare filename under a trusted root.

    Callers pass literal filenames only; reject anything that could climb out
    (absolute, '..' parts). No resolve(): joining a trusted constant with a
    checked literal cannot escape the root.
    """
    p = Path(name)
    if p.is_absolute() or ".." in p.parts:
        raise ValueError(f"output name must be a bare filename: {name}")
    return root / p


OUT_BOX_STATS = _fixed_out(PROC, "box_size_stats.csv")
OUT_RFS = _fixed_out(PROC, "rfs_table.csv")
OUT_MANIFEST = _fixed_out(PROC, "manifest.csv")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--max-aerial-share", type=float, default=MAX_AERIAL_SHARE)
    args = ap.parse_args()
    dry = args.dry_run
    rng = random.Random(SEED)

    # ---------- inputs ----------
    assign = {}
    for r in csv.DictReader(open(AUDIT / "split_assignment.csv", encoding="utf-8")):
        assign[(r["source"], r["group_id"])] = r["split"]
    fml_group = {r["original_path"]: r["group_id"] for r in
                 csv.DictReader(open(AUDIT / "manifest.csv", encoding="utf-8"))
                 if r["source"] == "fml"}

    man = []  # manifest rows
    stats = {"empties": Counter(), "box_counts": defaultdict(Counter),
             "box_px": defaultdict(list), "box_px_by_stem": defaultdict(lambda: defaultdict(list)),
             "orphans_skipped": Counter(),
             "tiles": Counter(),
             "per_split": defaultdict(Counter)}

    def add_px(source, W, H, boxes):
        for s in (640, 960):
            sc = s / max(W, H)
            for _, _, bw, bh in [(b[0], b[1], b[3] * W, b[4] * H) for b in boxes]:
                stats["box_px"][(source, s)].append(bw * sc)
                stats["box_px"][(source, s)].append(bh * sc)

    if not dry:
        for tree in ("merged2", "merged3"):
            shutil.rmtree(PROC / tree, ignore_errors=True)

    def emit(source, orig, ocls, split, group, tile, stem, img_src, b2, b3, W, H):
        place(img_src, stem, split, b2, b3, dry)
        man.append({"source": source, "original_path": orig, "original_class": ocls,
                    "mapped_class": "+".join(sorted({n for n, c in CLASS2.items() if c in {b[0] for b in b2}})),
                    "split": split, "group_id": group, "tile_info": tile,
                    "final_stem": stem, "n_boxes": len(b2), "width": W, "height": H})

    # ---------- FML (D1: full_dataset only) ----------
    fml = RAW / "120969" / "fml_version2" / "full_dataset"
    empties_fml, empties_tud = [], []
    for split_dir in ("train", "val", "test"):
        for p in sorted((fml / "images" / split_dir).glob("*.jpg")):
            rel = str(p.relative_to(RAW))
            g = fml_group[rel]
            split = assign[("fml", g)]
            boxes = yolo_boxes(fml / "labels" / "yolo_format" / split_dir / (p.stem + ".txt"))
            b2 = b3 = [(0, x, y, w, h) for _, x, y, w, h in boxes]
            add_px("fml", 1920, 1080, boxes)
            stats["box_counts"]["fml"][split] += len(boxes)
            if boxes:
                emit("fml", rel, "garbage", split, g, "", f"fml_{p.stem}", p, b2, b3, 1920, 1080)
            else:
                stats["empties"]["fml"] += 1
                empties_fml.append((rel, g, split, p))
            stats["per_split"][split]["fml"] += 1

    # ---------- TUD-GV ----------
    tud = RAW / "TUD-GV Dataset for Floating Litter Detection"
    for p in sorted(tud.rglob("*.jpg")):
        g = "tud_" + p.stem.split("_")[0]
        split = assign[("tud_gv", g)]
        cand = list(tud.rglob(p.stem + ".txt"))
        if not cand:
            stats["orphans_skipped"]["tud_gv"] += 1
            continue
        lab = cand[0]
        boxes = yolo_boxes(lab)
        b2 = b3 = [(0, x, y, w, h) for _, x, y, w, h in boxes]
        add_px("tud_gv", 1920, 1080, boxes)
        stats["box_counts"]["tud_gv"][split] += len(boxes)
        if boxes:
            emit("tud_gv", str(p.relative_to(RAW)), "litter", split, g, "",
                 f"tud_{p.stem}", p, b2, b3, 1920, 1080)
        else:
            stats["empties"]["tud_gv"] += 1
            empties_tud.append((str(p.relative_to(RAW)), g, split, p))
        stats["per_split"][split]["tud_gv"] += 1

    # ---------- Tiled aerial sources: Hagenbeek (D3d, D4, D9) + Saigon (CP2) ----------
    # Saigon groups: no reliable sequence id (DJI/Gopro stills) -> one group per
    # original image (same discipline as Hagenbeek); split assigned via assign_splits.py
    # and recorded in split_assignment.csv (D10 re-run for CP2).
    saigon_dir = PROC / "saigon_src" / "extracted"
    empty_imgs = []
    for source, img_dir, lab_dir in (
            ("hagenbeek", RAW / "Platic-water hyathin" / "Annotated_images_labels" / "Annotated Images All" / "images",
             RAW / "Platic-water hyathin" / "Annotated_images_labels" / "Annotated Images All" / "labels"),
            ("saigon", saigon_dir / "images", saigon_dir / "labels")):
        for p in sorted(img_dir.glob("*.[jJ][pP][gG]")):
            prefix = {"hagenbeek": "hag_", "saigon": "sai_"}[source]
            g = prefix + p.stem
            split = assign[(source, g)]
            boxes = yolo_boxes(lab_dir / (p.stem + ".txt"))
            if not boxes:
                stats["empties"][source] += 1
                if source == "hagenbeek":
                    empty_imgs.append(p)
                continue  # D9/CP3: verified-empty aerial images stay excluded (default)
        # -- tiling (identical rules for both aerial sources) --
            im = Image.open(p)
            W, H = im.size
            nc, nr = max(1, (W - TILE) // STRIDE + 1), max(1, (H - TILE) // STRIDE + 1)
            pos, emp = [], []
            for r_ in range(nr):
                for c_ in range(nc):
                    x0, y0 = min(c_ * STRIDE, W - TILE), min(r_ * STRIDE, H - TILE)
                    tb = []
                    for cls, cx, cy, bw, bh in boxes:
                        bx0, by0 = (cx - bw / 2) * W, (cy - bh / 2) * H
                        bx1, by1 = bx0 + bw * W, by0 + bh * H
                        ix0, iy0 = max(bx0, x0), max(by0, y0)
                        ix1, iy1 = min(bx1, x0 + TILE), min(by1, y0 + TILE)
                        inter = max(0.0, ix1 - ix0) * max(0.0, iy1 - iy0)
                        if inter / max(1e-9, (bx1 - bx0) * (by1 - by0)) >= MIN_AREA_FRAC:
                            tb.append((HAG_NAMES[cls], ix0, iy0, ix1, iy1))
                    tinfo = f"x{x0}_y{y0}_s{STRIDE}"
                    (pos if tb else emp).append((f"{prefix}{p.stem}__x{x0}_y{y0}", (x0, y0), tb, tinfo))
            k = min(len(emp), int(MAX_EMPTY_TILE_FRAC * len(pos)))
            keep_empty = set(s for s, *_ in rng.sample(emp, k)) if k else set()
            all_tiles = [(s, xy, tb, ti, True) for s, xy, tb, ti in pos] + \
                        [(s, xy, tb, ti, False) for s, xy, tb, ti in emp if s in keep_empty]
            for stem, (x0, y0), tb, ti, has_box in all_tiles:
                b2, b3 = [], []
                if has_box:
                    for name, ix0, iy0, ix1, iy1 in tb:
                        nx = ((ix0 + ix1) / 2 - x0) / TILE  # YOLO center-x within tile
                        ny = ((iy0 + iy1) / 2 - y0) / TILE
                        nw = (ix1 - ix0) / TILE
                        nh = (iy1 - iy0) / TILE
                        b2.append((CLASS2["litter" if name != "hyacinth" else "hyacinth"], nx, ny, nw, nh))
                        b3.append((CLASS3[{"ff_litter": "litter", "hyacinth": "hyacinth",
                                           "ent_litter": "entangled_plastic"}[name]], nx, ny, nw, nh))
                if not dry:
                    # unique temp per tile: hardlinking a reused temp inode would let
                    # the next crop.save() overwrite every earlier tile's pixels
                    tmp_dir = PROC / "_tile_tmp"
                    tmp_dir.mkdir(parents=True, exist_ok=True)
                    tmp = tmp_dir / f"{stem}.jpg"
                    im.crop((x0, y0, x0 + TILE, y0 + TILE)).save(tmp, quality=92)
                else:
                    tmp = p
                emit(source, str(p.relative_to(RAW)) if source == "hagenbeek" else str(p),
                     "+".join(sorted({n for n, *_ in tb})) if has_box else "",
                     split, g, ti + ("" if has_box else "_empty"), stem, tmp, b2, b3, TILE, TILE)
                if not dry:
                    tmp.unlink()
                stats["tiles"]["pos" if has_box else "empty"] += 1
                stats["box_counts"][source][split] += len(b2)
                stats["box_px_by_stem"][(source, 640)][stem] = [b[3] * TILE for b in b2] + [b[4] * TILE for b in b2]
                stats["box_px_by_stem"][(source, 960)][stem] = [b[3] * TILE * 960 / 640 for b in b2] + [b[4] * TILE * 960 / 640 for b in b2]
            stats["per_split"][split][source + "_orig"] += 1

    # ---------- aerial cap (D4) ----------
    n_train_before = sum(1 for r in man if r["split"] == "train")
    aerial_pool = [r for r in man if r["source"] in ("hagenbeek", "saigon")
                   and r["n_boxes"] > 0 and r["split"] == "train"]
    n_non_aerial_train = n_train_before - len(aerial_pool)
    share_pre = {}
    for aerial_src in ("hagenbeek", "saigon"):
        share_pre[aerial_src] = sum(1 for r in aerial_pool if r["source"] == aerial_src) / max(1, n_train_before)
    combined = len(aerial_pool) / max(1, n_train_before)
    cap_applied = False
    drop = set()
    if combined > args.max_aerial_share and aerial_pool:
        # D4 (CP2 extension): cap COMBINED aerial share; drop from the dominant
        # source (saigon) so aerial tiles cannot dominate the merged train set
        target = int(args.max_aerial_share * n_non_aerial_train / (1.0 - args.max_aerial_share))
        sai_pool = [r for r in aerial_pool if r["source"] == "saigon"]
        n_drop = min(len(sai_pool), max(0, len(aerial_pool) - target))
        if n_drop > 0:
            drop = {r["final_stem"] for r in rng.sample(sai_pool, n_drop)}
            cap_applied = True
    if cap_applied:
        dropped = 0
        keep_rows = []
        for r in man:
            if r["final_stem"] in drop and r["split"] == "train":
                dropped += 1
                stats["box_counts"][r["source"]]["train"] -= r["n_boxes"]
                if not dry:
                    for tree in ("merged2", "merged3"):
                        for sub in ("images", "labels"):
                            f = PROC / tree / sub / "train" / (f"{r['final_stem']}{'.jpg' if sub=='images' else '.txt'}")
                            f.unlink(missing_ok=True)
                continue
            keep_rows.append(r)
        man = keep_rows
        stats["tiles_dropped"] = dropped
        stats["_dropped_stems"] = drop
        stats["tiles"]["pos"] -= dropped
    n_train = sum(1 for r in man if r["split"] == "train")
    pos_after = sum(1 for r in man if r["source"] in ("hagenbeek", "saigon") and r["n_boxes"] > 0 and r["split"] == "train")
    share_post = pos_after / max(1, n_train)

    # ---------- background negatives (D8) ----------
    budget = int(MAX_BG_FRAC * n_train)
    half = budget // 2
    take_f = [e for e in empties_fml if e[2] == "train"][:half]
    take_t = [e for e in empties_tud if e[2] == "train"][:budget - len(take_f)]
    fml_bg = {x[0] for x in take_f}
    for rel, g, split, p in take_f + take_t:
        src = "fml" if rel in fml_bg else "tud_gv"
        emit(src, rel, "", split, g, "background",
             f"{src}_{p.stem}", p, [], [], 1920, 1080)
    stats["bg_used"] = {"fml": len(take_f), "tud_gv": len(take_t), "budget": budget}

    # ---------- RFS (D7) ----------
    train_rows = [r for r in man if r["split"] == "train"]
    img_classes = defaultdict(set)
    for r in train_rows:
        if r["mapped_class"]:
            for nm in r["mapped_class"].split("+"):
                img_classes[r["final_stem"]].add(nm)
    f_c = Counter()
    for r in train_rows:
        for nm, c in CLASS2.items():
            if c in {CLASS2[x] for x in img_classes[r["final_stem"]]}:
                f_c[nm] += 1
    N = len(train_rows)
    freq = {k: f_c[k] / N for k in CLASS2}
    rfs_rows, r_by_t = [], {}
    for t in RFS_TS:
        r = {k: min(6.0, max(1.0, (t / freq[k]) ** 0.5)) if freq[k] > 0 else 1.0
             for k in CLASS2}
        r_by_t[t] = r
        rfs_rows.append({"t": t, "f_litter": round(freq["litter"], 4),
                         "f_hyacinth": round(freq["hyacinth"], 4),
                         "r_litter": round(r["litter"], 2), "r_hyacinth": round(r["hyacinth"], 2)})
    ok_ts = [t for t in RFS_TS if 3 <= r_by_t[t]["hyacinth"] <= 6]
    t_use = ok_ts[0] if ok_ts else RFS_TS[-1]
    r_use = r_by_t[t_use]
    rep_counts = {}
    for r_ in train_rows:
        ri = max((r_use[nm] for nm in img_classes[r_["final_stem"]]), default=1.0)
        rep_counts[r_["final_stem"]] = max(1, int(ri))  # floor; repeats in list
    for r_ in man:
        r_["rfs_repeat"] = rep_counts.get(r_["final_stem"], 1) if r_["split"] == "train" else 1

    # ---------- lists + yamls (D10) ----------
    if not dry:
        (PROC / "lists").mkdir(parents=True, exist_ok=True)
        (PROC / "yamls").mkdir(exist_ok=True)

        def img_rel(r, tree="merged2"):
            return str((PROC / tree / "images" / r["split"] / f"{r['final_stem']}.jpg").relative_to(ROOT)).replace("\\", "/")

        for split in ("train", "val", "test"):
            rows = [r for r in man if r["split"] == split]
            base = [img_rel(r) for r in rows]
            (PROC / "lists" / f"{split}_base.txt").write_text("\n".join(base) + "\n")
            if split == "train":
                expanded = []
                for r in rows:
                    expanded += [img_rel(r)] * r["rfs_repeat"]
                (PROC / "lists" / "train.txt").write_text("\n".join(expanded) + "\n")
            for src in ("fml", "tud_gv", "hagenbeek", "saigon"):
                (PROC / "lists" / f"{src}_{split}.txt").write_text(
                    "\n".join(img_rel(r) for r in rows if r["source"] == src) + "\n")
        def yaml_for(name, val_list, test_list, names):
            for cfg, tree, nc in (("c2", "merged2", CLASS2), ("c3", "merged3", CLASS3)):
                y = PROC / "yamls" / f"{name}_{cfg}.yaml"
                y.write_text(f"path: {ROOT.as_posix()}\n"
                             f"train: {PROC.joinpath('lists','train.txt').as_posix()}\n"
                             f"val: {(PROC/'lists'/val_list).as_posix()}\n"
                             f"test: {(PROC/'lists'/test_list).as_posix()}\n"
                             f"nc: {nc if isinstance(nc, int) else len(nc)}\n"
                             f"names: [{', '.join(nc if isinstance(nc, list) else list(nc))}]\n")
        for src, nm in (("fml", "fml"), ("tud_gv", "tud_gv"), ("hagenbeek", "hagenbeek_tiles"), ("saigon", "saigon_tiles")):
            yaml_for(nm, f"{src}_val.txt", f"{src}_test.txt", None)
        yaml_for("combined", "val_base.txt", "test_base.txt", None)

    # ---------- AquaTrash OOD (D6) ----------
    if not dry:
        at = RAW / "AquaTrash-master" / "AquaTrash-master"
        ood = PROC / "ood_aquatrash"
        shutil.rmtree(ood, ignore_errors=True)
        (ood / "images").mkdir(parents=True)
        (ood / "labels").mkdir(parents=True)
        import csv as _csv
        for row in _csv.DictReader(open(at / "annotations.csv", encoding="utf-8-sig")):
            img = at / "Images" / row["image_name"]
            im = Image.open(img)
            W, H = im.size
            x1, y1, x2, y2 = (float(row[k]) for k in ("x_min", "y_min", "x_max", "y_max"))
            lines = []
            stem = "aqt_" + img.stem
            if not (ood / "images" / f"{stem}.jpg").exists():
                shutil.copy2(img, ood / "images" / f"{stem}.jpg")
                (ood / "labels" / f"{stem}.txt").unlink(missing_ok=True)
            (ood / "labels" / f"{stem}.txt").open("a").write(
                f"0 {(x1+x2)/2/W:.6f} {(y1+y2)/2/H:.6f} {(x2-x1)/W:.6f} {(y2-y1)/H:.6f}\n")
        (ood.parent / "lists").mkdir(exist_ok=True)
        (PROC / "lists" / "ood_aquatrash.txt").write_text(
            "\n".join(str(p.relative_to(ROOT)).replace("\\", "/")
                      for p in sorted((ood / "images").glob("*.jpg"))) + "\n")
        (PROC / "yamls" / "ood_aquatrash.yaml").write_text(
            f"path: {ROOT.as_posix()}\ntrain: {PROC.joinpath('lists','train.txt').as_posix()}\n"
            f"val: {PROC.joinpath('lists','ood_aquatrash.txt').as_posix()}\n"
            f"test: {PROC.joinpath('lists','ood_aquatrash.txt').as_posix()}\nnc: 1\nnames: [litter]\n")

    # ---------- D9 contact sheet ----------
    if empty_imgs and not dry:
        SAMPLES_OUT.mkdir(parents=True, exist_ok=True)
        cols = 5
        rows_n = (len(empty_imgs) + cols - 1) // cols
        sheet = Image.new("RGB", (cols * 480, rows_n * 400), "black")
        d = ImageDraw.Draw(sheet)
        try:
            font = ImageFont.load_default(18)
        except TypeError:
            font = ImageFont.load_default()
        for i, p in enumerate(empty_imgs):
            im = Image.open(p).convert("RGB")
            im.thumbnail((470, 350))
            x, y = (i % cols) * 480 + 5, (i // cols) * 400 + 5
            sheet.paste(im, (x, y))
            d.text((x, y + 352), p.name, fill="yellow", font=font)
        sheet.save(SAMPLES_OUT / "hagenbeek_empty_labels_contact_sheet.jpg", quality=88)

    # ---------- box-size stats (D11) ----------
    import numpy as np
    drop = stats.get("_dropped_stems", set())
    for (src, s), by_stem in stats["box_px_by_stem"].items():
        for stem, vals in by_stem.items():
            if stem not in drop:
                stats["box_px"][(src, s)].extend(vals)
    bx_rows = []
    for (src, s), vals in sorted(stats["box_px"].items()):
        a = np.array(vals)
        bx_rows.append({"source": src, "imgsz": s, "n": len(a),
                        "median_px": round(float(np.median(a)), 1),
                        "p10_px": round(float(np.percentile(a, 10)), 1),
                        "share_lt8px": round(float((a < 8).mean()), 3),
                        "share_lt16px": round(float((a < 16).mean()), 3)})
    if not dry:
        with open(OUT_BOX_STATS, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(bx_rows[0].keys()))
            w.writeheader(); w.writerows(bx_rows)
        with open(OUT_RFS, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(rfs_rows[0].keys()))
            w.writeheader(); w.writerows(rfs_rows)

    # ---------- manifest + stats ----------
    hdr = ["source", "original_path", "original_class", "mapped_class", "split",
           "group_id", "tile_info", "final_stem", "n_boxes", "width", "height", "rfs_repeat"]
    with open(OUT_MANIFEST, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=hdr, extrasaction="ignore")
        w.writeheader(); w.writerows(man)
    out = {"per_split": {k: dict(v) for k, v in stats["per_split"].items()},
           "empties_total": dict(stats["empties"]),
           "bg_used": stats["bg_used"],
           "box_counts": {k: dict(v) for k, v in stats["box_counts"].items()},
           "tiles": dict(stats["tiles"]),
           "aerial_share_pre_cap": {k: round(v, 3) for k, v in share_pre.items()},
           "aerial_share_post_cap": round(share_post, 3),
           "cap_applied": cap_applied,
           "tiles_dropped": stats.get("tiles_dropped", 0),
           "rfs": {"t_used": t_use, "r": {k: round(v, 2) for k, v in r_use.items()},
                   "freq": {k: round(v, 4) for k, v in freq.items()},
                   "train_images": N},
           "manifest_rows": len(man)}
    (PROC / "build_stats.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
