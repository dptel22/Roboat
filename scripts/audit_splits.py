"""T1 - Split leakage audit and grouping decision (D3).

Read-only on data/raw. Outputs:
  data/processed/audit/manifest.csv          - every image: source, path, class counts, original split, proposed group_id
  data/processed/audit/filename_analysis.csv - per-source filename/sequence-id analysis summary rows
  data/processed/audit/leakage_audit.csv     - per eval image: nearest train neighbour (pHash bits, CLIP cosine)
  data/processed/audit/group_proposal.csv    - per group: size, original split composition
Prints a summary and the proposed re-split plan. --dry-run processes a bounded sample.

NOTE: analysis only - does NOT write any split lists (that is build_dataset.py).
"""
import argparse
import csv
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np

SEED = 42
ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
AUDIT = ROOT / "data" / "processed" / "audit"

MANIFEST_HEADER = ["source", "original_path", "original_class", "mapped_class",
                   "split", "group_id", "tile_info"]


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


def _out(name: str) -> Path:
    return _fixed_out(AUDIT, name)


def _under(path: Path, root: Path) -> Path:
    """Read-side guard: resolve a (data-derived) path and refuse escape from root."""
    rp = path.resolve()
    root_r = root.resolve()
    if not rp.is_relative_to(root_r):
        raise ValueError(f"path escapes allowed root {root_r}: {path}")
    return rp


def _rel_to_raw(rel: str) -> Path:
    return _under(RAW / rel, RAW)


# ---------- pHash (no imagehash package; classic DCT hash via cv2) ----------
def phash(img_path, hash_size=8):
    img = cv2.imdecode(np.fromfile(str(img_path), dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
    if img is None:
        return None
    img = cv2.resize(img, (32, 32), interpolation=cv2.INTER_AREA)
    d = cv2.dct(np.float32(img))
    low = d[:hash_size, :hash_size].flatten()
    med = np.median(low[1:])
    return (low > med).astype(np.uint8)  # 64 bits; drop DC weighting nuance, fine for near-dup


def pack(bits):
    v = 0
    for b in bits:
        v = (v << 1) | int(b)
    return v


def hamming(a, b):
    return bin(a ^ b).count("1")


# ---------- CLIP ----------
def clip_embeddings(paths, batch=32, model_name="ViT-B-32-quickgelu", pretrained="openai"):
    import torch
    import open_clip
    device = "cpu"
    model, _, preprocess = open_clip.create_model_and_transforms(
        model_name, pretrained=pretrained, device=device)
    torch.manual_seed(SEED)
    embs = []
    with torch.no_grad():
        for i in range(0, len(paths), batch):
            import PIL.Image
            imgs = [preprocess(PIL.Image.open(p).convert("RGB")) for p in paths[i:i + batch]]
            e = model.encode_image(torch.stack(imgs).to(device))
            e = e / e.norm(dim=-1, keepdim=True)
            embs.append(e.cpu().numpy())
            if (i // batch) % 20 == 0:
                print(f"  CLIP {i + len(imgs)}/{len(paths)}", flush=True)
    return np.concatenate(embs)


# ---------- group proposals ----------
def propose_fml_groups(records, gap_seconds=600):
    """Split each day into sessions when the gap between consecutive frames > gap_seconds."""
    by_day = defaultdict(list)
    for r in records:
        p = _rel_to_raw(r["original_path"])
        parts = p.stem.split("_")
        secs = int(parts[2][:2]) * 3600 + int(parts[2][2:4]) * 60 + int(parts[2][4:6])
        by_day[parts[1]].append((secs, p.name))
    groups = {}
    for day, lst in by_day.items():
        lst.sort()
        sess = 0
        prev = None
        for secs, name in lst:
            if prev is not None and secs - prev > gap_seconds:
                sess += 1
            groups[name] = f"fml_{day}_s{sess:02d}"
            prev = secs
    return groups


def tud_group(path):
    return "tud_" + path.stem.split("_")[0]


def hag_group(path):
    """Hagenbeek: group = original image stem (tiles share it later); here per original image."""
    return "hag_" + path.stem


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true", help="sample 300 images per source")
    ap.add_argument("--skip-clip", action="store_true")
    ap.add_argument("--session-gap", type=int, default=600)
    args = ap.parse_args()

    AUDIT.mkdir(parents=True, exist_ok=True)
    manifest_rows = []
    groups_by_src = {}

    # ---------------- FML ----------------
    fml = RAW / "120969" / "fml_version2" / "full_dataset"
    fml_recs = []
    for split in ("train", "val", "test"):
        for p in sorted((fml / "images" / split).glob("*.jpg")):
            lf = fml / "labels" / "yolo_format" / split / (p.stem + ".txt")
            n_boxes = 0
            if lf.exists():
                n_boxes = sum(1 for l in lf.read_text().splitlines() if l.strip())
            fml_recs.append({"source": "fml", "original_path": str(p.relative_to(RAW)),
                             "original_class": "garbage" if n_boxes else "",
                             "mapped_class": "litter" if n_boxes else "",
                             "split": split, "group_id": None, "tile_info": "",
                             "_img": p, "_nboxes": n_boxes})
    if args.dry_run:
        fml_recs = fml_recs[::max(1, len(fml_recs) // 300)]
    sess_groups = propose_fml_groups(fml_recs, args.session_gap)
    for r in fml_recs:
        r["group_id"] = sess_groups[Path(r["original_path"]).name]
    groups_by_src["fml"] = fml_recs

    # ---------------- TUD-GV ----------------
    tud = RAW / "TUD-GV Dataset for Floating Litter Detection"
    tud_recs = []
    for p in sorted(tud.rglob("*.jpg")):
        lf = p.parent.parent / "labels" / (p.stem + ".txt")
        if not lf.exists():
            cand = list(tud.rglob(p.stem + ".txt"))
            lf = cand[0] if cand else None
        n_boxes = 0
        if lf is not None and lf.exists():
            n_boxes = sum(1 for l in lf.read_text().splitlines() if l.strip())
        tud_recs.append({"source": "tud_gv", "original_path": str(p.relative_to(RAW)),
                         "original_class": "litter" if n_boxes else "",
                         "mapped_class": "litter" if n_boxes else "",
                         "split": "", "group_id": tud_group(p), "tile_info": "",
                         "_img": p, "_nboxes": n_boxes})
    if args.dry_run:
        tud_recs = tud_recs[::max(1, len(tud_recs) // 300)]
    groups_by_src["tud_gv"] = tud_recs

    # ---------------- Hagenbeek ----------------
    hag = RAW / "Platic-water hyathin" / "Annotated_images_labels" / "Annotated Images All"
    hag_recs = []
    for p in sorted((hag / "images").glob("*.jpg")):
        lf = hag / "labels" / (p.stem + ".txt")
        n_boxes = 0
        if lf.exists():
            n_boxes = sum(1 for l in lf.read_text().splitlines() if l.strip())
        hag_recs.append({"source": "hagenbeek", "original_path": str(p.relative_to(RAW)),
                         "original_class": "yolo:ff_litter,hyacinth,ent_litter" if n_boxes else "",
                         "mapped_class": "(per-box after tiling)", "split": "",
                         "group_id": hag_group(p), "tile_info": "",
                         "_img": p, "_nboxes": n_boxes})
    groups_by_src["hagenbeek"] = hag_recs

    # ---------------- filename analysis ----------------
    rows = []
    fml_days = Counter(Path(r["original_path"]).stem.split("_")[1] for r in fml_recs)
    fml_gsizes = Counter(r["group_id"] for r in fml_recs)
    rows.append({"source": "fml", "pattern": "image_YYYYMMDD_HHMMSS_micro.jpg",
                 "n_images": len(fml_recs), "n_groups": len(fml_gsizes),
                 "group_sizes_min": min(fml_gsizes.values()), "group_sizes_max": max(fml_gsizes.values()),
                 "group_sizes_median": int(np.median(list(fml_gsizes.values()))),
                 "note": f"groups = day x session (>{args.session_gap}s gap); days={dict(fml_days)}"})
    tg = Counter(r["group_id"] for r in tud_recs)
    rows.append({"source": "tud_gv", "pattern": "expNN_KKK.jpg",
                 "n_images": len(tud_recs), "n_groups": len(tg),
                 "group_sizes_min": min(tg.values()), "group_sizes_max": max(tg.values()),
                 "group_sizes_median": int(np.median(list(tg.values()))),
                 "note": "groups = experiment id prefix expNN; act as sequence ids [Confirmed from names]"})
    rows.append({"source": "hagenbeek", "pattern": "<hash>-<DJI|GXXX>.jpg",
                 "n_images": len(hag_recs), "n_groups": len(hag_recs),
                 "group_sizes_min": 1, "group_sizes_max": 1, "group_sizes_median": 1,
                 "note": "each original image its own group; tiling keeps group (D3d)"})
    with open(_out("filename_analysis.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)

    # ---------------- leakage audit (FML original split) ----------------
    print("Computing pHash for FML ...", flush=True)
    hashes = {}
    for r in fml_recs:
        h = phash(r["_img"])
        if h is not None:
            hashes[r["original_path"]] = pack(h)
    train_h = [(hashes[r["original_path"]], r["original_path"]) for r in fml_recs
               if r["split"] == "train" and r["original_path"] in hashes]
    leak_rows = []
    for r in fml_recs:
        if r["split"] not in ("val", "test") or r["original_path"] not in hashes:
            continue
        q = hashes[r["original_path"]]
        best = min((hamming(q, h), p) for h, p in train_h)
        leak_rows.append({"source": "fml", "path": r["original_path"], "split": r["split"],
                          "nearest_train_phash_hamming": best[0],
                          "nearest_train_path": best[1],
                          "near_dup_phash_le8": int(best[0] <= 8)})
    if leak_rows and not args.skip_clip:
        print("Computing CLIP ViT-B/32 embeddings for FML ...", flush=True)
        paths = [r["_img"] for r in fml_recs]
        embs = clip_embeddings(paths)
        idx = {r["original_path"]: i for i, r in enumerate(fml_recs)}
        tr_idx = [idx[r["original_path"]] for r in fml_recs if r["split"] == "train"]
        tr_e = embs[tr_idx]
        leak_by_path = {r["path"]: r for r in leak_rows}
        for i, r in enumerate(fml_recs):
            if r["split"] not in ("val", "test") or r["original_path"] not in leak_by_path:
                continue
            e = embs[i]
            cos = tr_e @ e
            j = int(cos.argmax())
            leak_by_path[r["original_path"]].update({
                "nearest_train_clip_cosine": round(float(cos[j]), 4),
                "near_dup_clip_cos_ge095": int(cos[j] >= 0.95)})
    with open(_out("leakage_audit.csv"), "w", newline="", encoding="utf-8") as f:
        if leak_rows:
            w = csv.DictWriter(f, fieldnames=list(leak_rows[0].keys()))
            w.writeheader(); w.writerows(leak_rows)

    # ---------------- group proposal summary ----------------
    group_rows = []
    for src, recs in groups_by_src.items():
        gs = defaultdict(lambda: Counter())
        for r in recs:
            gs[r["group_id"]][r["split"] or "unsplit"] += 1
        for gid, c in sorted(gs.items()):
            group_rows.append({"source": src, "group_id": gid, "n_images": sum(c.values()),
                               "original_split_mix": json.dumps(dict(c), sort_keys=True)})
    with open(_out("group_proposal.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(group_rows[0].keys()))
        w.writeheader(); w.writerows(group_rows)

    # ---------------- manifest ----------------
    with open(_out("manifest.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(MANIFEST_HEADER)
        for src in ("fml", "tud_gv", "hagenbeek"):
            for r in groups_by_src[src]:
                w.writerow([r["source"], r["original_path"], r["original_class"],
                            r["mapped_class"], r["split"], r["group_id"], r["tile_info"]])

    # ---------------- summary ----------------
    print("\n==== SUMMARY ====")
    for src, recs in groups_by_src.items():
        g = Counter(r["group_id"] for r in recs)
        print(f"{src}: {len(recs)} images, {len(g)} groups, sizes {min(g.values())}-{max(g.values())}")
    if leak_rows:
        n = len(leak_rows)
        ph8 = sum(r["near_dup_phash_le8"] for r in leak_rows)
        print(f"LEAKAGE (FML original split, {n} val+test images):")
        print(f"  pHash nearest-train hamming<=8 (near-duplicate): {ph8}/{n} = {ph8/n:.1%}")
        d = np.array([r["nearest_train_phash_hamming"] for r in leak_rows])
        print(f"  hamming median={np.median(d):.0f} p10={np.percentile(d,10):.0f}")
        if "nearest_train_clip_cosine" in leak_rows[0]:
            c95 = sum(r["near_dup_clip_cos_ge095"] for r in leak_rows)
            print(f"  CLIP nearest-train cosine>=0.95: {c95}/{n} = {c95/n:.1%}")
    print(f"\nOutputs in {AUDIT}")


if __name__ == "__main__":
    sys.exit(main())
