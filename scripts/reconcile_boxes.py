"""Reconcile aerial-source box counts through tiling (MERGE_REPORT §9 evidence).

Replays scripts/build_dataset.py tiling geometry verbatim (tile 640, stride 512,
MIN_AREA_FRAC 0.40) over the raw label files of Hagenbeek and Saigon, then ties
the chain raw -> boundary-dropped -> overlap-duplicated -> aerial-cap-dropped ->
final against the on-disk merged2 labels and manifest.csv.

Read-only. No rng needed: box/tile geometry is deterministic, kept-empty-tile
sampling affects no box counts, and cap drops are positive tiles only, so
pre-cap empty-tile counts equal the manifest's.
"""
import csv
from collections import Counter, defaultdict
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
PROC = ROOT / "data" / "processed"
TILE, STRIDE, MIN_AREA_FRAC = 640, 512, 0.40

SOURCES = {
    "hagenbeek": (
        RAW / "Platic-water hyathin" / "Annotated_images_labels" / "Annotated Images All" / "images",
        RAW / "Platic-water hyathin" / "Annotated_images_labels" / "Annotated Images All" / "labels",
        "hag_",
    ),
    "saigon": (
        PROC / "saigon_src" / "extracted" / "images",
        PROC / "saigon_src" / "extracted" / "labels",
        "sai_",
    ),
}


def yolo_boxes(lab_path):
    out = []
    if lab_path.exists():
        for line in lab_path.read_text().splitlines():
            t = line.split()
            if len(t) >= 5:
                out.append((int(t[0]), float(t[1]), float(t[2]), float(t[3]), float(t[4])))
    return out


def main():
    assign = {}
    for r in csv.DictReader(open(PROC / "audit" / "split_assignment.csv", encoding="utf-8")):
        assign[(r["source"], r["group_id"])] = r["split"]
    man = list(csv.DictReader(open(PROC / "manifest.csv", encoding="utf-8")))

    for source, (img_dir, lab_dir, prefix) in SOURCES.items():
        app_hist = Counter()                 # raw box -> number of tiles it appears in
        precap_boxes = Counter()             # split -> boxes (post duplication, pre cap)
        precap_pos_tiles = Counter()         # split -> positive tiles
        for p in sorted(img_dir.glob("*.[jJ][pP][gG]")):
            split = assign[(source, prefix + p.stem)]
            boxes = yolo_boxes(lab_dir / (p.stem + ".txt"))
            if not boxes:
                continue  # empty originals skipped before tiling in build_dataset.py
            W, H = Image.open(p).size
            nc = max(1, (W - TILE) // STRIDE + 1)
            nr = max(1, (H - TILE) // STRIDE + 1)
            tiles = []
            for r_ in range(nr):
                for c_ in range(nc):
                    x0 = min(c_ * STRIDE, W - TILE)
                    y0 = min(r_ * STRIDE, H - TILE)
                    tiles.append((x0, y0))
            tile_has_box = [False] * len(tiles)
            for cls, cx, cy, bw, bh in boxes:
                bx0, by0 = (cx - bw / 2) * W, (cy - bh / 2) * H
                bx1, by1 = bx0 + bw * W, by0 + bh * H
                n_hits = 0
                for i, (x0, y0) in enumerate(tiles):
                    ix0, iy0 = max(bx0, x0), max(by0, y0)
                    ix1, iy1 = min(bx1, x0 + TILE), min(by1, y0 + TILE)
                    inter = max(0.0, ix1 - ix0) * max(0.0, iy1 - iy0)
                    if inter / max(1e-9, (bx1 - bx0) * (by1 - by0)) >= MIN_AREA_FRAC:
                        n_hits += 1
                        tile_has_box[i] = True
                app_hist[n_hits] += 1
                precap_boxes[split] += n_hits
            precap_pos_tiles[split] += sum(tile_has_box)

        raw_total = sum(app_hist.values())
        dropped_boundary = app_hist[0]
        kept = raw_total - dropped_boundary
        overlap_gain = sum((n - 1) * c for n, c in app_hist.items() if n >= 1)
        precap_total = sum(precap_boxes.values())

        # final: straight from the on-disk merged2 labels (ground truth)
        final_boxes = Counter()
        for split in ("train", "val", "test"):
            for lf in (PROC / "merged2" / "labels" / split).glob(f"{prefix}*.txt"):
                final_boxes[split] += sum(1 for ln in lf.read_text().splitlines() if len(ln.split()) == 5)
        final_total = sum(final_boxes.values())

        # manifest tile counts (drops are positive tiles only -> pre-cap empties == final empties)
        man_rows = [r for r in man if r["source"] == source]
        man_boxes = Counter()
        man_pos_tiles = Counter()
        man_empty_tiles = Counter()
        for r in man_rows:
            man_boxes[r["split"]] += int(r["n_boxes"])
            if r["tile_info"].endswith("_empty"):
                man_empty_tiles[r["split"]] += 1
            elif r["tile_info"] != "background":
                man_pos_tiles[r["split"]] += 1

        cap_drop_boxes = {s: precap_boxes[s] - final_boxes[s] for s in ("train", "val", "test")}
        tile_drops = precap_pos_tiles["train"] - man_pos_tiles["train"]

        # cap-replay inputs (combined replay printed after both sources)
        fml_pos_train = sum(1 for r in man if r["source"] == "fml" and int(r["n_boxes"]) > 0 and r["split"] == "train")
        tud_train = sum(1 for r in man if r["source"] == "tud_gv" and r["split"] == "train")
        aerial_train_tiles_precap = precap_pos_tiles["train"] + man_empty_tiles["train"]
        n_train_before = fml_pos_train + tud_train + aerial_train_tiles_precap

        print(f"=== {source} ===")
        print(f"raw boxes                        : {raw_total:>7,}")
        print(f"- dropped at tile boundary (0 tiles >= 40%): {dropped_boundary:>6,}")
        print(f"= kept in >=1 tile               : {kept:>7,}")
        print(f"+ gained via overlap duplication : {overlap_gain:>7,}")
        print(f"= pre-cap total (all splits)     : {precap_total:>7,}")
        for s in ("train", "val", "test"):
            print(f"   pre-cap {s:<5}                 : {precap_boxes[s]:>7,}   (pos tiles {precap_pos_tiles[s]:>5})")
        for s in ("train", "val", "test"):
            d = cap_drop_boxes[s]
            if d:
                print(f"- cap-drop boxes ({s})           : {d:>7,}   (tile drops {tile_drops:>4})")
        print(f"= final total (on-disk merged2)  : {final_total:>7,}")
        for s in ("train", "val", "test"):
            print(f"   final {s:<5}                   : {final_boxes[s]:>7,}")
        print(f"appearance histogram (appearances: boxes): "
              + ", ".join(f"{n}:{c:,}" for n, c in sorted(app_hist.items())))
        dup_check = sum((n - 1) * c for n, c in app_hist.items() if n >= 1)
        tie = (kept + overlap_gain == precap_total
               and precap_total - sum(v for k, v in cap_drop_boxes.items()) == final_total
               and final_total == sum(man_boxes.values())
               and dup_check == overlap_gain)
        print(f"TIE-OUT (raw-drop+gain-cap == final == manifest): {'PASS' if tie else 'FAIL'}")
        print(f"cap replay inputs: n_train_before={n_train_before:,} "
              f"(non-aerial {fml_pos_train + tud_train:,} + aerial tiles {aerial_train_tiles_precap:,})")
        print()
        # stash for combined replay
        globals().setdefault("per_source", {})[source] = {
            "pos_train_precap": precap_pos_tiles["train"], "empty_train": man_empty_tiles["train"]}

    # combined aerial-cap replay (as in build_dataset.py lines 238-257)
    ps = globals()["per_source"]
    fml_pos_train = sum(1 for r in man if r["source"] == "fml" and int(r["n_boxes"]) > 0 and r["split"] == "train")
    tud_train = sum(1 for r in man if r["source"] == "tud_gv" and r["split"] == "train")
    pool = sum(v["pos_train_precap"] for v in ps.values())
    n_train_before = fml_pos_train + tud_train + pool + sum(v["empty_train"] for v in ps.values())
    n_non_aerial = n_train_before - pool
    target = int(0.35 * n_non_aerial / 0.65)
    n_drop = max(0, pool - target)
    print("=== combined aerial cap replay (D4) ===")
    print(f"aerial pool pre-cap (pos train tiles): {pool:,}")
    print(f"n_train_before={n_train_before:,}  n_non_aerial={n_non_aerial:,}")
    print(f"target = int(0.35*{n_non_aerial:,}/0.65) = {target:,}  ->  n_drop = {n_drop:,}")
    final_pos_train = sum(
        1 for r in man if r["source"] in ("hagenbeek", "saigon") and int(r["n_boxes"]) > 0 and r["split"] == "train")
    post_cap_train = n_train_before - n_drop
    print(f"actual positive aerial train tiles in manifest: {final_pos_train:,} "
          f"(=> actual drops {pool - final_pos_train:,})")
    print(f"post-cap pre-background share: {final_pos_train:,}/{post_cap_train:,} = "
          f"{final_pos_train / post_cap_train:.4%}  (final train incl. bg: "
          f"{final_pos_train:,}/7,368 = {final_pos_train / 7368:.4%})")


if __name__ == "__main__":
    main()
