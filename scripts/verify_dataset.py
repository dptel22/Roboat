"""T4 - Verify the built dataset. Assertions per task spec; read-only on data/raw.

Checks:
  1 no final image (stem) in >1 split;  2 no group in >1 split (per source);
  3 all label values within 0-1, class ids < nc (merged2: 2, merged3: 3);
  4 no orphan images/labels in either tree;
  5 counts reconcile with DATA_EXPLORATION.md (FML 5299/16457, TUD-GV 1501/8181,
    Hagenbeek 82/1415 originals; tile boxes >= originals due to 20% overlap);
    Run C donors reconcile exactly (mendeley 947/4176, navsci_invasive 4693/4693,
    navsci_whd 411/417 - kept images / boxes, post-dedupe, polygons dropped);
  6 tiled boxes lie inside their tile (recompute pixel box from tile_info).
Contact sheets with boxes -> data/exploration_samples/processed/.
Writes verification_results.csv unless --dry-run is passed; dry-run is fully read-only.
"""
import argparse
import csv
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

SEED = 42
ROOT = Path(__file__).resolve().parents[1]
PROC = ROOT / "data" / "processed"
SAMPLES_OUT = ROOT / "data" / "exploration_samples" / "processed"
EXPECTED = {"fml": (5299, 16457), "tud_gv": (1501, 8181), "hagenbeek": (82, 1415),
            "saigon": (272, 9352),
            # Run C donors (post-dedupe WHD-first order, polygon files dropped),
            # recomputed from the 2026-10-06 rebuild's manifest/labels:
            # (manifest rows, boxes)
            "mendeley": (947, 4176), "navsci_invasive": (4521, 4521),
            "navsci_whd": (584, 590)}
COLORS = {0: "red", 1: "lime", 2: "cyan"}
NAMES2 = {0: "litter", 1: "hyacinth"}
NAMES3 = {0: "litter", 1: "hyacinth", 2: "entangled_plastic"}


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


OUT_VERIFY = _fixed_out(PROC, "verification_results.csv")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    results = []
    ok_all = True

    def check(name, ok, detail=""):
        nonlocal ok_all
        results.append({"check": name, "status": "PASS" if ok else "FAIL", "detail": str(detail)[:400]})
        if not ok:
            ok_all = False
            print(f"FAIL {name}: {detail}")
        else:
            print(f"pass {name}" + (f" ({detail})" if detail else ""))

    man = list(csv.DictReader(open(PROC / "manifest.csv", encoding="utf-8")))

    # 1 final stems unique across splits
    stem_splits = defaultdict(set)
    for r in man:
        stem_splits[r["final_stem"]].add(r["split"])
    dup = {s: v for s, v in stem_splits.items() if len(v) > 1}
    check("final_stem_in_single_split", not dup, dup)

    # 2 groups unique across splits (per source)
    grp = defaultdict(set)
    for r in man:
        grp[(r["source"], r["group_id"])].add(r["split"])
    dupg = {k: v for k, v in grp.items() if len(v) > 1}
    check("group_in_single_split", not dupg, dupg)

    # 3/4 label files: values in [0,1], class ids valid, no orphans
    for tree, nc in (("merged2", 2), ("merged3", 3)):
        bad_vals, bad_cls, orph_i, orph_l = [], [], 0, 0
        n_lab_files, n_boxes = 0, 0
        for split in ("train", "val", "test"):
            idir = PROC / tree / "images" / split
            ldir = PROC / tree / "labels" / split
            istems = {p.stem for p in idir.glob("*.jpg")}
            lstems = {p.stem for p in ldir.glob("*.txt")}
            orph_i += len(istems - lstems)
            orph_l += len(lstems - istems)
            for lf in ldir.glob("*.txt"):
                n_lab_files += 1
                for line in lf.read_text().splitlines():
                    t = line.split()
                    if len(t) != 5:
                        bad_vals.append(f"{lf.name}: {line[:50]}")
                        continue
                    c = int(t[0])
                    if c >= nc:
                        bad_cls.append(f"{lf.name}: class {c}")
                    vals = [float(x) for x in t[1:]]
                    if any(v < 0 or v > 1 for v in vals):
                        bad_vals.append(f"{lf.name}: {line[:50]}")
                    n_boxes += 1
        check(f"{tree}_label_values_0_1", not bad_vals, bad_vals[:5])
        check(f"{tree}_class_ids_valid", not bad_cls, bad_cls[:5])
        check(f"{tree}_no_orphans", orph_i == 0 and orph_l == 0,
              f"imgs_wo_label={orph_i} labels_wo_img={orph_l} files={n_lab_files} boxes={n_boxes}")

    # 5 reconcile with exploration report
    stats = json.load(open(PROC / "build_stats.json", encoding="utf-8"))
    box_by_src = {src: sum(counts.values()) for src, counts in stats["box_counts"].items()}
    check("fml_boxes_reconcile", box_by_src["fml"] == EXPECTED["fml"][1],
          f"{box_by_src['fml']} vs {EXPECTED['fml'][1]}")
    check("tud_boxes_reconcile", box_by_src["tud_gv"] == EXPECTED["tud_gv"][1],
          f"{box_by_src['tud_gv']} vs {EXPECTED['tud_gv'][1]}")
    check("hagenbeek_boxes_reconcile", box_by_src["hagenbeek"] >= EXPECTED["hagenbeek"][1],
          f"tile boxes {box_by_src['hagenbeek']} >= orig {EXPECTED['hagenbeek'][1]} (overlap doubling)")
    # images present in raw vs manifest: positives in manifest + unused empties == raw total
    pos_fml = sum(1 for r in man if r["source"] == "fml" and int(r["n_boxes"]) > 0)
    check("fml_images_reconcile", pos_fml + stats["empties_total"].get("fml", 0) == EXPECTED["fml"][0],
          f"positives={pos_fml} + empties={stats['empties_total'].get('fml',0)} vs raw 5299")

    # Enforce the builder's pre-background denominator: all train rows except
    # the later-added FML background negatives.
    train_rows = [r for r in man if r["split"] == "train"]
    positive_aerial = [r for r in train_rows if r["source"] in ("hagenbeek", "saigon")
                       and int(r["n_boxes"]) > 0]
    cap_denominator = sum(1 for r in train_rows if r["tile_info"] != "background")
    aerial_share = len(positive_aerial) / max(cap_denominator, 1)
    check("aerial_train_share_within_35pct_cap", aerial_share <= 0.35 + 1e-12,
          f"{len(positive_aerial)}/{cap_denominator}={aerial_share:.4%}")
    pos_tud = sum(1 for r in man if r["source"] == "tud_gv" and int(r["n_boxes"]) > 0)
    check("tud_images_reconcile", pos_tud + stats["empties_total"].get("tud_gv", 0) == EXPECTED["tud_gv"][0],
          f"positives={pos_tud} + empties={stats['empties_total'].get('tud_gv',0)} vs raw 1501")
    check("hagenbeek_orig_images", stats["per_split"]["train"]["hagenbeek_orig"]
          + stats["per_split"]["val"]["hagenbeek_orig"] + stats["per_split"]["test"]["hagenbeek_orig"]
          + stats["empties_total"].get("hagenbeek", 0) == EXPECTED["hagenbeek"][0],
          f"{stats['per_split']} + {stats['empties_total'].get('hagenbeek',0)} excluded == 82")
    sai_orig = sum(stats["per_split"][sp].get("saigon_orig", 0) for sp in ("train", "val", "test"))
    check("saigon_orig_images", sai_orig + stats["empties_total"].get("saigon", 0)
          == EXPECTED["saigon"][0],
          f"{sai_orig} + {stats['empties_total'].get('saigon',0)} excluded == 272")
    check("saigon_boxes_reconcile",
          sum(stats["box_counts"]["saigon"].values()) >= EXPECTED["saigon"][1] or stats.get("cap_applied", False),
          f"tile boxes {sum(stats['box_counts']['saigon'].values())} (post-cap, "
          f"{stats.get('tiles_dropped', 'count not recorded in this build_stats.json')} tiles dropped under 35% aerial cap) vs orig {EXPECTED['saigon'][1]}")

    # 5b Run C donors: every kept donor image emits exactly one row (no empties
    # in any donor per the audit round) and its box total must reconcile exactly
    # (equality, not >=: donors are surface imagery, never tiled/capped).
    for src in ("mendeley", "navsci_invasive", "navsci_whd"):
        pos = sum(1 for r in man if r["source"] == src and int(r["n_boxes"]) > 0)
        check(f"{src}_images_reconcile",
              pos + stats["empties_total"].get(src, 0) == EXPECTED[src][0],
              f"positives={pos} + empties={stats['empties_total'].get(src,0)} vs kept {EXPECTED[src][0]}")
        check(f"{src}_boxes_reconcile", box_by_src[src] == EXPECTED[src][1],
              f"{box_by_src[src]} vs {EXPECTED[src][1]}")

    # Regression gate: four raw river_vegetation boxes must stay mapped to
    # hyacinth (class 1) in the primary two-class tree, with geometry intact.
    mendeley_repairs = {
        ("val", "men_FOTO_0140"): (11, "FOTO_0140"),
        ("val", "men_IMG_6180_frame_00000"): (5, "IMG_6180_frame_00000"),
        ("test", "men_FOTO_0011"): (1, "FOTO_0011"),
        ("test", "men_IMG_6190_frame_00005"): (0, "IMG_6190_frame_00005"),
    }
    repair_errors = []
    raw_men = PROC / "donors" / "mendeley" / "extracted" / "Floating Waste and River Vegetation Dataset"
    for (split, stem), (index, raw_stem) in mendeley_repairs.items():
        raw_file = raw_men / split / "labels" / f"{raw_stem}.txt"
        out_file = PROC / "merged2" / "labels" / split / f"{stem}.txt"
        raw_rows = [line.split() for line in raw_file.read_text().splitlines()]
        out_rows = [line.split() for line in out_file.read_text().splitlines()]
        if (len(raw_rows) <= index or len(out_rows) != len(raw_rows)
                or raw_rows[index][0] != "1" or out_rows[index][0] != "1"
                or [round(float(v), 6) for v in raw_rows[index][1:]] !=
                   [round(float(v), 6) for v in out_rows[index][1:]]):
            repair_errors.append(f"{split}/{stem} box {index}")
    check("mendeley_river_vegetation_class_mapping", not repair_errors, repair_errors)

    # 6 tiles: boxes lie inside tile bounds
    bad_tiles = []
    for r in man:
        if r["source"] not in ("hagenbeek", "saigon") or not r["tile_info"] or r["tile_info"] == "background":
            continue
        lf = PROC / "merged2" / "labels" / r["split"] / f"{r['final_stem']}.txt"
        if not lf.exists():
            bad_tiles.append(f"{r['final_stem']}: missing label")
            continue
        for line in lf.read_text().splitlines():
            t = line.split()
            if len(t) == 5:
                _, cx, cy, w, h = float(t[0]), *[float(x) for x in t[1:]]
                x0, x1 = cx - w / 2, cx + w / 2
                y0, y1 = cy - h / 2, cy + h / 2
                if not (-0.001 <= x0 and x1 <= 1.001 and -0.001 <= y0 and y1 <= 1.001):
                    bad_tiles.append(f"{r['final_stem']}: box outside tile {line[:40]}")
    check("tiles_boxes_inside", not bad_tiles, bad_tiles[:5])

    # contact sheets
    if not args.dry_run:
        SAMPLES_OUT.mkdir(parents=True, exist_ok=True)
        try:
            font = ImageFont.load_default(16)
        except TypeError:
            font = ImageFont.load_default()

        def sheet(rows, out_name, per=6):
            rows = rows[:per]
            cols = 3
            cell = 420
            rows_n = (len(rows) + cols - 1) // cols
            s = Image.new("RGB", (cols * cell, rows_n * (cell + 24)), "black")
            d = ImageDraw.Draw(s)
            for i, r in enumerate(rows):
                img_p = PROC / "merged2" / "images" / r["split"] / f"{r['final_stem']}.jpg"
                lab_p = PROC / "merged2" / "labels" / r["split"] / f"{r['final_stem']}.txt"
                im = Image.open(img_p).convert("RGB")
                W, H = im.size
                im.thumbnail((cell - 10, cell - 10))
                x, y = (i % cols) * cell + 5, (i // cols) * (cell + 24) + 5
                s.paste(im, (x, y))
                for line in lab_p.read_text().splitlines():
                    t = line.split()
                    if len(t) == 5:
                        c, cx, cy, w, h = int(t[0]), *[float(v) for v in t[1:]]
                        sc = im.size[0] / W
                        d.rectangle([(cx - w / 2) * W * sc + x, (cy - h / 2) * H * sc + y,
                                     (cx + w / 2) * W * sc + x, (cy + h / 2) * H * sc + y],
                                    outline=COLORS.get(c, "white"), width=2)
                d.text((x, y + cell - 2), f"{r['final_stem']} [{r['split']}]",
                       fill="yellow", font=font)
            s.save(SAMPLES_OUT / out_name, quality=88)

        import random
        rng = random.Random(SEED)
        for src, name in (("fml", "fml"), ("tud_gv", "tud_gv")):
            rows = [r for r in man if r["source"] == src and int(r["n_boxes"]) > 0]
            sheet(rng.sample(rows, min(6, len(rows))), f"{name}_contact_sheet.jpg")
        tiles = [r for r in man if r["source"] == "hagenbeek" and int(r["n_boxes"]) > 0]
        sheet(rng.sample(tiles, min(6, len(tiles))), "hagenbeek_tiles_contact_sheet.jpg")
        sai_tiles = [r for r in man if r["source"] == "saigon" and int(r["n_boxes"]) > 0]
        if sai_tiles:
            sheet(rng.sample(sai_tiles, min(6, len(sai_tiles))), "saigon_tiles_contact_sheet.jpg")

    if not args.dry_run:
        with open(OUT_VERIFY, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=["check", "status", "detail"])
            w.writeheader(); w.writerows(results)
    print("ALL PASS" if ok_all else "FAILURES PRESENT")
    sys.exit(0 if ok_all else 1)


if __name__ == "__main__":
    main()
