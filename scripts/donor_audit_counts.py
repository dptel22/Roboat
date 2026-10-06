# NOTE: authored to run from data/processed/donors/ (HERE-relative paths for the
# donor extracts and output JSONs); kept here for reproducibility of the
# 2026-10-06 gated dataset round. Run it from that directory or fix HERE first.
"""Donor audit step 1 (read-only): per-donor label parsing, USER-DECISION class
mapping, sanity checks. No torch, no network.

Outputs (next to this script):
  _audit_records.json   - one record per donor image (stem, split, raw/mapped box counts)
  _audit_box_counts.json- per-donor/per-split/per-class summary + sanity findings

USER DECISION 2026-10-06 (binding):
  mendeley j26w4m645z.2 : floating_waste(0) -> litter(0); river_vegetation(1) -> hyacinth(1)
  navsci invasive       : Water Lettuce(0) + water_hyacinth(1) -> ALL hyacinth(1)
  navsci whd            : water_hyacinth(0) -> hyacinth(1)
Class-id ordering DISCOVERED from each archive's own data.yaml (not assumed).
"""
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]

DONORS = {
    "mendeley": {
        "dir": HERE / "mendeley" / "extracted" / "Floating Waste and River Vegetation Dataset",
        "splits": ("train", "val", "test"),
        "expected_ids": {0, 1},
        "id_names": {0: "floating_waste", 1: "river_vegetation"},
        "map": {0: "litter", 1: "hyacinth"},
    },
    "navsci_invasive": {
        "dir": HERE / "navsci_invasive",
        "splits": ("train", "valid", "test"),
        "expected_ids": {0, 1},
        "id_names": {0: "Water Lettuce", 1: "water_hyacinth"},
        "map": {0: "hyacinth", 1: "hyacinth"},
    },
    "navsci_whd": {
        "dir": HERE / "navsci_whd",
        "splits": ("train", "valid", "test"),
        "expected_ids": {0},
        "id_names": {0: "water_hyacinth"},
        "map": {0: "hyacinth"},
    },
}

RF_RE = re.compile(r"^(?P<src>.+)_(?:jpg|JPG|png|PNG)\.rf\.[0-9a-f]{32}$")


def rf_source_stem(stem: str) -> str:
    """Strip Roboflow `_jpg.rf.<32hex>` suffix -> original source stem."""
    m = RF_RE.match(stem)
    return m.group("src") if m else stem


def parse_label(path: Path):
    """Return (boxes, problems). boxes=(cls,cx,cy,w,h). Mirrors build_dataset.yolo_boxes
    strictness but records what it would silently swallow."""
    boxes, problems = [], []
    if not path.exists():
        return boxes, ["missing_label"]
    for i, line in enumerate(path.read_text(encoding="utf-8", errors="replace").splitlines(), 1):
        t = line.split()
        if not t:
            continue
        if len(t) < 5:
            problems.append(f"line{i}:short")
            continue
        try:
            c = int(t[0])
            cx, cy, w, h = (float(v) for v in t[1:5])
        except ValueError:
            problems.append(f"line{i}:parse")
            continue
        if not (0.0 <= cx <= 1.0 and 0.0 <= cy <= 1.0 and 0.0 <= w <= 1.0 and 0.0 <= h <= 1.0):
            problems.append(f"line{i}:coord_range")
        if w <= 0.0 or h <= 0.0:
            problems.append(f"line{i}:degenerate_wh")
        boxes.append((c, cx, cy, w, h))
    return boxes, problems


def main():
    records, summary = [], {}
    for donor, cfg in DONORS.items():
        d = cfg["dir"]
        imgs_by_split, pair_problems = {}, []
        per_split = {}
        raw_tot, map_tot = Counter(), Counter()
        prob_counter = Counter()
        boxes_per_file = Counter()
        src_stems = defaultdict(set)          # source stem -> {split}
        for split in cfg["splits"]:
            img_dir = d / split / "images"
            lab_dir = d / split / "labels"
            jpgs = sorted(p for p in img_dir.iterdir() if p.suffix.lower() == ".jpg")
            txts = {p.stem for p in lab_dir.glob("*.txt")}
            imgs_by_split[split] = jpgs
            no_lab = sorted(p.stem for p in jpgs if p.stem not in txts)
            no_img = sorted(txts - {p.stem for p in jpgs})
            if no_lab:
                pair_problems.append(f"{split}: {len(no_lab)} imgs_without_label")
            if no_img:
                pair_problems.append(f"{split}: {len(no_img)} labels_without_img")
            n_boxes_split = 0
            for p in jpgs:
                boxes, probs = parse_label(lab_dir / (p.stem + ".txt"))
                for pb in probs:
                    prob_counter[pb.split(":")[0]] += 1
                rc = Counter(b[0] for b in boxes)
                for cid, n in rc.items():
                    if cid not in cfg["expected_ids"]:
                        prob_counter[f"unexpected_class_id_{cid}"] += n
                raw_tot.update(rc)
                mc = Counter()
                for c, n in rc.items():
                    mc[cfg["map"].get(c, f"UNMAPPED_{c}")] += n
                map_tot.update(mc)
                boxes_per_file[len(boxes)] += 1
                n_boxes_split += len(boxes)
                src = rf_source_stem(p.stem)
                src_stems[src].add(split)
                records.append({
                    "donor": donor, "split": split, "stem": p.stem,
                    "img_rel": str((d / split / "images" / p.name).relative_to(ROOT)).replace("\\", "/"),
                    "n_boxes": len(boxes),
                    "raw_counts": {str(k): v for k, v in sorted(rc.items())},
                    "mapped_counts": dict(mc),
                    "src_stem": src,
                    "problems": probs,
                })
            per_split[split] = {"images": len(jpgs), "labels": len(txts), "boxes": n_boxes_split}
        # version files per source stem (within whole donor, not per split)
        vcount = Counter(r["src_stem"] for r in records if r["donor"] == donor)
        vhist = Counter(vcount.values())
        cross_split_srcs = sorted(s for s, sp in src_stems.items() if len(sp) > 1)
        summary[donor] = {
            "dir": str(d),
            "data_yaml_ids": cfg["id_names"],
            "mapping_applied": {str(k): v for k, v in cfg["map"].items()},
            "per_split": per_split,
            "total_images": sum(v["images"] for v in per_split.values()),
            "total_boxes": sum(v["boxes"] for v in per_split.values()),
            "raw_boxes_by_id": {str(k): v for k, v in sorted(raw_tot.items())},
            "mapped_boxes": dict(map_tot),
            "pair_problems": pair_problems,
            "label_problems": dict(prob_counter),
            "boxes_per_file_hist": {str(k): v for k, v in sorted(boxes_per_file.items())},
            "distinct_source_stems": len(vcount),
            "files_per_source_stem_hist": {str(k): v for k, v in sorted(vhist.items())},
            "source_stems_in_multiple_own_splits": cross_split_srcs[:20],
            "n_source_stems_in_multiple_own_splits": len(cross_split_srcs),
        }
    (HERE / "_audit_records.json").write_text(json.dumps(records, indent=0))
    (HERE / "_audit_box_counts.json").write_text(json.dumps(summary, indent=2))
    for donor, s in summary.items():
        print(f"== {donor}")
        print(f"   images={s['total_images']} boxes={s['total_boxes']} "
              f"per_split={ {k: (v['images'], v['boxes']) for k, v in s['per_split'].items()} }")
        print(f"   raw_boxes_by_id={s['raw_boxes_by_id']}  (names {s['data_yaml_ids']})")
        print(f"   mapped_boxes={s['mapped_boxes']}")
        print(f"   pair_problems={s['pair_problems']} label_problems={s['label_problems']}")
        print(f"   distinct_source_stems={s['distinct_source_stems']} "
              f"files_per_stem_hist={s['files_per_source_stem_hist']} "
              f"cross_split_srcs={s['n_source_stems_in_multiple_own_splits']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
