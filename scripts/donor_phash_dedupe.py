# NOTE: authored to run from data/processed/donors/ (HERE-relative paths for the
# hash cache and donor extracts); kept here for reproducibility of the 2026-10-06
# gated dataset round. Run it from that directory or fix HERE first.
"""Donor audit step 2 (read-only): pHash near-duplicate dedupe, hd<=8, keep first-seen.

Reuses phash()/pack()/hamming() from scripts/audit_splits.py (imported, not copied).
No torch, no network. Hash cache (_phash_cache.npz) makes reruns incremental.

FIRST-SEEN ORDER (stated in _dedupe_result.json):
  1. existing originals (FML -> TUD -> hagenbeek -> saigon; the frozen CP4 incumbent wins)
  2. mendeley  (train -> val -> test, filename-sorted)
  3. navsci_whd      (train -> valid -> test, filename-sorted)   [USER DECISION 2026-10-06:
     WHD's clean unaugmented originals win over invasive's pre-baked augmented copies]
  4. navsci_invasive (train -> valid -> test, filename-sorted)
A donor image within hd<=8 of ANY already-kept image is dropped and classified
(vs_existing / within_donor / cross_donor:<donor>).
"""
import json
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
from audit_splits import phash, pack, hamming  # noqa: E402  (repo's own functions)

CACHE = HERE / "_phash_cache.npz"
OUT = HERE / "_dedupe_result.json"
HD_MAX = 8

MEN = ROOT / "data/processed/donors/mendeley/extracted/Floating Waste and River Vegetation Dataset"
GROUPS = [
    # (donor_label, split, [image paths])  -- existing first (incumbent wins)
    ("existing_fml", None, sorted((ROOT / "data/raw/120969/fml_version2/full_dataset/images").glob("*/*.jpg"))),
    ("existing_tud_gv", None, sorted((ROOT / "data/raw/TUD-GV Dataset for Floating Litter Detection").rglob("*.jpg"))),
    ("existing_hagenbeek", None, sorted((ROOT / "data/raw/Platic-water hyathin/Annotated_images_labels/Annotated Images All/images").glob("*.jpg"))),
    ("existing_saigon", None, sorted((ROOT / "data/processed/saigon_src/extracted").rglob("*.jpg"))),
    ("mendeley", "train", sorted((MEN / "train/images").glob("*.jpg"))),
    ("mendeley", "val", sorted((MEN / "val/images").glob("*.jpg"))),
    ("mendeley", "test", sorted((MEN / "test/images").glob("*.jpg"))),
    ("navsci_whd", "train", sorted((HERE / "navsci_whd/train/images").glob("*.jpg"))),
    ("navsci_whd", "valid", sorted((HERE / "navsci_whd/valid/images").glob("*.jpg"))),
    ("navsci_whd", "test", sorted((HERE / "navsci_whd/test/images").glob("*.jpg"))),
    ("navsci_invasive", "train", sorted((HERE / "navsci_invasive/train/images").glob("*.jpg"))),
    ("navsci_invasive", "valid", sorted((HERE / "navsci_invasive/valid/images").glob("*.jpg"))),
    ("navsci_invasive", "test", sorted((HERE / "navsci_invasive/test/images").glob("*.jpg"))),
]


def load_cache():
    if CACHE.exists():
        z = np.load(CACHE, allow_pickle=False)
        return dict(zip(z["paths"].tolist(), z["hashes"].tolist()))
    return {}


def save_cache(cache):
    paths = np.array(sorted(cache))  # unicode array, no pickle needed
    hashes = np.array([cache[p] for p in sorted(cache)], dtype=np.uint64)
    np.savez(CACHE, paths=paths, hashes=hashes)


def popcount_hd(q: np.uint64, bank: np.ndarray) -> np.ndarray:
    """Vectorized equivalent of audit_splits.hamming() (verified against it below)."""
    return np.bitwise_count(np.bitwise_xor(bank, q)).astype(np.int32)


def main():
    t0 = time.time()
    cache = load_cache()
    print(f"cache: {len(cache)} hashes preloaded", flush=True)

    # ---- hash everything (incremental cache) ----
    todo = [(lbl, sp, p) for lbl, sp, paths in GROUPS for p in paths]
    print(f"total images to hash: {len(todo)}", flush=True)
    new = 0
    for i, (lbl, sp, p) in enumerate(todo):
        key = str(p.relative_to(ROOT)).replace("\\", "/")
        if key not in cache:
            h = phash(p)
            if h is None:
                cache[key] = np.uint64(0)  # sentinel; tracked as unreadable below
                print(f"  PHASH FAILED: {key}", flush=True)
            else:
                cache[key] = np.uint64(pack(h))
            new += 1
        if new and new % 500 == 0 and i % 500 == 0:
            print(f"  hashed {i + 1}/{len(todo)} ({time.time() - t0:.0f}s)", flush=True)
            save_cache(cache)
    save_cache(cache)
    print(f"hashing done: {new} new, {time.time() - t0:.0f}s", flush=True)

    # ---- verify vectorized hamming == repo hamming on 10k random pairs ----
    rng = np.random.default_rng(42)
    keys = sorted(cache)
    vals = np.array([cache[k] for k in keys], dtype=np.uint64)
    ia, ib = rng.integers(0, len(vals), size=(2, 10000))
    vec = np.bitwise_count(np.bitwise_xor(vals[ia], vals[ib])).astype(np.int64)
    ref = np.fromiter((hamming(int(vals[a]), int(vals[b])) for a, b in zip(ia, ib)), dtype=np.int64)
    assert (vec == ref).all(), "vectorized hamming != audit_splits.hamming"
    print(f"verified: vectorized popcount hamming == audit_splits.hamming on 10,000 random pairs", flush=True)

    # ---- first-seen dedupe ----
    bank_keys, bank_vals = [], []
    drops = []
    per_donor = defaultdict(lambda: {"kept": 0, "dropped": Counter(), "nearest_hd_to_bank": []})
    existing_end = sum(1 for lbl, _, _ in GROUPS if lbl.startswith("existing_"))
    for gi, (lbl, sp, paths) in enumerate(GROUPS):
        for p in paths:
            key = str(p.relative_to(ROOT)).replace("\\", "/")
            q = cache[key]
            per_donor[lbl]["n_images"] = per_donor[lbl].get("n_images", 0) + 1
            if len(bank_vals):
                hd = popcount_hd(q, np.array(bank_vals, dtype=np.uint64))
                j = int(hd.argmin())
                minh = int(hd[j])
                per_donor[lbl]["nearest_hd_to_bank"].append(minh)
                is_existing_match = j < existing_end
                if minh <= HD_MAX and not (lbl.startswith("existing_")):
                    match_lbl = bank_keys[j][0]
                    match_stem = bank_keys[j][1]
                    cat = ("vs_existing" if is_existing_match
                           else "within_donor" if match_lbl == lbl
                           else f"cross_donor:{match_lbl}")
                    per_donor[lbl]["dropped"][cat] += 1
                    drops.append({"donor": lbl, "split": sp, "stem": p.stem, "path": key,
                                  "matched_donor": match_lbl, "matched_stem": match_stem,
                                  "matched_path": bank_keys[j][2], "hamming": minh, "category": cat})
                    continue
            bank_keys.append((lbl, p.stem, key))
            bank_vals.append(q)
            per_donor[lbl]["kept"] += 1
        print(f"  processed {lbl}" + (f" [{sp}]" if sp else "") + f" @ {time.time() - t0:.0f}s", flush=True)

    # ---- join records for post-dedupe mapped-class yields ----
    recs = json.load(open(HERE / "_audit_records.json"))
    rec_by_rel = {r["img_rel"]: r for r in recs}
    kept_rels = {bk[2] for bk in bank_keys if not bk[0].startswith("existing_")}
    yields = {}
    for donor in ("mendeley", "navsci_invasive", "navsci_whd"):
        pre = Counter(); post = Counter()
        n_pre = n_post = 0
        for r in recs:
            if r["donor"] != donor:
                continue
            n_pre += 1
            for k, v in r["mapped_counts"].items():
                pre[k] += v
            if r["img_rel"] in kept_rels:
                n_post += 1
                for k, v in r["mapped_counts"].items():
                    post[k] += v
        yields[donor] = {"images_pre": n_pre, "images_post_dedupe": n_post,
                         "mapped_boxes_pre": dict(pre), "mapped_boxes_post_dedupe": dict(post)}

    # ---- Navsci overlap families (NNN shared between invasive and whd) ----
    fam = {}
    for r in recs:
        s = r["src_stem"]
        if s.startswith("water_hyacinth_"):
            fam.setdefault(s, {"navsci_invasive": [], "navsci_whd": []})[r["donor"]].append(r["split"])
    for d_ in drops:
        s = d_["stem"].split("_jpg.rf.")[0].split("_JPG.rf.")[0]
        if s in fam:
            fam[s][d_["donor"] + "_dropped"] = fam[s].get(d_["donor"] + "_dropped", 0) + 1
    shared = {k: v for k, v in fam.items() if v["navsci_invasive"] and v["navsci_whd"]}

    result = {
        "hd_threshold": HD_MAX,
        "first_seen_order": ["existing originals (fml,tud_gv,hagenbeek,saigon)",
                             "mendeley", "navsci_whd", "navsci_invasive"],
        "total_images_hashed": len(todo),
        "phash_failures": [k for k, v in cache.items() if v == 0],
        "hamming_impl_check": "vectorized np.bitwise_count == audit_splits.hamming on 10,000 random pairs (assert passed)",
        "per_donor": {lbl: {"n_images": v.get("n_images", 0), "kept": v["kept"],
                            "dropped_by_category": dict(v["dropped"]),
                            "nearest_hd_to_bank_min": int(min(v["nearest_hd_to_bank"])) if v["nearest_hd_to_bank"] else None,
                            "nearest_hd_to_bank_p10": int(np.percentile(v["nearest_hd_to_bank"], 10)) if v["nearest_hd_to_bank"] else None,
                            "nearest_hd_to_bank_median": int(np.median(v["nearest_hd_to_bank"])) if v["nearest_hd_to_bank"] else None}
                      for lbl, v in per_donor.items()},
        "mapped_class_yields": yields,
        "drops": drops,
        "navsci_shared_families": {k: {"invasive_files": len(v["navsci_invasive"]), "whd_files": len(v["navsci_whd"]),
                                       "invasive_dropped": v.get("navsci_invasive_dropped", 0),
                                       "whd_dropped": v.get("navsci_whd_dropped", 0)}
                                   for k, v in sorted(shared.items())},
    }
    OUT.write_text(json.dumps(result, indent=1))
    print(f"\n==== DEDUPE SUMMARY (hd<={HD_MAX}) ====")
    for lbl, v in per_donor.items():
        print(f"{lbl}: n={v.get('n_images', 0)} kept={v['kept']} dropped={dict(v['dropped'])}")
    print(f"drops total: {len(drops)} -> {OUT}")


if __name__ == "__main__":
    main()
