"""Deploy: build the Hailo INT8 calibration set (~1,024 images from TRAIN).

Samples from data/processed/manifest.csv train rows, balanced across sources and
guaranteeing a minimum share of hyacinth-containing images (the rare class).
Writes: data/processed/calib/images/*.jpg (hardlinks), calib_list.txt,
calib_manifest.csv. --dry-run prints the plan without writing.

Usage: python deploy/make_calib_set.py [--n 1024] [--min-hyacinth-share 0.25]
"""
import argparse
import csv
import random
from collections import Counter
from pathlib import Path
import shutil

SEED = 42
ROOT = Path(__file__).resolve().parents[1]
PROC = ROOT / "data" / "processed"
CALIB = PROC / "calib"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=1024)
    ap.add_argument("--min-hyacinth-share", type=float, default=0.25)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    rng = random.Random(SEED)

    man = [r for r in csv.DictReader(open(PROC / "manifest.csv", encoding="utf-8"))
           if r["split"] == "train"]
    hy = [r for r in man if "hyacinth" in r["mapped_class"]]
    rest = [r for r in man if "hyacinth" not in r["mapped_class"]]

    n_hy = min(len(hy), max(int(args.min_hyacinth_share * args.n), 1))
    n_rest = args.n - n_hy
    per_src = Counter(r["source"] for r in rest)
    picks = rng.sample(hy, n_hy)
    # proportional across sources for the remainder
    for src, cnt in per_src.items():
        pool = [r for r in rest if r["source"] == src]
        k = round(n_rest * cnt / len(rest))
        picks += rng.sample(pool, min(k, len(pool)))
    rng.shuffle(picks)

    counts = Counter(r["source"] for r in picks)
    hy_share = n_hy / len(picks)
    print(f"picked {len(picks)}: {dict(counts)}, hyacinth share {hy_share:.2f}")
    if args.dry_run:
        return
    (CALIB / "images").mkdir(parents=True, exist_ok=True)
    shutil.rmtree(CALIB, ignore_errors=True)
    (CALIB / "images").mkdir(parents=True, exist_ok=True)
    for r in picks:
        src = PROC / "merged2" / "images" / "train" / f"{r['final_stem']}.jpg"
        dst = CALIB / "images" / f"{r['final_stem']}.jpg"
        try:
            dst.hardlink_to(src)
        except OSError:
            shutil.copy2(src, dst)
    (CALIB / "calib_list.txt").write_text(
        "\n".join(str(p.relative_to(ROOT)).replace("\\", "/")
                  for p in sorted((CALIB / "images").glob("*.jpg"))) + "\n")
    with open(CALIB / "calib_manifest.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["source", "original_path", "final_stem", "has_hyacinth"])
        for r in picks:
            w.writerow([r["source"], r["original_path"], r["final_stem"],
                        int("hyacinth" in r["mapped_class"])])
    print(f"calib set at {CALIB}")


if __name__ == "__main__":
    main()
