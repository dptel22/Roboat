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

SEED = 42
ROOT = Path(__file__).resolve().parents[1]
PROC = ROOT / "data" / "processed"
CALIB = PROC / "calib"


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


OUT_CALIB_MANIFEST = _fixed_out(CALIB, "calib_manifest.csv")


def allocate_exact(total, capacities):
    """Proportional largest-remainder allocation, capped by source capacity."""
    keys = sorted(capacities)
    allocation = {key: 0 for key in keys}
    remaining = min(total, sum(capacities.values()))
    while remaining:
        active = [key for key in keys if allocation[key] < capacities[key]]
        if not active:
            break
        weight = sum(capacities[key] for key in active)
        quotas = {key: remaining * capacities[key] / weight for key in active}
        floors = {key: min(capacities[key] - allocation[key], int(quotas[key]))
                  for key in active}
        placed = sum(floors.values())
        for key, value in floors.items():
            allocation[key] += value
        remaining -= placed
        if not remaining:
            break
        order = sorted(active, key=lambda key: (-(quotas[key] - int(quotas[key])), key))
        for key in order:
            if remaining and allocation[key] < capacities[key]:
                allocation[key] += 1
                remaining -= 1
    return allocation

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
    picks = rng.sample(hy, n_hy)
    # Largest-remainder allocation avoids losing samples to independent rounding.
    per_src = Counter(r["source"] for r in rest)
    allocation = allocate_exact(n_rest, per_src)
    for src, k in allocation.items():
        pool = [r for r in rest if r["source"] == src]
        picks += rng.sample(pool, min(k, len(pool)))
    rng.shuffle(picks)

    counts = Counter(r["source"] for r in picks)
    hy_share = n_hy / len(picks)
    print(f"picked {len(picks)}: {dict(counts)}, hyacinth share {hy_share:.2f}")
    if len(picks) != args.n:
        raise SystemExit(f"could select only {len(picks)} of requested {args.n} calibration images")
    if args.dry_run:
        return
    (CALIB / "images").mkdir(parents=True, exist_ok=True)
    # Replace only generated image files; preserve any unrelated files in calib/.
    for old in (CALIB / "images").glob("*.jpg"):
        old.unlink()
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
    with open(OUT_CALIB_MANIFEST, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["source", "original_path", "final_stem", "has_hyacinth"])
        for r in picks:
            w.writerow([r["source"], r["original_path"], r["final_stem"],
                        int("hyacinth" in r["mapped_class"])])
    print(f"calib set at {CALIB}")


if __name__ == "__main__":
    main()
