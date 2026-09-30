"""Convert the legacy (ultralytics 8.0.36) Zenodo Saigon checkpoints to current-format .pt files.

A raw torch unpickle fails on these checkpoints (saved with the old `ultralytics.yolo.*`
module layout), but current ultralytics loads them directly: `ultralytics.nn.tasks.torch_safe_load`
wraps the pickle load in temporary_modules(), remapping "ultralytics.yolo.utils" -> "ultralytics.utils".
This script therefore uses the YOLO() loader path only (no direct torch usage anywhere) —
prints names/nc per checkpoint, re-saves each in the current format, then re-loads the
converted file and asserts names/nc match before printing PASS.
"""
import sys
from pathlib import Path

from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[1]
INPUT_DIR = ROOT / "data" / "processed" / "zenodo_12800597" / "extracted" / "trained_weights"
OUTPUT_DIR = ROOT / "data" / "processed" / "zenodo_12800597" / "converted"
EXPECTED = ("Model_resize_weights.pt", "Model_tiles_weights.pt")


def main():
    if not INPUT_DIR.is_dir():
        print(f"ERROR: input directory not found: {INPUT_DIR}", file=sys.stderr)
        return 1
    missing = [name for name in EXPECTED if not (INPUT_DIR / name).is_file()]
    if missing:
        print(f"ERROR: missing checkpoint(s) in {INPUT_DIR}: {', '.join(missing)}", file=sys.stderr)
        return 1
    checkpoints = sorted(INPUT_DIR.glob("*.pt"))
    if not checkpoints:
        print(f"ERROR: no *.pt checkpoints found in {INPUT_DIR}", file=sys.stderr)
        return 1

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    failures = []
    for ckpt in checkpoints:
        print("=" * 60)
        print(ckpt.name)
        model = YOLO(str(ckpt))
        names = model.names
        nc = model.model.yaml.get("nc")
        print(f"  model.names = {names}")
        print(f"  model.nc    = {nc}")

        out_path = OUTPUT_DIR / f"{ckpt.stem}_converted.pt"
        model.save(out_path)

        reloaded = YOLO(str(out_path))
        if reloaded.names != names or reloaded.model.yaml.get("nc") != nc:
            failures.append(ckpt.name)
            print(f"  FAIL: re-loaded {out_path.name} does not match source names/nc", file=sys.stderr)
            continue
        print(f"  saved + round-trip verified: {out_path}")
        print("  PASS")

    if failures:
        print(f"ERROR: {len(failures)} conversion(s) failed round-trip check: {', '.join(failures)}", file=sys.stderr)
        return 1
    print("=" * 60)
    print(f"All {len(checkpoints)} checkpoint(s) converted to {OUTPUT_DIR}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
