"""Bounded CPU smoke test for the Roboat training path.

Proves the Run B training path end-to-end on a CPU-only machine:
  1. Deterministically (seed 42) copy 16 image+label pairs from
     data/processed/merged2/{images,labels}/train into
     data/processed/smoke/{images,labels}/{train,val} (8/8 split),
     stratified to include at least 4 sai_* (saigon_tiles) and
     2 hag_* (hagenbeek_tiles) stems when available.
  2. Write data/processed/smoke/smoke_c2.yaml (nc=2, litter/hyacinth,
     absolute POSIX paths).
  3. Train ultralytics YOLO initialized from the converted Run B
     checkpoint (Model_tiles_weights_converted.pt, nc=3 -> auto-transfer
     into nc=2) for exactly 1 epoch at imgsz 320, batch 4, device cpu,
     workers 0.
  4. Run model.val() on the same yaml and print the mAP50 values.

Prints "SMOKE PASS" and exits 0 on success; exits nonzero on any failure.
No raw torch.load is used here -- checkpoints are loaded by the
ultralytics YOLO loader only.

Usage (from repo root, using the CPU venv):
    .venv/Scripts/python.exe scripts/smoke_train.py
"""

from __future__ import annotations

import random
import shutil
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC = REPO_ROOT / "data" / "processed" / "merged2"
DST = REPO_ROOT / "data" / "processed" / "smoke"
CKPT = (
    REPO_ROOT
    / "data"
    / "processed"
    / "zenodo_12800597"
    / "converted"
    / "Model_tiles_weights_converted.pt"
)

SEED = 42
N_TOTAL = 16
N_TRAIN = 8  # same 8/8 split
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}
STRATIFY = {"sai": 4, "hag": 2}  # min counts required if available


def fail(msg: str) -> "NoReturn":  # type: ignore[name-defined]
    print(f"SMOKE FAIL: {msg}", file=sys.stderr)
    raise SystemExit(1)


def stratified_pairs(images: list[Path]) -> list[Path]:
    """Pick N_TOTAL images deterministically, honoring STRATIFY minimums.

    random.Random(SEED) is intentional: this shuffles a file list for a
    reproducible smoke dataset, not for any security purpose.
    """
    rng = random.Random(SEED)

    buckets: dict[str, list[Path]] = {"sai": [], "hag": [], "other": []}
    for p in images:
        prefix = p.stem.split("_", 1)[0]
        buckets.setdefault(prefix if prefix in buckets else "other", []).append(p)

    for bucket in buckets.values():
        bucket.sort()  # deterministic order before shuffling
        rng.shuffle(bucket)

    chosen: list[Path] = []
    for prefix, min_count in STRATIFY.items():
        have = buckets.get(prefix, [])
        if len(have) < min_count:
            print(
                f"SMOKE NOTE: only {len(have)} '{prefix}_*' images available "
                f"(< {min_count} requested); using all of them"
            )
        chosen.extend(have[:min_count])

    rest = [p for p in buckets["other"]]
    chosen.extend(rest[: max(0, N_TOTAL - len(chosen))])

    if len(chosen) < N_TOTAL:
        fail(f"only {len(chosen)} image+label pairs available, need {N_TOTAL}")

    rng.shuffle(chosen)  # deterministic shuffle before the 8/8 split
    return chosen[:N_TOTAL]


def build_dataset() -> Path:
    img_dir = SRC / "images" / "train"
    lbl_dir = SRC / "labels" / "train"
    if not img_dir.is_dir() or not lbl_dir.is_dir():
        fail(f"source dataset not found under {SRC}")

    images = sorted(p for p in img_dir.iterdir() if p.suffix.lower() in IMG_EXTS)
    chosen = stratified_pairs(images)

    # Every chosen image must have a label file, else YOLO treats it as
    # background and the stratification silently degrades.
    for p in chosen:
        lbl = lbl_dir / (p.stem + ".txt")
        if not lbl.is_file():
            fail(f"missing label file for {p.name}")

    # Fresh, deterministic smoke dataset.
    if DST.exists():
        shutil.rmtree(DST)
    for split in ("train", "val"):
        (DST / "images" / split).mkdir(parents=True)
        (DST / "labels" / split).mkdir(parents=True)

    for i, p in enumerate(chosen):
        split = "train" if i < N_TRAIN else "val"
        shutil.copy2(p, DST / "images" / split / p.name)
        shutil.copy2(
            lbl_dir / (p.stem + ".txt"), DST / "labels" / split / (p.stem + ".txt")
        )

    # Verify what we just built: pair counts + stratification floor.
    for split in ("train", "val"):
        n_img = len(list((DST / "images" / split).iterdir()))
        n_lbl = len(list((DST / "labels" / split).iterdir()))
        if n_img != N_TRAIN or n_lbl != N_TRAIN:
            fail(f"split '{split}' has {n_img} images / {n_lbl} labels, want 8/8")
    prefixes: dict[str, int] = {}
    for p in (DST / "images").rglob("*"):
        if p.suffix.lower() in IMG_EXTS:
            prefixes[p.stem.split("_", 1)[0]] = prefixes.get(p.stem.split("_", 1)[0], 0) + 1
    for prefix, min_count in STRATIFY.items():
        if prefixes.get(prefix, 0) < min_count:
            fail(f"stratification failed: {prefixes.get(prefix, 0)} '{prefix}_*' "
                 f"images copied, wanted >= {min_count}")
    print(f"SMOKE dataset: 8/8 train/val pairs, stem prefixes: {prefixes}")
    return DST


def write_yaml(dataset_dir: Path) -> Path:
    # Absolute POSIX paths so the yaml works regardless of cwd (incl. Colab).
    yaml_path = dataset_dir / "smoke_c2.yaml"
    yaml_path.write_text(
        f"path: {dataset_dir.as_posix()}\n"
        f"train: {(dataset_dir / 'images' / 'train').as_posix()}\n"
        f"val: {(dataset_dir / 'images' / 'val').as_posix()}\n"
        "nc: 2\n"
        "names: [litter, hyacinth]\n",
        encoding="utf-8",
    )
    print(f"SMOKE yaml written: {yaml_path.as_posix()}")
    return yaml_path


def train_and_val(yaml_path: Path) -> None:
    if not CKPT.is_file():
        fail(f"Run B checkpoint not found: {CKPT}")

    from ultralytics import YOLO  # local import: fails fast with a clear traceback

    model = YOLO(str(CKPT))
    model.train(
        data=str(yaml_path),
        epochs=1,
        imgsz=320,
        batch=4,
        device="cpu",
        workers=0,
        seed=SEED,
        deterministic=True,
        project=str(DST / "runs"),
        name="smoke",
        exist_ok=True,
        plots=False,
        val=False,  # val is run explicitly below on the same yaml
        verbose=False,
    )

    metrics = model.val(
        data=str(yaml_path),
        split="val",
        imgsz=320,
        batch=4,
        device="cpu",
        workers=0,
        plots=False,
        verbose=False,
    )

    map50 = float(metrics.box.map50)
    print(f"SMOKE val mAP50 (all): {map50:.4f}")

    if not (0.0 <= map50 <= 1.0):
        fail(f"mAP50 out of range: {map50}")


def main() -> int:
    t0 = time.time()
    dataset_dir = build_dataset()
    yaml_path = write_yaml(dataset_dir)
    train_and_val(yaml_path)
    print(f"SMOKE PASS (elapsed {time.time() - t0:.1f}s)")
    return 0


if __name__ == "__main__":
    try:
        SystemExit(main())
    except Exception as exc:  # noqa: BLE001 - smoke test reports any failure
        print(f"SMOKE FAIL: {type(exc).__name__}: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
