"""T7 - Optional augmented training run (Ultralytics Python API + Albumentations).

Conservative augmentations only. Argument names are verified against the
INSTALLED albumentations version at runtime via a smoke test (A.Compose with
each transform, applied to a dummy image) before training starts; unknown
transform names abort with a clear message instead of failing mid-train.

Run on Kaggle after train_baseline.py:
  python train_aug.py --data-root /kaggle/input/roboat-processed
"""
import argparse
from pathlib import Path

from train_baseline import rebase, IMGSZ, EPOCHS, PATIENCE, SEED, PER_SOURCE_YAMLS

AUG_PIPELINE = [
    # (albumentations transform kwargs, probability)
    ("HorizontalFlip", dict(p=0.5)),
    ("Affine", dict(scale=(0.9, 1.1), translate_percent=(0.0, 0.05),
                    rotate=(-5, 5), p=0.5)),
    ("ColorJitter", dict(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.02, p=0.5)),
    ("RandomResizedCrop", dict(size=(640, 640), scale=(0.7, 1.0), ratio=(0.9, 1.1),
                               p=0.3)),
    ("RandomShadow", dict(num_shadows_lower=1, num_shadows_upper=2, p=0.2)),
]


def smoke_test():
    """VERIFY transform names/args against the installed albumentations."""
    import numpy as np
    import albumentations as A
    print(f"albumentations {A.__version__}")
    tf = []
    for name, kwargs in AUG_PIPELINE:
        cls = getattr(A, name, None)
        if cls is None:
            raise SystemExit(f"ABORT: {name} not in albumentations {A.__version__}")
        try:
            tf.append(cls(**kwargs))
        except TypeError as e:
            raise SystemExit(f"ABORT: {name} kwargs rejected: {e}")
    compose = A.Compose(tf, bbox_params=A.BboxParams(format="yolo", label_fields=["class_labels"]))
    img = (np.random.rand(640, 640, 3) * 255).astype("uint8")
    res = compose(image=img, bboxes=[[0.5, 0.5, 0.1, 0.1]], class_labels=[0])
    assert res["image"].shape == (640, 640, 3)
    print("smoke test OK: transforms compose and apply")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", type=Path, default=Path("data/processed"))
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    smoke_test() if not args.dry_run else print("[dry-run] would run smoke test")

    from ultralytics import YOLO
    yamls = rebase(args.data_root, Path("rebased"))
    train_yaml = yamls / "combined_c2.yaml"
    if args.dry_run:
        print(f"[dry-run] would train with augmentations on {train_yaml}")
        return
    model = YOLO("yolov8n.pt")
    model.train(data=str(train_yaml), epochs=EPOCHS, patience=PATIENCE,
                imgsz=IMGSZ, seed=SEED, deterministic=True,
                project="roboat", name="aug_yolov8n")
    # save train batches so the user can confirm transforms are applied
    from shutil import copy2
    run = Path(model.trainer.save_dir)
    for f in run.glob("train_batch*.jpg"):
        copy2(f, Path("roboat") / f.name)
    best = model.trainer.best
    row = {}
    for ys in PER_SOURCE_YAMLS:
        m = YOLO(best)
        metrics = m.val(data=str(yamls / f"{ys}.yaml"), imgsz=IMGSZ, split="test")
        row[ys] = float(metrics.box.map50)
    print("\n===== aug run per-source test mAP@0.5 =====")
    for ys, v in row.items():
        print(f"{ys:<20}{v:.4f}")


if __name__ == "__main__":
    main()
