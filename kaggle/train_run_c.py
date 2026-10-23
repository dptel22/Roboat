"""Run C training with auditable data checks and lake-condition augmentation.

Examples (Kaggle):
  python train_run_c.py --mode audit --data-root /kaggle/input/roboat-run-c
  python train_run_c.py --mode train --arm control --data-root /kaggle/input/roboat-run-c
  python train_run_c.py --mode train --arm control --resume --data-root /kaggle/input/roboat-run-c
  python train_run_c.py --mode train --arm zenodo --data-root /kaggle/input/roboat-run-c
  python train_run_c.py --mode evaluate --arm both --data-root /kaggle/input/roboat-run-c

Run `evaluate` only after training/configuration decisions are frozen. Training
uses the combined validation split for checkpoint selection; test and AquaTrash
are read only by the separate final-evaluation mode.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import shutil
import sys
from collections import Counter, defaultdict
from pathlib import Path

import yaml


ULTRALYTICS_VERSION = "8.4.165"
ALBUMENTATIONS_VERSION = "2.0.8"
IMG_SIZE = 960
EPOCHS = 100
PATIENCE = 20
SEED = 42
EXPECTED_SPLIT_COUNTS = {"train": 14_507, "val": 1_590, "test": 1_772}
EXPECTED_OOD_IMAGES = 369
CLASS_NAMES = ["litter", "hyacinth"]
SOURCE_YAMLS = [
    "saigon_tiles_c2", "hagenbeek_tiles_c2", "fml_c2", "tud_gv_c2",
    "mendeley_c2", "navsci_invasive_c2", "navsci_whd_c2", "combined_c2",
]
OOD_YAML = "ood_aquatrash"
ARMS = ("control", "zenodo")


def resolve_data_path(value: str | Path, data_root: Path) -> Path:
    """Rebase archived repo paths to the Kaggle dataset mount."""
    raw = str(value).replace("\\", "/")
    marker = "data/processed/"
    if marker in raw:
        return data_root / raw.rsplit(marker, 1)[1]
    if len(raw) > 2 and raw[1:3] == ":/":
        raise ValueError(f"Unrecognized Windows dataset path (missing {marker}): {raw}")
    path = Path(raw)
    return path if path.is_absolute() else data_root / path


def rebase_yamls(data_root: Path, workdir: Path) -> Path:
    """Write Kaggle-local YAML copies and return their directory."""
    source_dir = data_root / "yamls"
    if not source_dir.is_dir():
        raise FileNotFoundError(f"dataset yamls directory not found: {source_dir}")
    out_dir = workdir / "rebased_yamls"
    out_dir.mkdir(parents=True, exist_ok=True)
    rebased_lists = out_dir / "lists"
    rebased_lists.mkdir(parents=True, exist_ok=True)
    source_lists = data_root / "lists"
    if not source_lists.is_dir():
        raise FileNotFoundError(f"dataset lists directory not found: {source_lists}")
    for source_list in source_lists.glob("*.txt"):
        rebased = []
        for line in source_list.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line:
                rebased.append(str(resolve_data_path(line, data_root)))
        (rebased_lists / source_list.name).write_text("\n".join(rebased) + "\n", encoding="utf-8")
    for source in source_dir.glob("*.yaml"):
        config = yaml.safe_load(source.read_text(encoding="utf-8"))
        if not isinstance(config, dict):
            raise ValueError(f"invalid dataset YAML: {source}")
        for key in ("train", "val", "test"):
            if config.get(key):
                split_path = resolve_data_path(config[key], data_root)
                if split_path.suffix.lower() == ".txt":
                    split_path = rebased_lists / split_path.name
                config[key] = str(split_path)
        config["path"] = str(data_root)
        (out_dir / source.name).write_text(
            yaml.safe_dump(config, sort_keys=False), encoding="utf-8"
        )
    return out_dir


def _names(config: dict) -> list[str]:
    raw = config.get("names")
    if isinstance(raw, dict):
        return [str(raw[k]) for k in sorted(raw, key=lambda v: int(v))]
    if isinstance(raw, list):
        return [str(v) for v in raw]
    raise ValueError("dataset YAML must provide names as a list or integer-keyed mapping")


def _read_images(split_value: str | Path) -> list[Path]:
    path = Path(split_value)
    if not path.exists():
        raise FileNotFoundError(f"dataset split path does not exist: {path}")
    if path.is_file():
        rows = [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
        result = [Path(row) for row in rows]
    else:
        result = sorted(p for p in path.rglob("*") if p.suffix.lower() in {".jpg", ".jpeg", ".png"})
    missing = [p for p in result if not p.is_file()]
    if missing:
        raise FileNotFoundError(f"{len(missing)} listed image(s) are missing; first: {missing[0]}")
    return result


def _label_path(image: Path) -> Path:
    parts = list(image.parts)
    try:
        index = len(parts) - 1 - list(reversed(parts)).index("images")
    except ValueError as exc:
        raise ValueError(f"image path has no 'images' component: {image}") from exc
    parts[index] = "labels"
    return Path(*parts).with_suffix(".txt")


def _audit_label(label: Path, nc: int) -> Counter:
    if not label.is_file():
        raise FileNotFoundError(f"label file missing for image: {label}")
    counts: Counter = Counter()
    for line_no, line in enumerate(label.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        fields = line.split()
        if len(fields) != 5:
            raise ValueError(f"{label}:{line_no}: expected class plus normalized xywh")
        class_id = int(fields[0])
        if not 0 <= class_id < nc:
            raise ValueError(f"{label}:{line_no}: class id {class_id} outside [0,{nc})")
        box = [float(v) for v in fields[1:]]
        if any(v < 0 or v > 1 for v in box) or box[2] <= 0 or box[3] <= 0:
            raise ValueError(f"{label}:{line_no}: invalid normalized box {box}")
        if box[0] - box[2] / 2 < -1e-4 or box[0] + box[2] / 2 > 1 + 1e-4:
            raise ValueError(f"{label}:{line_no}: x extent falls outside image")
        if box[1] - box[3] / 2 < -1e-4 or box[1] + box[3] / 2 > 1 + 1e-4:
            raise ValueError(f"{label}:{line_no}: y extent falls outside image")
        counts[class_id] += 1
    return counts


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def audit_dataset(data_root: Path, rebased: Path, run_root: Path) -> dict:
    """Fail-closed audit of split lists, labels, exact duplicates and groups."""
    report: dict = {"data_root": str(data_root), "checks": {}, "sources": {}}
    combined_path = rebased / "combined_c2.yaml"
    if not combined_path.is_file():
        raise FileNotFoundError(f"required dataset YAML missing: {combined_path}")
    combined = yaml.safe_load(combined_path.read_text(encoding="utf-8"))
    if int(combined.get("nc", -1)) != 2 or _names(combined) != CLASS_NAMES:
        raise ValueError(f"combined_c2 must map nc=2 and names={CLASS_NAMES}")

    combined_lists: dict[str, list[Path]] = {}
    split_paths: dict[str, set[str]] = {}
    class_counts: dict[str, Counter] = {}
    images_by_size: dict[int, list[tuple[str, Path]]] = defaultdict(list)
    for split in ("train", "val", "test"):
        images = _read_images(combined[split])
        combined_lists[split] = images
        expected = EXPECTED_SPLIT_COUNTS[split]
        if len(images) != expected:
            raise ValueError(f"combined {split}: {len(images)} images; expected {expected}")
        normalized = [str(p.resolve()).casefold() for p in images]
        if len(normalized) != len(set(normalized)):
            raise ValueError(f"duplicate image path(s) in combined {split} list")
        split_paths[split] = set(normalized)
        counts: Counter = Counter()
        for image_index, image in enumerate(images, 1):
            counts.update(_audit_label(_label_path(image), 2))
            images_by_size[image.stat().st_size].append((split, image))
            if image_index % 1_000 == 0:
                print(f"Audit {split}: checked labels for {image_index:,}/{len(images):,} images", flush=True)
        class_counts[split] = counts

    for a, b in (("train", "val"), ("train", "test"), ("val", "test")):
        path_overlap = split_paths[a] & split_paths[b]
        if path_overlap:
            raise ValueError(f"path leakage across {a}/{b}: {len(path_overlap)} images")
    report["checks"]["combined_split_counts"] = {k: len(v) for k, v in combined_lists.items()}
    report["checks"]["combined_class_boxes"] = {
        split: {CLASS_NAMES[c]: class_counts[split][c] for c in range(2)}
        for split in combined_lists
    }

    # Per-source split files must agree with the corresponding combined split.
    for name in SOURCE_YAMLS:
        path = rebased / f"{name}.yaml"
        if not path.is_file():
            raise FileNotFoundError(f"required per-source YAML missing: {path}")
        config = yaml.safe_load(path.read_text(encoding="utf-8"))
        if int(config.get("nc", -1)) != 2 or _names(config) != CLASS_NAMES:
            raise ValueError(f"{name} must use the same two-class mapping as combined_c2")
        source_report = {}
        for split in ("val", "test"):
            images = _read_images(config[split])
            keys = {str(p.resolve()).casefold() for p in images}
            if not keys <= split_paths[split]:
                raise ValueError(f"{name} {split} contains images outside combined {split}")
            counts: Counter = Counter()
            for image in images:
                counts.update(_audit_label(_label_path(image), 2))
            source_report[split] = {
                "images": len(images),
                "boxes": {CLASS_NAMES[c]: counts[c] for c in range(2)},
            }
        report["sources"][name] = source_report

    # Run C bundle carries this manifest so the audit can independently verify
    # that video/session groups never cross train, validation, and test.
    manifest = data_root / "audit" / "manifest.csv"
    if not manifest.is_file():
        # Local data/processed trees keep the same file at their root; the
        # bundle builder places it under audit/ for the Kaggle input mount.
        manifest = data_root / "manifest.csv"
    if not manifest.is_file():
        raise FileNotFoundError(
            f"group audit manifest missing: {manifest}. Rebuild the Run C bundle "
            "with scripts/build_kaggle_bundle.py before training."
        )
    by_group: dict[tuple[str, str], set[str]] = defaultdict(set)
    by_stem: dict[str, set[str]] = defaultdict(set)
    manifest_counts = Counter()
    with manifest.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        required = {"source", "group_id", "split", "final_stem"}
        if not required <= set(reader.fieldnames or []):
            raise ValueError(f"manifest is missing required columns: {sorted(required - set(reader.fieldnames or []))}")
        for row in reader:
            if row["split"] not in EXPECTED_SPLIT_COUNTS:
                raise ValueError(f"unexpected manifest split: {row['split']}")
            by_group[(row["source"], row["group_id"])].add(row["split"])
            by_stem[row["final_stem"]].add(row["split"])
            manifest_counts[row["split"]] += 1
    group_leaks = [key for key, splits in by_group.items() if len(splits) > 1]
    stem_leaks = [key for key, splits in by_stem.items() if len(splits) > 1]
    if group_leaks or stem_leaks:
        raise ValueError(f"manifest leakage: {len(group_leaks)} groups and {len(stem_leaks)} stems cross splits")
    if dict(manifest_counts) != EXPECTED_SPLIT_COUNTS:
        raise ValueError(f"manifest split counts differ from bundle: {dict(manifest_counts)}")
    report["checks"]["manifest_group_disjoint"] = True
    report["checks"]["manifest_split_counts"] = dict(manifest_counts)

    # AquaTrash is an eval-only, single-class benchmark, intentionally repeated
    # under val and test in its YAML. Check its labels and train disjointness.
    ood_path = rebased / f"{OOD_YAML}.yaml"
    if not ood_path.is_file():
        raise FileNotFoundError(f"required OOD YAML missing: {ood_path}")
    ood = yaml.safe_load(ood_path.read_text(encoding="utf-8"))
    if int(ood.get("nc", -1)) != 1 or _names(ood) != ["litter"]:
        raise ValueError("ood_aquatrash must be a one-class litter evaluation YAML")
    ood_images = _read_images(ood["test"])
    if len(ood_images) != EXPECTED_OOD_IMAGES:
        raise ValueError(f"AquaTrash has {len(ood_images)} images; expected {EXPECTED_OOD_IMAGES}")
    ood_path_keys = {str(p.resolve()).casefold() for p in ood_images}
    if len(ood_path_keys) != len(ood_images):
        raise ValueError("duplicate image path(s) in AquaTrash evaluation list")
    ood_boxes = Counter()
    for image in ood_images:
        ood_boxes.update(_audit_label(_label_path(image), 1))
        images_by_size[image.stat().st_size].append(("ood", image))

    # Exact hashes are needed only for files whose byte sizes occur in more
    # than one split. This catches exact copies without reading all 4.5 GB.
    candidates = {
        size: rows for size, rows in images_by_size.items()
        if len({split for split, _ in rows}) > 1
    }
    sha_splits: dict[str, set[str]] = defaultdict(set)
    for size_index, (size, rows) in enumerate(candidates.items(), 1):
        for split, image in rows:
            sha_splits[_sha256(image)].add(split)
        if size_index % 100 == 0:
            print(f"Exact-duplicate audit: hashed {size_index:,}/{len(candidates):,} shared-size groups", flush=True)
    cross_split_hashes = [sha for sha, splits in sha_splits.items() if len(splits) > 1]
    if cross_split_hashes:
        raise ValueError(f"byte-identical images found across splits: {len(cross_split_hashes)}")
    report["checks"]["exact_hash_candidate_files"] = sum(len(rows) for rows in candidates.values())
    report["checks"]["split_path_and_sha256_disjoint"] = True
    report["checks"]["ood_aquatrash"] = {
        "images": len(ood_images), "boxes": ood_boxes[0], "disjoint_from_train": True,
    }

    run_root.mkdir(parents=True, exist_ok=True)
    audit_path = run_root / "audit_report.json"
    audit_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(f"AUDIT PASS — report saved to {audit_path}")
    return report


def normalized_image_key(path: Path) -> str:
    return str(path.resolve()).casefold()


def build_augmentations():
    """Low-probability, photometric deployment shifts; no crop/box destruction."""
    os.environ.setdefault("NO_ALBUMENTATIONS_UPDATE", "1")
    try:
        import albumentations as A
    except ImportError as exc:
        raise RuntimeError(
            f"Albumentations {ALBUMENTATIONS_VERSION} is required; install the "
            "pinned dependencies before training."
        ) from exc
    if A.__version__ != ALBUMENTATIONS_VERSION:
        raise RuntimeError(
            f"Expected Albumentations {ALBUMENTATIONS_VERSION}, found {A.__version__}"
        )
    return [
        A.RandomBrightnessContrast(brightness_limit=0.15, contrast_limit=0.15, p=0.20),
        A.RandomFog(fog_coef_range=(0.05, 0.22), alpha_coef=0.06, p=0.05),
        A.RandomRain(slant_range=(-8, 8), drop_length=12, drop_width=1,
                     blur_value=3, brightness_coefficient=0.95, p=0.05),
        A.RandomShadow(shadow_roi=(0.0, 0.5, 1.0, 1.0), num_shadows_limit=(1, 2),
                       shadow_dimension=5, shadow_intensity_range=(0.15, 0.35), p=0.05),
        A.RandomSunFlare(flare_roi=(0.0, 0.0, 1.0, 0.45), p=0.03),
        A.MotionBlur(blur_limit=(3, 5), p=0.03),
    ]


def smoke_test_augmentations(augmentations: list) -> None:
    """Prove configured photo transforms preserve the sample bbox and image shape."""
    import numpy as np
    import albumentations as A

    compose = A.Compose(
        augmentations,
        bbox_params=A.BboxParams(format="yolo", label_fields=["class_labels"],
                                 min_visibility=0.0, clip=True),
    )
    image = np.full((960, 960, 3), 127, dtype=np.uint8)
    box = [0.5, 0.5, 0.02, 0.02]
    result = compose(image=image, bboxes=[box], class_labels=[1])
    if result["image"].shape != image.shape:
        raise RuntimeError("augmentation changed image dimensions")
    if len(result["bboxes"]) != 1 or not np.allclose(result["bboxes"][0], box, atol=1e-6):
        raise RuntimeError("photometric augmentation changed or removed the tiny test bbox")
    print(f"Albumentations {A.__version__}: {len(augmentations)} custom transforms active; bbox smoke test passed")


def write_augmentation_preview(rebased: Path, run_root: Path, augmentations: list) -> Path:
    """Save original and one-transform previews from a real labeled train image."""
    import albumentations as A
    import numpy as np
    from PIL import Image, ImageDraw

    config = yaml.safe_load((rebased / "combined_c2.yaml").read_text(encoding="utf-8"))
    sample = None
    sample_labels = None
    for candidate in _read_images(config["train"]):
        label = _label_path(candidate)
        if label.is_file() and label.read_text(encoding="utf-8").strip():
            sample, sample_labels = candidate, label
            break
    if sample is None:
        raise RuntimeError("could not find a labeled training image for augmentation preview")

    source = np.asarray(Image.open(sample).convert("RGB"))
    boxes = []
    for line in sample_labels.read_text(encoding="utf-8").splitlines():
        fields = line.split()
        if len(fields) == 5:
            boxes.append((int(fields[0]), *[float(v) for v in fields[1:]]))

    tiles = [("Original", source, boxes)]
    for transform in augmentations:
        cloned = A.from_dict(A.to_dict(transform))
        cloned.p = 1.0
        result = cloned(image=source)
        tiles.append((type(transform).__name__, result["image"], boxes))

    tile_w, tile_h, columns = 480, 390, 3
    rows = (len(tiles) + columns - 1) // columns
    canvas = Image.new("RGB", (columns * tile_w, rows * tile_h), "#101820")
    draw = ImageDraw.Draw(canvas)
    for index, (title, array, label_boxes) in enumerate(tiles):
        image = Image.fromarray(array).convert("RGB")
        image.thumbnail((tile_w - 16, tile_h - 42))
        col, row = index % columns, index // columns
        x = col * tile_w + (tile_w - image.width) // 2
        y = row * tile_h + 30 + (tile_h - 42 - image.height) // 2
        canvas.paste(image, (x, y))
        draw.text((col * tile_w + 10, row * tile_h + 8), title, fill="white")
        sx, sy = image.width / array.shape[1], image.height / array.shape[0]
        for cls, cx, cy, width, height in label_boxes:
            color = "#ff6b6b" if cls == 0 else "#32e875"
            x0 = x + (cx - width / 2) * array.shape[1] * sx
            y0 = y + (cy - height / 2) * array.shape[0] * sy
            x1 = x + (cx + width / 2) * array.shape[1] * sx
            y1 = y + (cy + height / 2) * array.shape[0] * sy
            draw.rectangle((x0, y0, x1, y1), outline=color, width=2)
    out = run_root / "augmentation_preview.jpg"
    canvas.save(out, quality=92)
    print(f"Augmentation preview (real train image): {out}")
    return out


def train_arm(arm: str, args, rebased: Path) -> Path:
    from ultralytics import YOLO
    import ultralytics

    if ultralytics.__version__ != ULTRALYTICS_VERSION:
        raise RuntimeError(f"Expected Ultralytics {ULTRALYTICS_VERSION}, found {ultralytics.__version__}")
    aug = build_augmentations()
    smoke_test_augmentations(aug)
    write_augmentation_preview(rebased, args.run_root, aug)
    arm_name = f"run_c_{arm}"
    arm_dir = args.run_root / arm_name
    weights_dir = arm_dir / "weights"
    last_pt = weights_dir / "last.pt"
    best_pt = weights_dir / "best.pt"

    if args.resume_from and not args.resume:
        raise ValueError("--resume-from requires --resume")
    if args.resume_from:
        source = args.resume_from.expanduser().resolve()
        if not (source / "weights" / "last.pt").is_file():
            raise FileNotFoundError(f"--resume-from must point to a run folder containing weights/last.pt: {source}")
        if arm_dir.exists() and any(arm_dir.iterdir()):
            raise FileExistsError(f"target run folder already exists; use it with --resume or choose another --run-root: {arm_dir}")
        arm_dir.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(source, arm_dir, dirs_exist_ok=True)

    if args.resume:
        if not last_pt.is_file():
            raise FileNotFoundError(f"--resume requested but {last_pt} does not exist")
        probe = YOLO(str(last_pt))  # Use the supported loader; never torch.load checkpoints directly.
        checkpoint = getattr(probe, "ckpt", {}) or {}
        if checkpoint.get("epoch", -1) >= 0 and checkpoint.get("optimizer") is not None:
            print(f"Resuming {arm_name} from epoch {checkpoint['epoch'] + 1}")
            model = probe
            model.train(resume=True, augmentations=aug)
        else:
            if not best_pt.is_file():
                raise RuntimeError(f"completed checkpoint has no best.pt: {best_pt}")
            print(f"{arm_name} is already complete; preserving and reusing {best_pt}")
            return best_pt
    else:
        if arm_dir.exists() and any(arm_dir.iterdir()):
            raise FileExistsError(
                f"run directory already contains data: {arm_dir}; use --resume "
                "or set a new --run-root to avoid overwriting a prior run"
            )
        weights = (
            "yolov8n.pt" if arm == "control" else
            str(args.data_root / "zenodo_12800597" / "converted" / "Model_tiles_weights_converted.pt")
        )
        if arm == "zenodo" and not Path(weights).is_file():
            raise FileNotFoundError(f"converted Zenodo checkpoint missing: {weights}")
        model = YOLO(weights)
        model.train(
            data=str(rebased / "combined_c2.yaml"),
            epochs=EPOCHS,
            patience=PATIENCE,
            imgsz=IMG_SIZE,
            seed=SEED,
            deterministic=True,
            project=str(args.run_root),
            name=arm_name,
            exist_ok=False,
            augmentations=aug,
            # Keep the already-small tiled targets visible after scale jitter.
            scale=0.2,
            flipud=0.0,
            fliplr=0.5,
        )
    if not best_pt.is_file():
        raise FileNotFoundError(f"training did not produce expected checkpoint: {best_pt}")
    print(f"Best checkpoint: {best_pt}")
    return best_pt


def evaluate_arm(arm: str, args, rebased: Path) -> dict:
    from ultralytics import YOLO

    model_path = args.run_root / f"run_c_{arm}" / "weights" / "best.pt"
    if not model_path.is_file():
        raise FileNotFoundError(f"best checkpoint missing for {arm}: {model_path}")
    output: dict = {"checkpoint": str(model_path), "sources": {}}
    model = YOLO(str(model_path))
    eval_names = SOURCE_YAMLS + [OOD_YAML]
    for name in eval_names:
        yaml_path = rebased / f"{name}.yaml"
        if not yaml_path.is_file():
            raise FileNotFoundError(f"required final-evaluation YAML missing: {yaml_path}")
        config = yaml.safe_load(yaml_path.read_text(encoding="utf-8"))
        expected_names = _names(config)
        metrics = model.val(data=str(yaml_path), imgsz=IMG_SIZE, split="test")
        seen = {int(c): float(v) for c, v in zip(metrics.box.ap_class_index, metrics.box.ap50)}
        # A class with no ground-truth instances is N/A, never a fabricated zero.
        per_class = {label: seen.get(index) for index, label in enumerate(expected_names)}
        output["sources"][name] = {
            "map50": float(metrics.box.map50),
            "per_class_map50": per_class,
            "class_names": expected_names,
        }
        print(f"{arm:<8} {name:<22} mAP50={metrics.box.map50:.4f} " +
              " ".join(f"{label}={value:.4f}" if value is not None else f"{label}=N/A" for label, value in per_class.items()))
    return output


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("audit", "train", "evaluate"), required=True)
    parser.add_argument("--arm", choices=("control", "zenodo", "both"), default="both")
    parser.add_argument("--data-root", type=Path, default=Path("/kaggle/input/roboat-run-c"))
    parser.add_argument("--run-root", type=Path, default=Path("/kaggle/working/roboat"))
    parser.add_argument("--resume", action="store_true", help="resume an interrupted run, or reuse its completed best.pt")
    parser.add_argument("--resume-from", type=Path, help="previous Kaggle output run folder; copied from the read-only input before resuming")
    parser.add_argument("--dry-run", action="store_true", help="audit and print the plan; never train or evaluate")
    args = parser.parse_args(argv)
    args.data_root = args.data_root.expanduser().resolve()
    args.run_root = args.run_root.expanduser().resolve()
    rebased = rebase_yamls(args.data_root, args.run_root)
    audit_dataset(args.data_root, rebased, args.run_root)
    if args.mode == "audit" or args.dry_run:
        if args.mode != "audit":
            print(f"[dry-run] would {args.mode} arm(s): {args.arm}")
        return 0

    arms = list(ARMS) if args.arm == "both" else [args.arm]
    if args.resume_from and len(arms) > 1:
        parser.error("--resume-from applies to one arm per invocation; run each arm separately")
    if args.mode == "train":
        for arm in arms:
            train_arm(arm, args, rebased)
        return 0

    results = {arm: evaluate_arm(arm, args, rebased) for arm in arms}
    out_path = args.run_root / "final_test_results.json"
    out_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"FINAL TEST/OOD RESULTS saved to {out_path}")
    print("Treat these as final reported metrics; do not tune augmentation or checkpoints against them.")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (FileNotFoundError, FileExistsError, ValueError, RuntimeError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(2)
