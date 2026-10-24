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
import concurrent.futures
import csv
import hashlib
import json
import os
import shutil
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import yaml


ULTRALYTICS_VERSION = "8.4.165"
ALBUMENTATIONS_VERSION = "2.0.8"
IMG_SIZE = 960
EPOCHS = 100
PATIENCE = 20
SEED = 42
DEFAULT_MAX_HOURS = 10.3
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


def stage_dataset_to_ssd(
    data_root: Path,
    target_dir: Path = Path("/tmp/roboat_ssd"),
    num_workers: int = 16,
) -> Path:
    """Stage dataset from slow network mount to local NVMe SSD storage (/tmp)."""
    if data_root.resolve() == target_dir.resolve():
        return data_root

    # Verify if target already contains a valid, fully staged dataset with yamls
    if (target_dir / "yamls").is_dir() and any((target_dir / "yamls").glob("*.yaml")):
        print(f"[ssd-stage] Dataset already staged at {target_dir}; reusing.")
        return target_dir

    print(
        f"[ssd-stage] Staging dataset from {data_root} to fast SSD {target_dir}...",
        flush=True,
    )
    t0 = time.time()
    try:
        target_dir.mkdir(parents=True, exist_ok=True)
        # Attempt fast system copy first on Linux (cp -r -L follows all symlinks)
        cp_ok = False
        if os.name != "nt":
            try:
                ret = os.system(f"cp -r -L '{data_root}/.' '{target_dir}/'")
                if ret == 0 and (target_dir / "yamls").is_dir() and any((target_dir / "yamls").glob("*.yaml")):
                    cp_ok = True
            except Exception:
                cp_ok = False

        if not cp_ok:
            shutil.copytree(data_root, target_dir, symlinks=False, dirs_exist_ok=True)

        elapsed = time.time() - t0
        print(f"[ssd-stage] Staged dataset to {target_dir} in {elapsed:.1f}s.", flush=True)
    except Exception as exc:
        print(f"[ssd-stage] Warning: Staging failed ({exc}); falling back to {data_root}.", flush=True)
        return data_root

    # Fail-safe verification: if yamls wasn't copied, fallback to data_root
    if not (target_dir / "yamls").is_dir() or not any((target_dir / "yamls").glob("*.yaml")):
        print(f"[ssd-stage] Warning: Staged yamls directory missing; falling back to {data_root}.", flush=True)
        return data_root

    return target_dir


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
    run_root.mkdir(parents=True, exist_ok=True)
    audit_path = run_root / "audit_report.json"
    if audit_path.is_file():
        try:
            cached = json.loads(audit_path.read_text(encoding="utf-8"))
            checks = cached.get("checks", {})
            if (
                checks.get("split_path_and_sha256_disjoint")
                and checks.get("manifest_group_disjoint")
                and checks.get("combined_split_counts") == EXPECTED_SPLIT_COUNTS
            ):
                print(f"[audit] Reusing cached audit report: {audit_path}")
                print(f"AUDIT PASS (cached) — {audit_path}")
                return cached
        except Exception as exc:
            print(f"[audit] Cached report unreadable or invalid ({exc}); running fresh audit...")

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

    def _audit_image_item(item: tuple[str, Path, int]) -> tuple[str, Path, int, Counter]:
        split, img, nc = item
        counts = _audit_label(_label_path(img), nc)
        size = img.stat().st_size
        return split, img, size, counts

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
        items = [(split, img, 2) for img in images]
        with concurrent.futures.ThreadPoolExecutor(max_workers=16) as pool:
            for _, img, size, c in pool.map(_audit_image_item, items):
                counts.update(c)
                images_by_size[size].append((split, img))
        print(f"Audit {split}: checked labels for {len(images):,} images", flush=True)
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
            items = [(split, img, 2) for img in images]
            with concurrent.futures.ThreadPoolExecutor(max_workers=16) as pool:
                for _, _, _, c in pool.map(_audit_image_item, items):
                    counts.update(c)
            source_report[split] = {
                "images": len(images),
                "boxes": {CLASS_NAMES[c]: counts[c] for c in range(2)},
            }
        report["sources"][name] = source_report

    # Group leakage audit via manifest
    manifest = data_root / "audit" / "manifest.csv"
    if not manifest.is_file():
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

    # AquaTrash OOD
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
    ood_items = [("ood", img, 1) for img in ood_images]
    with concurrent.futures.ThreadPoolExecutor(max_workers=16) as pool:
        for _, img, size, c in pool.map(_audit_image_item, ood_items):
            ood_boxes.update(c)
            images_by_size[size].append(("ood", img))

    # Exact duplicates check
    candidates = {
        size: rows for size, rows in images_by_size.items()
        if len({split for split, _ in rows}) > 1
    }
    sha_splits: dict[str, set[str]] = defaultdict(set)
    if candidates:
        all_candidate_items = [
            (split, img) for rows in candidates.values() for split, img in rows
        ]
        def _hash_worker(item: tuple[str, Path]) -> tuple[str, str]:
            split, img = item
            return _sha256(img), split
        with concurrent.futures.ThreadPoolExecutor(max_workers=16) as pool:
            for digest, split in pool.map(_hash_worker, all_candidate_items):
                sha_splits[digest].add(split)
        cross_split_hashes = [sha for sha, splits in sha_splits.items() if len(splits) > 1]
        if cross_split_hashes:
            raise ValueError(f"byte-identical images found across splits: {len(cross_split_hashes)}")

    report["checks"]["exact_hash_candidate_files"] = sum(len(rows) for rows in candidates.values())
    report["checks"]["split_path_and_sha256_disjoint"] = True
    report["checks"]["ood_aquatrash"] = {
        "images": len(ood_images), "boxes": ood_boxes[0], "disjoint_from_train": True,
    }

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


def register_session_guard_callbacks(
    model,
    max_hours: float,
    arm_dir: Path,
    arm_name: str,
) -> None:
    """Register wall-clock timer and per-epoch safe checkpoint backup callback."""
    start_time = time.time()
    max_seconds = max_hours * 3600.0

    def on_fit_epoch_end(trainer):
        weights_dir = Path(trainer.save_dir) / "weights"
        last_pt = weights_dir / "last.pt"
        safe_pt = weights_dir / "resume_safe.pt"

        # 1. Back up last.pt to resume_safe.pt every epoch
        if last_pt.is_file():
            try:
                shutil.copy2(last_pt, safe_pt)
            except Exception as e:
                print(f"[SESSION GUARD] Warning: could not backup last.pt to resume_safe.pt: {e}")

        # 2. Check wall-clock elapsed time
        elapsed = time.time() - start_time
        remaining = max_seconds - elapsed
        epoch_idx = getattr(trainer, "epoch", 0)
        total_epochs = getattr(trainer, "epochs", EPOCHS)

        # Halt training cleanly before container limit
        if elapsed >= max_seconds or remaining < 900:  # < 15 min remaining
            print("\n" + "=" * 72)
            print(f"[SESSION GUARD] Session boundary guard triggered after epoch {epoch_idx + 1}/{total_epochs}!")
            print(f"[SESSION GUARD] Elapsed: {elapsed / 3600:.2f}h / Limit: {max_hours:.2f}h (remaining: {remaining / 60:.1f}m).")
            print(f"[SESSION GUARD] Checkpoint safely preserved at {safe_pt}.")
            print(f"[SESSION GUARD] Halting training cleanly to allow safe container completion.")
            print("=" * 72 + "\n", flush=True)
            trainer.stop = True

    model.add_callback("on_fit_epoch_end", on_fit_epoch_end)


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
    safe_pt = weights_dir / "resume_safe.pt"
    results_csv = arm_dir / "results.csv"

    if args.resume_from and not args.resume:
        raise ValueError("--resume-from requires --resume")
    if args.resume_from:
        source = args.resume_from.expanduser().resolve()
        has_checkpoint = (
            (source / "weights" / "last.pt").is_file()
            or (source / "weights" / "resume_safe.pt").is_file()
            or (source / "weights" / "best.pt").is_file()
        )
        if not has_checkpoint:
            raise FileNotFoundError(
                f"--resume-from must point to a run folder containing weights/(last.pt, resume_safe.pt, or best.pt): {source}"
            )
        if arm_dir.exists() and any(arm_dir.iterdir()):
            raise FileExistsError(
                f"target run folder already exists; use it with --resume or choose another --run-root: {arm_dir}"
            )
        arm_dir.parent.mkdir(parents=True, exist_ok=True)
        print(f"[resume] Copying previous run folder from {source} to {arm_dir}...")
        shutil.copytree(source, arm_dir, dirs_exist_ok=True)
        if not last_pt.is_file() and safe_pt.is_file():
            shutil.copy2(safe_pt, last_pt)

    if args.resume:
        resume_target = None
        if last_pt.is_file():
            resume_target = last_pt
        elif safe_pt.is_file():
            shutil.copy2(safe_pt, last_pt)
            resume_target = last_pt

        standard_resume_done = False
        if resume_target is not None:
            probe = YOLO(str(resume_target))
            checkpoint = getattr(probe, "ckpt", {}) or {}
            if checkpoint.get("epoch", -1) >= 0 and checkpoint.get("optimizer") is not None:
                completed_epoch = checkpoint["epoch"] + 1
                if completed_epoch >= EPOCHS:
                    print(f"{arm_name} is already complete ({completed_epoch}/{EPOCHS} epochs); preserving {best_pt}")
                    return best_pt
                print(f"[resume] Resuming {arm_name} from epoch {completed_epoch + 1}/{EPOCHS} via standard resume")
                model = probe
                register_session_guard_callbacks(model, args.max_hours, arm_dir, arm_name)
                model.train(resume=True, augmentations=aug)
                standard_resume_done = True
            else:
                print(f"[resume] Checkpoint {resume_target} has no optimizer state; activating resilient resume fallback.")

        if not standard_resume_done:
            if not best_pt.is_file():
                raise FileNotFoundError(f"--resume requested but neither valid last.pt nor best.pt exists in {weights_dir}")

            completed_epochs = 0
            if results_csv.is_file():
                try:
                    lines = [line.strip() for line in results_csv.read_text(encoding="utf-8").splitlines() if line.strip()]
                    if len(lines) > 1:
                        completed_epochs = len(lines) - 1
                except Exception as e:
                    print(f"Warning: could not parse results.csv: {e}")

            if completed_epochs >= EPOCHS:
                print(f"{arm_name} is already complete ({completed_epochs}/{EPOCHS} epochs); preserving {best_pt}")
                return best_pt

            remaining_epochs = EPOCHS - completed_epochs
            print("\n" + "=" * 72)
            print(f"[RESILIENT RESUME] Standard last.pt optimizer state unavailable.")
            print(f"[RESILIENT RESUME] Recovered {completed_epochs} completed epochs from results.csv.")
            print(f"[RESILIENT RESUME] Resuming training from best.pt ({best_pt}) for {remaining_epochs} remaining epochs.")
            print("=" * 72 + "\n", flush=True)

            model = YOLO(str(best_pt))
            register_session_guard_callbacks(model, args.max_hours, arm_dir, arm_name)
            model.train(
                data=str(rebased / "combined_c2.yaml"),
                epochs=remaining_epochs,
                patience=PATIENCE,
                imgsz=IMG_SIZE,
                seed=SEED,
                deterministic=True,
                project=str(args.run_root),
                name=arm_name,
                exist_ok=True,
                augmentations=aug,
                scale=0.2,
                flipud=0.0,
                fliplr=0.5,
            )
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
        register_session_guard_callbacks(model, args.max_hours, arm_dir, arm_name)
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
            scale=0.2,
            flipud=0.0,
            fliplr=0.5,
        )

    target_best = best_pt if best_pt.is_file() else last_pt
    if not target_best.is_file():
        raise FileNotFoundError(f"training did not produce expected checkpoint: {best_pt}")
    print(f"Checkpoint ready: {target_best}")
    return target_best


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
    parser.add_argument("--resume-from", type=Path, help="previous Kaggle output run folder; copied from read-only input before resuming")
    parser.add_argument("--max-hours", type=float, default=DEFAULT_MAX_HOURS, help="maximum wall-clock training hours before safe graceful stop (default: 10.3)")
    parser.add_argument("--stage-ssd", action="store_true", default=None, help="stage dataset to fast local NVMe SSD (/tmp/roboat_ssd)")
    parser.add_argument("--no-stage-ssd", action="store_true", help="disable SSD staging")
    parser.add_argument("--ssd-dir", type=Path, default=Path("/tmp/roboat_ssd"), help="SSD staging directory (default: /tmp/roboat_ssd)")
    parser.add_argument("--dry-run", action="store_true", help="audit and print the plan; never train or evaluate")
    args = parser.parse_args(argv)
    args.data_root = args.data_root.expanduser().resolve()
    args.run_root = args.run_root.expanduser().resolve()

    # If resume_from is given, copy cached audit_report.json if available
    if args.resume_from:
        rf = args.resume_from.expanduser().resolve()
        rf_audit = rf.parent / "audit_report.json" if rf.name.startswith("run_c_") else rf / "audit_report.json"
        if rf_audit.is_file() and not (args.run_root / "audit_report.json").is_file():
            args.run_root.mkdir(parents=True, exist_ok=True)
            shutil.copy2(rf_audit, args.run_root / "audit_report.json")
            print(f"[audit] Copied cached audit report from {rf_audit} to {args.run_root / 'audit_report.json'}")

    # SSD Staging: auto-stage when running inside Kaggle (/kaggle/input and /tmp available) unless explicitly disabled
    should_stage = not args.no_stage_ssd and (
        args.stage_ssd or (str(args.data_root).startswith("/kaggle/input") and Path("/tmp").is_dir())
    )
    if should_stage:
        args.data_root = stage_dataset_to_ssd(args.data_root, target_dir=args.ssd_dir, num_workers=16)

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
