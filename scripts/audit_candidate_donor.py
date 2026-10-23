"""Audit a candidate detection dataset without merging or downloading it.

The command is intentionally fail-closed.  It audits a locally extracted
candidate, records provenance and every exclusion, and writes only under a
run-specific audit directory.  It never copies files into ``data/processed``
and never changes the existing Run C dataset.

Example::

    .venv/Scripts/python.exe scripts/audit_candidate_donor.py \
        --candidate-root C:/datasets/candidate \
        --slug candidate_name \
        --source-url https://example.org/dataset \
        --license "CC BY 4.0" \
        --version 1

The output status is ``READY_FOR_MERGE`` only when the source, annotation,
semantic, deduplication, grouping, and provenance gates pass.  A ready audit
is still not a merge: it is the immutable input for a later, run-scoped build.
"""

from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import json
import re
import sys
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable


ROOT = Path(__file__).resolve().parents[1]
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp"}
TARGET_CLASSES = {"litter", "hyacinth", "entangled_plastic"}
HYACINTH_ALIASES = {
    "hyacinth",
    "water_hyacinth",
    "water hyacinth",
    "eichhornia crassipes",
}
LITTER_ALIASES = {"litter", "floating_waste", "floating waste", "garbage"}
ENTANGLED_ALIASES = {"ent_litter", "entangled_plastic", "entangled plastic"}
RF_SUFFIX_RE = re.compile(r"^(?P<base>.+)_[A-Za-z0-9]+\.rf\.[0-9a-f]{32}$", re.I)
FRAME_RE = re.compile(r"^(?P<base>.+)_frame_\d+(?:_(?:jpg|jpeg|png|webp))?$", re.I)


@dataclass
class ImageRecord:
    relative_path: str
    label_path: str
    official_split: str
    group_id: str
    sha256: str
    phash: int | None
    width: int
    height: int
    n_boxes: int
    mapped_classes: str
    empty_label: bool
    issues: str


def _norm(value: str) -> str:
    return re.sub(r"\s+", " ", value.strip().lower().replace("-", "_"))


def _safe_slug(value: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9._-]+", "_", value.strip()).strip("._-")
    if not slug:
        raise ValueError("slug must contain at least one alphanumeric character")
    return slug


def _relative(path: Path, root: Path) -> str:
    return path.resolve().relative_to(root.resolve()).as_posix()


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _load_phash_backend():
    try:
        import cv2
        import numpy as np
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise RuntimeError(
            "perceptual dedupe requires cv2 and numpy; refusing to audit without them"
        ) from exc
    return cv2, np


def phash(path: Path, cv2, np) -> int:
    data = np.fromfile(str(path), dtype=np.uint8)
    image = cv2.imdecode(data, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError(f"cannot decode image: {path}")
    image = cv2.resize(image, (32, 32), interpolation=cv2.INTER_AREA)
    dct = cv2.dct(np.float32(image))
    low = dct[:8, :8].flatten()
    median = float(np.median(low[1:]))
    value = 0
    for bit in low > median:
        value = (value << 1) | int(bit)
    return value


def hamming(a: int, b: int) -> int:
    return (a ^ b).bit_count()


def _parse_names_text(path: Path) -> list[str]:
    text = path.read_text(encoding="utf-8", errors="replace").strip()
    if path.name.lower() == "classes.txt":
        return [line.strip() for line in text.splitlines() if line.strip()]
    match = re.search(r"^names\s*:\s*(.+)$", text, re.M)
    if not match:
        return []
    raw = match.group(1).strip()
    try:
        parsed = ast.literal_eval(raw)
    except (SyntaxError, ValueError):
        parsed = None
    if isinstance(parsed, (list, tuple)):
        return [str(x) for x in parsed]
    if isinstance(parsed, dict):
        return [str(parsed[k]) for k in sorted(parsed, key=lambda x: int(x))]
    # Minimal fallback for YAML's one-name-per-line form.
    names = []
    for line in text.splitlines():
        m = re.match(r"^\s*[-\d]+\s*:\s*['\"]?(.+?)['\"]?\s*$", line)
        if m:
            names.append(m.group(1).strip("'\""))
    return names


def discover_class_names(root: Path) -> tuple[list[str], list[str]]:
    candidates = sorted(
        [p for p in root.rglob("classes.txt") if p.is_file()]
        + [p for p in root.rglob("data.yaml") if p.is_file()]
        + [p for p in root.rglob("data.yml") if p.is_file()]
    )
    for path in candidates:
        names = _parse_names_text(path)
        if names:
            return names, [_relative(path, root)]
    return [], []


def parse_mapping(items: Iterable[str]) -> dict[str, str]:
    mapping: dict[str, str] = {}
    for item in items:
        if "=" not in item:
            raise ValueError(f"--map requires SOURCE=TARGET, got {item!r}")
        source, target = item.split("=", 1)
        source, target = _norm(source), _norm(target)
        if target not in TARGET_CLASSES:
            raise ValueError(f"unsupported target class {target!r}")
        mapping[source] = target
    return mapping


def infer_mapping(names: list[str], explicit: dict[str, str]) -> dict[str, str]:
    result = dict(explicit)
    for name in names:
        key = _norm(name)
        if key in result:
            continue
        if key in HYACINTH_ALIASES:
            result[key] = "hyacinth"
        elif key in LITTER_ALIASES:
            result[key] = "litter"
        elif key in ENTANGLED_ALIASES:
            result[key] = "entangled_plastic"
    return result


def find_images(root: Path) -> list[Path]:
    return sorted(p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTS)


def official_split(path: Path, root: Path) -> str:
    parts = path.resolve().relative_to(root.resolve()).parts
    lowered = [p.lower() for p in parts]
    for candidate in ("train", "val", "valid", "validation", "test"):
        if candidate in lowered:
            return "val" if candidate in {"valid", "validation"} else candidate
    return "unspecified"


def group_id(path: Path, root: Path, mode: str) -> str:
    stem = path.stem
    match = RF_SUFFIX_RE.match(stem)
    if match:
        stem = match.group("base")
    if mode == "per-image":
        return "img_" + _relative(path, root)
    frame = FRAME_RE.match(stem)
    if frame:
        return "frame_" + frame.group("base").lower()
    return "img_" + _relative(path, root) if mode == "auto" else "stem_" + stem.lower()


def locate_label(image: Path, root: Path) -> tuple[Path | None, bool]:
    candidates = []
    direct = image.with_suffix(".txt")
    candidates.append(direct)
    parts = list(image.parts)
    for i, part in enumerate(parts):
        if part.lower() == "images":
            candidates.append(Path(*parts[:i], "labels", *parts[i + 1 :]).with_suffix(".txt"))
            break
    existing = sorted({p.resolve() for p in candidates if p.exists() and p.is_file()})
    if len(existing) == 1:
        return existing[0], False
    if len(existing) > 1:
        return None, True
    matches = sorted(p.resolve() for p in root.rglob(image.stem + ".txt") if p.is_file())
    if len(matches) == 1:
        return matches[0], False
    return None, len(matches) > 1


def parse_yolo_label(path: Path, n_classes: int) -> tuple[list[tuple[int, float, float, float, float]], list[str]]:
    boxes = []
    issues = []
    for line_no, line in enumerate(path.read_text(encoding="utf-8", errors="replace").splitlines(), 1):
        if not line.strip():
            continue
        fields = line.split()
        if len(fields) != 5:
            issues.append(f"line{line_no}:expected_5_fields")
            continue
        try:
            cls = int(fields[0])
            values = [float(x) for x in fields[1:]]
        except ValueError:
            issues.append(f"line{line_no}:not_numeric")
            continue
        cx, cy, width, height = values
        if not 0 <= cls < n_classes:
            issues.append(f"line{line_no}:class_{cls}_out_of_range")
        if any(value < 0 or value > 1 for value in values):
            issues.append(f"line{line_no}:coordinate_out_of_range")
        if width <= 0 or height <= 0:
            issues.append(f"line{line_no}:non_positive_box")
        boxes.append((cls, cx, cy, width, height))
    return boxes, issues


def existing_images(root: Path, candidate_root: Path) -> list[Path]:
    roots = [root / "data" / "raw", root / "data" / "processed"]
    paths = []
    candidate_resolved = candidate_root.resolve()
    for scan_root in roots:
        if not scan_root.exists():
            continue
        for path in scan_root.rglob("*"):
            if not path.is_file() or path.suffix.lower() not in IMAGE_EXTS:
                continue
            if candidate_resolved in path.resolve().parents:
                continue
            # Archives are not images, and audit output should not be treated as
            # an incumbent source if a future run root sits under processed.
            if "candidate_audits" in path.resolve().parts:
                continue
            paths.append(path)
    return sorted(set(paths))


def write_csv(path: Path, rows: list[ImageRecord]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(asdict(rows[0]).keys()) if rows else list(ImageRecord.__annotations__.keys())
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(asdict(row) for row in rows)


def audit(args: argparse.Namespace) -> int:
    candidate_root = Path(args.candidate_root).resolve()
    slug = _safe_slug(args.slug)
    out_root = (ROOT / "data" / "processed" / "candidate_audits" / slug).resolve()
    if args.output:
        out_root = Path(args.output).resolve()
    if not candidate_root.exists() or not candidate_root.is_dir():
        out_root.mkdir(parents=True, exist_ok=True)
        report = {
            "schema_version": 1,
            "status": "QUARANTINED",
            "candidate": {"slug": slug, "root": str(candidate_root)},
            "quarantine_reasons": ["candidate_root_missing_or_not_directory"],
        }
        (out_root / "candidate_audit.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"QUARANTINED: {report['quarantine_reasons'][0]}")
        return 2

    cv2, np = _load_phash_backend()
    names, metadata_files = discover_class_names(candidate_root)
    explicit_mapping = parse_mapping(args.map_class)
    mapping = infer_mapping(names, explicit_mapping)
    images = find_images(candidate_root)
    blockers: list[str] = []
    warnings: list[str] = []
    if not args.source_url:
        blockers.append("missing_source_url")
    if not args.license:
        blockers.append("missing_license")
    if not args.version:
        blockers.append("missing_version")
    if not images:
        blockers.append("no_images")
    if not names:
        blockers.append("class_names_not_found")
    normalized_names = {_norm(n) for n in names}
    unknown_names = sorted(normalized_names - set(mapping))
    if unknown_names:
        blockers.append("unmapped_classes:" + ",".join(unknown_names))

    exact_existing: dict[str, str] = {}
    existing_hashes: dict[str, Path] = {}
    existing_phashes: list[tuple[int, Path]] = []
    incumbent = existing_images(ROOT, candidate_root)
    for path in incumbent:
        try:
            digest = _sha256(path)
            existing_hashes.setdefault(digest, path)
            existing_phashes.append((phash(path, cv2, np), path))
        except (OSError, ValueError) as exc:
            warnings.append(f"incumbent_hash_failed:{path}:{exc}")

    seen_hashes: dict[str, Path] = {}
    seen_phashes: list[tuple[int, Path]] = []
    records: list[ImageRecord] = []
    duplicate_rows = []
    group_to_splits: defaultdict[str, set[str]] = defaultdict(set)
    class_box_counts: Counter[str] = Counter()
    total_boxes = 0
    tiny_boxes_960 = 0
    all_box_min_960 = []

    for image in images:
        rel = _relative(image, candidate_root)
        label, ambiguous_label = locate_label(image, candidate_root)
        issues: list[str] = []
        if label is None:
            issues.append("ambiguous_label" if ambiguous_label else "missing_label")
        try:
            from PIL import Image
            with Image.open(image) as im:
                width, height = im.size
                im.verify()
        except Exception as exc:  # PIL raises several format-specific errors
            width = height = 0
            issues.append("image_decode_failed:" + type(exc).__name__)
        digest = _sha256(image)
        image_hash = phash(image, cv2, np)
        if digest in existing_hashes:
            exact_existing[rel] = _relative(existing_hashes[digest], ROOT)
            issues.append("exact_duplicate_of_incumbent")
            duplicate_rows.append({"candidate": rel, "matched": exact_existing[rel], "kind": "sha256"})
        elif digest in seen_hashes:
            issues.append("exact_duplicate_within_candidate")
            duplicate_rows.append({"candidate": rel, "matched": _relative(seen_hashes[digest], candidate_root), "kind": "sha256"})
        else:
            nearest = min(((hamming(image_hash, h), p) for h, p in existing_phashes), default=None)
            nearest_within = min(((hamming(image_hash, h), p) for h, p in seen_phashes), default=None)
            for kind, pair, base in (("phash_incumbent", nearest, ROOT), ("phash_candidate", nearest_within, candidate_root)):
                if pair is not None and pair[0] <= args.phash_threshold:
                    issues.append(f"near_duplicate_{kind}")
                    duplicate_rows.append({
                        "candidate": rel,
                        "matched": _relative(pair[1], base),
                        "kind": kind,
                        "hamming": pair[0],
                    })
            seen_hashes[digest] = image
            seen_phashes.append((image_hash, image))

        boxes = []
        if label is not None:
            boxes, label_issues = parse_yolo_label(label, len(names))
            issues.extend(label_issues)
        if not boxes:
            if label is not None:
                issues.append("empty_label")
                if not args.allow_empty_labels:
                    blockers.append("empty_labels_present")
        mapped = set()
        for cls, _, _, bw, bh in boxes:
            if 0 <= cls < len(names) and _norm(names[cls]) in mapping:
                target = mapping[_norm(names[cls])]
                mapped.add(target)
                class_box_counts[target] += 1
                total_boxes += 1
                if width and height:
                    px_w = bw * width * 960 / max(width, height)
                    px_h = bh * height * 960 / max(width, height)
                    min_px = min(px_w, px_h)
                    all_box_min_960.append(min_px)
                    tiny_boxes_960 += int(min_px < 4)
        if any(issue.startswith("line") or "decode_failed" in issue for issue in issues):
            blockers.append(f"invalid_annotations:{rel}")
        if any(issue.startswith("near_duplicate") or issue.startswith("exact_duplicate") for issue in issues):
            blockers.append(f"duplicate_image:{rel}")
        split = official_split(image, candidate_root)
        group = group_id(image, candidate_root, args.group_mode)
        group_to_splits[group].add(split)
        records.append(ImageRecord(
            relative_path=rel,
            label_path=_relative(label, candidate_root) if label else "",
            official_split=split,
            group_id=group,
            sha256=digest,
            phash=image_hash,
            width=width,
            height=height,
            n_boxes=len(boxes),
            mapped_classes="+".join(sorted(mapped)),
            empty_label=not boxes,
            issues=";".join(sorted(set(issues))),
        ))

    multi_split_groups = sorted(group for group, splits in group_to_splits.items() if len(splits - {"unspecified"}) > 1)
    if multi_split_groups:
        warnings.append(f"official_split_group_overlap:{len(multi_split_groups)}")
    auto_one_group_per_image = (
        args.group_mode == "auto" and len(images) > 1 and len(group_to_splits) == len(images)
    )
    if auto_one_group_per_image:
        warnings.append("auto_grouping_used_one_group_per_image; inspect_for_video_sequences")
        if not args.allow_ambiguous_grouping:
            blockers.append("grouping_requires_review")
    if total_boxes == 0:
        blockers.append("no_valid_boxes")
    tiny_fraction = tiny_boxes_960 / total_boxes if total_boxes else 0.0
    if tiny_fraction >= 0.25:
        warnings.append(f"tiny_boxes_at_960:{tiny_fraction:.3f}")
        blockers.append("box_quality_requires_review")
    if multi_split_groups:
        blockers.append("official_groups_span_splits")

    # Proposed splits are group-based and independent of any donor-provided split.
    groups = defaultdict(list)
    for row in records:
        groups[row.group_id].append(row)
    ordered_groups = sorted(groups, key=lambda key: (-len(groups[key]), key))
    target = {"train": 0.8, "val": 0.1, "test": 0.1}
    split_counts = Counter()
    proposed = {}
    total_images = len(records)
    for group in ordered_groups:
        candidates = sorted(target, key=lambda split: (split_counts[split] / max(1, total_images), split))
        proposed[group] = candidates[0]
        split_counts[candidates[0]] += len(groups[group])
    for row in records:
        row.issues = ";".join(filter(None, [row.issues, "proposed_split=" + proposed[row.group_id]]))

    status = "READY_FOR_MERGE" if not blockers else "QUARANTINED"
    provenance = {
        "schema_version": 1,
        "slug": slug,
        "source_url": args.source_url,
        "license": args.license,
        "version": args.version,
        "candidate_root": str(candidate_root),
        "candidate_root_sha256_note": "individual image SHA-256 values are in candidate_records.csv",
        "class_metadata_files": metadata_files,
        "class_names": names,
        "mapping": mapping,
        "group_mode": args.group_mode,
        "phash_threshold": args.phash_threshold,
        "incumbent_image_count": len(incumbent),
        "audit_tool": "scripts/audit_candidate_donor.py",
    }
    report = {
        "schema_version": 1,
        "status": status,
        "candidate": provenance,
        "counts": {
            "images": len(records),
            "boxes": total_boxes,
            "class_boxes": dict(class_box_counts),
            "groups": len(groups),
            "proposed_split_images": dict(split_counts),
            "empty_labels": sum(row.empty_label for row in records),
            "duplicate_rows": len(duplicate_rows),
            "tiny_boxes_at_960": tiny_boxes_960,
            "tiny_box_fraction_at_960": round(tiny_fraction, 6),
        },
        "gates": {
            "source_license_provenance": not any(x in blockers for x in ("missing_source_url", "missing_license", "missing_version")),
            "class_semantics": bool(names) and not unknown_names,
            "annotation_integrity": not any(x.startswith("invalid_annotations:") for x in blockers),
            "exact_and_perceptual_dedupe": not any(x.startswith("duplicate_image:") for x in blockers),
            "group_split_integrity": not multi_split_groups,
            "box_quality": "box_quality_requires_review" not in blockers,
        },
        "warnings": warnings,
        "quarantine_reasons": sorted(set(blockers)),
        "duplicate_matches": duplicate_rows,
    }
    out_root.mkdir(parents=True, exist_ok=True)
    (out_root / "provenance.json").write_text(json.dumps(provenance, indent=2), encoding="utf-8")
    (out_root / "candidate_audit.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    write_csv(out_root / "candidate_records.csv", records)
    (out_root / "proposed_group_splits.json").write_text(json.dumps(proposed, indent=2, sort_keys=True), encoding="utf-8")
    (out_root / "duplicate_matches.json").write_text(json.dumps(duplicate_rows, indent=2), encoding="utf-8")
    print(json.dumps({"status": status, "output": str(out_root), "reasons": report["quarantine_reasons"]}, indent=2))
    return 0 if status == "READY_FOR_MERGE" else 2


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-root", required=True, help="locally extracted candidate directory; never downloaded by this tool")
    parser.add_argument("--slug", required=True)
    parser.add_argument("--source-url", default="")
    parser.add_argument("--license", default="")
    parser.add_argument("--version", default="")
    parser.add_argument("--map-class", action="append", default=[], help="explicit SOURCE=TARGET mapping; repeatable")
    parser.add_argument("--group-mode", choices=("auto", "per-image", "stem"), default="auto")
    parser.add_argument(
        "--allow-ambiguous-grouping",
        action="store_true",
        help="explicitly allow auto mode to use one group per image",
    )
    parser.add_argument("--phash-threshold", type=int, default=8)
    parser.add_argument("--allow-empty-labels", action="store_true")
    parser.add_argument("--output", help="optional run-specific output directory")
    return parser


if __name__ == "__main__":
    try:
        raise SystemExit(audit(build_parser().parse_args()))
    except (ValueError, RuntimeError) as exc:
        print(f"QUARANTINED: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc
