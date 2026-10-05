"""Build a single upload-ready Kaggle dataset zip for the Run A/B training.

Bundle layout (top level, exactly what kaggle/train_baseline.py rebase()
expects at --data-root: it globs <data_root>/lists/*.txt and
<data_root>/yamls/*.yaml and joins "data/processed/..." list entries onto
the data root):
  merged2/                        images+labels for train/val/test (PRIMARY
                                  2-class tree)
  lists/                          *.txt (train.txt is the RFS-expanded
                                  10,070-line list)
  yamls/                          *.yaml (per-source + combined, c2 and c3)
  ood_aquatrash/                  eval-only AquaTrash OOD tree (369 images +
                                  369 labels, flat images/ + labels/ - no
                                  split subdirs); shipped so the D6 OOD
                                  benchmark is evaluable on Kaggle/Colab
  zenodo_12800597/converted/      Zenodo checkpoints converted by
                                  scripts/convert_zenodo_weights.py (Run B
                                  --pretrained uses
                                  zenodo_12800597/converted/
                                  Model_tiles_weights_converted.pt, see
                                  kaggle/RUN_PLAN.md) - the same nested path
                                  kaggle/README.md §1 documents
  BUNDLE_README.txt               mount-path convention + launch commands

ood_aquatrash note: the AquaTrash OOD tree IS bundled (369 images + 369
labels) together with lists/ood_aquatrash.txt and yamls/ood_aquatrash.yaml, so
the D6 OOD benchmark is evaluable on Kaggle/Colab (its yaml is 1-class
litter-only - that is expected for the OOD eval).

Hardlink note (kaggle/README.md §1): the dataset trees use hardlinks
(Windows-compatible, no symlinks - scripts/build_dataset.py place()). A plain
zipfile archive reads every member by path, so each hardlink becomes an
independent copy of the file contents inside the zip - which is exactly what
the Kaggle mount needs. (A symlink-preserving format would NOT work on Kaggle;
that is why the trees are hardlinked rather than symlinked.)

Sanity assertions gate the zip (all read-only):
  - merged2 holds 9,428 images AND 9,428 labels (7,368/900/1,160 per split),
    with identical stem sets per split;
  - ood_aquatrash holds 369 images AND 369 labels with identical stem sets;
  - every yaml in yamls/ parses and carries nc + names;
  - lists/train.txt has exactly 10,070 non-empty lines;
  - extra guard: every merged2 and ood_aquatrash entry in the resolved lists
    exists on disk.

--dry-run: prints file counts and total sizes per top-level dir and writes
nothing (the assertions still run, since they write nothing either).
"""
import argparse
import zipfile
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
PROC = ROOT / "data" / "processed"
OUT_DIR = PROC / "kaggle_bundle"
ZIP_NAME = "roboat-processed.zip"  # name expected by training/colab_train.ipynb

# top-level bundle path -> on-disk source dir under data/processed.
# "zenodo_12800597/converted" keeps the NESTED path documented in
# kaggle/README.md §1 and kaggle/RUN_PLAN.md (Run B's --pretrained path).
BUNDLE_DIRS = {
    "merged2": PROC / "merged2",
    "lists": PROC / "lists",
    "yamls": PROC / "yamls",
    "ood_aquatrash": PROC / "ood_aquatrash",
    "zenodo_12800597/converted": PROC / "zenodo_12800597" / "converted",
}
README_NAME = "BUNDLE_README.txt"

EXPECTED_IMAGES = 9_428  # 7,368 train + 900 val + 1,160 test (CP4 verifier)
EXPECTED_PER_SPLIT = {"train": 7_368, "val": 900, "test": 1_160}
EXPECTED_OOD = 369  # eval-only AquaTrash OOD tree (images == labels)
EXPECTED_TRAIN_LINES = 10_070  # RFS-expanded (lists/train.txt)

LISTS_RESOLVE = ["train.txt", "train_base.txt", "val_base.txt", "test_base.txt",
                 "fml_val.txt", "fml_test.txt", "tud_gv_val.txt", "tud_gv_test.txt",
                 "hagenbeek_val.txt", "hagenbeek_test.txt",
                 "saigon_val.txt", "saigon_test.txt", "ood_aquatrash.txt"]

README_TEXT = f"""RoBoat Kaggle dataset bundle
============================
Built by scripts/build_kaggle_bundle.py from data/processed (CP4-approved,
16/16 verifier checks). 2 classes: 0 litter, 1 hyacinth.

Contents (top level = the Kaggle dataset mount root, i.e. --data-root):
  merged2/images/{{train,val,test}}  +  merged2/labels/{{train,val,test}}
      {EXPECTED_PER_SPLIT['train']:,} / {EXPECTED_PER_SPLIT['val']} / {EXPECTED_PER_SPLIT['test']:,} images
  lists/*.txt   train.txt is the RFS-expanded {EXPECTED_TRAIN_LINES:,}-line training list
                (train_base.txt is the un-expanded 7,368-image base)
  yamls/*.yaml  per-source (fml, tud_gv, hagenbeek_tiles, saigon_tiles,
                ood_aquatrash) + combined; *_c2.yaml = primary 2-class,
                *_c3.yaml = ablation
  ood_aquatrash/images + ood_aquatrash/labels
                369 eval-only AquaTrash OOD images (D6 benchmark; the OOD
                yaml is 1-class litter-only - expected for the OOD eval)
  zenodo_12800597/converted/
                Zenodo Model_* checkpoints converted by
                scripts/convert_zenodo_weights.py (Run B --pretrained)

Mount-path convention (what kaggle/train_baseline.py rebase() expects):
the lists and yamls carry repo-absolute "data/processed/..." entries;
rebase() rewrites whatever follows "data/processed/" onto --data-root.
So --data-root must be the directory that DIRECTLY contains merged2/,
lists/, yamls/, ood_aquatrash/ and zenodo_12800597/converted/ - e.g. upload
this zip as a private Kaggle Dataset named "roboat-processed", which mounts
at /kaggle/input/roboat-processed.

Launch (GPU T4 x2 or P100, imgsz=960 per the D11 box-size audit):
  pip install -q ultralytics
  python train_baseline.py --data-root /kaggle/input/roboat-processed
Run B (Zenodo Model_tiles init, yolov8n leg only - see kaggle/RUN_PLAN.md):
  python train_baseline.py --data-root /kaggle/input/roboat-processed \\
      --pretrained /kaggle/input/roboat-processed/zenodo_12800597/converted/Model_tiles_weights_converted.pt

Caveats:
  - ood_aquatrash.yaml evaluates only class 0 (litter; nc=1) against the
    2-class model - that is expected for the OOD benchmark (D6), so per-class
    numbers on it are not comparable with the 2-class per-source sets.
  - *_c3.yaml belong to the merged3 ablation tree (not bundled); the
    primary runs use the *_c2.yaml files.
  - The dataset trees use hardlinks; this zip stored each hardlink as a real,
    independent copy of the file contents (kaggle/README.md section 1).
  - The val split is same-site; the Bengaluru Capture Set (T9) is the real test.
  - Contains licensed research data (TUD-GV, FML v2, Hagenbeek): keep the
    Kaggle dataset PRIVATE; per-source licenses in reports/DECISIONS_LOG.md.
"""


def human(n: int) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024 or unit == "GB":
            return f"{n:,.1f} {unit}" if unit != "B" else f"{n:,} B"
        n /= 1024
    return f"{n:,.1f} GB"


def collect_bundle_files() -> dict:
    """Map top-level dir name -> sorted list of (source_path, arcname)."""
    bundle = {}
    for top, src_dir in BUNDLE_DIRS.items():
        if not src_dir.is_dir():
            raise SystemExit(f"missing bundle source dir: {src_dir}")
        files = sorted(p for p in src_dir.rglob("*") if p.is_file())
        bundle[top] = [(p, f"{top}/{p.relative_to(src_dir).as_posix()}")
                       for p in files]
    return bundle


def sanity_checks(bundle: dict) -> dict:
    """Gate the zip: dataset counts, yaml parseability, list line counts."""
    report = {}

    # 1. merged2 image/label counts per split, plus stem-set equality
    n_img = n_lab = 0
    for split, expected in EXPECTED_PER_SPLIT.items():
        imgs = sorted((PROC / "merged2" / "images" / split).glob("*.jpg"))
        labs = sorted((PROC / "merged2" / "labels" / split).glob("*.txt"))
        assert len(imgs) == expected, \
            f"merged2/images/{split}: {len(imgs)} != expected {expected}"
        assert len(labs) == expected, \
            f"merged2/labels/{split}: {len(labs)} != expected {expected}"
        assert {p.stem for p in imgs} == {p.stem for p in labs}, \
            f"merged2 {split}: image/label stem sets differ"
        n_img += len(imgs)
        n_lab += len(labs)
    assert n_img == EXPECTED_IMAGES, f"images {n_img} != {EXPECTED_IMAGES}"
    assert n_lab == EXPECTED_IMAGES, f"labels {n_lab} != {EXPECTED_IMAGES}"
    report["images"] = n_img
    report["labels"] = n_lab

    # 1b. ood_aquatrash eval-only tree: 369 images + 369 labels, flat
    ood_imgs = sorted((PROC / "ood_aquatrash" / "images").glob("*.jpg"))
    ood_labs = sorted((PROC / "ood_aquatrash" / "labels").glob("*.txt"))
    assert len(ood_imgs) == EXPECTED_OOD, \
        f"ood_aquatrash/images: {len(ood_imgs)} != expected {EXPECTED_OOD}"
    assert len(ood_labs) == EXPECTED_OOD, \
        f"ood_aquatrash/labels: {len(ood_labs)} != expected {EXPECTED_OOD}"
    assert {p.stem for p in ood_imgs} == {p.stem for p in ood_labs}, \
        "ood_aquatrash: image/label stem sets differ"
    report["ood_images"] = len(ood_imgs)
    report["ood_labels"] = len(ood_labs)

    # 2. every yaml parseable, with nc + names
    yamls = sorted((PROC / "yamls").glob("*.yaml"))
    parsed = {}
    for f in yamls:
        y = yaml.safe_load(f.read_text(encoding="utf-8"))
        assert isinstance(y, dict) and "nc" in y and "names" in y, \
            f"{f.name}: missing nc/names or unparseable"
        parsed[f.name] = y
    assert len(parsed) == len(list((PROC / "yamls").glob('*.yaml')))
    report["yamls"] = sorted(parsed)

    # 3. train.txt line count
    lines = [ln for ln in (PROC / "lists" / "train.txt").read_text().splitlines()
             if ln.strip()]
    assert len(lines) == EXPECTED_TRAIN_LINES, \
        f"lists/train.txt: {len(lines)} lines != {EXPECTED_TRAIN_LINES}"
    report["train_txt_lines"] = len(lines)

    # 4. extra guard: every merged2 and ood_aquatrash list entry resolves on disk
    resolved = missing = 0
    for name in LISTS_RESOLVE:
        for line in (PROC / "lists" / name).read_text().splitlines():
            if not line.strip():
                continue
            if (ROOT / line).exists():
                resolved += 1
            else:
                missing += 1
    assert missing == 0, f"{missing} list entries do not resolve on disk"
    report["list_entries_resolved"] = resolved

    # the converted Run B checkpoint must be present in the bundle, at the
    # nested path RUN_PLAN.md documents for --pretrained
    arcnames = {a for files in bundle.values() for _, a in files}
    assert "zenodo_12800597/converted/Model_tiles_weights_converted.pt" in arcnames, \
        "Run B checkpoint missing from zenodo_12800597/converted/"
    # the OOD tree + its yaml/txt must ship now (D6 benchmark evaluability)
    assert "yamls/ood_aquatrash.yaml" in arcnames, \
        "yamls/ood_aquatrash.yaml missing from the bundle (D6 benchmark unevaluable)"
    assert "lists/ood_aquatrash.txt" in arcnames, \
        "lists/ood_aquatrash.txt missing from the bundle (D6 benchmark unevaluable)"
    assert any(a.startswith("ood_aquatrash/images/") for a in arcnames) and \
        any(a.startswith("ood_aquatrash/labels/") for a in arcnames), \
        "ood_aquatrash image/label tree missing from the bundle"
    return report


def build(out_zip: Path, dry_run: bool) -> None:
    bundle = collect_bundle_files()
    report = sanity_checks(bundle)

    print("sanity checks passed:")
    for split, expected in EXPECTED_PER_SPLIT.items():
        print(f"  merged2 {split:<5} {expected:>6,} images + {expected:>6,} labels")
    # raw (ungrouped) digits so the count is greppable by automated gates
    print(f"  images total   {report['images']:>6} (labels {report['labels']})")
    print(f"  ood_aquatrash  {report['ood_images']:>6} images + "
          f"{report['ood_labels']:>6} labels (eval-only, D6)")
    print(f"  yamls parsed   {len(report['yamls'])}: {', '.join(report['yamls'])}")
    print(f"  train.txt      {report['train_txt_lines']:,} lines")
    print(f"  list entries   {report['list_entries_resolved']:,} resolved")

    total_files = 0
    total_bytes = 0
    for top in BUNDLE_DIRS:
        files = bundle[top]
        size = sum(p.stat().st_size for p, _ in files)
        total_files += len(files)
        total_bytes += size
        print(f"{top:<26} {len(files):>6,} files  {human(size):>12}")
    total_files += 1  # BUNDLE_README.txt
    total_bytes += len(README_TEXT.encode("utf-8"))
    print(f"{'TOTAL':<26} {total_files:>6,} files  {human(total_bytes):>12} "
          f"(uncompressed)")

    if dry_run:
        print("[dry-run] no zip written")
        return

    out_zip.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(out_zip, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as zf:
        for top, files in bundle.items():
            for src, arc in files:
                zf.write(src, arc)
        zf.writestr(README_NAME, README_TEXT)
    print(f"wrote {out_zip} ({human(out_zip.stat().st_size)}, "
          f"{total_files} files)")


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", type=Path, default=OUT_DIR / ZIP_NAME,
                    help=f"output zip path (default {OUT_DIR / ZIP_NAME})")
    ap.add_argument("--dry-run", action="store_true",
                    help="print counts/sizes only, write nothing")
    args = ap.parse_args()
    build(args.out, args.dry_run)


if __name__ == "__main__":
    main()
