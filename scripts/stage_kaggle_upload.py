"""Stage a versioned Kaggle directory tree using hardlinks to the verified Run C files.

The stage is separate from the frozen source trees and the generated ZIP. Refuses
an existing destination so it cannot silently replace a prior upload payload.
"""
import argparse
import json
import os
import shutil
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PROC = ROOT / "data" / "processed"
DEFAULT_OUT = PROC / "kaggle_bundle_v2" / "upload_payload"
DIRS = {
    "merged2": PROC / "merged2",
    "lists": PROC / "lists",
    "yamls": PROC / "yamls",
    "ood_aquatrash": PROC / "ood_aquatrash",
    "zenodo_12800597/converted": PROC / "zenodo_12800597" / "converted",
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    out = args.out.resolve()
    if out.exists():
        raise SystemExit(f"refusing to reuse existing staging directory: {out}")
    out.mkdir(parents=True)
    files = 0
    try:
        for target_name, source_root in DIRS.items():
            target_root = out / target_name
            for src in source_root.rglob("*"):
                if not src.is_file():
                    continue
                dst = target_root / src.relative_to(source_root)
                dst.parent.mkdir(parents=True, exist_ok=True)
                try:
                    os.link(src, dst)
                except OSError:
                    shutil.copy2(src, dst)
                files += 1
        bundle_zip = PROC / "kaggle_bundle_v2" / "roboat-run-c.zip"
        with zipfile.ZipFile(bundle_zip) as zf:
            (out / "BUNDLE_README.txt").write_bytes(zf.read("BUNDLE_README.txt"))
        metadata = PROC / "kaggle_bundle_v2" / "dataset-metadata.json"
        (out / metadata.name).write_text(metadata.read_text(encoding="utf-8"), encoding="utf-8")
    except Exception:
        # Keep failed staging artifacts for inspection/recovery; do not delete
        # potentially valuable uploaded content automatically.
        raise
    total = sum(p.stat().st_size for p in out.rglob("*") if p.is_file())
    print(json.dumps({"path": str(out), "files": files + 2,
                      "bytes": total, "hardlinks_or_copies": files}, indent=2))


if __name__ == "__main__":
    main()
