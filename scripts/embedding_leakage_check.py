"""Embedding near-duplicate leakage check: test/val vs train (Run C dataset).

Embeds every merged2 image with CLIP ViT-B-32 (quickgelu, the repo's audited
model, CPU-only) and reports, for each val and test image, the maximum cosine
similarity to any train image. Crop-different copies of the same source photo
(e.g. the 371 shared-Navsci sources whose pHash distance exceeds 8) survive the
pHash gate but sit at very high embedding cosine - this scan is what settles
whether any of them straddle the train/test split.

Read-only on the dataset; writes a JSON report. Usage:
    .venv/Scripts/python.exe scripts/embedding_leakage_check.py
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
PROC = ROOT / "data" / "processed"
OUT = PROC / "embedding_leakage_report.json"
BATCH = 64
THRESHOLDS = (0.95, 0.90)


def main():
    import torch
    import open_clip

    model, _, preprocess = open_clip.create_model_and_transforms(
        "ViT-B-32-quickgelu", pretrained="openai", cache_dir=ROOT / ".clip_cache")
    model.eval()

    def embed_dir(split: str) -> tuple[list[str], np.ndarray]:
        paths = sorted((PROC / "merged2" / "images" / split).glob("*.jpg"))
        feats = []
        t0 = time.time()
        with torch.no_grad():
            for i in range(0, len(paths), BATCH):
                batch = torch.stack([preprocess(Image.open(p).convert("RGB"))
                                     for p in paths[i:i + BATCH]])
                f = model.encode_image(batch).float()
                f = f / f.norm(dim=-1, keepdim=True)
                feats.append(f.numpy())
                if (i // BATCH) % 20 == 0:
                    print(f"  {split}: {i + len(paths[i:i+BATCH])}/{len(paths)}"
                          f" ({time.time() - t0:.0f}s)", flush=True)
        return [p.stem for p in paths], np.concatenate(feats)

    result = {"model": "ViT-B-32-quickgelu", "thresholds": THRESHOLDS, "splits": {}}
    bank_stems, bank = embed_dir("train")
    for split in ("val", "test"):
        stems, feats = embed_dir(split)
        sims = feats @ bank.T  # (n_query, n_bank), both L2-normalized
        max_idx = sims.argmax(axis=1)
        max_cos = sims[np.arange(len(stems)), max_idx]
        pairs = []
        for th in THRESHOLDS:
            hit = [(stems[i], bank_stems[j], round(float(sims[i, j]), 4))
                   for i, j in enumerate(max_idx) if sims[i, j] >= th]
            result["splits"].setdefault(split, {})[f"pairs_ge_{th}"] = hit
            print(f"{split}: {len(hit)} images with max train-cosine >= {th}", flush=True)
        result["splits"][split]["max_cosine_stats"] = {
            "median": round(float(np.median(max_cos)), 4),
            "p99": round(float(np.percentile(max_cos, 99)), 4),
            "max": round(float(max_cos.max()), 4),
        }
        for (s, b, c) in sorted((p for p in result["splits"][split]["pairs_ge_0.95"]),
                                key=lambda x: -x[2]):
            pairs.append(f"{s} <-> {b}: {c}")
        result["splits"][split]["top_pairs_ge_0.95"] = pairs[:50]

    OUT.write_text(json.dumps(result, indent=1), encoding="utf-8")
    print(f"report: {OUT}")


if __name__ == "__main__":
    try:
        main()
    except ImportError as exc:
        print(f"MISSING DEPENDENCY: {exc}\ninstall with: "
              f".venv/Scripts/python.exe -m pip install open-clip-torch",
              file=sys.stderr)
        raise SystemExit(1) from exc
