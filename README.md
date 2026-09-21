# RoBoat — Floating Litter & Water Hyacinth Detection

USV (unmanned surface vehicle) detection pipeline for floating litter and water
hyacinth on Bengaluru lakes. Train YOLO on Kaggle → export ONNX → compile to
Hailo-8L HEF → run on Raspberry Pi 5.

## Pipeline

```
data/raw (read-only, licensed datasets)
  └─ scripts/explore_data.py ............ dataset audit → data/DATA_EXPLORATION.md
  └─ scripts/audit_splits.py ............ leakage audit (pHash + CLIP), grouping
  └─ scripts/assign_splits.py ........... group→split (greedy vs KK), re-check
  └─ scripts/build_dataset.py ........... merged YOLO trees, tiling, RFS, yamls
  └─ scripts/verify_dataset.py .......... assertions + contact sheets
  └─ scripts/profile_saigon.py .......... Saigon River dataset profile (D5)
data/processed (gitignored)
  ├─ merged2/  PRIMARY 2-class tree   (0 litter, 1 hyacinth)
  ├─ merged3/  ABLATION 3-class tree  (+ 2 entangled_plastic)
  ├─ lists/ yamls/ manifest.csv calib/ ood_aquatrash/
kaggle/ ....... train_baseline.py, train_aug.py, README (upload steps)
deploy/ ....... export_onnx.py, make_calib_set.py, EXPORT_HAILO.md
docs/ ......... BENGALURU_CAPTURE.md (field capture protocol)
reports/ ...... DATA_EXPLORATION, MERGE_REPORT, DECISIONS_LOG, AGENTS_LOG
```

## Quickstart

```bash
# 1. audit + split + build + verify (data/raw must be populated)
python scripts/audit_splits.py            # leakage audit + grouping (use --skip-clip to skip CLIP)
python scripts/assign_splits.py           # assign 80/10/10 by group + leakage re-check
python scripts/build_dataset.py           # --dry-run first; writes data/processed
python scripts/verify_dataset.py          # must end ALL PASS
python deploy/make_calib_set.py           # 1,024-image INT8 calibration set

# 2. train on Kaggle — see kaggle/README.md (zip data/processed, upload, run)
python kaggle/train_baseline.py --data-root /kaggle/input/roboat-processed

# 3. deploy — see deploy/EXPORT_HAILO.md (ONNX → hailomz → HEF, Hailo-8L)
```

## Key decisions (details + citations in reports/DECISIONS_LOG.md)

- Group-based splits only (video-frame leakage: original FML split leaked 74%
  CLIP near-duplicates; group re-split → 0.8% pHash).
- imgsz **960** from box-size audit (FML <8px share 18.1%@640 → 2.6%@960).
- 2-class primary (`litter`, `hyacinth`) + 3-class ablation
  (`entangled_plastic`); RFS t=0.75 (r_hyacinth≈3.1) for imbalance.
- Hagenbeek aerial images tiled 640/512 (≥40% area rule), capped share.
- AquaTrash = out-of-domain eval only. Saigon profiled, not merged (pending).

## Reproducibility

Seed 42 everywhere; every generator script supports `--dry-run` and writes a
manifest CSV. Dataset rebuilds are deterministic given the same data/raw.

## License / data

Code: project's own. Datasets in data/raw keep their upstream licenses (TUD-GV
Zenodo, FML v2, Hagenbeek/Wageningen, AquaTrash MIT, Saigon CC-BY-4.0) — keep
the Kaggle dataset private. Model weights derived from them inherit the
strictest applicable terms.
