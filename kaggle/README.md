# Kaggle — RoBoat training

## 1. Zip and upload `data/processed` as a Kaggle Dataset

The dataset trees use **hardlinks** (Windows-compatible, no symlinks), so a plain
zip resolves them to real file copies — just zip normally:

```powershell
# from repo root, EXCLUDING intermediate/audit stuff Kaggle doesn't need
cd data\processed
tar -a -c -f roboat-processed.zip merged2 lists yamls ood_aquatrash zenodo_12800597/converted
```

- Size ≈ 3.7k unique FML images + 1.2k TUD-GV + 1.2k Hagenbeek tiles (JPEG).
- Simpler alternative: run `scripts/build_kaggle_bundle.py` — it produces
  `data/processed/kaggle_bundle/roboat-processed.zip` with the same top-level
  layout (`merged2 lists yamls` + `zenodo_12800597/converted`), minus
  `ood_aquatrash` (that OOD image tree is not shipped; `train_baseline.py`
  skips the missing per-source yaml, so the report still prints).
- `zenodo_12800597/converted` carries the modern-format Zenodo checkpoints
  produced by `scripts/convert_zenodo_weights.py` (needed for Run B's
  `--pretrained`, see `kaggle/RUN_PLAN.md`).
- Go to https://www.kaggle.com/datasets → **New Dataset**, upload
  `roboat-processed.zip`, name it `roboat-processed`, make it **private**
  (contains licensed research data: TUD-GV (Zenodo), FML v2, Hagenbeek).
- License metadata: keep private; individual source licenses are in
  `reports/DECISIONS_LOG.md`.

## 2. Enable GPU

Notebook settings → Accelerator → **GPU T4 x2** or P100. (CPU-only will take ~a
day per run; imgsz=960 needs the GPU.)

## 3. Run the baseline

```bash
pip install -q ultralytics
python train_baseline.py --data-root /kaggle/input/roboat-processed
```

- Models: `yolov8n`, `yolo11n` — DEFAULT hyperparameters only (epochs=100,
  patience=20), `imgsz=960` (from the D11 box-size audit), seed 42.
- After each model it validates on every per-source test yaml
  (`saigon_tiles_c2`, `hagenbeek_tiles_c2`, `fml_c2`, `tud_gv_c2`,
  `combined_c2`, `ood_aquatrash`) and prints per-class AND per-source
  mAP@0.5 tables. A source whose yaml is missing or fails to validate is
  skipped with a printed note (never crashes the run).
- Outputs in `roboat/baseline_<model>/` → download `best.pt`.

## 4. Optional augmented run

```bash
python train_aug.py --data-root /kaggle/input/roboat-processed
```

Runs a smoke test verifying the Albumentations transform names against the
installed version first, then trains yolov8n with a conservative pipeline
(flip / small affine / color jitter / light crop / shadows) and copies
`train_batch*.jpg` into `roboat/` so you can visually confirm the transforms.

## 5. What to bring back

`best.pt` from each run + the printed per-source mAP tables (also saved in the
run dirs). Deploy path continues in `deploy/EXPORT_HAILO.md`.
