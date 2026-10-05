# A/B training run plan (prepared, NOT yet run)

Two controlled runs on the verified 2-class dataset (16/16 checks, 2026-09-29),
driven by `kaggle/train_baseline.py`. No training has been launched; this file
only fixes the configuration.

## Common settings (locked, do not tune)

seed 42, epochs 100, patience 20, imgsz 960, deterministic=True
(`kaggle/train_baseline.py:25` for `EPOCHS, PATIENCE, SEED`, line 24 for
`IMGSZ`, and the `model.train(...)` call). Primary train yaml is the 2-class
`combined_c2.yaml`; all val/test yamls are the 2-class variants
(`*_c2.yaml`).

## Run A — YOLOv8n, COCO init

```bash
python train_baseline.py --data-root /kaggle/input/roboat-processed
```

`yolov8n` starts from its stock COCO `yolov8n.pt` (the script default,
`kaggle/train_baseline.py:74`).

## Run B — YOLOv8n, init from converted Model_tiles checkpoint

```bash
python train_baseline.py --data-root /kaggle/input/roboat-processed \
    --pretrained /kaggle/input/roboat-processed/zenodo_12800597/converted/Model_tiles_weights_converted.pt
```

- The Zenodo checkpoint (`Model_tiles_weights.pt`, ultralytics 8.0.36, nc=3)
  must first be converted via the `YOLO()` loader and re-saved with
  `model.save()` — raw `torch.load` fails on the old `ultralytics.yolo.utils`
  module paths. Conversion + re-load already verified working.
- The converter (`scripts/convert_zenodo_weights.py`) writes converted copies
  to `data/processed/zenodo_12800597/converted/<stem>_converted.pt` — that is
  the path above, **not** `extracted/trained_weights/`. When building the
  Kaggle dataset zip, include that `converted/` directory alongside
  `merged2 lists yamls ood_aquatrash` (see `kaggle/README.md` §1).
  The ready-made zip from `scripts/build_kaggle_bundle.py`
  (`data/processed/kaggle_bundle/roboat-processed.zip`) keeps exactly this
  layout — `merged2 lists yamls ood_aquatrash` + `zenodo_12800597/converted` —
  including the `ood_aquatrash` OOD tree (369 images + 369 labels, eval-only)
  so the D6 benchmark is evaluable on Kaggle/Colab.
- Ultralytics transfers what fits: for nc=2 that is 349/355 state-dict keys,
  skipping exactly the 6 `model.22.*` Detect cls-conv keys (verified). No
  manual surgery needed — pass the checkpoint to `--pretrained` and let
  `YOLO()` + `model.train()` do the transfer.
- Note: the converted checkpoint is a yolov8n architecture, so the script
  runs only the yolov8n leg when `--pretrained` is set (it prints the skipped
  entries and never trains the `yolo11n` entry under a wrong label) — Run B
  is the yolov8n run.

## P1 gate (written requirement)

Report **test-split mAP50 per class AND per source** (`saigon_tiles_c2`,
`hagenbeek_tiles_c2`, `fml_c2`, `tud_gv_c2`, `combined_c2`, `ood_aquatrash`;
all six per-source yamls exist under `data/processed/yamls/` and are in the
script's `PER_SOURCE_YAMLS`), not one aggregate. The gate is evaluated on
`split="test"` — unified with `training/colab_train.ipynb` so the numbers are
comparable — because the per-source VAL sets are too small for stable
per-class numbers (Hagenbeek hyacinth has just 64 val instances); val remains
the early-stopping split only. The script prints both tables itself: an
aggregate per-source table plus a per-class mAP@0.5 line for
every source × model. The gate passes only when both tables have been produced
for both runs.

## P2 note (interpretation caveat)

The val split is same-site and will flatter the model. The Bengaluru Capture
Set (T9) is the real test — score it BEFORE any fine-tune on it.

## Planned ablations (post-gate, do not fold into Run A/B)

- **A3 "scale fix"**: `scale 0.5 → 0.2` — the default zoom-out can push tiled boxes
  back under the D11 detectability floor (median 39.8 px @640 in-tile; a 0.5 zoom-out
  halves box size).
- **A4 "aerial flips"**: `flipud 0 → 0.5` — aerial imagery has no preferred
  up-direction (small-object detection practice, cf. SAHI).
- **A2 "weather-light"** and later packages: see `docs/WEATHER_AUGMENTATION.md` —
  USV cameras face sun glare, rain, fog, spray and turbidity that our fair-weather
  training data does not contain; Albumentations 2.x maps each condition to a
  transform (`RandomSunFlare`, `RandomRain`, `RandomFog`, `RandomShadow`, `Spatter`),
  wired via the `Albumentations(transforms=[...])` hook that ultralytics 8.4.165
  natively supports. Seasonal/night conditions are NOT augmented — the Bengaluru
  Capture Set (T9) captures them for real.

## Wall-clock budget & resume

One run trains the RFS-expanded 10,070-image list at imgsz 960 on a T4 ≈
15–20 min/epoch → 100 epochs ≈ **25–35 h per run**. That outlasts a single
free Colab session (~12 h) and a single Kaggle GPU session (~9–12 h): plan
**~2–3 free sessions per run with resume**, or use Kaggle's ~30 GPU-h/week.

Resume, don't restart — `deterministic=True` + `resume=True` restores the RNG
state, so a resumed run continues the same augmentation/schedule sequence as
an uninterrupted one:

- **Colab** (`training/colab_train.ipynb`): runs write to Drive
  (`project=/content/drive/MyDrive/roboat_runs`), so `last.pt` survives the
  session. In a fresh session re-run the setup cells (1–2c), then the §3/§4
  training cells auto-detect `last.pt` and call `model.train(resume=True)`
  (there is also a dedicated "Resume an interrupted run" cell after §4).
- **Kaggle**: `/kaggle/working` persists between Save-Version runs —
  re-attach the previous session's output as an input and call
  `model.train(resume=True)` on its `last.pt`, or run
  `python train_baseline.py --resume`.

## Colab caveat

At the estimate above (25–35 h per run), 100 epochs at imgsz 960 does **not**
fit one free Colab session (~12 h) — use the resume path in
"Wall-clock budget & resume" and plan ~2–3 sessions per run. (Compilation of
HEFs on free Colab is separately known to work via the DFC linux_x86_64 wheel;
that does not answer the training-fit question.)

## License note

Ultralytics-pretrained weights (including `yolov8n.pt` and anything derived
from the Zenodo checkpoints trained on them) are AGPL-3.0 licensed, which
requires open-sourcing derivative model code for the commercial USV unless an
Ultralytics Enterprise License is purchased.
