# A/B training run plan (prepared, NOT yet run)

Two controlled runs on the verified 2-class dataset (16/16 checks, 2026-09-29),
driven by `kaggle/train_baseline.py`. No training has been launched; this file
only fixes the configuration.

## Common settings (locked, do not tune)

seed 42, epochs 100, patience 20, imgsz 960, deterministic=True
(`kaggle/train_baseline.py:22` for `EPOCHS, PATIENCE, SEED`, line 21 for
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
  layout — `merged2 lists yamls` + `zenodo_12800597/converted` — but omits
  `ood_aquatrash` (that OOD image tree is not shipped); the script's
  per-source val loop skips the missing `ood_aquatrash` entry, so the Run B
  command above works unchanged on the bundle mount.
- Ultralytics transfers what fits: for nc=2 that is 349/355 state-dict keys,
  skipping exactly the 6 `model.22.*` Detect cls-conv keys (verified). No
  manual surgery needed — pass the checkpoint to `--pretrained` and let
  `YOLO()` + `model.train()` do the transfer.
- Note: the converted checkpoint is a yolov8n architecture, so the script
  runs only the yolov8n leg when `--pretrained` is set (it prints the skipped
  entries and never trains the `yolo11n` entry under a wrong label) — Run B
  is the yolov8n run.

## P1 gate (written requirement)

Report val mAP50 **per class** AND **per source** (the `saigon_tiles_c2.yaml`
and `hagenbeek_tiles_c2.yaml` yamls already exist under
`data/processed/yamls/`; `saigon_tiles_c2` is in the script's
`PER_SOURCE_YAMLS`), not one aggregate. The script prints both tables itself:
an aggregate per-source table plus a per-class mAP@0.5 line for every
source × model. The gate passes only when both tables have been produced for
both runs.

## P2 note (interpretation caveat)

The val split is same-site and will flatter the model. The Bengaluru Capture
Set (T9) is the real test — score it BEFORE any fine-tune on it.

## Colab caveat

Whether 100 epochs at imgsz 960 fits a free Colab session (RAM and ~12 h
session limit) is **unverified** — check before P1 launches. (Compilation of
HEFs on free Colab is separately known to work via the DFC linux_x86_64 wheel;
that does not answer the training-fit question.)

## License note

Ultralytics-pretrained weights (including `yolov8n.pt` and anything derived
from the Zenodo checkpoints trained on them) are AGPL-3.0 licensed, which
requires open-sourcing derivative model code for the commercial USV unless an
Ultralytics Enterprise License is purchased.
