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

## Run C — next dataset version (donor merge, 2026-10-06 rebuild)

Run C is a DATA change only: the locked Run A/B settings (seed 42, epochs 100,
patience 20, imgsz 960) are untouched, and the frozen CP4 dataset (7,368/900/
1,160; 34,048 boxes; the archived `roboat-processed.zip` + private Kaggle
dataset) remains the Run A/B artifact. **Run A/B numbers are NOT comparable to
Run C**: the train/val/test sets, the RFS decision and the aerial-cap outcome
all change.

Verified Run C dataset (verifier ALL PASS, 24 checks, 2026-10-06):

- 17,869 images (train 14,507 / val 1,590 / test 1,772), 49,694 boxes
  (litter(0) 37,986 / hyacinth(1) 11,708; per split: train 29,828+9,858,
  val 4,112+867, test 4,044+985).
- Added CC BY 4.0 donors (surface imagery, post pHash-dedupe hd<=8
  first-seen; 762 navsci_invasive segmentation-polygon label files DROPPED,
  not converted): Mendeley j26w4m645z.2 -> 947 imgs / 4,176 boxes (own split
  honored: 665/179/103 — every group sits in exactly one donor split);
  Navsci invasive-aquatic-plants v12 -> 4,693 imgs / 4,693 boxes; Navsci
  water-hyacinth-detection v1 -> 411 imgs / 417 boxes. The two Navsci donors
  share one group per original image (rf-hash-stripped stem, case-insensitive)
  and are split-assigned JOINTLY (0.8/0.1/0.1 group-level) so the 371
  shared-source photos still present in both donors cannot leak across splits.
- **"Operational mat class" mapping (USER DECISION 2026-10-06)**: Mendeley
  floating_waste -> litter(0), river_vegetation -> hyacinth(1); Navsci
  water_hyacinth, Giant Salvinia and Water Lettuce -> ALL hyacinth(1) — i.e.
  class 1 is the "floating vegetation mat" operational class, not the species
  (invasive v12 has only Water Lettuce + water_hyacinth; 'Giant Salvinia' is
  absent from that archive version, so that rule is vacuous there). Raw class
  ids are mapped through each archive's own data.yaml, never assumed
  (mendeley river_vegetation=1, invasive water_hyacinth=1, whd
  water_hyacinth=0).
- **RFS DROPPED for Run C (D7 override, USER DECISION 2026-10-06)**: the donor
  merge IS the rebalancing — stacking RFS on top would double-correct. The build
  now defaults to no RFS (`train.txt` = `train_base` = 14,505 lines, one pass);
  `build_dataset.py --rfs` re-enables the extended grid (2.5/3.0/4.0 added) as an
  ablation. Balance wording, level-explicit: **image-level** f_litter 0.5235 /
  f_hyacinth 0.4949 (near parity — that is what retired RFS); **box-level**
  hyacinth is 11,708 / 49,694 = 23.6% (~1:3 vs litter), which is expected since
  hyacinth boxes are larger clusters. Do not call the dataset "balanced" without
  saying which level. The old [3,6] r_hyacinth window (D7) is superseded for Run
  C; the rfs_table.csv still records the full grid for reference.
- **Aerial cap**: the 35% cap no longer fires — donors (surface imagery,
  deliberately excluded from the cap) dilute the combined aerial share to
  0.326, so all 2,049 saigon train tiles CP4 had dropped are retained (0
  dropped in Run C).
- **Background negatives**: the D8 background budget scales with train size
  (`budget = int(0.10 * n_train)`), so it grew 701 → 1,381 with the donors and
  the emitted fml background tiles went 350 → 690 (+340; `bg_used` in
  build_stats.json). Reconciliation: 9,428 (CP4) + 6,051 donors + 2,049
  retained saigon train tiles + 340 extra background = 17,868.
- Launch commands are unchanged (`kaggle/train_baseline.py --data-root ...`);
  per-source eval gained `mendeley_c2`, `navsci_invasive_c2`, `navsci_whd_c2`
  yamls. If Run C is uploaded, build a NEW versioned Kaggle dataset (do not
  overwrite the CP4 `roboat-processed` archive Run A/B train from).
- **Baselines for Run C (a lone Run C mAP proves nothing)**: train at least the
  COCO-init control on the SAME Run C data (`train_baseline.py` default legs =
  Run C-control), and report metrics **per donor** (`mendeley_c2`,
  `navsci_invasive_c2`, `navsci_whd_c2`) as well as per source — the new test
  split is dominated by web-donor imagery rather than deployment-like water.
  Comparators for the Run C gate: Run C-control vs Run C-zenodo-init, and the
  CP4-trained Run A/B re-scored on the CP4 test split for the architecture
  story (they cannot be compared to Run C numbers directly).

## Wall-clock budget & resume

Run C's un-RFS'd train list is 14,507 images at imgsz 960 on a T4 ≈
20–28 min/epoch → 100 epochs ≈ **33–47 h per run**. That outlasts a single
free Colab session (~12 h) and a single Kaggle GPU session (~9–12 h): plan
**~3–4 free sessions per run with resume**, or use Kaggle's ~30 GPU-h/week.

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

## Run C leakage — CLIP embedding check (closed, 2026-10-06)

`scripts/embedding_leakage_check.py` (CLIP ViT-B-32-quickgelu, CPU) embedded all
17,869 images and scored every val/test image against the full train bank:
val max-cosine median 0.9358 / p99 0.9773 / max 0.9854; **381 of 1,772 test
images (21.5%) at >= 0.95 to some train image**. Pair classification:
**zero same-video/same-group cross-split pairs** (Mendeley: 0; the Navsci joint
group rule holds) — the mass is Saigon-tile vs Saigon-tile (259), TUD-GV/FML/
Hagenbeek same-river different-session frames (288), Mendeley same-river
different-video shots (135), and 5 whd<->invasive shared-plant pairs (joint-
grouped). This is the CP4-documented "same-looking water" phenomenon (median
0.9358 then, 0.9358 now) amplified by tiling and 1 fps video frames — visual
similarity, not copy leakage. Report: `data/processed/embedding_leakage_report.json`.
Label fixes: 4 val/test hyacinth->litter relabels on litter-dominant crops
(`data/processed/donors/_label_fixes_valtest.json`); 44/48 smallest val/test
hyacinth crops reviewed and confirmed genuine.
