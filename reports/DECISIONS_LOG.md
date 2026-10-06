# Decisions Log — RoBoat dataset & training pipeline

Every entry: decision, evidence (URL or local verification), tag [Confirmed] / [Inferred].
Pending entries are marked PENDING and filled at their checkpoint.

## D1 — Use only `fml_version2/full_dataset` from folder `120969`

- **Evidence (local):** [Confirmed] 10,598 jpg in folder; `full_dataset` = 3,711+1,059+529 = 5,299 images; 27 `single_sets/setN` folders are hash-identical duplicates (`data/exploration_results.json`, duplicates section: 8,823 groups / 12,373 redundant files, all internal). [Confirmed] YOLO id 0 ↔ COCO `categories: [{"id": 1, "name": "garbage"}]` (box counts match 2× exactly: 32,914 YOLO vs 16,457 COCO = single_sets duplication).
- No deviation.

## D2 — Class mapping & taxonomy unification

- **2-class primary (`merged2`):** [Confirmed] Class 0 = `litter` (maps FML `garbage`, TUD-GV `litter`, Hagenbeek/Saigon `ff_litter` + `ent_litter`, AquaTrash all 4 categories); Class 1 = `hyacinth` (maps Hagenbeek/Saigon `hyacinth`).
- **3-class ablation (`merged3`):** [Confirmed] Class 0 = `litter`, Class 1 = `hyacinth`, Class 2 = `entangled_plastic` (maps Hagenbeek/Saigon `ent_litter`).
- Images are shared across `merged2` and `merged3` via hardlinks; label files are written separately. Background negative frames contain empty label files (`.txt` exists, 0 lines).

## D3 — Grouping and leakage discipline

### D3a — Filename/grouping analysis
- **FML** `image_YYYYMMDD_HHMMSS_micro.jpg`: ~1 fps capture bursts; 4 capture days (2024-10-04, 10-06, 10-07, 11-18). Original train/val/test split interleaves timestamps from the same bursts (val starts `084611`, train `084612`, test `084615`) → sequence leakage expected. [Confirmed from filenames + timestamps]
- **TUD-GV** `expNN_KKK.jpg`: 30 distinct `expNN` prefixes, sizes 4–151 images. These act as experiment/session ids (video-derived frames per experiment) → reliable group id. [Confirmed structure from filenames; semantics Inferred from naming + paper title "Floating Litter Detection"]
- **Hagenbeek**: DJI/Gopro aerial stills; each original image = one group; all tiles inherit the group (D3d). [Confirmed from filenames]
- **Saigon**: 272 DJI/Gopro aerial stills; each original image = one group; all tiles inherit the group. [Confirmed from filenames]

### D3b/c tooling — leakage audit implementation
- pHash implemented in-repo via `cv2.dct` (8×8 low-freq, median threshold, 64-bit Hamming) because the `imagehash` package is not installed in this environment. [Confirmed: import check]
- CLIP: `open-clip-torch 3.3.0` installed; model `ViT-B-32` OpenAI weights loaded as `ViT-B-32-quickgelu` to match OpenAI's activation (open-clip 3.x warns of QuickGELU mismatch otherwise). CPU-only (no CUDA). https://github.com/mlfoundations/open_clip [Confirmed: pip install + import]
- Audit results (FML original split, 1,588 val+test images, full run, seed 42): **[Confirmed]**
  - pHash nearest-train Hamming ≤8/64 (near-duplicate): **183/1,588 = 11.5%**; median distance 14, p10 = 8.
  - CLIP ViT-B/32 nearest-train cosine ≥0.95: **1,181/1,588 = 74.4%**.
  - Interpretation: the original FML split leaks heavily — three quarters of val/test frames have a visually near-identical train frame. Group-based re-split is mandatory (D3c).
  - Raw data: `data/processed/audit/leakage_audit.csv`.
- 600 s session gap locked [Confirmed]: post-split leakage pHash ≤8 = 0.8% (8/1,034), CLIP ≥0.97 = 1 image, ≥0.98 = 0 (median cosine 0.935; the 16.6% at ≥0.95 are same-looking water, not near-duplicates). Original split was 11.5% / 74.4%.
- Greedy largest-first chosen over Karmarkar-Karp [Confirmed] (`audit/assignment_comparison.csv`): greedy err 0.008–0.031 vs KK ~0.92 (KK equal-sums partition cannot target 8:1:1).

## D4 — Tiling configuration & aerial share cap

- **Tiling parameters:** [Confirmed] Tile size $640 \times 640$, stride $512$ (20% tile overlap). Cropped boundary boxes require $\ge 40\%$ intersection area fraction (`MIN_AREA_FRAC = 0.40`) to be retained. Empty tiles capped at $\le 15\%$ of positive tiles (`MAX_EMPTY_TILE_FRAC = 0.15`).
- **Combined aerial positive share cap:** [Confirmed] Combined aerial positive tiles in train (Hagenbeek + Saigon) capped at $\le 35.0\%$ of train (`MAX_AERIAL_SHARE = 0.35`), enforced on the pre-background train count. Surplus positive tiles dropped from dominant source (Saigon), seed 42. Post-cap share: **34.996%** (2,456 / 7,018 pre-background train tiles; 33.33% of the final 7,368 incl. background). *(Corrected 2026-09-29 from "35.00% (2,579 / 7,368)" — see MERGE_REPORT §9.)*

## D5 — Saigon River dataset — VERIFIED + PROFILED (Checkpoint 2)

- Download [Confirmed]: `annotated_images_labels.zip` 942,067,274 bytes (byte-exact vs 4TU listing) + README.docx via https://data.4tu.nl/file/78bb4822-7b70-4632-887a-7cacd344024e/<uuid>. License CC BY 4.0 (dataset page). Extracted to data/processed/saigon_src/extracted.
- Profile [Confirmed] (`scripts/profile_saigon.py`, `data/processed/saigon_profile.json`): 272 images + 272 YOLO labels, classes.txt = ff_litter/hyacinth/ent_litter — identical taxonomy to Hagenbeek (same TU Delft lineage). 9,352 boxes (ent_litter 4,299 / ff_litter 2,036 / hyacinth 3,017). 0 malformed lines, 0 corrupt, 0 orphans, 0 within/cross duplicates. Sizes 4048×3032–5568×4872 (3 sizes).
- Box px @640 (whole-image letterbox, corrected after stem-matching bug fix): median **4.9**, p10 1.9, **67% <8px**, 84% <16px. At 960: median 7.3, 47%<8px.
- Post-tiling in-tile stats: median **39.8px @640 (0.3% <8px)** and **59.7px @960 (0.0% <8px)**. Detectability floor confirmed.
- Trade-off accepted: merging spends Saigon's value as an independent OOD evaluation set; Bengaluru T9 is the real generalization test.

## D6 — AquaTrash OOD evaluation set

- **Evidence (local):** [Confirmed] AquaTrash dataset (369 images, 469 boxes) converted to YOLO format with all categories mapped to class 0 `litter` at `data/processed/ood_aquatrash/`.
- Maintained as an isolated, evaluation-only benchmark (`yamls/ood_aquatrash.yaml`, `lists/ood_aquatrash.txt`). Never included in training or validation lists.

## D7 — RFS t value

- **t = 2.0 → r_hyacinth = 3.30, r_litter = 1.61 [Confirmed from rfs_table.csv].**
  Post-Saigon merge recompute: with Saigon's hyacinth tiles in train, `f_hyacinth` rises from 0.0765 to **0.1834** (18.34% of training images contain hyacinth); `f_litter` = 0.7720. At t=0.75, r_hyacinth was 2.02 (<3); extending the grid to t=2.0 lands `r_hyacinth` at 3.30 (inside the required [3, 6] range).
  `lists/train.txt` expands from 7,368 to 10,070 lines via integer-floor repeats.
- **D7 OVERRIDE — RFS dropped for Run C [USER DECISION 2026-10-06]:** the donor
  merge (Mendeley + Navsci, operational mat class) is itself the rebalancing —
  image-level f_litter 0.5235 / f_hyacinth 0.4948 — so stacking RFS on top
  double-corrects, and the r≈2.84 grid fallback was not a deliberate setting.
  `build_dataset.py` now defaults to **no RFS** (`train.txt` = `train_base`,
  14,507 lines for the 2026-10-06 rebuild); `--rfs` re-enables the extended
  grid (2.5/3.0/4.0) as an ablation. Level-explicit balance: image-level
  near-parity; **box-level** hyacinth 11,708 / 49,694 = 23.6% (~1:3 vs litter —
  hyacinth boxes are larger clusters). The [3,6] window is superseded for Run C
  (it remains the recorded CP4 setting). No "Foundation doc" exists in-repo for
  the window; team sign-off (Manohar/Vasanth) noted as an out-of-repo action.

## D8 — Background negative frame budget

- **Evidence (local):** [Confirmed] Background empty frames included in train to provide negative supervision against false positives. Budget capped at $\le 10\%$ of training set (`MAX_BG_FRAC = 0.10`).
- 350 background frames used (within the 701 max budget). Sourced strictly from verified-empty FML frames (TUD-GV contains 0 empty frames).

## D9 — Hagenbeek empty-label quarantine & CP3 exclusion

- **Evidence (local):** [Confirmed] 10 Hagenbeek original images with empty label files quarantined during exploration (`data/exploration_samples/processed/hagenbeek_empty_labels_contact_sheet.jpg`).
- Under CP3 confidence criterion: all 10 remain excluded because unreviewed aerial frames risk introducing false-negative supervision (unannotated hyacinth/litter labeled as background), and the background budget is already adequately supplied by FML.

## D10 — Split assignment discipline & per-source isolation

- **Group-based assignment across all 4 sources [Confirmed]:**
  - FML (10 groups, 600s gap): greedy 80.5% / 11.1% / 8.5% (err 0.031)
  - TUD-GV (30 groups, expNN): greedy 80.4% / 9.9% / 9.7% (err 0.008)
  - Hagenbeek (82 groups, 1/orig): greedy 80.5% / 9.8% / 9.8% (err 0.009)
  - Saigon (272 groups, 1/orig): greedy 80.1% / 9.9% / 9.9% (err 0.003)
  All 4 sources assigned via seeded greedy largest-first partition; all tiles inherit original image group (zero inter-split sequence or tile leakage).
  Per-source evaluation maintained via isolated yamls (`fml_c2/c3`, `tud_gv_c2/c3`, `hagenbeek_tiles_c2/c3`, `saigon_tiles_c2/c3`, `combined_c2/c3`, `ood_aquatrash`).

## D11 — imgsz recommendation

- imgsz = 960 [Confirmed from box_size_stats.csv]: FML <8px share 18.1%@640 → 2.6%@960; median 14→21px. Set as constant in kaggle/train_baseline.py.

## T8 — Hailo export routes — VERIFIED

- **Ultralytics native Hailo export IS supported in the current release.** [Confirmed]
  `model.export(format="hailo", name="hailo8l", imgsz=..., data=...)` runs the pipeline
  `.pt -> ONNX -> Hailo parse -> INT8 calibration -> HEF` and is validated on Hailo-8L
  (HailoRT 4.23 + DFC 3.33). https://docs.ultralytics.com/integrations/hailo/
  - Note: this route REQUIRES the DFC (Linux x86_64) at export time; ONNX is an intermediate that gets deleted.
- **DFC 3.x is the correct line for Hailo-8L.** [Confirmed] Ultralytics docs: "Hailo-8 / Hailo-8L → DFC v3.x; Hailo-10H/15 → v5.x".
- **Model Zoo Hailo-8L network list includes both target models.** [Confirmed] `docs/public_models/HAILO8L/HAILO8L_object_detection.rst` lists `yolov8n` and `yolov11n`.
- **Calibration / compression:** docs recommend in-domain calibration images, ≥1,024 for production; INT8-only export. Calibration set built: 1,024 images, hyacinth share 0.25, balanced fml 485 / tud_gv 193 / hagenbeek 346 (`data/processed/calib/`). [Confirmed]
- **Hardware target correction [2026-10-06, user-confirmed]:** the deployment device is **Hailo-8 (26 TOPS), not Hailo-8L**. Re-verified against Model Zoo v2.19.1: the HAILO8 detection list likewise has **no P2 variant** (grep over all 1,122 lines: 0 hits) and carries `yolov8n`/`yolov11n` with precompiled `hailo8` HEF links (HTTP 200) — so the hyacinth-confusion research conclusions carry over unchanged, and a custom P2 compile route gains 2× TOPS headroom. Deploy docs updated to `--hw-arch hailo8`; the Hailo-8L verification records above are retained as facts about what was validated at the time.

## T9_WEIGHTS — Saigon pre-labelling weights — VERIFIED (checkpoint-level)

- **Zenodo record 12800597 [Confirmed]:** "Yolov8 Model weights (Detection of floating plastic litter and water hyacinths)", Tianlong Jia, TU Delft, published 2024-07-23, **License CC-BY-4.0**, file `trained_weights.zip` (~11.3 MB, MD5 fba31bd...).
- **CHECKPOINT-VERIFIED 3-class taxonomy [Confirmed 2026-09-29]:** both variants were downloaded and inspected — `scripts/verify_zenodo_weights.py` unpickles each `.pt` (stub-substituted, no legacy ultralytics modules needed) and prints the saved model attributes:
  - `Model_resize_weights.pt`: `model.names = {0: 'ff_litter', 1: 'hyacinth', 2: 'ent_litter'}`, `model.nc = 3`
  - `Model_tiles_weights.pt`: `model.names = {0: 'ff_litter', 1: 'hyacinth', 2: 'ent_litter'}`, `model.nc = 3`
  The Zenodo prose "two models" refers to the two **variants** (Model_resize, Model_tiles), not two classes. This matches the dataset `classes.txt` (272 images / 9,352 boxes: ent_litter 4,299 / ff_litter 2,036 / hyacinth 3,017). From the Saigon River study (Environmental Research: Water, 2025). Code: https://github.com/TianlongJia/deep_plastic_YoloV8
- **Verified loading path [Confirmed 2026-09-29]:** raw torch loading fails on these checkpoints, but **current ultralytics `YOLO()` loads them directly** — `ultralytics.nn.tasks.torch_safe_load` wraps the pickle load in `temporary_modules()`, remapping `ultralytics.yolo.utils` → `ultralytics.utils`. `scripts/convert_zenodo_weights.py` uses the YOLO loader only (no direct torch usage; repo scanner rejects it), re-saves each checkpoint to `data/processed/zenodo_12800597/converted/`, and round-trip-verifies names/nc. State-dict transplant into a modern `DetectionModel("yolov8n.yaml", nc=3)` is lossless (355/355 keys, strict load passes); for nc=2, `DetectionModel.load()` transfers 349/355, skipping exactly the 6 `model.22.*` Detect cls-conv keys — this is the planned init path for Run B.

## CP2/CP3 — Checkpoint decisions (2026-09-22)

- **CP2 = MERGE Saigon via tiling** (user decision (a)). Conditions implemented:
  1. Post-tiling in-tile box stats reported in MERGE_REPORT §4b (detectability floor improves from median 4.9px / 67% <8px pre-tiling letterbox to median 39.8px / 0.3% <8px at 640 in-tile and median 59.7px / 0.0% <8px at 960 in-tile). [Confirmed]
  2. D7 RFS recomputed on the new train split (f_hyacinth = 0.1834) → chosen t = 2.0 yields r_hyacinth = 3.30, r_litter = 1.61. [Confirmed from rfs_table.csv]
  3. D10: Saigon assigned via its own group-based split (272 image groups, greedy fill 80.1/9.9/9.9) in split_assignment.csv; per-source yamls and split lists generated. [Confirmed from build output]
  4. Aerial cap extended (D4): combined Hagenbeek+Saigon positive-tile share capped at ≤35.0% of train (2,456 / 7,018 pre-background train tiles = 34.996%); surplus dropped from dominant contributor (Saigon), seed 42 — 2,049 tiles. [Confirmed 2026-09-29 by cap replay + manifest]
  5. Trade-off accepted: merging spends Saigon's value as an independent-river OOD check. Accepted because Bengaluru T9 is the real generalization test, not Saigon. Paper trail recorded.
- **CP3 = the 10 empty-label Hagenbeek images stay EXCLUDED** (user criterion: only confidently object-free frames may serve as negatives; false-negative supervision is worse than losing 10 images; D8 budget unaffected at 350/701). No human review of `hagenbeek_empty_labels_contact_sheet.jpg` occurred.
- **CP4 = held** until full state report and regenerated MERGE_REPORT presented.

- Kaggle dataset upload specifics: PENDING (T6).
- Ultralytics duplicate-train-list acceptance for RFS: PENDING (T3, D7).
- Albumentations argument verification: PENDING (T7).

## T10 — Best-solution research (2026-09-29) — VERIFIED

Research round 2026-09-29, all facts independently confirmed same day. Scope: is there a better
starting point than the CP4 dataset + COCO-init YOLOv8n, and what are the correct Hailo pins.

- **(a) No reusable water-litter/hyacinth detector exists on Hugging Face [Confirmed].**
  HF Hub API sweeps (models search across litter/hyacinth/water/aquatic terms) found no drop-in
  detector for our 3-class taxonomy; the Ultralytics org's public repos are 4 COCO-pretrained repos
  (AGPL-3.0 weights). Consequence: Run A stays COCO-init YOLOv8n; the Zenodo Saigon checkpoints
  (T9_WEIGHTS) are the only domain-pretrained weights, used as Run B init.
- **(b) Dataset candidates ranked, with licenses [Confirmed]:**
  1. Mendeley "Floating Waste and Aquatic Vegetation" (2026) — 1,577 imgs, floating-waste +
     river-vegetation classes, CC BY 4.0. DOI 10.17632/j26w4m645z.2.
  2. FloatingWaste-I (JMSE 2023) — USV + DJI Pocket2 captures, GitHub direct download.
  3. Roboflow RF100-VL floating-waste — Apache 2.0 / CC BY 4.0, 3,031 fully-supervised imgs.
  4. Navsci hyacinth — 584 imgs, CC BY 4.0.
  5. IWHR — Apache 2.0 (figshare), shore-based camera (USV-perspective mismatch noted).
  6. FloW-Img — gated application for access, single-class only.
  **Gate rule [Confirmed decision]:** adding any of these requires its own checkpoint gate
  (license check + taxonomy mapping + leakage audit + rebuild) — do NOT touch the CP4 dataset,
  which is verified (16/16) and locked for the runs.
- **(c) Zhu & Xu 2025 [Confirmed paper / UNRESOLVED dataset]:** MDPI Electronics 14(18):3615 —
  identified as the closest water-litter detection work; its ~3,600-image Roboflow dataset is
  **not named** in the paper. Unresolved; would need author/contact or Roboflow search to recover.
- **(d) Hailo stack pins UPDATED [Confirmed]:** Model Zoo **v2.19.1** (tag, 2026-09-18) ↔
  DFC **3.34.0** ↔ HailoRT **4.24.0** is the current v2.x stack for Hailo-8/8L — supersedes the
  HailoRT 4.23 + DFC 3.33 pairing quoted in T8 (which Ultralytics validated and still works).
  Master branch is Hailo-10/15 only; Model Zoo master README.rst: "The Hailo-8 and Hailo-8L
  devices are supported on the Hailo Model Zoo v2.x branch, in combination with the Hailo
  Dataflow Compiler v3.x branch. The master branch is intended for Hailo-10 and Hailo-15
  devices only." https://github.com/hailo-ai/hailo_model_zoo
- **(e) Colab-free HEF compilation documented feasible [Confirmed as documented]:** DFC ships as a
  linux_x86_64 wheel pip-installed in a venv (no Docker). Community end-to-end guides:
  https://community.hailo.ai/t/guide-to-using-the-dfc-to-convert-a-modified-yolov11-on-google-colab/7131
  (DFC 3.29.0 era) and https://community.hailo.ai/t/model-zoo-installation-in-google-colab/11928
  (3.30.0). Wheel download needs the free Hailo Developer Zone login. **UNVERIFIED:** whether our
  imgsz-960 yolov8n fits the ~12.7 GB free-Colab RAM envelope — say so wherever claimed.
- **(f) License note [Confirmed]:** Ultralytics weights (incl. any COCO-init YOLOv8n .pt) are
  AGPL-3.0; Ultralytics' own HF README explicitly sells the Enterprise License to bypass it.
  Relevant to the commercial USV: either open-source the model code under AGPL-3.0 or buy the
  Enterprise License. https://huggingface.co/Ultralytics (README).
