# Decisions Log — RoBoat dataset & training pipeline

Every entry: decision, evidence (URL or local verification), tag [Confirmed] / [Inferred].
Pending entries are marked PENDING and filled at their checkpoint.

## D1 — Use only `fml_version2/full_dataset` from folder `120969`

- **Evidence (local):** [Confirmed] 10,598 jpg in folder; `full_dataset` = 3,711+1,059+529 = 5,299 images; 27 `single_sets/setN` folders are hash-identical duplicates (`data/exploration_results.json`, duplicates section: 8,823 groups / 12,373 redundant files, all internal). [Confirmed] YOLO id 0 ↔ COCO `categories: [{"id": 1, "name": "garbage"}]` (box counts match 2× exactly: 32,914 YOLO vs 16,457 COCO = single_sets duplication).
- No deviation.

## D3a — Filename/grouping analysis

- **FML** `image_YYYYMMDD_HHMMSS_micro.jpg`: ~1 fps capture bursts; 4 capture days (2024-10-04, 10-06, 10-07, 11-18). Original train/val/test split interleaves timestamps from the same bursts (val starts `084611`, train `084612`, test `084615`) → sequence leakage expected. [Confirmed from filenames + timestamps]
- **TUD-GV** `expNN_KKK.jpg`: 30 distinct `expNN` prefixes, sizes 4–151 images. These act as experiment/session ids (video-derived frames per experiment) → reliable group id. [Confirmed structure from filenames; semantics Inferred from naming + paper title "Floating Litter Detection"]
- **Hagenbeek**: DJI/Gopro aerial stills; each original image = one group; all tiles inherit the group (D3d). [Confirmed from filenames]

## D3b/c tooling — leakage audit implementation

- pHash implemented in-repo via `cv2.dct` (8×8 low-freq, median threshold, 64-bit Hamming) because the `imagehash` package is not installed in this environment. [Confirmed: import check]
- CLIP: `open-clip-torch 3.3.0` installed; model `ViT-B-32` OpenAI weights loaded as `ViT-B-32-quickgelu` to match OpenAI's activation (open-clip 3.x warns of QuickGELU mismatch otherwise). CPU-only (no CUDA). https://github.com/mlfoundations/open_clip [Confirmed: pip install + import]
- Audit results (FML original split, 1,588 val+test images, full run, seed 42): **[Confirmed]**
  - pHash nearest-train Hamming ≤8/64 (near-duplicate): **183/1,588 = 11.5%**; median
    distance 14, p10 = 8.
  - CLIP ViT-B/32 nearest-train cosine ≥0.95: **1,181/1,588 = 74.4%**.
  - Interpretation: the original FML split leaks heavily — three quarters of val/test
    frames have a visually near-identical train frame. Group-based re-split is mandatory (D3c).
  - Raw data: `data/processed/audit/leakage_audit.csv`.

## T8 — Hailo export routes — VERIFIED

- **Ultralytics native Hailo export IS supported in the current release.** [Confirmed]
  `model.export(format="hailo", name="hailo8l", imgsz=..., data=...)` runs the pipeline
  `.pt -> ONNX -> Hailo parse -> INT8 calibration -> HEF` and is validated on Hailo-8L
  (HailoRT 4.23 + DFC 3.33). https://docs.ultralytics.com/integrations/hailo/
  - Note: this route REQUIRES the DFC (Linux x86_64) at export time; ONNX is an
    intermediate that gets deleted. Per the task instruction we still document
    ONNX + `hailomz` as the primary explicit route and the native export as the
    integrated alternative. (Task deviation none — D-route choice was mandated.)
- **DFC 3.x is the correct line for Hailo-8L.** [Confirmed] Ultralytics docs:
  "Hailo-8 / Hailo-8L → DFC v3.x; Hailo-10H/15 → v5.x". Same statement in the Hailo
  Model Zoo README: "Hailo-8 and Hailo-8L devices are supported on the Hailo Model Zoo
  v2.x branch, in combination with the Hailo Dataflow Compiler v3.x branch."
  https://github.com/hailo-ai/hailo_model_zoo
- **Model Zoo Hailo-8L network list includes both target models.** [Confirmed]
  `docs/public_models/HAILO8L/HAILO8L_object_detection.rst` lists `yolov8n` and
  `yolov11n` (spelled yolov11n, i.e. Ultralytics yolo11n). Many more (yolov5s,
  yolov8s/m/l, yolov10n/s, yolov12n...). HEF compilation is Linux x86_64-only;
  Raspberry Pi 5 only runs the compiled HEF via HailoRT. [Confirmed, same docs]
- **Calibration / compression:** docs recommend in-domain calibration images, ≥1,024
  for production; INT8-only export. `compression_level=0` / 16-bit fallback to be set
  in the model script per task; forum-level citation to be added at T8 build time.
  [Confirmed for ≥1024 recommendation]

## T9 — Saigon pre-labelling weights — VERIFIED

- **Zenodo record 12800597 [Confirmed]:** "Yolov8 Model weights (Detection of floating
  plastic litter and water hyacinths)", Tianlong Jia, TU Delft, published 2024-07-23,
  **License CC-BY-4.0**, file `trained_weights.zip` (~11.3 MB, MD5 fba31bd...). Two
  models: plastic litter and water hyacinth, Saigon River study (Environmental
  Research: Water, 2025). Code: https://github.com/TianlongJia/deep_plastic_YoloV8
  https://zenodo.org/records/12800597

## D5 — Saigon River dataset — VERIFIED + PROFILED (Checkpoint 2)

- Download [Confirmed]: `annotated_images_labels.zip` 942,067,274 bytes (byte-exact
  vs 4TU listing) + README.docx via https://data.4tu.nl/file/78bb4822-7b70-4632-887a-7cacd344024e/<uuid>.
  License CC BY 4.0 (dataset page). Extracted to data/processed/saigon_src/extracted
  (data/raw is read-only, so new downloads live under data/processed).
- Profile [Confirmed] (`scripts/profile_saigon.py`, `data/processed/saigon_profile.json`):
  272 images + 272 YOLO labels, classes.txt = ff_litter/hyacinth/ent_litter —
  identical taxonomy to Hagenbeek (same TU Delft lineage). 9,352 boxes
  (ent_litter 4,299 / ff_litter 2,036 / hyacinth 3,017). 0 malformed lines,
  0 corrupt, 0 orphans, 0 within/cross duplicates. Sizes 4048×3032–5568×4872 (3 sizes).
  Box px @640: median 7.5, 54% <8px → this source is very small-object; if merged
  it needs the same tiling treatment as Hagenbeek. NOT merged (per D5) — proposed
  mapping ff_litter→litter, hyacinth→hyacinth, ent_litter→entangled_plastic
  [Inferred from identical names].
- README.docx excerpt in profile JSON.

## D3 — final grouping/assignment (locked at Checkpoint 1 follow-up)

- 600 s session gap locked [Confirmed]: post-split leakage pHash ≤8 = 0.8%
  (8/1,034), CLIP ≥0.97 = 1 image, ≥0.98 = 0 (median cosine 0.935; the 16.6%
  at ≥0.95 are same-looking water, not near-duplicates). Original split was
  11.5% / 74.4%.
- Greedy largest-first chosen over Karmarkar-Karp [Confirmed]
  (`audit/assignment_comparison.csv`): greedy err 0.008–0.031 vs KK ~0.92
  (KK equal-sums partition cannot target 8:1:1).

## D7 — RFS t value

- t = 0.75 → r_hyacinth = 3.13 [Confirmed from rfs_table.csv]. Grid extended
  beyond the LVIS-style values because f_hyacinth = 0.077 (LVIS t=0.001 scale is
  ~77× smaller); logged as deviation in MERGE_REPORT §5.

## D11 — imgsz recommendation

- imgsz = 960 [Confirmed from box_size_stats.csv]: FML <8px share 18.1%@640 →
  2.6%@960; median 14→21px. Set as constant in kaggle/train_baseline.py.

## Deployment verification summary

- All Hailo facts (native export, DFC 3.x for Hailo-8L, yolov8n+yolov11n in
  HAILO8L model-zoo list, ≥1,024 calib images, Linux x86_64 compile-only) —
  see T8 section above with URLs. Compression_level=0 note: Hailo docs state
  default calibration uses 4-bit weight quantization above 1,024 images — to be
  set explicitly in the model script (EXPORT_HAILO.md). Calibration set built:
  1,024 images, hyacinth share 0.25, balanced fml 485 / tud_gv 193 / hagenbeek
  346 (`data/processed/calib/`). [Confirmed]

## D8/Hailo/Kaggle — remaining PENDING items

- Kaggle dataset upload specifics: PENDING (T6).
- Ultralytics duplicate-train-list acceptance for RFS: PENDING (T3, D7).
- Albumentations argument verification: PENDING (T7).
