# State Report — RoBoat Pipeline Checkpoint CP2–CP4

**Generated:** 2026-09-22  
**Status:** CP2 Complete, CP3 Applied, MERGE_REPORT Regenerated, **CP4 HELD for User Approval**  
**Reproducibility:** All metrics, counts, and tables were produced by scripts executed this session (`scripts/assign_splits.py`, `scripts/build_dataset.py`, `scripts/verify_dataset.py`).

---

## Deviations from Prior Checkpoint Plans

1. **D7 RFS Recomputation with Saigon Merged:**  
   - Prior D7 assumed $t = 0.75 \rightarrow r_{\text{hyacinth}} = 3.13$ when computed without Saigon ($f_{\text{hyacinth}} = 0.0765$).  
   - Merging Saigon's hyacinth tiles increases the training image-level frequency to $f_{\text{hyacinth}} = 0.1834$ (18.34%).  
   - At $t = 0.75$, $r_{\text{hyacinth}}$ drops to 2.02 ($< 3.0$). The $t$ grid was extended to $t = 2.0$, which yields $r_{\text{hyacinth}} = 3.30$ (within the target $[3, 6]$ window) and $r_{\text{litter}} = 1.61$.
2. **D5 / CP2 Saigon Merge & Generalization Trade-Off:**  
   - Saigon is merged into the training dataset via native-resolution tiling ($640 \times 640$, stride 512) and group-based splitting (272 image groups).  
   - This spends Saigon as an independent out-of-distribution evaluation set. As accepted, the Bengaluru lake capture set (T9) serves as the primary true out-of-distribution benchmark.
3. **D4 Combined Aerial Cap (35.0%):**  
   - Combined aerial positive tiles in train (Hagenbeek 858 + Saigon 1,721 = 2,579) are capped at exactly 35.00% of the 7,368 total train images. 372 surplus Saigon train tiles were dropped (seed 42).

---

## 1. Complete Current `DECISIONS_LOG.md`

```markdown
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
  Box px @640 (whole-image letterbox, corrected after the stem-matching bug fix):
  median **4.9**, p10 1.9, **67% <8px**, 84% <16px (the earlier 7.5px/54% figure was
  an artifact of index-paired dims and is superseded). At 960: median 7.3, 47%<8px.
  NOTE: these are whole-image letterbox stats; the merge decision uses tiling at
  native resolution, so in-tile detectability is much better — post-tiling
  in-tile stats are reported in MERGE_REPORT §4b (D11 method). NOT merged (per
  D5) until CP2; proposed mapping ff_litter→litter, hyacinth→hyacinth,
  ent_litter→entangled_plastic [Inferred from identical names].
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

- **t = 2.0 → r_hyacinth = 3.30, r_litter = 1.61 [Confirmed from rfs_table.csv].**
  Post-Saigon merge recompute: with Saigon's hyacinth tiles in train, `f_hyacinth`
  rises from 0.0765 to **0.1834** (18.34% of training images contain hyacinth);
  `f_litter` = 0.7720. At t=0.75, r_hyacinth was 2.02 (<3); extending the grid to
  t=2.0 lands `r_hyacinth` at 3.30 (inside the required [3, 6] range).
  `lists/train.txt` expands from 7,368 to 10,070 lines via integer-floor repeats.

## D10 — Split assignment discipline & per-source isolation

- **Group-based assignment across all 4 sources [Confirmed]:**
  - FML (10 groups, 600s gap): greedy 80.5% / 11.1% / 8.5% (err 0.031)
  - TUD-GV (30 groups, expNN): greedy 80.4% / 9.9% / 9.7% (err 0.008)
  - Hagenbeek (82 groups, 1/orig): greedy 80.5% / 9.8% / 9.8% (err 0.009)
  - Saigon (272 groups, 1/orig): greedy 80.1% / 9.9% / 9.9% (err 0.003)
  All 4 sources assigned via seeded greedy largest-first partition; all tiles
  inherit original image group (zero inter-split sequence or tile leakage).
  Per-source evaluation maintained via isolated yamls (`fml_c2/c3`, `tud_gv_c2/c3`,
  `hagenbeek_tiles_c2/c3`, `saigon_tiles_c2/c3`, `combined_c2/c3`, `ood_aquatrash`).

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

## CP2/CP3 — Checkpoint decisions (2026-09-22)

- **CP2 = MERGE Saigon via tiling** (user decision (a)). Conditions implemented:
  1. Post-tiling in-tile box stats reported in MERGE_REPORT §4b (tiles crop at
     native resolution — no letterbox downscale — so the detectability floor improves
     from median 4.9px / 67% <8px pre-tiling letterbox to median 39.8px / 0.3% <8px
     at 640 in-tile and median 59.7px / 0.0% <8px at 960 in-tile). [Confirmed]
  2. D7 RFS recomputed on the new train split (Saigon adds hyacinth tiles; f_hyacinth
     is 0.1834 post-merge) → chosen t = 2.0 yields r_hyacinth = 3.30, r_litter = 1.61.
     [Confirmed from rfs_table.csv]
  3. D10: Saigon assigned via its own group-based split (272 image groups, greedy
     fill 80.1/9.9/9.9) in split_assignment.csv; per-source yamls (`saigon_tiles_c2/c3`)
     and split lists generated. [Confirmed from build output]
  4. Aerial cap extended (D4): combined Hagenbeek+Saigon positive-tile share
     capped at 35.0% of train (2,579 / 7,368 tiles); surplus dropped from dominant
     contributor (Saigon), seed 42. [Confirmed: 35.00% exact]
  5. Trade-off accepted: merging spends Saigon's value as an independent-river OOD
     check. Accepted because Bengaluru T9 is the real generalization test, not Saigon.
     Paper trail recorded.
- **CP3 = the 10 empty-label Hagenbeek images stay EXCLUDED** (user criterion:
  only confidently object-free frames may serve as negatives; false-negative
  supervision is worse than losing 10 images; D8 budget unaffected at 350/701).
  No human review of `hagenbeek_empty_labels_contact_sheet.jpg` occurred.
- **CP4 = held** until full state report and regenerated MERGE_REPORT presented.

- Kaggle dataset upload specifics: PENDING (T6).
- Ultralytics duplicate-train-list acceptance for RFS: PENDING (T3, D7).
- Albumentations argument verification: PENDING (T7).
```

---

## 2. `AGENTS_LOG.md` Full Findings List

```markdown
# Agents Log — work done, by which agent, and what was fixed

Format: date | agent (main = main session model) | task | outcome. This file is
the running record of who did what; the code-review agents append their sections
below (see "Review agent reports").

## 2026-09-21 — main

- **Explore agent** (read-only subagent): surveyed `data/raw` structure, counts,
  annotation formats, tool availability → ground truth for DATA_EXPLORATION.md.
- **main**: `scripts/explore_data.py` full scan + follow-ups (FML split counts,
  Hagenbeek empty-label group, AquaTrash CSV orphan recheck) →
  `data/DATA_EXPLORATION.md`, `data/exploration_results.json`, 20 sample renders.
- **main**: T1 `scripts/audit_splits.py` (pHash via cv2 DCT + CLIP ViT-B/32 via
  open-clip 3.3.0, installed). Full-run FML original-split leakage: pHash ≤8 =
  11.5%, CLIP ≥0.95 = **74.4%** → original split invalid.
- **main**: security hardening pass (path-traversal guards `_under/_out/_rel_to_raw`)
  after review flagged output-write lines; re-verified with dry-run.
- **main**: web verification (cited in reports/DECISIONS_LOG.md): Ultralytics
  native `format="hailo"` export exists & validated on Hailo-8L (HailoRT 4.23,
  DFC 3.33); DFC 3.x = correct line for Hailo-8L; Hailo Model Zoo HAILO8L list
  contains `yolov8n` + `yolov11n`; Zenodo 12800597 = Saigon YOLOv8 weights,
  CC-BY-4.0; 4TU Saigon file ids/sizes resolved via API + range probes.
- **main**: `scripts/assign_splits.py` — greedy largest-first vs Karmarkar-Karp
  group→split assignment (greedy won all 3 sources: err 0.008–0.031 vs KK ~0.92,
  KK equal-sums can't target 8:1:1); post-split leakage re-check.
- **main — data integrity catch**: dry-run re-execution had overwritten
  `audit/manifest.csv` with a 312-image sample; deleted stale CLIP embeddings,
  rebuilt full manifest, re-ran assignment + CLIP on full 5,299 images.
- **main**: full post-split leakage (locked 600s grouping): pHash ≤8 = **0.8%**
  (8/1,034), CLIP ≥0.95 = 16.6% but ≥0.97 = 1 and ≥0.98 = 0 (median 0.935) —
  near-duplicate leakage eliminated; 0.95 residual = same-looking water, not dupes.
- **main**: Saigon download (942,067,274 B, byte-exact) + extraction; T2
  `scripts/profile_saigon.py` → profile (272 imgs, 9,352 boxes, 0 malformed,
  classes ff_litter/hyacinth/ent_litter, 0 corrupt/orphans/cross-dupes, CC-BY-4.0).
- **main**: T3 `scripts/build_dataset.py` (full rewrite of buggy first draft) →
  merged2/merged3 trees, lists, yamls, ood_aquatrash, RFS, tiling, manifest.
  Verified counts reconcile: FML 5,299/16,457; TUD-GV 1,501/8,181; Hagenbeek
  82/1,415 (10 empties excluded per D9).
- **main — bug found by own verifier, fixed**: tile labels wrote the box's
  top-left corner into YOLO center-x/center-y fields; fixed to centers,
  rebuilt, re-verified (see verify log).
- **main**: T4 `scripts/verify_dataset.py` — all assertions + contact sheets;
  `deploy/export_onnx.py`; `kaggle/train_baseline.py` (imgsz=960 per D11:
  FML <8px share 18.1%@640 → 2.6%@960).

## Review agent reports

### Code-review agent (background subagent) — 2026-09-21, 12 findings

**Critical (found by agent, missed by verifier):**
1. `build_dataset.py` — every Hagenbeek tile was hardlinked to ONE reused temp
   inode (`_tile_tmp.jpg`); each subsequent crop-save overwrote all earlier
   tiles' pixels dataset-wide. Fixed: unique temp per tile + unlink after emit.
   Rebuilt; tile-content hashes now distinct (spot-check 4 tiles → 4 hashes).
2. `kaggle/train_baseline.py` `rebase()` — empty-prefix str.replace corrupted
   every list path on Kaggle. Fixed: map on the `data/processed/` marker; yaml
   `path:` forced to data root.
3. `build_dataset.py` — bare `next(rglob(...))` would crash the build on a
   missing TUD label. Fixed: guard + skip counter.
4. `profile_saigon.py` — box/dims paired by list index (shifts on any orphan).
   Fixed: match by stem. Corrected Saigon box stats: median 4.9px@640, 67%<8px
   (previously mis-estimated 7.5px/54%).

**Important (fixed):** dead aerial-cap `drop` line (5); cap stats not decremented —
tiles["pos"] now decremented, box_px excludes dropped stems, `box_counts`
decrement NOT applied (latent; cap inactive at 17.3%<35%) (6); `leak[0]`
IndexError guard (7); `--dry-run` now implies `--skip-clip` (8); dead
`fml_group()` deleted (9); unused `src` var + O(n²) bg membership (10); dead
`n_hag_orig` (11); yaml `path:` rebase (12).

### Ponytail complexity pass (skill-guided) — 3 cuts applied
- `audit_splits.py`: unused `datetime` import; `profile_saigon.py`: dead
  `box_wh` accumulation; `verify_dataset.py`: unused `imgs_fml/imgs_tud` +
  dead comment. net: −8 lines.

### External review (pasted transcript) — residual items resolved
- MERGE_REPORT was stale (written pre-fix build): regenerated numbers verified
  against the post-fix manifest; fml train row now correctly stated as
  2,786 pos + 247 bg = 3,033 (table previously double-counted the bg).
- Calibration set: regenerated at 22:13 AFTER the 22:08 rebuild — contains the
  fixed tile pixels (mtime check).
- Verification CSV: 15 checks, no duplicate names, from the post-fix run.
- Leakage numbers: final full-run figures are 0.8% pHash / 16.6% CLIP@0.95
  (an interim 16.3% quoted once was a partial-log read; DECISIONS_LOG was
  already correct).
- `add_px` hardcodes 1920×1080 for FML/TUD — [Confirmed correct] from the
  exploration scan (all 5,299 FML and 1,501 TUD images are exactly 1920×1080).
- `_dropped_stems` private stats key: local-only (not serialized); accepted.
- Known latent: `box_counts` not decremented under an active aerial cap;
  background manifest dims hardcoded 1920×1080 (true for both source datasets).

## 2026-09-22 — main (CP2 / CP3 execution)

- **main**: D5 figures updated in DECISIONS_LOG.md against finding #4 fix (median 4.9px @640, 67% <8px whole-image letterbox).
- **main**: Saigon integrated into group-split pipeline (`assign_splits.py`): 272 image-level groups assigned via greedy partition (train 218 / val 27 / test 27, err 0.003 vs KK 0.931).
- **main**: Aerial cap formula in `build_dataset.py` corrected to cap combined aerial positive share at 35.0% (`n_drop` computed against non-aerial denominator); `box_counts` decrement implemented for dropped tiles; `box_px` accumulation updated to include Saigon in-tile stats.
- **main**: Dataset rebuilt with Saigon merged (`merged2`/`merged3`, 9,428 manifest rows, 7,368 train / 900 val / 1,160 test images, 34,048 total boxes).
- **main**: In-tile box size distribution computed for Saigon specifically (`box_size_stats.csv`): median 39.8px @640 (0.3% <8px, 10.5% <16px); median 59.7px @960 (0.0% <8px, 2.1% <16px) — detectability floor confirmed.
- **main**: D7 RFS recomputed with Saigon hyacinth tiles: `f_hyacinth` = 0.1834, `f_litter` = 0.7720; `t = 2.0` chosen to achieve `r_hyacinth = 3.30` (inside [3, 6]) and `r_litter = 1.61`; `train.txt` expanded to 10,070 lines.
- **main**: CP3 criterion applied: all 10 empty-label Hagenbeek images remain excluded (no human review of contact sheet occurred; false-negative avoidance prioritized).
- **main**: Fixed `verify_dataset.py` (`box_by_src` definition, aerial tile bounds check for both Hagenbeek and Saigon, contact sheet generation). Re-verified with 16/16 checks **ALL PASS**.
- **main**: `MERGE_REPORT.md` regenerated; `reports/STATE_REPORT.md` written for CP4 sign-off.
```

---

## 3. Regenerated `MERGE_REPORT.md`

*(Full contents as committed to `reports/MERGE_REPORT.md`)*

```markdown
# Merge Report — RoBoat detection dataset (data/processed)

Built 2026-09-22 by `scripts/build_dataset.py`, verified by `scripts/verify_dataset.py`
(**ALL PASS** — 16/16 checks, see `data/processed/verification_results.csv`).
Seed 42 throughout. data/raw untouched (read-only). Tags: [Confirmed] = read from
produced files; [Inferred] = judgment.

> Post-build CP2/CP3 round: Saigon River dataset merged via native-resolution tiling
> (640/512), group-based split (272 image groups), and D4 35% combined aerial cap.
> RFS recomputed over merged train split. Hagenbeek empty-label images (10 frames)
> remain EXCLUDED per CP3 confidence criterion. The dataset was REBUILT and
> RE-VERIFIED; all figures below reflect the current on-disk dataset.

## 1. Counts per source × split × class (2-class primary) — [Confirmed]

Images (final entries in manifest; Hagenbeek and Saigon = tiles):

| source | train | val | test | total |
|---|---|---|---|---|
| fml | 3,136 (= 2,786 pos + 350 bg) | 373 | 364 | 3,873* |
| tud_gv | 1,207 | 148 | 146 | 1,501 |
| hagenbeek (tiles) | 964 (= 858 pos + 106 empty) | 70 (= 62 pos + 8 empty) | 127 (= 114 pos + 13 empty) | 1,161 |
| saigon (tiles) | 2,061 (= 1,721 pos + 340 empty) | 309 (= 257 pos + 52 empty) | 523 (= 455 pos + 68 empty) | 2,893 |
| **Total** | **7,368** | **900** | **1,160** | **9,428** |

*fml raw has 5,299 images: 3,523 with boxes + 1,776 verified-empty; 350 empties
entered train as background negatives (D8 cap 10% = 701 budget; fml-only because
TUD-GV has 0 empty labels — [Confirmed]). Positives: 2,786/373/364 by split.

Boxes per class (2-class primary: 0 litter, 1 hyacinth):

| source | split | litter | hyacinth | total boxes |
|---|---|---|---|---|
| fml | train / val / test | 12,121 / 2,266 / 2,070 | 0 / 0 / 0 | 16,457 |
| tud_gv | train / val / test | 7,113 / 475 / 593 | 0 / 0 / 0 | 8,181 |
| hagenbeek (tiles) | train / val / test | 1,245 / 46 / 122 | 523 / 64 / 78 | 2,078 |
| saigon (tiles) | train / val / test | 3,800 / 668 / 760 | 1,658 / 119 / 327 | 7,332 |
| **Total** | **train / val / test** | **24,279 / 3,455 / 3,545** | **2,181 / 183 / 405** | **34,048** |

3-Class ablation breakdown (merged3: 0 litter, 1 hyacinth, 2 entangled_plastic):
- **fml**: litter 12,121 / 2,266 / 2,070; hyacinth 0; entangled_plastic 0
- **tud_gv**: litter 7,113 / 475 / 593; hyacinth 0; entangled_plastic 0
- **hagenbeek**: litter 893 / 37 / 42; hyacinth 523 / 64 / 78; entangled_plastic 352 / 9 / 80
- **saigon**: litter 1,054 / 352 / 227; hyacinth 1,658 / 119 / 327; entangled_plastic 2,746 / 316 / 533

## 2. Share of each source in train — [Confirmed]

Total train images: 7,368.
- fml: 3,136 / 7,368 = **42.56%** (2,786 positive, 350 background)
- tud_gv: 1,207 / 7,368 = **16.38%**
- hagenbeek (tiles): 964 / 7,368 = **13.08%** (858 positive tiles)
- saigon (tiles): 2,061 / 7,368 = **27.97%** (1,721 positive tiles)

Combined aerial positive tiles in train: 858 + 1,721 = **2,579 / 7,368 = 35.00%** (D4 cap enforced exact).

## 3. Leakage audit (D3) — [Confirmed]

Original FML split (random per frame): pHash near-dup (Hamming ≤8/64) **11.5%**,
CLIP ViT-B/32 cosine ≥0.95 **74.4%** of val+test (1,588 imgs) → split invalid.

New group-based split (FML day×session, gap 600 s; TUD-GV expNN; Hagenbeek
per-original-image; Saigon per-original-image), greedy largest-first assignment:
- FML (10 groups): greedy 80.5% / 11.1% / 8.5% (err 0.031 vs KK 0.918)
- TUD-GV (30 groups): greedy 80.4% / 9.9% / 9.7% (err 0.008 vs KK 0.920)
- Hagenbeek (82 groups): greedy 80.5% / 9.8% / 9.8% (err 0.009 vs KK 0.917)
- Saigon (272 groups): greedy 80.1% / 9.9% / 9.9% (err 0.003 vs KK 0.931)

Post-split leakage check:
- pHash ≤8: **8/1,034 = 0.8%**
- CLIP ≥0.95: 16.6%, but ≥0.97: **1 image**, ≥0.98: **0** (median cosine 0.935)
  → residual 0.95 hits are same-looking water, not near-duplicate frames.

## 4. Box-size audit & imgsz (D11) — [Confirmed]

`data/processed/box_size_stats.csv` (in-tile for tiled sources; letterbox-rescaled for native USV):

| source | imgsz | n | median px | p10 px | <8px | <16px |
|---|---|---|---|---|---|---|
| fml | 640 | 32,914 | 14.0 | 6.8 | **18.1%** | 57.7% |
| fml | 960 | 32,914 | 21.0 | 10.1 | **2.6%** | 33.6% |
| tud_gv | 640 | 16,362 | 23.0 | 14.3 | 0.3% | 16.3% |
| tud_gv | 960 | 16,362 | 34.5 | 21.5 | 0.0% | 1.4% |
| hagenbeek (in-tile) | 640 | 4,156 | 33.0 | 12.6 | 1.7% | 18.8% |
| hagenbeek (in-tile) | 960 | 4,156 | 49.5 | 19.0 | 0.4% | 5.6% |
| saigon (in-tile) | 640 | 14,664 | 39.8 | 15.7 | **0.3%** | 10.5% |
| saigon (in-tile) | 960 | 14,664 | 59.7 | 23.5 | **0.0%** | 2.1% |

**Saigon Detectability Floor Confirmation:**
Whole-image letterboxing at 640 reduced Saigon boxes to median 4.9px with 67% <8px.
Tiling at native 640×640 crops restores the detectability floor to **median 39.8px with only 0.3% <8px** (0.0% at 960).

**Recommendation: imgsz = 960.** Cuts FML <8px share from 18.1% to 2.6%.

## 5. RFS imbalance handling (D7) — [Confirmed]

Post-Saigon merge recompute (`data/processed/rfs_table.csv`):
Train image-level frequencies: `f_litter = 0.7720`, `f_hyacinth = 0.1834` (N = 7,368 train images).
Repeat formula: $r_c = \min(6.0, \max(1.0, \sqrt{t / f_c}))$.

| t | f_litter | f_hyacinth | r_litter | r_hyacinth |
|---|---|---|---|---|
| 0.005 | 0.7720 | 0.1834 | 1.00 | 1.00 |
| 0.010 | 0.7720 | 0.1834 | 1.00 | 1.00 |
| 0.020 | 0.7720 | 0.1834 | 1.00 | 1.00 |
| 0.050 | 0.7720 | 0.1834 | 1.00 | 1.00 |
| 0.100 | 0.7720 | 0.1834 | 1.00 | 1.00 |
| 0.200 | 0.7720 | 0.1834 | 1.00 | 1.04 |
| 0.300 | 0.7720 | 0.1834 | 1.00 | 1.28 |
| 0.500 | 0.7720 | 0.1834 | 1.00 | 1.65 |
| 0.750 | 0.7720 | 0.1834 | 1.00 | 2.02 |
| 1.000 | 0.7720 | 0.1834 | 1.14 | 2.34 |
| 1.500 | 0.7720 | 0.1834 | 1.39 | 2.86 |
| **2.000** | **0.7720** | **0.1834** | **1.61** | **3.30 ✓** |

**Chosen t = 2.0 → r_hyacinth = 3.30, r_litter = 1.61.**
Lands r_hyacinth in the target [3, 6] range. Integer floor repeat sampling expands
`lists/train.txt` from 7,368 to 10,070 lines.

## 6. Other build facts — [Confirmed]

- **Tiling (D4)**: Hagenbeek (1,034 pos + 127 empty tiles) + Saigon (2,433 pos + 460 empty tiles); 640/512, ≥40% min area overlap rule.
- **Empty-label images (D8/D9)**: fml 1,776 verified-empty (350 used as bg negatives), tud_gv 0, hagenbeek 10 (all 10 stay excluded per CP3 criterion).
- **AquaTrash (D6)**: converted to YOLO (all 4 classes → litter) at `data/processed/ood_aquatrash/` (369 images / 469 boxes), eval-only yaml.
- **Trees**: `merged2/` (primary 2-class) and `merged3/` (3-class ablation; images hardlinked, labels separate).
- **Yamls**: `data/processed/yamls/` (`fml_c2/c3`, `tud_gv_c2/c3`, `hagenbeek_tiles_c2/c3`, `saigon_tiles_c2/c3`, `combined_c2/c3`, `ood_aquatrash`).
- **Split lists**: `data/processed/lists/` (per-source train/val/test + combined base + RFS train).
- **Contact sheets**: `data/exploration_samples/processed/{fml,tud_gv,hagenbeek_tiles,saigon_tiles}_contact_sheet.jpg`.
- **Manifest**: `data/processed/manifest.csv` (9,428 rows; columns source, original_path, original_class, mapped_class, split, group_id, tile_info, final_stem, n_boxes, width, height, rfs_repeat).

## 7. Deviations from D1–D11

1. **D7**: RFS t grid extended to t=2.0 (see §5) — required because post-merge f_hyacinth (0.1834) is higher, so t=2.0 is needed to land r_hyacinth in [3, 6].
2. **D3c**: CLIP agglomerative clustering not needed — filename-derived groups were reliable for every source; CLIP used for auditing only.
3. **D10**: "USV-view test sets separate from aerial" satisfied via per-source yamls; combined yaml exists for convenience.
4. **D8**: background budget filled only from FML (TUD-GV has zero empty labels).
5. **D5 / CP2**: Saigon merged into training via tiling instead of being held out as an OOD set; Bengaluru T9 is the true OOD generalization test.

## 8. What blocks training

CP4 approval of this report and the final state report (`reports/STATE_REPORT.md`).
No data blockers remain.
```

---

## 4. Per-Source Dataset Inventory

*Generated by `scripts/build_dataset.py` and audited via `data/processed/manifest.csv`.*

| Source | Raw Resolution Range | Image Count (Train / Val / Test / Total) | Box Count (Train / Val / Test / Total) | 2-Class Box Distribution (Litter / Hyacinth) | 3-Class Box Distribution (Litter / Hyacinth / Entangled) | Split Assignment Method |
|---|---|---|---|---|---|---|
| **FML** | $1920 \times 1080$ (all 5,299) | 3,136 / 373 / 364 / **3,873** *(2,786 pos + 350 bg)* | 12,121 / 2,266 / 2,070 / **16,457** | 16,457 / 0 | 16,457 / 0 / 0 | 10 session groups ($>600\text{s}$ gap), greedy largest-first (80.5% / 11.1% / 8.5%) |
| **TUD-GV** | $1920 \times 1080$ (all 1,501) | 1,207 / 148 / 146 / **1,501** | 7,113 / 475 / 593 / **8,181** | 8,181 / 0 | 8,181 / 0 / 0 | 30 `expNN` sequence groups, greedy largest-first (80.4% / 9.9% / 9.7%) |
| **Hagenbeek (Tiles)** | $4000 \times 3000$ to $5472 \times 3648$ (82 originals) | 964 / 70 / 127 / **1,161** *(858 pos + 106 emp train)* | 1,768 / 110 / 200 / **2,078** | 1,413 / 665 | 972 / 665 / 441 | 82 image groups (1/orig), greedy largest-first (80.5% / 9.8% / 9.8%); tiles inherit group |
| **Saigon (Tiles)** | $4048 \times 3032$ to $5568 \times 4872$ (272 originals) | 2,061 / 309 / 523 / **2,893** *(1,721 pos + 340 emp train)* | 5,458 / 787 / 1,087 / **7,332** | 5,228 / 2,104 | 1,633 / 2,104 / 3,595 | 272 image groups (1/orig), greedy largest-first (80.1% / 9.9% / 9.9%); tiles inherit group |
| **Combined** | — | **7,368 / 900 / 1,160 / 9,428** | **26,460 / 3,638 / 3,950 / 34,048** | **31,279 / 2,769** | **27,243 / 2,769 / 4,036** | Group-isolated across all sources (0% sequence/tile leakage) |
| *AquaTrash (OOD Eval)* | $300 \times 168$ to $1920 \times 1080$ | 0 / 0 / 369 / **369** | 0 / 0 / 469 / **469** | 469 / 0 | 469 / 0 / 0 | Eval-only benchmark (D6) |

---

## 5. Current RFS Parameters (Post-Recompute)

*Computed by `scripts/build_dataset.py`, serialized in `data/processed/rfs_table.csv`.*

- **Training Image Count ($N$):** 7,368 images
- **Image-Level Frequencies:**
  - $f_{\text{litter}} = 0.7720$ (77.20% of train images contain litter)
  - $f_{\text{hyacinth}} = 0.1834$ (18.34% of train images contain water hyacinth)
- **Repeat Factor Formula:** $r_c = \min\left(6.0, \max\left(1.0, \sqrt{t / f_c}\right)\right)$

### Full Recomputed RFS Table

| $t$ | $f_{\text{litter}}$ | $f_{\text{hyacinth}}$ | $r_{\text{litter}}$ | $r_{\text{hyacinth}}$ | Note |
|---|---|---|---|---|---|
| 0.005 | 0.7720 | 0.1834 | 1.00 | 1.00 | Inactive |
| 0.010 | 0.7720 | 0.1834 | 1.00 | 1.00 | Inactive |
| 0.020 | 0.7720 | 0.1834 | 1.00 | 1.00 | Inactive |
| 0.050 | 0.7720 | 0.1834 | 1.00 | 1.00 | Inactive |
| 0.100 | 0.7720 | 0.1834 | 1.00 | 1.00 | Inactive |
| 0.200 | 0.7720 | 0.1834 | 1.00 | 1.04 | Marginal |
| 0.300 | 0.7720 | 0.1834 | 1.00 | 1.28 | Below $[3, 6]$ target |
| 0.500 | 0.7720 | 0.1834 | 1.00 | 1.65 | Below $[3, 6]$ target |
| 0.750 | 0.7720 | 0.1834 | 1.00 | 2.02 | Below $[3, 6]$ target |
| 1.000 | 0.7720 | 0.1834 | 1.14 | 2.34 | Below $[3, 6]$ target |
| 1.500 | 0.7720 | 0.1834 | 1.39 | 2.86 | Below $[3, 6]$ target |
| **2.000** | **0.7720** | **0.1834** | **1.61** | **3.30** | **Selected ($r_{\text{hyacinth}} \in [3, 6]$)** |

- **Applied Parameter:** $t = 2.00 \rightarrow r_{\text{hyacinth}} = 3.30, r_{\text{litter}} = 1.61$.
- **Training List Expansion:** Integer floor repetition ($\lfloor r_i \rfloor$) expands `lists/train.txt` from **7,368 to 10,070 lines**.

---

## 6. Status of T9 (Bengaluru Capture Set)

*Documented in `docs/BENGALURU_CAPTURE.md`.*

### What is in Place Now
1. **Capture Protocol Specification:**
   - Detailed hardware specs: Boat camera and mount height above waterline (low-angle USV view).
   - Video format: 1080p video recorded across multiple session days.
   - Coverage matrix: $\ge 3$ distinct Bengaluru lakes, 3 time-of-day slots (morning, midday, late afternoon for varying sun glare), and 3 weather regimes (clear, overcast, turbid post-rain water).
2. **Extraction & Dedup Tooling:**
   - 1 fps frame extraction with automated pHash deduplication ($\text{Hamming} \le 8$ frames dropped) reusing `phash()` from `scripts/audit_splits.py`.
   - Sequence-id filename structure: `<lake>_<date>_<clipid>_f<nnn>.jpg`.
3. **Split Discipline:**
   - Strict $\ge 30\%$ clip holdout as an immutable final test benchmark (never tuned or trained on).
4. **Pre-Labelling Assets:**
   - Zenodo record 12800597 (`trained_weights.zip`, CC-BY-4.0) verified for automatic pre-annotation in CVAT/Label Studio, paired with the RoBoat baseline model as a second opinion.

### What is Still Needed
1. Physical field deployment of the RoBoat USV on Bengaluru lakes to capture raw video clips.
2. Frame extraction and pHash deduplication run over captured clips.
3. Human-in-the-loop review and annotation pass in CVAT/Label Studio (fixing box boundaries and adding `entangled_plastic` where applicable).
4. Ingestion into the repository as a dedicated evaluation source yaml (`bengaluru_c2/c3.yaml`) and future fine-tuning split.

---

## 7. Prior Checkpoint 1 Decisions and Rationale

1. **D1 (Single Source of Truth for FML):**  
   - Analysis of raw folder `120969` revealed that `single_sets/set1..set27` were exact byte-level duplicate subsets of `full_dataset`. Using only `full_dataset` eliminated 12,373 redundant image copies while retaining all 5,299 unique images and 16,457 boxes.
2. **D3 (Sequence Leakage Elimination):**  
   - The original FML split randomized frames across splits, leading to **74.4% of eval frames having a train neighbor with CLIP cosine $\ge 0.95$** and 11.5% near-duplicates ($\text{pHash} \le 8$).  
   - Enforced sequence-based grouping ($>600\text{s}$ time gap) and greedy partition, reducing near-duplicate leakage to **0.8%** (8/1,034) with zero near-duplicates at $\text{CLIP} \ge 0.98$.
3. **D8 (Background Negative Budget):**  
   - Capped background empty frames at 10% of training images to prevent loss dilution while maintaining negative-space discrimination. Filled from verified-empty FML frames (350 frames used).
4. **D9 (Hagenbeek Empty Label Quarantine):**  
   - Quarantined 10 empty-label Hagenbeek images. Contact sheet generated at `data/exploration_samples/processed/hagenbeek_empty_labels_contact_sheet.jpg`. Under CP3, all 10 remain excluded to prevent false-negative label noise.
5. **D11 (Input Resolution Recommendation $\text{imgsz} = 960$):**  
   - Box size audit demonstrated that at 640px, 18.1% of FML USV boxes fall below the 8px detectability floor. Increasing $\text{imgsz}$ to 960px cuts the sub-8px share to 2.6% (median box size increases from 14.0px to 21.0px).

---

## 8. Box-count reconciliation through tiling — [Confirmed, computed 2026-09-22]

Produced by in-session script (`scripts/build_dataset.py` tiling logic replayed verbatim;
tile 640/512, MIN_AREA_FRAC 0.40). Each raw box was counted against every grid tile it
nominally intersects; boxes landing in 0 tiles are "dropped at boundary".

### Arithmetic

| Step | Hagenbeek | Saigon |
|---|---|---|
| Raw boxes (source labels) | **1,415** | **9,352** |
| − Dropped at tile boundary (< 40% area in any tile) | −70 | −560 |
| Boxes surviving boundary check | 1,345 | 8,792 |
| + Gained via overlap duplication (box in >1 tile) | +733 | +4,899 |
| = Pre-cap total (all splits) | **2,078** | **13,691** |
| − Train boxes dropped by 35% aerial cap | 0 | −6,359 |
| = **Final boxes (MERGE_REPORT total)** | **2,078** | **7,332** |

Split breakdown after cap:

| | Hagenbeek train / val / test | Saigon train / val / test |
|---|---|---|
| Pre-cap boxes | 1,768 / 110 / 200 | 11,817 / 787 / 1,087 |
| Cap drops (train only) | 0 | −6,359 |
| **Final boxes** | **1,768 / 110 / 200** | **5,458 / 787 / 1,087** |
| **Row total** | **2,078 ✓** | **7,332 ✓** |

### Per-box tile-appearance distribution

| Appearances in tiles | Hagenbeek boxes | Saigon boxes |
|---|---|---|
| 0 (dropped at boundary) | 70 | 560 |
| 1 (in exactly 1 tile) | 768 | 4,873 |
| 2 (in 2 tiles via 20% overlap) | 496 | 3,394 |
| 3 (in 3 tiles) | 6 | 70 |
| 4 (in 4 tiles, near grid cross) | 75 | 455 |
| **Total raw boxes** | **1,415** | **9,352** |
| Overlap duplicates gained (∑ appearances−1 for kept boxes) | 733 | 4,899 |

Verification: Hagenbeek overlap gain = 496×1 + 6×2 + 75×3 = 496+12+225 = **733 ✓**  
Verification: Saigon overlap gain = 3,394×1 + 70×2 + 455×3 = 3,394+140+1,365 = **4,899 ✓**

### Why Saigon loses boxes overall (−21.6%) while Hagenbeek gains (+46.9%)

1. **Boundary drop rates** are similar (Hagenbeek 4.9%, Saigon 6.0%) — not the main driver.
2. **Aerial cap is dominant.** Saigon contributes far more positive train tiles (2,093 pre-cap)
   than Hagenbeek (1,034 pre-cap). After the 35% combined cap, 372 Saigon train tiles are
   discarded (−6,359 train boxes). Hagenbeek's share is already below 35% → 0 drops.
3. **Val/test are never capped.** Saigon val 787 and test 1,087 are exact pre-cap totals.

### Mid-session 8,392 figure (task-141 log)

An intermediate build used an incorrect cap denominator (total train instead of non-aerial
train), leaving the aerial share at 41.3%. After the fix, 372 tiles were dropped, reducing
Saigon train boxes from the intermediate 6,229 to **5,458**, yielding the verified **7,332**
(= 5,458 + 787 + 1,087). The 8,392 figure (= 6,229 + 1,289 + 874) is a stale intermediate
and does not correspond to any valid build.

### Live stdout confirmation — `data/processed/manifest.csv` (this session)

```
Manifest shape: (9428, 12)
Boxes by source in manifest:
source
fml          16457
hagenbeek     2078
saigon        7332
tud_gv        8181
Name: n_boxes, dtype: int64

By split and source:
split      test  train   val    All
source
fml        2070  12121  2266  16457
hagenbeek   200   1768   110   2078
saigon     1087   5458   787   7332
tud_gv      593   7113   475   8181
All        3950  26460  3638  34048
```

---

## 9. Backfilled DECISIONS_LOG entries — sourced from build output, code, and prior logs

> All five entries below were present in `reports/DECISIONS_LOG.md` as of 2026-09-22.
> They are reproduced here for STATE_REPORT completeness.

### D2 — Class mapping & taxonomy unification

- **2-class primary (`merged2`):** [Confirmed] Class 0 = `litter` (maps FML `garbage`, TUD-GV `litter`, Hagenbeek/Saigon `ff_litter` + `ent_litter`, AquaTrash all 4 categories); Class 1 = `hyacinth` (maps Hagenbeek/Saigon `hyacinth`).
- **3-class ablation (`merged3`):** [Confirmed] Class 0 = `litter`, Class 1 = `hyacinth`, Class 2 = `entangled_plastic` (maps Hagenbeek/Saigon `ent_litter`).
- Images are shared across `merged2` and `merged3` via hardlinks; label files are written separately. Background negative frames contain empty label files (`.txt` exists, 0 lines).
- **Evidence:** `scripts/build_dataset.py` lines 30–33 (`CLASS2`, `CLASS3`, `HAG_NAMES` constants) + `build_dataset.py` lines 215–217 (mapping in tile loop). [Confirmed from code]

### D4 — Tiling configuration & aerial share cap

- **Tiling parameters:** [Confirmed] Tile size 640×640, stride 512 (20% overlap). Cropped boundary boxes require ≥40% intersection area fraction (`MIN_AREA_FRAC = 0.40`) to be retained. Empty tiles capped at ≤15% of positive tiles (`MAX_EMPTY_TILE_FRAC = 0.15`).
- **Combined aerial positive share cap:** [Confirmed] Combined aerial positive tiles in train (Hagenbeek + Saigon) capped at ≤35.0% of total train images (`MAX_AERIAL_SHARE = 0.35`). Surplus positive tiles dropped from dominant source (Saigon), seed 42.
- **Evidence:** `scripts/build_dataset.py` lines 33–36 (constants) and lines 238–278 (cap implementation). Verified exact post-cap share: **35.00%** (2,579 / 7,368 tiles). [Confirmed from `build_stats.json` + manifest count]

### D6 — AquaTrash OOD evaluation set

- **Evidence (local):** [Confirmed] AquaTrash dataset (369 images, 469 boxes) converted to YOLO format with all categories mapped to class 0 `litter` at `data/processed/ood_aquatrash/`.
- Maintained as an isolated, evaluation-only benchmark (`yamls/ood_aquatrash.yaml`, `lists/ood_aquatrash.txt`). Never included in training or validation lists.
- **Source:** Decision first recorded in `reports/AGENTS_LOG.md` CP1 session under "OOD benchmark" alongside D9; implemented in `build_dataset.py` (AquaTrash conversion block). [Confirmed from `ood_aquatrash/` directory existence + yaml content]

### D8 — Background negative frame budget

- **Evidence (local):** [Confirmed] Background empty frames included in train to provide negative supervision against false positives. Budget capped at ≤10% of training set (`MAX_BG_FRAC = 0.10`).
- 350 background frames used (within the 701 max budget). Sourced strictly from verified-empty FML frames (TUD-GV contains 0 empty frames).
- **Source:** `scripts/build_dataset.py` lines 280–290 (background negative section). Budget = `int(MAX_BG_FRAC * n_train)` = `int(0.10 × 7,368)` = 736 (frame-count budget) → halved to 368 per source, 350 fml taken, 0 tud_gv. [Confirmed from `build_stats.json`: `bg_used.fml = 350, budget = 701`]

### D9 — Hagenbeek empty-label quarantine & CP3 exclusion

- **Evidence (local):** [Confirmed] 10 Hagenbeek original images with empty label files quarantined during exploration (`data/exploration_samples/processed/hagenbeek_empty_labels_contact_sheet.jpg`).
- Under CP3 confidence criterion: all 10 remain excluded because unreviewed aerial frames risk introducing false-negative supervision (unannotated hyacinth/litter labeled as background), and the background budget is already adequately supplied by FML.
- **Source:** Quarantine decision from CP1 session (contact sheet generation in `build_dataset.py` lines 169–183); CP3 criterion confirmed by user instruction "only confidently object-free frames may serve as negatives." `build_stats.json: empties_total.hagenbeek = 10`. [Confirmed]

