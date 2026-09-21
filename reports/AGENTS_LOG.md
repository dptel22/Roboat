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
