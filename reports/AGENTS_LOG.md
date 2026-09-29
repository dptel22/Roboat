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
  against the post-fix manifest; fml train row in CP1 was correctly stated as
  2,786 pos + 247 bg = 3,033 (table previously double-counted the bg; updated in CP2 to 350 bg).
- Calibration set: regenerated at 22:13 AFTER the 22:08 rebuild — contains the
  fixed tile pixels (mtime check).
- Verification CSV: 15 checks, no duplicate names, from the post-fix run.
- Leakage numbers: final full-run figures are 0.8% pHash / 16.6% CLIP@0.95
  (an interim 16.3% quoted once was a partial-log read; DECISIONS_LOG was
  already correct).
- `add_px` hardcodes 1920×1080 for FML/TUD — [Confirmed correct] from the
  exploration scan (all 5,299 FML and 1,501 TUD images are exactly 1920×1080).
- `_dropped_stems` private stats key: local-only (not serialized); accepted.
- Known latent: `box_counts` not decremented under an active aerial cap
  **[RESOLVED 2026-09-22 — decrement implemented; the cap was active in the final
  build, dropping 2,049 Saigon train tiles / 6,359 boxes]**;
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

## 2026-09-29 — main (CP4 re-verification round)

- **main**: Re-ran `build_dataset.py --dry-run` on the repopulated raw data: reproduces
  the on-disk dataset exactly (Saigon 5,458/787/1,087 = 7,332; Hagenbeek 1,768/110/200
  = 2,078; budget 701) — confirms 7,332 is the live output of the current code, and
  the task-141 "8,392" figure corresponds to no valid build.
- **main**: Committed `scripts/reconcile_boxes.py` (the 2026-09-22 §9 table had been
  produced by an uncommitted in-session script). Independent geometry replay ties out:
  Hagenbeek 1,415 − 70 + 733 = 2,078; Saigon 9,352 − 560 + 4,899 = 13,691 pre-cap −
  6,359 cap = 7,332; combined cap replay target 2,456 == manifest 2,456.
- **main — discrepancies found in the 2026-09-22 reports, corrected**: tile-level
  figures were wrong while all box figures were correct. Manifest ground truth:
  Saigon train tiles 1,598 pos + 463 empty (not 1,721 + 340), val 276 + 33, test
  465 + 58; Hagenbeek val 63 + 7, test 113 + 14; pre-cap positive train tiles 3,647
  (Saigon) / 858 (Hagenbeek), pool 4,505, cap target 2,456, **2,049 tiles dropped
  (not 372)**; post-cap share 34.996% of 7,018 pre-background train (33.33% of 7,368
  incl. bg). Fixed in MERGE_REPORT §1/§2/§6/§9 and STATE_REPORT §3/§4/§8/§9.
- **main**: `build_dataset.py` now serializes `tiles_dropped` in `build_stats.json`
  (dry-run confirms 2,049); `verify_dataset.py` saigon_boxes_reconcile detail no
  longer prints a misleading "0 tiles dropped" when the key is absent.
- **main**: STATE_REPORT §1/§2 embedded snapshots (2026-09-22) annotated as superseded
  by the live DECISIONS_LOG (D2/D4/D6/D8/D9 backfilled, T9_WEIGHTS rename) and live
  AGENTS_LOG (350 bg, resolved latent). Zenodo record 12800597 re-checked: it
  publishes two model **variants** (Model_resize, Model_tiles) detecting the 3-class
  taxonomy ff_litter/hyacinth/ent_litter — the old "two models: plastic litter and
  water hyacinth" phrasing conflated variants with classes.
- **main**: D8 budget arithmetic corrected in STATE_REPORT §9: budget = int(0.10 ×
  7,018 post-cap pre-background train) = 701 (not int(0.10 × 7,368) = 736).
- **main**: `verify_dataset.py --dry-run` re-run: 16/16 checks **ALL PASS**.

## 2026-09-29 — main (CP4 re-verification round 2 — review gaps closed)

- **main — checkpoint taxonomy VERIFIED (was Derived):** downloaded Zenodo 12800597
  `trained_weights.zip` (11,345,833 B) and inspected both checkpoints directly.
  Modern ultralytics cannot unpickle the legacy `ultralytics.yolo` layout, so
  `scripts/verify_zenodo_weights.py` unpickles with inert stub classes (no legacy
  modules reachable; builtins whitelisted) and prints the real saved attributes:
  both `Model_resize_weights.pt` and `Model_tiles_weights.pt` carry
  `model.names = {0: 'ff_litter', 1: 'hyacinth', 2: 'ent_litter'}` and
  `model.nc = 3`. T9_WEIGHTS upgraded from Derived to checkpoint-Verified in
  DECISIONS_LOG; BENGALURU_CAPTURE.md's 3-class pre-labelling text confirmed correct.
- **main — background count shown, not just the cap:** manifest query: train rows
  with `tile_info == "background"` = **350** (all FML); all 350 corresponding
  `merged2/labels/train/*.txt` files verified empty. Denominators: 7,368 final
  train = 7,018 post-cap pre-background + 350 background. The exact one-third
  aerial share (2,456/7,368) is a coincidence (2,456 × 3 = 7,368); the cap was
  enforced on 7,018 at 34.996%. Recorded in MERGE_REPORT §6.
- **main — pool composition stated explicitly:** the 4,505 combined aerial pool =
  Saigon 3,647 + Hagenbeek 858 positive train tiles; the 569 kept empty aerial
  train tiles are outside the pool but count in `n_train_before`. One-liner added
  to MERGE_REPORT §9.
- **main — output-path hardening (Mimosa gate round):** the Mimosa git gate blocks
  commits while its Semgrep-derived path-traversal rule reports highs. Probe files
  proved the rule fires on **every** form of file-write path — helper-returned,
  module constant, inline literal join, and even a pure relative string
  `open("data/processed/x.csv", "w")` — so no code style can clear it; the 12
  findings are false positives on fixed output filenames (`manifest.csv`,
  `verification_results.csv`, ...) written inside the repo. Refactored the five
  pipeline scripts to a `_fixed_out(root, literal)` helper (bare-filename check,
  rejects absolute/`..`; read-side `_rel_to_raw` keeps the resolve+containment
  guard). Compile-checked; `verify_dataset.py --dry-run` ALL PASS; helper rejects
  `../escape.csv`. No suppression path exists in-tool (`mimosa validate` covers
  only readDoc/remote-runner contracts), and the gate mode (`MIMOSA_GIT_GATE_MODE`,
  default `graded`) is user security configuration — **commit is left to the
  user** (external terminal, or gate-mode decision); findings evidence above.
