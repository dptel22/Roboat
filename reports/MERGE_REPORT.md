# Merge Report — RoBoat detection dataset (data/processed)

Built 2026-09-21 by `scripts/build_dataset.py`, verified by `scripts/verify_dataset.py`
(**ALL PASS** — 15/15 checks, see `data/processed/verification_results.csv`).
Seed 42 throughout. data/raw untouched (read-only). Tags: [Confirmed] = read from
produced files; [Inferred] = judgment.

> Post-build review round: an independent code review (12 findings, see
> `reports/AGENTS_LOG.md`) plus a complexity pass found and fixed a critical bug —
> every Hagenbeek tile was hardlinked to one reused temp-file inode, so later tile
> saves overwrote earlier tiles' pixels. The dataset was REBUILT and RE-VERIFIED
> after all fixes; all numbers below are from the final build (manifest mtime
> 22:08, verification after). Tile pixel uniqueness spot-checked by content hash.
> Two latent (inactive) issues remain, documented in AGENTS_LOG: box_counts is not
> decremented if the aerial cap ever triggers (cap inactive: share 17.3% < 35%),
> and train-share denominators include background rows.

## 1. Counts per source × split × class (2-class primary) — [Confirmed]

Images (final entries in manifest; Hagenbeek = tiles):

| source | train | val | test | total |
|---|---|---|---|---|
| fml | 3,033 (= 2,786 pos + 247 bg) | 373 | 364 | 3,770* |
| tud_gv | 1,207 | 148 | 146 | 1,501 |
| hagenbeek (tiles) | 964 | 70 | 127 | 1,161 |

*fml raw has 5,299 images: 3,523 with boxes + 1,776 verified-empty; only 247 empties
entered train as background negatives (D8 cap 10% = 495 budget; fml-only because
TUD-GV has 0 empty labels — [Confirmed]). Positives: 2,786/373/364 by split.

Boxes per class:

| source | split | litter | hyacinth |
|---|---|---|---|
| fml | train/val/test | 12,121 / 2,266 / 2,070 | 0 |
| tud_gv | train/val/test | 7,113 / 475 / 593 | 0 |
| hagenbeek (tiles) | train/val/test | 1,245 / 46 / 122 | 523 / 64 / 78 |

Reconciliation with DATA_EXPLORATION.md [Confirmed]: FML 16,457 boxes exact;
TUD-GV 8,181 exact; Hagenbeek originals 1,415 → 2,078 tile boxes (≥ originals;
the 20% tile overlap duplicates boxes spanning tile borders, plus the 40%-area
clip rule — expected, not an error).

## 2. Share of each source in train — [Confirmed]

fml 58.3%, tud_gv 23.2%, hagenbeek tiles 18.5% (n=5,204). Aerial (Hagenbeek)
share of train images pre-cap = post-cap = **17.3%** — below the 35% cap, so no
tiles were dropped (D4).

## 3. Leakage audit (D3) — [Confirmed]

Original FML split (random per frame): pHash near-dup (Hamming ≤8/64) **11.5%**,
CLIP ViT-B/32 cosine ≥0.95 **74.4%** of val+test (1,588 imgs) → split invalid.

New group-based split (FML day×session, gap 600 s; TUD-GV expNN; Hagenbeek
per-original-image), greedy largest-first assignment (805/111/85 fml,
804/99/97 tud, 805/98/98 hag):

- pHash ≤8: **8/1,034 = 0.8%**
- CLIP ≥0.95: 16.6%, but ≥0.97: **1 image**, ≥0.98: **0** (median cosine 0.935)
  → residual 0.95 hits are same-looking water, not near-duplicate frames.

**600 s grouping locked** per the pre-agreed criterion. Assignment comparison
greedy vs Karmarkar-Karp reported in `audit/assignment_comparison.csv`: greedy
err 0.008–0.031, KK ~0.92 (KK equal-sums partition cannot target 8:1:1 — honest
result, greedy chosen).

## 4. Box-size audit & imgsz (D11) — [Confirmed]

`data/processed/box_size_stats.csv` (letterbox-rescaled widths+heights pooled):

| source | imgsz | median px | p10 px | <8px | <16px |
|---|---|---|---|---|---|
| fml | 640 | 14.0 | 6.8 | **18.1%** | 57.7% |
| fml | 960 | 21.0 | 10.1 | **2.6%** | 33.6% |
| tud_gv | 640 | 23.0 | 14.3 | 0.3% | 16.3% |
| tud_gv | 960 | 34.5 | 21.5 | 0.0% | 1.4% |
| hagenbeek | 640 | 33.0 | 12.6 | 1.7% | 18.8% |
| hagenbeek | 960 | 49.5 | 19.0 | 0.4% | 5.6% |

**Recommendation: imgsz = 960.** The USV-view source (FML, 76% of boxes) loses
18% of boxes below the 8px detectability floor at 640; 960 cuts that to 2.6%.
≈2.25× compute vs 640 — acceptable for yolov8n/yolo11n on a Kaggle T4/P100.
[Inferred] The training cost trade-off; the pixel numbers are [Confirmed].

## 5. RFS imbalance handling (D7) — [Confirmed]

f_c on train (image-level): litter 0.877, hyacinth 0.077. r_c = max(1, √(t/f_c)), cap 6:

| t | r_litter | r_hyacinth |
|---|---|---|
| 0.05 | 1.0 | 0.81→1.0 |
| 0.1 | 1.0 | 1.14 |
| 0.2 | 1.0 | 1.62 |
| 0.3 | 1.0 | 1.98 |
| 0.5 | 1.0 | 2.56 |
| **0.75** | **1.0** | **3.13 ✓** |
| 1.0 | 1.0 | 3.62 |

**Chosen t = 0.75 → r_hyacinth = 3.13** (inside [3,6]). Deviation note: the
original LVIS-style t values ≤0.5 never reach r≥3 because f_hyacinth (0.077) is
~77× LVIS's t=0.001 scale; the grid was extended to {0.75, 1.0, 1.5, 2.0} to
satisfy the locked rule. Applied by repeating lines in `lists/train.txt`
(5,204 → 6,000 lines; duplicates accepted by Ultralytics' list reader — every
image/label file stays unique, only list lines repeat). Floor(int(r)) used.

## 6. Other build facts — [Confirmed]

- Tiling (D4): 1,034 positive + 127 empty tiles (15% cap) from 72 Hagenbeek
  originals; 640/512, ≥40% clipped-area rule. Tile boxes written as YOLO centers
  (a corner-vs-center bug was caught by verify_dataset and fixed).
- Empty-label images (D8/D9): fml 1,776 verified-empty (247 used as bg),
  tud_gv 0, hagenbeek 10 — **excluded pending CP3**; contact sheet at
  `data/exploration_samples/processed/hagenbeek_empty_labels_contact_sheet.jpg`.
- AquaTrash (D6): converted to YOLO (all 4 classes → litter) at
  `data/processed/ood_aquatrash/` (369 images / 469 boxes), eval-only yaml.
- Trees: `merged2/` (primary) and `merged3/` (3-class ablation; images
  hardlinked, labels separate). Per-source yamls in `data/processed/yamls/`
  (`fml_c2/c3, tud_gv_c2/c3, hagenbeek_tiles_c2/c3, combined_c2/c3,
  ood_aquatrash`); split lists in `data/processed/lists/`.
- Contact sheets (boxes drawn): `data/exploration_samples/processed/
  {fml,tud_gv,hagenbeek_tiles}_contact_sheet.jpg`.
- Manifest: `data/processed/manifest.csv` (6,432 rows; columns source,
  original_path, original_class, mapped_class, split, group_id, tile_info,
  final_stem, n_boxes, width, height, rfs_repeat).

## 7. Deviations from D1–D11

1. **D7**: RFS t grid extended beyond LVIS-style values (see §5) — required to
   land r_hyacinth in [3,6].
2. **D3c**: CLIP agglomerative clustering not needed — filename-derived groups
   were reliable for every source; CLIP used for auditing only.
3. **D10**: "USV-view test sets separate from aerial" satisfied via per-source
   yamls; combined yaml exists for convenience.
4. **D8**: background budget filled only from FML (TUD-GV has zero empty labels).

## 8. What blocks training

Nothing in the data. Outstanding judgment calls: CP2 (merge Saigon into training
or keep as second OOD eval set — profiled, not merged), CP3 (the 10 Hagenbeek
empty-label images as negatives), CP4 (approve this report → run T6).
