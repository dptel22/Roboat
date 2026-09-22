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

---

## 9. Box-count reconciliation through tiling — [Confirmed, computed 2026-09-22]

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

Both directions are legitimate from identical tiling parameters:

1. **Boundary drop rate differs by image geometry.** Hagenbeek's 72 labeled images are
   4048–5568px wide; boxes cluster near river-patch centres well inside tile boundaries
   (70/1,415 = 4.9% drop). Saigon's 243 labeled images have dense near-edge annotations
   (560/9,352 = 6.0% drop) — a slightly higher boundary rate, but not the main driver.

2. **Aerial cap is the dominant cause.** Saigon contributes far more total positive tiles
   than Hagenbeek (2,093 vs 1,034 pre-cap positive train tiles). After the 35% combined
   aerial cap kicks in, 372 Saigon positive train tiles are discarded, removing 6,359 train
   boxes. Hagenbeek's share (858 train tiles) is already below 35%, so it receives 0 cap
   drops. The net result is Hagenbeek ends up at pre-cap total (+46.9%) while Saigon ends
   below raw count (−21.6%) despite tiling adding boxes through overlap duplication.

3. **Val/test are never capped.** Saigon val (787) and test (1,087) totals are exact
   pre-cap box counts for those splits and tie out correctly.

### Mid-session 8,392 figure (task-141 log)

During an intermediate task-141 build, the aerial cap denominator formula computed
the target as a fraction of the *total* train count (including aerial tiles), which
under-dropped Saigon tiles and left the combined aerial share at 41.3% instead of 35.0%.
Once the denominator was corrected to `n_non_aerial_train / (1.0 - MAX_AERIAL_SHARE)`,
372 tiles were dropped — reducing Saigon train boxes from the intermediate 6,229 to
**5,458**, giving the verified 7,332 total (5,458 + 787 + 1,087).
The 8,392 figure (= 6,229 + 1,289 + 874) was a stale intermediate log; it does not
correspond to any valid build.

