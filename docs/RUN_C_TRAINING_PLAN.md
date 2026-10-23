# Roboat Run C — Training Plan (consolidated, for review)

Status: DRAFT for review — 2026-10-23. No Run C training is currently running or
has ever completed. This document consolidates: (1) the complete data lineage
("what has been done to the data"), (2) the exact current state of remote and
local artifacts, (3) the training plan, and (4) known open items. A reviewer
should verify every claim in §1–§2 against the repository and critique §3–§5.

## 0. Mission and hard constraints

- **Goal**: floating-litter + water-hyacinth (vegetation-mat) detection for an
  autonomous cleanup USV operating on Bengaluru lakes.
- **Deployment target**: Hailo-8 (26 TOPS) edge accelerator (corrected from
  Hailo-8L, commit 41a76fa). Model must be nano-scale (YOLOv8n), INT8 HEF
  export, input 960 px.
- **License constraint**: Ultralytics AGPL-3.0 applies to COCO-pretrained
  YOLOv8n and everything derived from it (incl. the Zenodo-initialized arm).
  Commercial USV use needs an Ultralytics Enterprise License or a non-AGPL
  replacement later.
- **Compute budget**: Kaggle free tier (T4/P100, ~30 GPU-h/week, 9–12 h
  sessions). No local GPU. Training must be resumable across sessions.
- **Hardware research done (T8/T10)**: DFC 3.34.0 + HailoRT pins, ONNX opset 11,
  Zenodo→modern ultralytics converter, 1,024-image INT8 calibration set built
  (`data/processed/calib/`, hyacinth share 0.25).

## 1. What has been done to the data (complete lineage)

### 1.1 Raw sources and exploration (CP1, 2026-09-21)

| Source | Kind | Raw inventory | Notes |
|---|---|---|---|
| FML v2 (folder 120969) | USV surface video frames, 1920×1080 | 5,299 imgs / 16,457 boxes (1 class `garbage`) + 1,776 verified-empty | 27 `single_sets` folders were byte-identical duplicates → deduped (D1); original random split **invalid** (see 1.2) |
| TUD-GV | USV surface frames, 1920×1080 | 1,501 imgs / 8,181 boxes | 30 `expNN` session groups |
| Hagenbeek | Aerial stills (DJI/GoPro) | 82 originals / 1,415 boxes | 10 empty-label originals excluded per CP3 (false-negative risk) |
| Saigon (4TU, CC BY 4.0) | Aerial stills, 4–5.5 K | 272 originals / 9,352 boxes | Same TU Delft taxonomy as Hagenbeek (`ff_litter`/`hyacinth`/`ent_litter`) |
| AquaTrash (D6) | OOD eval-only | 369 imgs / 469 boxes | All 4 categories → `litter`; never in train/val lists |

**Leakage finding that forced the re-split (D3)**: the original FML split
leaked heavily — pHash near-dup (Hamming ≤ 8/64) 11.5% and CLIP ViT-B/32
cosine ≥ 0.95 on **74.4%** of val+test frames.

### 1.2 Group-based re-split + tiling (CP2–CP4, 2026-09-22 → 09-29)

- **Groups** (leak-free unit): FML 600 s session gaps (10 groups), TUD-GV
  `expNN` (30), Hagenbeek per-original (82), Saigon per-original (272).
  Greedy largest-first 80/10/10 assignment (beat Karmarkar-Karp on all
  sources). Post-split leakage: pHash ≤ 8 = 0.8%, CLIP ≥ 0.98 = 0 images.
- **Tiling (D4)**: Hagenbeek + Saigon tiled at native resolution 640×640,
  stride 512 (20% overlap), boundary boxes kept at ≥ 40% area fraction, empty
  tiles ≤ 15% of positive. Tiling restored Saigon detectability: median box
  4.9 px (letterboxed @640) → **39.8 px in-tile @640**.
- **Aerial cap (D4)**: combined Hagenbeek+Saigon positive train tiles capped
  at 35.0% → 2,049 surplus Saigon train tiles dropped (seed 42) in CP4.
- **RFS (D7)**: repeat-factor sampling t = 2.0 → r_hyacinth 3.30, train list
  7,368 → 10,070 lines (CP4 only).
- **Background negatives (D8)**: 350 verified-empty FML frames = 10% budget.
- **imgsz decision (D11)**: 960 px cuts FML < 8 px boxes 18.1% → 2.6%.
- **CP4 frozen dataset** (Run A/B artifact): 9,428 imgs (7,368/900/1,160) /
  34,048 boxes; 16/16 verifier checks; bundle `roboat-processed.zip` (2.77 GB)
  → private Kaggle dataset `dptel22/roboat-processed`.
- **Taxonomy (D2)**: 2-class primary `merged2` (0 litter ← garbage/ff_litter/
  ent_litter, 1 hyacinth); 3-class ablation `merged3` (+ entangled_plastic).

### 1.3 Run C donor merge (2026-10-06, commit 3177c4a) — the current dataset

Added three CC BY 4.0 web donors of **surface** imagery. All counts below are
from the authoritative `data/processed/manifest.csv` (23-check verifier ALL
PASS, `data/processed/verification_results.csv`):

| Source | Raw imgs | After pHash dedupe (hd ≤ 8, first-seen) | In manifest (train/val/test) | Boxes |
|---|---|---|---|---|
| Mendeley j26w4m645z.2 | 1,577 | 947 (630 within-donor drops) | 947 (665/179/103 — own split honored) | 4,176 (litter 2,475 + hyacinth 1,701) |
| Navsci invasive-aquatic-plants v12 | 5,815 | 5,194 (421 within-donor + 200 cross-donor vs WHD) | 4,521 (3,870/362/289) | 4,521 (all hyacinth-mat, 1 box/img) |
| Navsci water-hyacinth-detection v1 | 584 | 584 (0 drops) | 584 (215/149/220) | 590 |

Donor-merge decisions (all user-approved 2026-10-06):

1. **pHash dedupe before merge** — first-seen order: incumbent originals →
   Mendeley → navsci_whd → navsci_invasive. Zero incumbent images dropped.
2. **762 navsci_invasive segmentation-polygon label files DROPPED, not
   converted** — their polygon→bbox conversions span ~100% of the image
   (documented in `data/processed/donors/_audit_polygon_labels.json`); they
   would have been full-image "objects".
3. **Operational mat-class mapping**: class 1 = "floating vegetation mat",
   not the species. Mendeley `river_vegetation` → hyacinth(1),
   `floating_waste` → litter(0); Navsci `water_hyacinth`, Water Lettuce,
   Giant Salvinia → all hyacinth(1). Raw class ids are mapped through each
   archive's own `data.yaml`, never assumed.
4. **Joint group split for the two Navsci donors**: one group per original
   image (rf-hash-stripped stem, case-insensitive), split-assigned jointly
   0.8/0.1/0.1 so shared-source photos present in both archives cannot leak
   across splits.
5. **RFS DROPPED for Run C** (D7 override): the donor merge is the
   rebalancing; stacking RFS would double-correct. `train.txt` =
   `train_base` = one pass. Level-explicit balance: image-level
   f_litter 0.5235 / f_hyacinth 0.4949 (near parity); **box-level hyacinth
   23.6%** (11,708/49,694) — hyacinth boxes are large mats.
6. **Aerial cap no longer fires**: surface donors dilute the aerial share to
   0.326 → all 2,049 Saigon train tiles that CP4 had dropped are RETAINED
   (Saigon: 13,691 boxes pre-cap = 13,691 in manifest).
7. **Background budget grew with train size** (10% rule): 701 → 1,381
   budget, FML background tiles 350 → 690.
8. **Label review**: 4 val/test hyacinth→litter relabels on litter-dominant
   crops; 44/48 smallest val/test hyacinth crops reviewed and confirmed.

**Run C dataset (authoritative, from manifest.csv)**: 17,869 images
(train 14,507 / val 1,590 / test 1,772), 49,694 boxes
(litter 37,986 / hyacinth 11,708).

Per-source image totals (manifest): fml 4,213 / tud_gv 1,501 / hagenbeek
tiles 1,161 / saigon tiles 4,942 / mendeley 947 / navsci_invasive 4,521 /
navsci_whd 584.

**CLIP leakage check (closed, 2026-10-06)**: CLIP ViT-B-32-quickgelu over all
17,869 images — **zero same-video/same-group cross-split pairs**. 381/1,772
test images (21.5%) have ≥ 0.95 cosine to some train image, but pair analysis
shows these are same-looking-water / different-session similarities (Saigon
tile↔tile 259, same-river different-session 288, Mendeley same-river
different-video 135, 5 whd↔invasive shared-plant pairs — all joint-grouped),
not copy leakage. Report: `data/processed/embedding_leakage_report.json`.

### 1.4 Bundles and remote state (2026-10-02 → 10-08)

- CP4 bundle `roboat-processed.zip` (2.77 GB) → `dptel22/roboat-processed`
  (Run A/B artifact, frozen — do not overwrite).
- Run C bundle v2 `roboat-run-c.zip` (4.79 GB, 36,524 entries) → private
  `dptel22/roboat-run-c`. **This zip does NOT contain `audit/manifest.csv`.**
- Paired kernels `dptel22/run-c-control` and `dptel22/run-c-zenodo-init`
  pushed 2026-10-08 03:53 — **both are in ERROR state**. Root cause (from the
  kernel log): the pushed kernel version wrote rebased YAMLs to read-only
  `/kaggle/src/rebased` → `OSError: [Errno 30] Read-only file system`. The
  local kernel copies (`kaggle/kernels/run_c_*/run_c_*.py`) were since fixed
  to write under `/kaggle/working`, but were never re-pushed — and are now
  superseded anyway (see §3.1).

## 2. Local working-tree state (uncommitted, 2026-10-23)

Uncommitted changes on `main` (last commit 2ded5ac "Launch Run C training on
Kaggle: kernels + dataset live"):

- `kaggle/train_run_c.py` (new) — the Run C training CLI: `--mode
  audit|train|evaluate`, two arms, fail-closed audit, pinned versions
  (ultralytics 8.4.165, Albumentations 2.0.8), augmentation, resume logic.
- `kaggle/run_c_training.ipynb` (new) — self-contained Kaggle notebook
  embedding the CLI script; supersedes the paired kernels.
- `kaggle/RUN_PLAN.md` (modified) — documents the new entry points and the
  audit-manifest bundle requirement.
- `scripts/build_kaggle_bundle.py` (modified) — now bundles
  `audit/manifest.csv` and asserts group/stem/split disjointness at build
  time. The v2 zip predates this change.
- `scripts/audit_candidate_donor.py` + `scripts/test_audit_candidate_donor.py`
  + `docs/CANDIDATE_INGESTION.md` (new) — candidate-donor ingestion gate
  (fail-closed audit; 5 candidates currently quarantined, none merged).

**Net**: the Run C launch of 2026-10-08 failed and was superseded by a better
uncommitted implementation. Nothing needs to be re-done on the data side; the
pending work is: commit, rebuild the bundle with the audit manifest, re-upload,
train.

## 3. Training plan (Run C)

### 3.1 Design — two-arm controlled experiment

Both arms are identical except initialization, isolating the effect of
domain-specific initialization on the merged donor dataset:

| | Arm 1 `control` | Arm 2 `zenodo` |
|---|---|---|
| Init | stock COCO `yolov8n.pt` | converted Zenodo `Model_tiles_weights.pt` (TU Delft Saigon-domain, ultralytics 8.0.36, nc=3; ultralytics transfers 349/355 keys, skips the 6 cls-conv head keys) |
| Entry | `kaggle/run_c_training.ipynb` or `kaggle/train_run_c.py --mode train --arm control` | same, `--arm zenodo` |

Locked settings (do not tune): seed 42, epochs 100, patience 20, imgsz 960,
`deterministic=True`, data = `combined_c2.yaml` (2-class), checkpoint
selection on **combined val only**.

### 3.2 Augmentation (new for Run C — this makes Run C incomparable to Run A/B)

- Albumentations 2.0.8 (pinned, photometric only):
  RandomBrightnessContrast(±0.15, p=0.20), RandomFog(coef 0.05–0.22, p=0.05),
  RandomRain(p=0.05), RandomShadow(p=0.05), RandomSunFlare(p=0.03),
  MotionBlur(3–5, p=0.03).
- Native YOLO: `scale=0.2` (halved from default 0.5 — protects the small-box
  floor per D11), `fliplr=0.5`, `flipud=0.0` (flipud=0.5 stays a post-gate
  ablation A4). "No geometric crops" applies to the **custom Albumentations
  pipeline only**: the Ultralytics native geometric defaults remain active —
  `mosaic=1.0` (disabled for the last `close_mosaic=10` epochs) and
  `translate=0.1` — alongside `scale=0.2`/`fliplr=0.5`. This matches the
  native-transform regime Run A/B used; `scale=0.2` bounds the zoom-out so
  mosaic/translate cannot push boxes under the detectability floor the way
  the 0.5 default could.
- Smoke test proves the transforms preserve a 2%-of-image bbox, and a preview
  grid from a real labeled train image is saved for visual confirmation.
- Rationale: deployment conditions (glare, rain, fog, shadows, spray) are
  absent from fair-weather training imagery. Seasonal/night conditions are
  NOT synthesized — the Bengaluru capture set covers them for real.

### 3.3 Fail-closed pre-flight audit (runs before any training, on Kaggle)

`--mode audit` verifies against the bundle: exact split counts
(14,507/1,590/1,772), every label file present and well-formed (class ids in
range, normalized boxes inside the image), per-source val/test ⊆ combined,
**group/stem split disjointness via `audit/manifest.csv`**, byte-identical
cross-split duplicates via SHA-256 (only for sizes shared across splits), and
the 369-image AquaTrash eval-only tree. Any failure aborts before training.

### 3.4 Wall-clock budget and resume

14,507 train images at imgsz 960 on a T4 ≈ 20–28 min/epoch → 100 epochs ≈
33–47 h per arm; both arms ≈ 66–94 GPU-h ≈ 2–3 weeks of Kaggle free-tier
quota. Resume, don't restart: `deterministic=True` + `resume=True` restores
model/optimizer/epoch state; the dataloader RNG (shuffle + augmentation
sampling) restarts on each session boundary, so resumed epochs are not
bit-identical to an uninterrupted run. The notebook/CLI support
`--resume-from` to copy a previous session's output off the read-only input
mount before resuming, and refuse to overwrite an existing run directory.

### 3.5 Evaluation protocol

1. **During training**: combined-val mAP only (early stopping + best.pt).
2. **After both arms complete** (`--mode evaluate`, run once, frozen — no
   tuning against it): per-source test mAP@0.5 AND per-class for all 8 yamls
   (`saigon_tiles_c2`, `hagenbeek_tiles_c2`, `fml_c2`, `tud_gv_c2`,
   `mendeley_c2`, `navsci_invasive_c2`, `navsci_whd_c2`, `combined_c2`) plus
   `ood_aquatrash` (litter-only, nc=1). Results → `final_test_results.json`.
   A class with no ground truth in a source is reported N/A, never 0.
3. **Comparators**: Run C-control vs Run C-zenodo on the same data (primary);
   CP4-trained Run A/B re-scored on the CP4 test split (architecture story
   only — numbers are NOT comparable to Run C: splits, RFS-off, aerial-cap
   outcome and augmentation all changed).
4. **Reading the numbers**: the new test split is dominated by web-donor
   imagery; report deployment-like sources (fml, tud_gv, saigon_tiles,
   hagenbeek_tiles) separately from web donors (mendeley, navsci_*). Val is
   same-site and will flatter the model.

### 3.6 Decision rule (proposed)

Select the arm with the better test mAP@0.5 profile on the deployment-like
sources + AquaTrash OOD litter; break ties in favor of `control` (simpler
provenance). If `zenodo` wins only on web donors but loses on
deployment-like sources, prefer `control`. Post-gate ablations (A3 scale,
A4 flipud, weather packages) may follow, each as a new arm vs the frozen
winner — never folded into the gate runs.

### 3.7 After Run C

1. **Bengaluru Capture Set (T9)** — the real generalization test. Score the
   chosen checkpoint on it BEFORE any fine-tune on it (protocol in
   `docs/BENGALURU_CAPTURE.md`; ≥ 30% clips held out forever).
2. **Hailo-8 export** of the chosen checkpoint (INT8 HEF, DFC 3.34.0, opset
   11, 1,024-image calibration set already built) → on-boat A/B.
3. **New donors** only through the candidate ingestion gate
   (`scripts/audit_candidate_donor.py`, fail-closed; registry in
   `docs/CANDIDATE_INGESTION.md`).

## 4. Known discrepancies and risks (for the reviewer)

1. **~~RUN_PLAN.md donor prose vs manifest~~ FIXED 2026-10-23**: RUN_PLAN
   previously said invasive "4,693 imgs / 4,693 boxes" and whd "411 imgs /
   417 boxes" and its reconciliation line summed to 17,868; it now carries
   the authoritative manifest numbers (invasive 4,521/4,521, whd 584/590,
   reconciliation 9,428 + 6,052 + 2,049 + 340 = 17,869) and the correct
   23-check verifier count.
2. **Class-1 semantic drift**: class 1 is now the operational "floating
   vegetation mat" class (mixed species: hyacinth, water lettuce, salvinia).
   Downstream reporting must not call it "water hyacinth detection" without
   the mat caveat.
3. **Test split is donor-heavy**: 584 of 1,772 test images are navsci/mendeley
   web imagery vs deployment-like USV/aerial. Per-source reporting (3.5.4) is
   the mitigation; do not quote a single combined mAP.
4. **AGPL**: both arms are AGPL-entangled (COCO-pretrained base). Fine for
   research/competition; commercial deployment needs the license decision.
5. **Kaggle launch history**: both 2026-10-08 kernels errored (read-only FS
   bug). No training time has been consumed. The new entry points write only
   under `/kaggle/working`.
6. **Hyacinth box-level minority (23.6%)**: image-level parity but box-level
   1:3 — expected (mats are large); if hyacinth recall is the deployment
   priority, a post-gate RFS ablation (`build_dataset.py --rfs`) is the
   documented lever, NOT a Run C change.

## 5. Execution checklist (ordered)

1. ~~Fix the RUN_PLAN prose discrepancies (§4.1)~~ — DONE 2026-10-23.
2. Commit the working tree (bundle builder + entry points + ingestion gate +
   this plan).
3. Rebuild the Run C bundle (now includes `audit/manifest.csv`; builder
   asserts group/stem split disjointness) and upload as a NEW VERSION of
   `dptel22/roboat-run-c` (do not touch `roboat-processed`).
4. Run the notebook's audit cell on Kaggle (or `--mode audit`) — must pass
   before training.
5. Train arm `control`; resume across sessions until completion.
6. Train arm `zenodo`; resume across sessions.
7. Run `--mode evaluate` once; freeze `final_test_results.json`.
8. Apply the decision rule (§3.6); record the decision in
   `reports/DECISIONS_LOG.md`.
9. Post-gate (optional, each vs frozen winner): A3 scale, A4 flipud, weather
   augmentation packages.
10. Bengaluru capture → score chosen checkpoint → fine-tune decision.
11. Hailo-8 export + on-boat validation.
