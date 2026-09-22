# Bengaluru Lake Capture Protocol — RoBoat USV data collection

Goal: a Bengaluru-domain USV-view test/pretraining set of floating litter and
water hyacinth, collected with the actual boat camera and mount, grouped by
clip for leak-free splitting. Target: **300–500 labeled frames** minimum,
including hyacinth in every split's sources.

## 1. Hardware & setup

- Camera: the boat's actual low-angle camera and mount height (do NOT prototype
  with a phone at eye level — viewpoint/mount height drive scale and horizon).
- Record 1080p video; note mount height above waterline and camera pitch in a
  log sheet per session.

## 2. Coverage matrix (plan sessions to fill every cell)

| axis | values |
|---|---|
| lakes | several Bengaluru lakes (≥3; rotate across sessions) |
| time of day | morning / midday / late afternoon (glare differences) |
| weather | clear / overcast / post-rain (turbid water) |
| content | open water, shoreline, hyacinth patches, near outfalls |

Each session: 10–20 min of cruising that passes litter and hyacinth naturally;
do not stage debris.

## 3. Frame extraction

- Extract **~1 fps** from each clip with pHash dedup (skip frames whose 64-bit
  pHash is within Hamming ≤8 of any kept frame from the same clip):
  reuse `phash()` from `scripts/audit_splits.py`.
- Keep the source **clip id in the filename** (`<lake>_<date>_<clipid>_f<nnn>.jpg`)
  — clips are the group id for grouped splitting (D3 discipline extends to
  Bengaluru data).

## 4. Split discipline (locked)

- **At least 30% of clips are held out as the final test set — never trained on,
  never tuned on, not even for model selection.** Decide the held-out clips once,
  up front, by hash-seeded random selection over the clip list, and record them
  in this file's addendum.
- Never split at frame level. A clip is train or test, whole.

## 5. Pre-labelling then human review

1. Pre-label with the **Saigon YOLOv8 weights** (Zenodo record 12800597 —
   "Yolov8 Model weights (Detection of floating plastic litter and water
   hyacinths)", TU Delft, **CC-BY-4.0**, `trained_weights.zip`; code at
   github.com/TianlongJia/deep_plastic_YoloV8) — detects 3 classes: `ff_litter`,
   `hyacinth`, and `ent_litter`. Map outputs to our taxonomy (`litter`/`hyacinth`/`entangled_plastic`).
   - *Note on cross-river accuracy:* This checkpoint's litter-class accuracy drops sharply across rivers (48%→23% mAP50 in the source paper's own cross-river test), hyacinth holds up better. Whoever does the CVAT pass should expect to correct litter pre-labels heavily and trust hyacinth ones more.
2. Pre-label with our Kaggle baseline as a second opinion.
3. Human review + correction in **CVAT or Label Studio**: accept/fix/delete
   boxes; correct the class, and add `entangled_plastic` boxes where visible
   (the Saigon models have an `ent_litter` class — check those predictions too).
4. Reject (don't fix) frames where the boat is turning fast (motion blur).

## 6. Volume target & quality bar

- ≥300–500 reviewed frames total; ensure ≥50 frames contain hyacinth (it is the
  rare class everywhere else in the corpus).
- Log per session: lake, date, time, weather, mount height, clip count, kept
  frame count. Store the log next to the capture folder.
- Bengaluru frames enter the corpus as a NEW source with clip-based groups;
  30% of clips reserved as described. Do not fold them into the current
  train/val/test lists — they form their own eval source + future train source
  (same per-source yaml pattern as D10).
