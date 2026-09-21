# Data Exploration Report — `data/raw`

Generated 2026-09-21 by `scripts/explore_data.py` (full machine-readable results in
`data/exploration_results.json`; box-drawn samples in `data/exploration_samples/<dataset>/`).

Every finding is tagged **[Confirmed]** (verified by reading file contents) or
**[Inferred]** (deduced from names/structure without opening content). This report
describes what is in the data only — no training/model recommendations.

---

## Summary (cross-dataset table)

| Dataset (folder) | Expected name | Images (unique) | Format | Classes (literal) | Instances | Usable as-is for YOLO? | Conversion needed |
|---|---|---|---|---|---|---|---|
| `120969/` | Lazzerini / Floating Marine Litter (FML) v2 | 5,299 (10,598 files incl. duplicated `single_sets`) | YOLO txt **and** COCO json, in parallel | `garbage` (COCO id 1 = YOLO id 0) | 16,457 boxes (unique) | **Yes** (YOLO tree already exists) | Remove/dedup the `single_sets` copy; no class-names file in folder |
| `TUD-GV Dataset for Floating Litter Detection/` | TUD-GV litter subset | 1,501 | YOLO txt | `litter` (1 class) | 8,181 | **Yes** | None (delete redundant zips) |
| `Platic-water hyathin/` | Hagenbeek / Wageningen | 82 | YOLO txt | `ff_litter`, `hyacinth`, `ent_litter` | 1,415 | **Yes** | None structural; 10 empty label files to decide on |
| `AquaTrash-master/` | AquaTrash | 369 | CSV (`image_name,x_min,y_min,x_max,y_max,class_name`, absolute pixels) | `glass`, `paper`, `metal`, `plastic` | 469 | **No** | CSV → YOLO txt conversion script required |
| *(missing)* | Custom / Bengaluru capture | 0 | — | — | — | — | Not present in `data/raw` |

**[Confirmed]** All of the above was verified by reading the actual label files, COCO
JSONs, CSV, and `classes.txt` files. **[Inferred]** The mapping of folder names to the
"expected dataset" names is by content similarity (e.g. `120969` is a Roboflow-style
dataset id whose COCO metadata says "fml_version2"); no README confirms the naming.

---

## 1. `120969/` — Lazzerini / Floating Marine Litter (FML v2)

- **Image count** — **[Confirmed]** 10,598 `.jpg` files total. Of these, 5,299 are
  unique: `full_dataset` has train 3,711 / val 1,059 / test 529 (= 5,299), and 27
  `single_sets/setN` folders repeat the same 5,299 images (hash-identical).
  Breakdown: 10,598 `.jpg`, 10,598 `.txt`, 3 `.json`.
- **Annotation format** — **[Confirmed]** Both YOLO and COCO ship in parallel:
  - `fml_version2/full_dataset/labels/yolo_format/{train,val,test}/*.txt` — normalized
    `class cx cy w h` (e.g. `0 0.70234375 0.16796875 0.02578125 0.0234375`).
  - `fml_version2/full_dataset/labels/coco_format/{train,val,test}.json` — COCO with
    `images`/`annotations`/`categories`; annotations: train 11,333 / val 3,476 /
    test 1,648 = 16,457, `categories = [{"id": 1, "name": "garbage"}]`.
  - YOLO txt count across the whole folder is 32,914 boxes = 2 × 16,457, exactly
    matching the `single_sets` duplication. **[Confirmed]**
- **Class list** — **[Confirmed]** Single class: `garbage` (COCO id 1; YOLO files use
  id 0). **[Inferred]** YOLO id 0 ↔ COCO id 1 `garbage` (consistent box counts).
  No `classes.txt`/`data.yaml` in the folder — the name is only recoverable from the
  COCO JSON.
- **Class balance** — **[Confirmed]** 16,457 instances, all one class.
- **Image dimensions** — **[Confirmed]** All 10,598 images are exactly 1920×1080.
- **Orphan check** — **[Confirmed]** 0 images without label, 0 labels without image
  (within `full_dataset`).
- **Corrupt files** — **[Confirmed]** 0 (PIL verify on all 10,598).
- **Duplicates** — **[Confirmed]** 8,823 duplicate-hash groups / 12,373 redundant
  files, all internal (`full_dataset` ↔ `single_sets`). See Decisions Needed.
- **Samples** — 5 rendered to `data/exploration_samples/120969/` (class label drawn as
  `garbage`, recovered from COCO categories). **[Confirmed]**
- **README/metadata** — **[Confirmed]** None (no README, license, or metadata file).

## 2. `TUD-GV Dataset for Floating Litter Detection/`

- **Image count** — **[Confirmed]** 1,501 `.jpg` + 1,502 `.txt` (one is `classes.txt`)
  + 2 `.zip` archives (`images.zip`, `labels_txt.zip`) that duplicate the extracted
  content.
- **Annotation format** — **[Confirmed]** YOLO txt, normalized
  `class cx cy w h` (sample `exp11_102.txt`: `0 0.119531 0.459722 0.074479 0.173148`).
- **Class list** — **[Confirmed]** `classes.txt` = `litter` (single class, id 0).
  Mismatch vs target taxonomy: generic `litter`, no `water_hyacinth`/
  `entangled_plastic` distinction. **[Confirmed]**
- **Class balance** — **[Confirmed]** 8,181 boxes, all `litter`.
- **Image dimensions** — **[Confirmed]** All 1,501 images exactly 1920×1080.
- **Orphan check** — **[Confirmed]** 0 / 0.
- **Corrupt files** — **[Confirmed]** 0.
- **Duplicates** — **[Confirmed]** 0 exact duplicates among extracted files; the two
  zips are redundant archives of the same content (not counted as file dupes).
- **Samples** — 5 rendered to `data/exploration_samples/TUD-GV Dataset for Floating Litter Detection/`. **[Confirmed]**
- **README/metadata** — **[Confirmed]** None.

## 3. `Platic-water hyathin/` — Hagenbeek / Wageningen (hyacinth set)

- **Image count** — **[Confirmed]** 82 `.jpg` (uppercase `.JPG` lowercased in scan),
  82 YOLO label `.txt` + `classes.txt`, 1 `notes.json`, 14 `.py` scripts, 1 `.ipynb`,
  `README.docx`, 1 `.cache`.
- **Annotation format** — **[Confirmed]** YOLO txt, normalized, many boxes extremely
  small (sample: `2 0.16230 0.29126 0.00448 0.00573` — sub-0.5% of frame width).
  Aerial/drone imagery (DJI/Gopro filenames). **[Confirmed]**
- **Class list** — **[Confirmed]** `classes.txt` = `ff_litter`, `hyacinth`,
  `ent_litter`. Direct 1:1 match potential with target taxonomy:
  `ff_litter`↔`plastic_debris`, `hyacinth`↔`water_hyacinth`, `ent_litter`↔`entangled_plastic`
  (**Inferred** — mapping by name similarity, not by any documented spec).
- **Class balance** — **[Confirmed]** `ff_litter` 632, `hyacinth` 485, `ent_litter` 298
  (total 1,415). Mild imbalance (~2.1× between extremes), not severe.
- **Image dimensions** — **[Confirmed]** Mixed: 5280×3956 to 5568×4872 (mode
  5568×4872, 44/82 images). Very large images.
- **Orphan check** — **[Confirmed]** 0 / 0 by stem match.
- **Corrupt files** — **[Confirmed]** 0.
- **Duplicates** — **[Confirmed]** One hash-group of 10 identical label files — all
  **empty** files (10 images have empty labels). Not a data-quality problem, but
  those 10 images contribute zero positive instances.
- **Samples** — 5 rendered to `data/exploration_samples/Platic-water hyathin/`. **[Confirmed]**
- **README/metadata** — **[Confirmed]** `README.docx` present but not plain-text
  readable (binary Word file; text not extracted here). `notes.json` present
  (Roboflow-style export notes). 14 author Python scripts + notebook for two separate
  models (hyacinth / plastic) ship alongside. **[Confirmed]**

## 4. `AquaTrash-master/`

- **Image count** — **[Confirmed]** 369 `.jpg` in the double-nested
  `AquaTrash-master/AquaTrash-master/Images/`.
- **Annotation format** — **[Confirmed]** Single CSV `annotations.csv`, header
  `image_name,x_min,y_min,x_max,y_max,class_name` with absolute-pixel boxes, 469 rows
  (multiple boxes per image). Roboflow-exported filenames (`*.rf.<hash>.jpg`).
- **Class list** — **[Confirmed]** `glass` (44), `paper` (116), `metal` (118),
  `plastic` (191). Matches the bundled README ("4 classes {(0: glass), (1: paper),
  (2: metal), (3: plastic)}"). Mismatch vs target taxonomy: only `plastic` maps
  cleanly; `glass`/`paper`/`metal` have no target equivalent.
- **Class balance** — **[Confirmed]** See class list; ~4.3× between `plastic` and
  `glass` within the dataset.
- **Image dimensions** — **[Confirmed]** Mixed: 416×416 to 3264×3264 (mode 416×416,
  179/369 images).
- **Orphan check** — **[Confirmed]** 0 images without a CSV row, 0 CSV rows without an
  image (369 unique image names in CSV = 369 jpg files). (An initial automated check
  reported orphans — that was an artifact of checking against YOLO txts; corrected here.)
- **Corrupt files** — **[Confirmed]** 0.
- **Duplicates** — **[Confirmed]** 0 exact hash duplicates (similar stems like
  `000000_jpg.rf.*` differ in hash — Roboflow augmentation variants, **Inferred**).
- **Samples** — 5 rendered to `data/exploration_samples/AquaTrash-master/`. **[Confirmed]**
- **README/metadata** — **[Confirmed]** `README.md` ("AquaVision", 369 images, 470
  boxes, 4 classes; Elsevier 2020 paper) and `LICENSE` (MIT, © 2020 Harsh Panwar).

## 5. Custom / Bengaluru capture folder

- **[Confirmed]** No such folder exists in `data/raw` (only the 4 folders above).

---

## Cross-dataset checks

- **Cross-folder duplicates** — **[Confirmed]** 5,299 duplicate-hash groups across the
  whole `data/raw` tree; all are internal to `120969` (`single_sets` ↔ `full_dataset`,
  group count matches the single_sets image count exactly). **No image appears in more
  than one dataset folder.**
- **Formats requiring conversion** — Only **AquaTrash** (CSV absolute-pixel → YOLO
  normalized). All others are already YOLO.
- **Estimated merged per-class counts (unique images only, after name mapping):**

  | Target class | Sources | Est. instances |
  |---|---|---|
  | `plastic_debris` | FML `garbage` (16,457) + TUD-GV `litter` (8,181) + Hagenbeek `ff_litter` (632) + AquaTrash `plastic` (191) | **25,461** |
  | `water_hyacinth` | Hagenbeek `hyacinth` | **485** |
  | `entangled_plastic` | Hagenbeek `ent_litter` | **298** |

  **[Confirmed]** per-dataset counts; **[Inferred]** the mapping (FML `garbage` and
  TUD-GV `litter` are generic debris classes, not verified to be "plastic" specifically).
- **Class imbalance** — **[Confirmed]** Severe: `water_hyacinth` and
  `entangled_plastic` have ~1.7% and ~1.0% of the instances that `plastic_debris`
  would have (~50–85× fewer). Also note only 82 images in the entire corpus contain
  any hyacinth/entanglement labels at all.

---

## Decisions Needed

1. **`120969` duplication** — the 27 `single_sets` folders exactly duplicate
   `full_dataset` (12,373 redundant files). Merge should use only `full_dataset`
   (5,299 unique images). Confirm and drop `single_sets`.
2. **FML class name** — YOLO labels use id 0 with no `classes.txt`; name `garbage` is
   recoverable only from the COCO JSONs. Confirm 0↔`garbage` before any merge.
3. **Missing Bengaluru capture folder** — expected but absent. Add the data or
   proceed without it?
4. **Taxonomy mapping** — is FML `garbage` / TUD-GV `litter` acceptable as
   `plastic_debris` (they are generic debris classes, not verified plastic-only)?
   And accept `ff_litter`→`plastic_debris`, `ent_litter`→`entangled_plastic`?
5. **AquaTrash class fate** — `glass`/`paper`/`metal` have no target-taxonomy
   equivalent: drop those instances, fold into `plastic_debris`, or skip the dataset?
   (Requires the CSV→YOLO conversion regardless.)
6. **Hagenbeek empty labels** — 10 of 82 images have empty label files: keep as
   negatives/background images or exclude?
7. **TUD-GV redundant zips** — `images.zip` / `labels_txt.zip` duplicate the extracted
   content; safe to delete (needs confirmation since data/raw is treated read-only).
8. **Hagenbeek source scripts** — 14 author `.py` scripts + notebook live inside the
   dataset folder; exclude them from any merge tree (they are not data).
9. **Sample-verification of label quality** — box-drawn samples are in
   `data/exploration_samples/<dataset>/`; eyeball whether labels align (especially
   Hagenbeek's sub-pixel-scale aerial boxes) before trusting the annotations.
