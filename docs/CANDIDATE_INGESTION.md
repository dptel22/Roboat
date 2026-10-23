# Candidate Dataset Ingestion Gate

Candidate datasets are audited in isolation before they can enter the RoBoat
training tree. The existing Run C dataset and donor archives are not modified
by the audit.

## Current registry

| Candidate | Decision | Reason |
|---|---|---|
| FML v2, TUD-GV, Hagenbeek, Saigon | Excluded from new-source work | Already represented in the current pipeline |
| Mendeley Floating Waste and River Vegetation | Excluded from new-source work | Already represented as the Run C donor |
| Navsci invasive-aquatic-plants | Excluded from new-source work | Already represented as the Run C donor |
| Navsci water-hyacinth-detection / WHD-7282 | Excluded from new-source work | Already represented as the Run C donor |
| FloatingWaste-I | Quarantined | Candidate requires a fresh local archive, annotation inspection, checksum, and dedupe audit |
| RF100-VL floating-waste | Quarantined | Primarily a litter candidate; no verified hyacinth class in the repository record |
| IWHR | Quarantined | License is recorded, but the camera/domain mismatch and annotation details require a separate audit |
| FloW-Img | Quarantined | Access is gated and the repository has no verified local annotations |
| Zhu & Xu 2025 dataset | Quarantined | The paper does not identify a retrievable dataset or license |

“Quarantined” means no download, conversion, or merge was performed. A source
can leave quarantine only after a local archive is supplied and the audit below
returns `READY_FOR_MERGE`.

## Audit command

```powershell
.venv/Scripts/python.exe scripts/audit_candidate_donor.py `
  --candidate-root C:/datasets/candidate `
  --slug candidate_name `
  --source-url https://example.org/dataset `
  --license "CC BY 4.0" `
  --version "v1" `
  --group-mode auto
```

The command writes only to
`data/processed/candidate_audits/<slug>/` (or `--output`). It produces:

- `provenance.json` — source, license, version, class mapping, and audit parameters;
- `candidate_records.csv` — one row per image, with label path, group, SHA-256,
  pHash, dimensions, box count, and issues;
- `duplicate_matches.json` — exact and pHash matches against incumbent data;
- `proposed_group_splits.json` — deterministic group-aware train/val/test proposal;
- `candidate_audit.json` — gate results and quarantine reasons.

The tool fails closed when source provenance, class names, annotation integrity,
deduplication, group integrity, or box quality is unresolved. It never treats
polygon segmentation labels as bounding boxes and never silently maps unknown
vegetation classes to `hyacinth`.

## Merge requirements after a passing audit

A later run-scoped builder must still:

1. convert to both `merged2` and `merged3` explicitly;
2. preserve the original candidate tree and its archive checksum;
3. include the source in the combined aerial cap when it is aerial;
4. inherit splits from the audited groups before tiling;
5. re-run exact/pHash and CLIP leakage checks against the full incumbent corpus;
6. reconcile per-source image and box counts; and
7. prove that disabling the candidate reproduces the existing Run C manifest and labels.

No candidate is accepted based only on a paper title, a search result, or an
unverified license claim.
