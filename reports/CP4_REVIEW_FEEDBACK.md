# CP4 Review Feedback (external review, 2026-09-29)

> Paste-source for the other session: resolve blockers 1–2, then re-present for CP4.
> Verbatim reviewer message below.

---

**CP4 still held. Two blockers below need resolving before this goes back for approval — both are "show your work," not redesigns.**

**1. Reconcile Saigon's box count through tiling.** Raw 9,352 → final 7,332 in MERGE_REPORT is a 21.6% loss; Hagenbeek gained 46.9% (1,415→2,078) through the *same* tiling config. Those can move in opposite directions legitimately, but nobody's shown the arithmetic. Produce an explicit table for both sources: raw boxes → dropped at tile boundary → gained via overlap duplication → dropped by the 35% aerial cap → final count. Numbers must tie out exactly to MERGE_REPORT's 7,332 and 2,078.

Also: the task-141 log mid-session shows Saigon boxes as 8,392 (train 6229 + val 1289 + test 874) — a third number matching neither raw nor final. Confirm 7,332 is the actual current output of the latest verified run, not a stale transcription, by re-running whatever script produces the per-source box counts and pasting the literal stdout.

**2. `DECISIONS_LOG.md` is missing D2, D4, D6, D8, D9.** Only D1, D3a, D3b/c, T8, T9, D5, D3, D7, D10, D11 exist as headers. But D4 (aerial cap), D6 (AquaTrash), D8 (background budget), D9 (Hagenbeek empty-label quarantine) are cited by number as settled decisions throughout MERGE_REPORT, AGENTS_LOG, and STATE_REPORT — STATE_REPORT even writes out D9's full rationale despite D9 not existing in the log it's supposedly summarizing. Backfill all five missing entries with evidence/tags in the same format as the existing ones, sourced from wherever the actual decision was made (build output, code comments, prior conversation) — not reconstructed from memory of what they probably said.

**Also fix while in there, non-blocking:**

3. `T9` is used as an ID for two unrelated things (Saigon pre-labelling weights in DECISIONS_LOG; Bengaluru Capture Set everywhere else). Rename one.

4. DECISIONS_LOG's T9 says the Zenodo checkpoint is "two models: plastic litter and water hyacinth." BENGALURU_CAPTURE.md step 3 says to also check its `ent_litter` predictions. Confirm which is actually true — two classes or three — and fix whichever doc is wrong.

5. Add a line to BENGALURU_CAPTURE.md's pre-labelling section: this checkpoint's litter-class accuracy drops sharply across rivers (48%→23% mAP50 in the source paper's own cross-river test), hyacinth holds up better. Whoever does the CVAT pass should expect to correct litter pre-labels heavily and trust hyacinth ones more.

6. AGENTS_LOG's "External review" section still quotes fml background count as 247; MERGE_REPORT correctly says 350. Delete the stale line or date-stamp it so it's not read as current.

**Once 1 and 2 are done, append the reconciliation table and the backfilled D-entries to MERGE_REPORT/STATE_REPORT — no full regen needed — then re-present for CP4.**
