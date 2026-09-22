# CP2–CP4 — Final Direction Prompt

Approved. Proceed as follows:

## CP2 — approved as (a), merge Saigon via tiling. Before marking this done:

1. **Patch `DECISIONS_LOG.md` D5** with the corrected figures ("median 4.9px @640, 67% <8px") — it's currently stale against the `AGENTS_LOG.md` finding #4 fix. Show me the diff, not just the patched text.
2. **Report post-tiling box-size-in-tile stats for Saigon specifically**, same table shape as the D11 audit. Don't cite the pre-tiling number as justification — confirm the detectability floor actually improves. This must be computed by a script you run this session, not assumed by analogy to Hagenbeek.
3. **Recompute D7's RFS table** (f_hyacinth, t/r) now that Saigon's 3,017 hyacinth boxes are in train. t=0.75 / r=3.13 was computed without Saigon and is no longer valid.
4. **Redo D10's per-source split for Saigon** as its own group-based train/val/test assignment by original image — not a raw append to existing lists. Same leakage discipline as the other three sources.
5. **Note accepted:** this spends Saigon's value as an independent OOD check. That's fine — Bengaluru T9 is the real generalization test, not Saigon. Record this trade-off in the decisions log so the paper trail shows it was considered, not overlooked.

## CP3 — apply this criterion:

Only keep a frame as a negative if it's **confidently** object-free. Default to excluding anything uncertain — at 247/495 against the D8 10% cap, there's no upside to a marginal inclusion, and false-negative supervision (an unannotated hyacinth/litter frame labeled "nothing here") is worse than just dropping it.

The contact sheet still hasn't been uploaded, so apply the criterion yourself: list which frames you excluded and the reason for each, and note in the report that no human review of `hagenbeek_empty_labels_contact_sheet.jpg` occurred. If any individual frame genuinely can't be adjudicated from the data available, flag it by name rather than guessing.

## CP4 — hold.

Regenerate `MERGE_REPORT.md`, the RFS table, the box-size table, and the training yamls **with Saigon merged** before presenting this checkpoint again. As written, the report explicitly says Saigon is "NOT merged" — approving it now would mean approving a version of the dataset you're about to change.

Sequence: **CP2 (merge + D7/D10 recompute) → CP3 → regenerate MERGE_REPORT → then CP4 for real approval.** Don't reorder, and don't present CP4 until the regenerated report exists on disk.

## Full state report — before I sign off on CP4

Send me all seven items below, in this order, each as its own clearly labeled section:

1. Complete current `DECISIONS_LOG.md` (all D-numbered decisions, not just the ones touched this round)
2. `AGENTS_LOG.md` full findings list (not just #4)
3. Regenerated `MERGE_REPORT.md`
4. Per-source dataset inventory: image count, box count, class distribution, resolution range, split assignment method — for every source currently merged
5. Current RFS parameters (t, r) post-recompute, with f_hyacinth shown
6. Status of T9 (Bengaluru capture set) — what's in it now vs. what's still needed
7. Any prior CP1 decision/rationale I haven't seen

## Execution and delivery rules

- **Each numbered item is a gate.** If any step fails or a number can't be reproduced, stop and report the failure — don't paper over it and move on.
- **Every figure in the state report must come from a script you actually ran this session** (post-tiling box stats, RFS recompute, split counts). Show the command or script name next to each table so I can re-run it. Transcribe numbers from tool output; don't round or paraphrase from memory.
- **Write the state report to `reports/STATE_REPORT.md`** and paste it in full in your reply.
- **If a recomputed number contradicts an earlier decision** (e.g., RFS t/r lands outside what D7 assumed, or post-tiling box stats don't actually improve), flag it at the top of the report under "Deviations" — don't silently overwrite the log.
- **Do not mark CP4 approved yourself, and do not start training.** Nothing is approved until I've read the state report and replied.

Send the state report when — and only when — every item above is complete.
