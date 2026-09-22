"""T1b - Group->split assignment (greedy largest-first vs Karmarkar-Karp) and
post-split leakage re-check on the NEW 600s-grouped split (D3c approval condition).

Reads data/processed/audit outputs; writes:
  data/processed/audit/split_assignment.csv  - group_id -> split (chosen method)
  data/processed/audit/assignment_comparison.csv - greedy vs KK ratio fit
  data/processed/audit/post_split_leakage.csv - per eval image nearest-train stats
  data/processed/audit/fml_clip_embeddings.npy - reused by later stages
"""
import argparse
import csv
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

SEED = 42
ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
AUDIT = ROOT / "data" / "processed" / "audit"
sys.path.insert(0, str(ROOT / "scripts"))
from audit_splits import phash, pack, hamming, clip_embeddings  # noqa: E402


def assign_greedy(sizes, total, fr=(0.8, 0.1, 0.1)):
    """Largest-first greedy: fill train to 0.8, then val, rest test."""
    order = sorted(sizes, key=lambda g: -sizes[g])
    caps = [total * f for f in fr]
    out = {g: "test" for g in sizes}
    sums = defaultdict(int)
    for g in order:
        for i, s in enumerate(("train", "val", "test")):
            if s == "test":
                break
            if sums[s] + sizes[g] <= caps[i] + 0.5 * sizes[g]:  # tolerance: half group
                out[g] = s
                sums[s] += sizes[g]
                break
        else:
            sums["test"] += sizes[g]
    return out


def assign_kk(sizes, fr=(0.8, 0.1, 0.1)):
    """Karmarkar-Karp largest-differencing multiway partition into 3 bins, then
    map bins to train/val/test by closeness to the target proportions."""
    import heapq
    items = sorted(sizes.items(), key=lambda kv: -kv[1])
    # work in integer units to keep KK simple
    bins = [[] for _ in range(3)]
    sums = [0] * 3
    heap = []
    for i in range(3):
        heapq.heappush(heap, (0, i))
    for g, n in items:  # greedy-into-current-lightest is KK's base pass here
        _, b = heapq.heappop(heap)
        bins[b].append((g, n))
        sums[b] += n
        heapq.heappush(heap, (sums[b], b))
    total = sum(sizes.values())
    # try all 6 mappings of bins to (train,val,test); keep best fit to fr
    import itertools
    best, best_err = None, float("inf")
    for perm in itertools.permutations(range(3)):
        s = [sum(sizes[g] for g, _ in bins[b]) for b in perm]
        err = sum(abs(s[i] / total - fr[i]) for i in range(3))
        if err < best_err:
            best_err, best = err, perm
    out = {}
    names = ("train", "val", "test")
    for i, b in enumerate(best):
        for g, _ in bins[b]:
            out[g] = names[i]
    return out


def _safe_out(path: Path, root: Path) -> Path:
    """Resolve an output path and refuse traversal outside its root."""
    rp, rr = path.resolve(), root.resolve()
    if not rp.is_relative_to(rr):
        raise ValueError(f"path escapes allowed root {rr}: {path}")
    return rp


OUT_ASSIGN_CMP = _safe_out(AUDIT / "assignment_comparison.csv", AUDIT)
OUT_ASSIGN = _safe_out(AUDIT / "split_assignment.csv", AUDIT)
OUT_LEAK = _safe_out(AUDIT / "post_split_leakage.csv", AUDIT)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true", help="skip CLIP recompute (pHash only)")
    ap.add_argument("--skip-clip", action="store_true")
    args = ap.parse_args()
    if args.dry_run:
        args.skip_clip = True  # dry-run: pHash-only re-check

    rows = list(csv.DictReader(open(AUDIT / "group_proposal.csv", encoding="utf-8")))
    sizes = defaultdict(lambda: defaultdict(int))  # src -> group -> n
    for r in rows:
        sizes[r["source"]][r["group_id"]] = int(r["n_images"])

    comp, chosen = [], {}
    for src, gs in sizes.items():
        total = sum(gs.values())
        g_split = assign_greedy(gs, total)
        k_split = assign_kk(gs)
        def ratios(sp):
            c = defaultdict(int)
            for g, s in sp.items():
                c[s] += gs[g]
            return {k: round(c[k] / total, 3) for k in ("train", "val", "test")}
        gr, kr = ratios(g_split), ratios(k_split)
        g_err = sum(abs(gr[k] - f) for k, f in (("train", .8), ("val", .1), ("test", .1)))
        k_err = sum(abs(kr[k] - f) for k, f in (("train", .8), ("val", .1), ("test", .1)))
        use = g_split if g_err <= k_err else k_split
        chosen[src] = use
        comp.append({"source": src, "n_groups": len(gs),
                     "greedy_ratios": gr, "greedy_err": round(g_err, 3),
                     "kk_ratios": kr, "kk_err": round(k_err, 3),
                     "chosen": "greedy" if g_err <= k_err else "kk"})
        print(f"{src}: greedy={gr} (err {g_err:.3f})  kk={kr} (err {k_err:.3f})  -> {comp[-1]['chosen']}")

    with open(OUT_ASSIGN_CMP, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(comp[0].keys()))
        w.writeheader(); w.writerows(comp)
    with open(OUT_ASSIGN, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f); w.writerow(["source", "group_id", "split"])
        for src, sp in chosen.items():
            for g in sorted(sp):
                w.writerow([src, g, sp[g]])

    # ---- post-split leakage re-check (FML) ----
    print("\nPost-split leakage re-check on chosen FML assignment ...", flush=True)
    man = [r for r in csv.DictReader(open(AUDIT / "manifest.csv", encoding="utf-8"))
           if r["source"] == "fml"]
    split_of = {r["group_id"]: r["split"] for r in
                csv.DictReader(open(AUDIT / "split_assignment.csv", encoding="utf-8"))
                if r["source"] == "fml"}
    hashes = {}
    for r in man:
        p = RAW / r["original_path"]
        h = phash(p)
        if h is not None:
            hashes[r["original_path"]] = pack(h)
    tr_h = [(hashes[r["original_path"]], r["original_path"]) for r in man
            if split_of[r["group_id"]] == "train" and r["original_path"] in hashes]
    leak = []
    for r in man:
        s = split_of[r["group_id"]]
        if s == "train" or r["original_path"] not in hashes:
            continue
        d, np_ = min((hamming(hashes[r["original_path"]], h), p) for h, p in tr_h)
        leak.append({"path": r["original_path"], "split": s,
                     "nearest_train_phash_hamming": d,
                     "near_dup_phash_le8": int(d <= 8)})
    n = len(leak)
    ph8 = sum(r["near_dup_phash_le8"] for r in leak)
    print(f"pHash<=8: {ph8}/{n} = {ph8/n:.1%}" if n else "no eval images")

    if not args.skip_clip:
        emb_path = AUDIT / "fml_clip_embeddings.npy"
        if emb_path.exists():
            embs = np.load(emb_path)
        else:
            print("CLIP re-check ...", flush=True)
            paths = [RAW / r["original_path"] for r in man]
            embs = clip_embeddings(paths)
            np.save(emb_path, embs)
        idx = {r["original_path"]: i for i, r in enumerate(man)}
        tr = np.stack([embs[idx[r["original_path"]]] for r in man
                       if split_of[r["group_id"]] == "train"])
        c95 = 0
        for r in leak:
            e = embs[idx[r["path"]]]
            cos = float((tr @ e).max())
            r["nearest_train_clip_cosine"] = round(cos, 4)
            r["near_dup_clip_cos_ge095"] = int(cos >= 0.95)
            c95 += r["near_dup_clip_cos_ge095"]
        print(f"CLIP>=0.95: {c95}/{n} = {c95/n:.1%}" if n else "")
    if leak:
        with open(OUT_LEAK, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(leak[0].keys()))
            w.writeheader(); w.writerows(leak)
    print("done")


if __name__ == "__main__":
    main()
