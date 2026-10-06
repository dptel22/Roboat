# Hugging Face as a P1 training venue — decision note

Question: can Ultralytics YOLO training (P1: yolov8n/yolo11n, imgsz 960, 100 epochs,
10,070-line RFS train list) run on *free* HF infrastructure, as a third option
beside free Colab and Kaggle? Researched 2026-10-02. Everything below was read
from official docs/this session; nothing was executed on real GPUs (this machine
is CPU-only), so all runtime/cost figures for our runs are labeled estimates.

## 1. ZeroGPU (Spaces) — not a training venue

ZeroGPU is the GPU backend for **Gradio inference demos** in Spaces, not for
long-running jobs: GPU is granted per `@spaces.GPU` function call (default
**60-second duration cap**, extendable per-call), the Space must be Gradio-SDK,
and quota is a **daily minutes budget**, not hours ([ZeroGPU docs](https://huggingface.co/docs/hub/en/spaces-zerogpu)).

- Daily quota: unauthenticated 2 min, free account **5 min/day**, PRO 40 min/day
  (extensible with credits at $1/10 min); resets 24 h after first use.
- Even one YOLO epoch at imgsz 960 over 10k images takes minutes-to-tens-of-minutes;
  100 epochs need GPU-hours. The free daily quota of minutes makes training
  arithmetically impossible; the per-call duration model and Gradio-only SDK rule
  out wrapping a training script anyway.
- **Verdict: ruled out for P1 training.** ZeroGPU is fine later for a *demo*
  Space serving `best.pt` (that is what it is designed for).

## 2. HF Jobs / auto-training — works technically, but there is no free GPU

[HF Jobs](https://huggingface.co/docs/hub/en/jobs-pricing) run an arbitrary
Docker image + command on managed hardware, billed **per minute** with **no free
tier**: "available to any user or organization with a positive credit balance";
the monthly credits included with PRO ($9/mo), Team, Enterprise count toward it.
Default timeout is 30 min — a long training run needs `--timeout` set explicitly.
Cheapest GPU flavors: T4-small **$0.40/h**, T4-medium $0.60/h, L4 $0.80/h.

Two sub-options:

- **Custom Job with Ultralytics** (possible but *untested* by us): a Job is just
  a Docker command, so `pip install ultralytics` + `python train_baseline.py`
  should run unchanged. Cost estimate (not measured, this machine is CPU-only):
  if yolov8n @ 960 runs ~5–7 min/epoch on a T4-class GPU, 100 epochs ≈ 8–12 h ≈
  **$3.50–$5 per run** on T4-small; Runs A+B ≈ **$7–$10**. Checkpoints + resume
  make multi-day runs fine (no session kills).
- **`huggingface-vision-trainer` skill / AutoTrain** (the "managed" route):
  the official [vision-trainer skill](https://github.com/huggingface/skills)
  (`skills/huggingface-vision-trainer/SKILL.md`, read this session) trains
  **D-FINE, RT-DETR v2, DETR, YOLOS** — *not* Ultralytics YOLO — on HF Jobs, and
  states "Jobs require paid plan". [AutoTrain object detection](https://huggingface.co/docs/autotrain/en/tasks/object_detection)
  is the same Transformers-family (DETR/YOLOS) story. Using it would change the
  detector architecture, break the Run A/B A/B plan, and orphan the converted
  Zenodo checkpoint init (`Model_tiles_weights_converted.pt`, 349/355-key
  transfer — `kaggle/RUN_PLAN.md:24-47`).

**Verdict: not free.** Viable only as a paid fallback (~$4/run est.) via a
custom Job; the skill/AutoTrain route is incompatible with the Ultralytics P1 plan.

## 3. Comparison against the planned venues

| Dimension | Colab free | Kaggle | HF ZeroGPU | HF Jobs |
|---|---|---|---|---|
| Free GPU hours | T4, variable daily cap; fit at imgsz 960 × 100 epochs **unverified** (already flagged, `kaggle/RUN_PLAN.md 'Wall-clock budget & resume' section`) | ~30 h/week GPU quota, sessions ~9–12 h ([Kaggle docs](https://www.kaggle.com/docs/efficient-gpu-usage); session cap varies 9→12 h across doc versions — not re-verified here) | 5 min/day — unusable for training | $0 free GPU; billed from first minute |
| Private dataset (licensed data must stay private) | Google Drive, private by default | Private Kaggle dataset, already designed for (`kaggle/README.md` §1: "make it private (contains licensed research data)") | n/a | Private Hub dataset: free accounts get **100 GB private storage** ([storage limits](https://huggingface.co/docs/hub/en/storage-limits)) — plenty for the ~few-GB `roboat-processed` zip |
| Session limits @ imgsz 960, 100 epochs | ~12 h session (RUN_PLAN caveat); risk of mid-run kill; unverified fit | Session cap fine for yolov8n @ 960 (est.); resume available if a session ends; 30 h/wk covers both runs | N/A | No session limit concern; 30-min default timeout must be overridden |
| Friction | Notebook copy-paste + Drive mount; rebase() equivalent needed | **Lowest** — `kaggle/train_baseline.py` `rebase()`, dataset zip recipe, `--pretrained` Run B, per-class/per-source tables all already built and documented (`kaggle/README.md`, `kaggle/RUN_PLAN.md`) | N/A | Highest: new tooling (Docker image/HF token/secrets, `--timeout`, YOLO-format data either uploaded as-is or converted to Hub `objects`-column format for the skill route), plus a billing account |

## 4. One-line verdict

**Launch P1 on Kaggle** (free ~30 GPU-h/week, T4×2, private-dataset flow and
`train_baseline.py` tooling already built and Run-B-ready), keep free Colab as
the backup venue, and treat HF only as a paid fallback (HF Jobs, ~$4/run est.)
plus the eventual ZeroGPU demo Space for `best.pt` — HF has no free GPU path for
this training.

## What was and wasn't verified

- **Verified by reading this session**: ZeroGPU quotas/duration model ([docs](https://huggingface.co/docs/hub/en/spaces-zerogpu)); Jobs per-minute pricing table, no free tier, 30-min default timeout ([docs](https://huggingface.co/docs/hub/en/jobs-pricing)); vision-trainer skill's supported models and "Jobs require paid plan" ([SKILL.md](https://github.com/huggingface/skills), fetched raw); 100 GB free private storage ([storage limits](https://huggingface.co/docs/hub/en/storage-limits)); project tooling claims (`kaggle/README.md`, `kaggle/RUN_PLAN.md`).
- **Not verified here** (reported as such): any YOLO training runtime on GPU (this machine is CPU-only — the Jobs cost figures are estimates); whether an Ultralytics custom Job actually runs end-to-end on HF Jobs; current Kaggle session cap (9 vs 12 h; Kaggle doc page not fetched directly); Colab free daily GPU caps and 960px/100-epoch fit (already flagged unverified in `kaggle/RUN_PLAN.md 'Wall-clock budget & resume' section`).
