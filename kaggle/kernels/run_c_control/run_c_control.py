"""RoBoat Run C-control kernel: YOLOv8n COCO initialization, Run C dataset.

Generated from kaggle/train_baseline.py (commit 3177c4a state) with the RFS-off
Run C dataset at /kaggle/input/roboat-run-c. Push with:
  .venv/Scripts/python.exe -m kaggle kernels push -p kaggle/kernels/run_c_control
"""
import subprocess, sys as _sys

subprocess.run(
    [_sys.executable, "-m", "pip", "install", "-q", "ultralytics==8.4.165"], check=True)

import argparse
import json
from pathlib import Path

import yaml

ROOT = Path("/kaggle/working")
MODELS = ["yolov8n"]
IMGSZ = 960  # D11 recommendation - do not tune in the baseline
EPOCHS, PATIENCE, SEED = 100, 20, 42
PER_SOURCE_YAMLS = ["saigon_tiles_c2", "hagenbeek_tiles_c2", "fml_c2",
                    "tud_gv_c2", "mendeley_c2", "navsci_invasive_c2",
                    "navsci_whd_c2", "combined_c2"]


def rebase(data_root: Path, workdir: Path):
    """Map ROOT-relative list/yaml entries onto the Kaggle data root."""
    lists = workdir / "lists"
    yamls = workdir / "yamls"
    lists.mkdir(parents=True, exist_ok=True)
    yamls.mkdir(parents=True, exist_ok=True)
    marker = "data/processed/"
    for f in (data_root / "lists").glob("*.txt"):
        out = []
        for line in f.read_text().splitlines():
            if marker in line:
                out.append(f"{data_root.as_posix()}/{line.split(marker)[-1]}")
            elif line:
                out.append(f"{data_root.as_posix()}/{line}")
        (lists / f.name).write_text("\n".join(out) + "\n")
    for f in (data_root / "yamls").glob("*.yaml"):
        y = yaml.safe_load(f.read_text())
        for key in ("train", "val", "test"):
            v = y.get(key)
            if v:
                sv = str(v).replace("\\", "/")
                y[key] = (f"{data_root.as_posix()}/{sv.split(marker)[-1]}"
                          if marker in sv else
                          (f"{data_root.as_posix()}/{sv}" if sv.startswith("data/") else sv))
        y["path"] = data_root.as_posix()
        (yamls / f.name).write_text(yaml.safe_dump(y))
    return yamls


def map50_row(metrics) -> dict:
    """Aggregate + per-class mAP@0.50 from an ultralytics DetMetrics."""
    names = metrics.names
    idx = metrics.box.ap_class_index
    ap50 = metrics.box.ap50
    per_class = ({names[int(c)]: float(ap) for c, ap in zip(idx, ap50)}
                 if idx is not None and len(idx) else {})
    return {"all": float(metrics.box.map50), "per_class": per_class}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", type=Path, default=Path("/kaggle/input/roboat-run-c"))
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--resume", action="store_true",
                    help="Resume from roboat/run_c_control/weights/last.pt; "
                         "fails fast when absent. Resume restores model/optimizer/"
                         "EMA/epoch; the augmentation stream restarts.")
    args = ap.parse_args()
    from ultralytics import YOLO

    yamls = rebase(args.data_root, ROOT / "rebased")
    train_yaml = yamls / "combined_c2.yaml"
    models = list(MODELS)
    results = {}
    for model_name in models:
        weights = "yolov8n.pt"
        print(f"\n===== {model_name} (init={weights}, imgsz={IMGSZ}, "
              f"epochs={EPOCHS}) =====")
        if args.dry_run:
            print(f"[dry-run] would train {model_name} (init={weights}) "
                  f"on {train_yaml}")
            continue
        last_pt = ROOT / "roboat" / "run_c_control" / "weights" / "last.pt"
        if args.resume and not last_pt.exists():
            raise SystemExit(f"--resume: {last_pt} not found. Copy roboat/ from "
                             f"the previous output into the working directory, "
                             f"or drop --resume.")
        if args.resume:
            # Guard: ultralytics strips the optimizer from last.pt at final_eval;
            # train(resume=True) on a COMPLETED run silently starts a fresh COCO8 run.
            probe = YOLO(str(last_pt))
            ck = getattr(probe, "ckpt", {}) or {}
            if ck.get("epoch", -1) >= 0 and ck.get("optimizer") is not None:
                print(f"resuming {model_name} from {last_pt}")
                model = probe
                model.train(resume=True)
            else:
                print(f"{model_name} already finished - evaluating its best.pt")
                model = YOLO(str(ROOT / "roboat" / "run_c_control" / "weights" / "best.pt"))
        else:
            model = YOLO(weights)
            model.train(data=str(train_yaml), epochs=EPOCHS, patience=PATIENCE,
                        imgsz=IMGSZ, seed=SEED, deterministic=True,
                        project=str(ROOT / "roboat"), name="run_c_control")
        best = model.trainer.best
        row = {}
        skipped = []
        for ys in PER_SOURCE_YAMLS:
            yf = yamls / f"{ys}.yaml"
            if not yf.exists():
                skipped.append(ys)
                print(f"note: skipping per-source val on {ys}: {yf} not found")
                continue
            try:
                m = YOLO(best)
                metrics = m.val(data=str(yf), imgsz=IMGSZ, split="val")
            except Exception as exc:
                skipped.append(ys)
                print(f"note: per-source val on {ys} failed ({exc}); continuing")
                continue
            row[ys] = map50_row(metrics)
        if skipped:
            print(f"note: {len(skipped)} per-source val entries skipped: "
                  f"{', '.join(skipped)}")
        results[model_name] = row
    if results and not args.dry_run:
        sources = [y for y in PER_SOURCE_YAMLS if all(y in row for row in results.values())]
        print("\n===== per-source validation mAP@0.5 (aggregate) =====")
        print(f"{'model':<12}" + "".join(f"{y:<20}" for y in sources))
        for mname, row in results.items():
            print(f"{mname:<12}" + "".join(f"{row[y]['all']:<20.4f}" for y in sources))
        print("\n===== per-class mAP@0.5 per source (P1 gate: per class AND per source) =====")
        for mname, row in results.items():
            for ys, vals in row.items():
                pc = "  ".join(f"{cls}={ap:.4f}" for cls, ap in vals["per_class"].items())
                print(f"{mname:<12}{ys:<20}{pc}")
        out = ROOT / "roboat" / "run_c_control" / "p1_gate_results.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
