"""T6 - Kaggle baseline training: yolov8n and yolo11n, DEFAULT hyperparameters.

No hyperparameter tuning (locked decision): epochs=100, patience=20, imgsz=960
(from D11 box-size audit: FML median box 14px/<8px share 18% at 640 vs 21px/2.6%
at 960). After each model trains, runs val on every per-source yaml and prints
per-class AND per-source mAP@0.5 tables (the P1 gate: not one aggregate).

Run on Kaggle with the dataset uploaded per kaggle/README.md:
  python train_baseline.py --data-root /kaggle/input/roboat-processed
Run B additionally passes --pretrained <converted Model_tiles_weights.pt>
(see kaggle/RUN_PLAN.md). The converted checkpoint is a yolov8n architecture,
so Run B trains the yolov8n leg only (the script skips the other MODELS entries).
Pass --resume to continue an interrupted run from its last.pt
(roboat/baseline_<model>/weights/last.pt) instead of starting fresh - 100
epochs at imgsz 960 outlasts a single free session.
"""
import argparse
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
MODELS = ["yolov8n", "yolo11n"]
IMGSZ = 960  # D11 recommendation - do not tune in the baseline
EPOCHS, PATIENCE, SEED = 100, 20, 42
PER_SOURCE_YAMLS = ["saigon_tiles_c2", "hagenbeek_tiles_c2", "fml_c2",
                    "tud_gv_c2", "mendeley_c2", "navsci_invasive_c2",
                    "navsci_whd_c2", "combined_c2", "ood_aquatrash"]


def rebase(data_root: Path, workdir: Path):
    """Map ROOT-relative list/yaml entries onto the Kaggle data root.

    build_dataset.py writes ROOT-relative paths ("data/processed/merged2/..."),
    so the mapping is: whatever follows "data/processed/" joined onto data_root.
    Yaml path/train/val/test values get the same treatment.
    """
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
    ap.add_argument("--data-root", type=Path, default=Path("data/processed"))
    ap.add_argument("--pretrained", default="yolov8n.pt",
                    help="Initial weights passed to YOLO(). With the default, each "
                         "model in MODELS starts from its own COCO init "
                         "({model_name}.pt). Pass an explicit checkpoint (e.g. the "
                         "converted Model_tiles_weights.pt, loaded via YOLO() - never "
                         "raw torch.load) to start from it; because that checkpoint is "
                         "a yolov8n architecture, only the yolov8n leg is then run "
                         "(other MODELS entries are skipped, so no leg is mislabeled), "
                         "and ultralytics transfers the compatible layers "
                         "automatically (349/355 for nc=2).")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--resume", action="store_true",
                    help="If the run's last.pt exists "
                         "(roboat/baseline_<model>/weights/last.pt), resume from "
                         "it instead of starting fresh. Resume restores model/"
                         "optimizer/EMA/epoch and the epoch-seeded shuffle order; "
                         "the augmentation stream restarts (no RNG state is "
                         "checkpointed), so a resumed run is valid but not "
                         "sequence-identical. On Kaggle, re-attach the previous "
                         "session's output and COPY roboat/ into /kaggle/working "
                         "first - /kaggle/input is read-only.")
    args = ap.parse_args()
    from ultralytics import YOLO

    yamls = rebase(args.data_root, Path("rebased"))
    train_yaml = yamls / "combined_c2.yaml"
    models = list(MODELS)
    if args.pretrained != "yolov8n.pt":
        # The converted Zenodo checkpoints are yolov8n-architecture: running the
        # yolo11n entry with them would silently train yolov8n under a yolo11n
        # label. Run B is therefore the yolov8n leg only.
        skipped = ", ".join(m for m in models if m != "yolov8n")
        models = [m for m in models if m == "yolov8n"]
        print(f"note: --pretrained is a yolov8n-architecture checkpoint; running "
              f"the yolov8n leg only (skipped: {skipped})")
    results = {}
    for model_name in models:
        weights = (args.pretrained if args.pretrained != "yolov8n.pt"
                   else f"{model_name}.pt")
        print(f"\n===== {model_name} (init={weights}, imgsz={IMGSZ}, "
              f"epochs={EPOCHS}) =====")
        if args.dry_run:
            print(f"[dry-run] would train {model_name} (init={weights}) "
                  f"on {train_yaml}")
            continue
        last_pt = Path("roboat") / f"baseline_{model_name}" / "weights" / "last.pt"
        if args.resume and not last_pt.exists():
            # fail-fast instead of silently training 100 fresh epochs while the
            # user believes the run resumed
            raise SystemExit(f"--resume: {last_pt} not found. Re-attach the "
                             f"previous session's output and copy roboat/ into "
                             f"the working directory, or drop --resume.")
        if args.resume:
            # mirror of the notebook resume logic: YOLO(last.pt) + resume=True.
            # Guard against resuming a COMPLETED run: ultralytics strips the
            # optimizer from last.pt at final_eval, and train(resume=True) then
            # silently starts a fresh COCO8 run. Resume restores model/optimizer/
            # EMA/epoch; the augmentation stream restarts (no RNG continuation).
            # YOLO loader only, never raw torch.load.
            probe = YOLO(str(last_pt))
            ck = getattr(probe, "ckpt", {}) or {}
            if ck.get("epoch", -1) >= 0 and ck.get("optimizer") is not None:
                print(f"resuming {model_name} from {last_pt}")
                model = probe
                model.train(resume=True)
            else:
                print(f"{model_name} already finished (no optimizer state in "
                      f"last.pt) - evaluating its best.pt instead of retraining")
                model = YOLO(str(Path("roboat") / f"baseline_{model_name}" / "weights" / "best.pt"))
        else:
            model = YOLO(weights)
            model.train(data=str(train_yaml), epochs=EPOCHS, patience=PATIENCE,
                        imgsz=IMGSZ, seed=SEED, deterministic=True,
                        project="roboat", name=f"baseline_{model_name}")
        best = model.trainer.best
        row = {}
        skipped = []
        for ys in PER_SOURCE_YAMLS:
            yf = yamls / f"{ys}.yaml"
            if not yf.exists():
                # yamls ship in the bundle (incl. ood_aquatrash) - a missing one
                # is still skipped gracefully, never losing the whole report
                skipped.append(ys)
                print(f"note: skipping per-source val on {ys}: {yf} not found")
                continue
            try:
                m = YOLO(best)
                metrics = m.val(data=str(yf), imgsz=IMGSZ, split="test")
            except Exception as exc:  # one bad source must not lose the report
                skipped.append(ys)
                print(f"note: per-source val on {ys} failed ({exc}); "
                      f"continuing with the remaining sources")
                continue
            row[ys] = map50_row(metrics)
        if skipped:
            print(f"note: {len(skipped)} per-source val entries skipped: "
                  f"{', '.join(skipped)}")
        results[model_name] = row
    if results and not args.dry_run:
        # only print sources that actually produced metrics (skipped ones
        # would KeyError the table)
        sources = [y for y in PER_SOURCE_YAMLS if all(y in row for row in results.values())]
        print("\n===== per-source test mAP@0.5 (aggregate) =====")
        print(f"{'model':<12}" + "".join(f"{y:<20}" for y in sources))
        for mname, row in results.items():
            print(f"{mname:<12}" + "".join(f"{row[y]['all']:<20.4f}" for y in sources))
        print("\n===== per-class mAP@0.5 per source (P1 gate: per class AND per source) =====")
        for mname, row in results.items():
            for ys, vals in row.items():
                pc = "  ".join(f"{cls}={ap:.4f}" for cls, ap in vals["per_class"].items())
                print(f"{mname:<12}{ys:<20}{pc}")


if __name__ == "__main__":
    main()
