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
"""
import argparse
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
MODELS = ["yolov8n", "yolo11n"]
IMGSZ = 960  # D11 recommendation - do not tune in the baseline
EPOCHS, PATIENCE, SEED = 100, 20, 42
PER_SOURCE_YAMLS = ["saigon_tiles_c2", "hagenbeek_tiles_c2", "fml_c2",
                    "tud_gv_c2", "combined_c2", "ood_aquatrash"]


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
        model = YOLO(weights)
        model.train(data=str(train_yaml), epochs=EPOCHS, patience=PATIENCE,
                    imgsz=IMGSZ, seed=SEED, deterministic=True,
                    project="roboat", name=f"baseline_{model_name}")
        best = model.trainer.best
        row = {}
        for ys in PER_SOURCE_YAMLS:
            m = YOLO(best)
            metrics = m.val(data=str(yamls / f"{ys}.yaml"), imgsz=IMGSZ, split="test")
            row[ys] = map50_row(metrics)
        results[model_name] = row
    if results and not args.dry_run:
        print("\n===== per-source test mAP@0.5 (aggregate) =====")
        print(f"{'model':<12}" + "".join(f"{y:<20}" for y in PER_SOURCE_YAMLS))
        for mname, row in results.items():
            print(f"{mname:<12}" + "".join(f"{row[y]['all']:<20.4f}" for y in PER_SOURCE_YAMLS))
        print("\n===== per-class mAP@0.5 per source (P1 gate: per class AND per source) =====")
        for mname, row in results.items():
            for ys, vals in row.items():
                pc = "  ".join(f"{cls}={ap:.4f}" for cls, ap in vals["per_class"].items())
                print(f"{mname:<12}{ys:<20}{pc}")


if __name__ == "__main__":
    main()
