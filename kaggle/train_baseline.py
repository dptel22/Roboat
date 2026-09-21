"""T6 - Kaggle baseline training: yolov8n and yolo11n, DEFAULT hyperparameters.

No hyperparameter tuning (locked decision): epochs=100, patience=20, imgsz=960
(from D11 box-size audit: FML median box 14px/<8px share 18% at 640 vs 21px/2.6%
at 960). After each model trains, runs val on every per-source yaml and prints a
per-source mAP table.

Run on Kaggle with the dataset uploaded per kaggle/README.md:
  python train_baseline.py --data-root /kaggle/input/roboat-processed
"""
import argparse
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
MODELS = ["yolov8n", "yolo11n"]
IMGSZ = 960  # D11 recommendation - do not tune in the baseline
EPOCHS, PATIENCE, SEED = 100, 20, 42
PER_SOURCE_YAMLS = ["fml_c2", "tud_gv_c2", "hagenbeek_tiles_c2", "combined_c2",
                    "ood_aquatrash"]


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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", type=Path, default=Path("data/processed"))
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    from ultralytics import YOLO

    yamls = rebase(args.data_root, Path("rebased"))
    train_yaml = yamls / "combined_c2.yaml"
    results = {}
    for model_name in MODELS:
        print(f"\n===== {model_name} (defaults, imgsz={IMGSZ}, epochs={EPOCHS}) =====")
        if args.dry_run:
            print(f"[dry-run] would train {model_name} on {train_yaml}")
            continue
        model = YOLO(f"{model_name}.pt")
        model.train(data=str(train_yaml), epochs=EPOCHS, patience=PATIENCE,
                    imgsz=IMGSZ, seed=SEED, deterministic=True,
                    project="roboat", name=f"baseline_{model_name}")
        best = model.trainer.best
        row = {}
        for ys in PER_SOURCE_YAMLS:
            m = YOLO(best)
            metrics = m.val(data=str(yamls / f"{ys}.yaml"), imgsz=IMGSZ, split="test")
            row[ys] = float(metrics.box.map50)
        results[model_name] = row
    if results and not args.dry_run:
        print("\n===== per-source test mAP@0.5 =====")
        print(f"{'model':<12}" + "".join(f"{y:<20}" for y in PER_SOURCE_YAMLS))
        for mname, row in results.items():
            print(f"{mname:<12}" + "".join(f"{row[y]:<20.4f}" for y in PER_SOURCE_YAMLS))


if __name__ == "__main__":
    main()
