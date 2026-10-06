"""Deploy: export trained .pt to ONNX for the Hailo toolchain (primary route).

Run on the training machine (Kaggle) or locally after training - NOT on the Pi.
Produces <weights_stem>.onnx (static shapes, opset 11, simplified) suitable for
hailomz / Hailo Dataflow Compiler ingestion. Opset 11 follows Hailo's own yolov8
retrain guide (hailo_model_zoo v2.19.1, training/yolov8/README.rst line 94:
"yolo export ... format=onnx opset=11").

Usage:
  python deploy/export_onnx.py --weights runs/detect/train/weights/best.pt [--imgsz 960]
"""
import argparse
from pathlib import Path

from ultralytics import YOLO


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True, type=Path)
    ap.add_argument("--imgsz", type=int, default=960)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    if args.dry_run:
        print(f"[dry-run] would export {args.weights} to ONNX "
              f"(imgsz={args.imgsz}, opset=11, dynamic=False, simplify=True)")
        return

    model = YOLO(str(args.weights))
    out = model.export(format="onnx", imgsz=args.imgsz, opset=11,
                       dynamic=False, simplify=True, half=False)
    print(f"ONNX written: {out}")
    print("Next (Linux x86_64 with Hailo DFC 3.x, see deploy/EXPORT_HAILO.md):")
    print(f'  hailomz compile --hw-arch hailo8 --calib-path <calib_dir> '
          f'--classes 2 --performance "..." {out}')


if __name__ == "__main__":
    main()
