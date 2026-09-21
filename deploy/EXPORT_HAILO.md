# Exporting RoBoat YOLO → Hailo-8L (Raspberry Pi 5)

Target: Hailo-8L on Pi 5, running a 2-class (litter / hyacinth) nano detector.
Two routes are documented. **Route A (ONNX + hailomz) is primary** per project
decision; Route B is Ultralytics' integrated export.

Verified facts (full citations in `reports/DECISIONS_LOG.md`):
- DFC **3.x** is the correct line for Hailo-8/8L (Model Zoo v2.x + DFC 3.x;
  Ultralytics docs validate Hailo-8L with HailoRT 4.23 + DFC 3.33).
- Hailo Model Zoo HAILO8L object-detection list includes **yolov8n and
  yolov11n** — both our baseline models are compilable.
- HEF compilation is **Linux x86_64 only** (WSL2 Ubuntu or Hailo Docker on your
  machine, or a Kaggle/Linux box). The Pi 5 only runs the compiled HEF via
  HailoRT.
- DFC wheels come from the **Hailo Developer Zone** (free registration).

## Route A (primary): ONNX → hailomz

1. Export ONNX (works anywhere, including Windows/Kaggle):
   ```bash
   python deploy/export_onnx.py --weights best.pt --imgsz 960
   ```
   (static shapes, opset 12, simplified — what the DFC parser expects).

2. Build the calibration set (from the training machine, before zipping):
   ```bash
   python deploy/make_calib_set.py            # 1,024 train images, source-balanced
   ```
   → `data/processed/calib/images/`. Include it in the Kaggle dataset zip.

3. In WSL2/Docker with DFC 3.x installed:
   ```bash
   pip install hailo-dataflow-compiler==3.* hailo-model-zoo   # from Developer Zone
   hailomz compile --hw-arch hailo8l --ckpt best.onnx \
       --calib-path data/processed/calib/images \
       --classes 2 --performance
   ```

4. **compression_level=0 must be set explicitly.** Hailo's default calibration
   enables 4-bit weight quantization when the calibration set exceeds 1,024
   images; with exactly 1,024 you are at the boundary — set it in the model
   script / compile call to force 8-bit weights. If mAP drops beyond tolerance,
   retry with 16-bit outputs (`--output-format-type HAILO_OUTPUT_TYPE_UINT16`
   / `HAILO_QUANTIZED_FLOAT16`) at the cost of more FPS bandwidth. [Confirmed:
   Hailo docs/forum state the default >1024-image behavior; per-task instruction
   the explicit flag is mandatory — exact CLI flag varies by DFC release, check
   `hailomz compile --help` on your installed version.]

5. Result: `best.hef` + `hailo_model_zoo` metadata → copy to the Pi 5.

## Route B (alternative): Ultralytics native export

Current Ultralytics supports `model.export(format="hailo", name="hailo8l",
imgsz=960, data=dataset.yaml)` — validated on Hailo-8L (DFC 3.33/HailoRT 4.23).
It still requires DFC on a Linux x86_64 host (internally: .pt → ONNX → parse →
INT8 calibration → HEF, then deletes the intermediate ONNX). Use Route A when
you want explicit control over calibration images and quantization flags; use
Route B for convenience.

## Validation plan (do this after every export step)

Compare mAP@0.5 for `.pt` → `.onnx` → `.hef` on the SAME per-source test lists
(`data/processed/yamls/*_c2.yaml` test splits, run on a GPU box with the ONNX
model; HEF via HailoRT on the Pi):

```bash
yolo val model=best.pt  data=<yaml> imgsz=960
yolo val model=best.onnx data=<yaml> imgsz=960
# on the Pi:
hailortcli run best.hef   # + custom postprocess/NMS script, compare per-class AP
```

Record where the drop appears. **Small objects are the usual casualty of INT8
quantization** — if FML (USV view) mAP drops much more than TUD-GV, that is the
small-box signature: revisit compression_level=0 / 16-bit outputs before
accepting the HEF. Note the HEF runs fixed 960 input; val must use imgsz=960.
