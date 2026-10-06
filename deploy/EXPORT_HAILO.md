# Exporting RoBoat YOLO → Hailo-8 (Raspberry Pi 5)

Target: Hailo-8 on Pi 5 (26 TOPS), running a 2-class (litter / hyacinth) nano
detector. *(Hardware target corrected from Hailo-8L to Hailo-8 by the user,
2026-10-06 — the research conclusions carry over: the HAILO8 detection list
likewise has no P2 variant (grep over all 1,122 lines: 0 hits), and the extra
TOPS headroom makes the custom P2 compile route more practical, not less.)*
Two routes are documented. **Route A (ONNX + hailomz) is primary** per project
decision; Route B is Ultralytics' integrated export.

Verified facts (full citations in `reports/DECISIONS_LOG.md`):
- **Pinned stack (verified 2026-09-29): Hailo Model Zoo v2.19.1 (tag, 2026-09-18)
  ↔ DFC 3.34.0 ↔ HailoRT 4.24.0** — the current v2.x stack for Hailo-8/8L.
  There is no `model-zoo-v2.x` branch anymore; the v2.x line lives as **tags**.
  The Ultralytics-validated pairing (HailoRT 4.23 + DFC 3.33) still works but is
  **superseded** — usable as a known-good fallback.
- Master branch is Hailo-10/15 only. From the Model Zoo master `README.rst`
  (verbatim): "The Hailo-8 and Hailo-8L devices are supported on the Hailo Model
  Zoo v2.x branch, in combination with the Hailo Dataflow Compiler v3.x branch.
  The master branch is intended for Hailo-10 and Hailo-15 devices only."
- ONNX **opset 11** is what Hailo itself uses: the Model Zoo v2.19.1 yolov8
  retrain guide (`hailo_model_zoo` v2.19.1, `training/yolov8/README.rst` line 94)
  runs `yolo export ... format=onnx opset=11`.
- Hailo Model Zoo **HAILO8** object-detection list includes **yolov8n and
  yolov11n** — both our baseline models are compilable (verified 2026-10-06:
  both rows carry precompiled `hailo8` HEF links, HTTP 200).
- DFC wheels come from the **Hailo Developer Zone** (free registration).

## Route A (primary): ONNX → hailomz

1. Export ONNX (works anywhere, including Windows/Kaggle):
   ```bash
   python deploy/export_onnx.py --weights best.pt --imgsz 960
   ```
   (static shapes, opset 11, simplified — what the DFC parser expects; opset 11
   follows Hailo's own yolov8 retrain guide, `training/yolov8/README.rst` line 94
   in `hailo_model_zoo` v2.19.1).

2. Build the calibration set (from the training machine, before zipping):
   ```bash
   python deploy/make_calib_set.py            # 1,024 train images, source-balanced
   ```
   → `data/processed/calib/images/`. Include it in the Kaggle dataset zip.

3. In WSL2/Docker with DFC 3.x installed:
   ```bash
   pip install hailo-dataflow-compiler==3.* hailo-model-zoo   # from Developer Zone
   hailomz compile --hw-arch hailo8 --ckpt best.onnx \
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

## Calibration

Ultralytics' `export(format="hailo")` defaults calibration to **COCO128** for
detection — wrong data for our model. Always override with `data=` pointing at a
dataset yaml or a calibration image directory. Use our existing calibration set
at `data/processed/calib/` (1,024 images, hyacinth share 0.25). Hailo recommends
**≥ 1,024 calibration images**; we meet that exactly.

## Where to compile

DFC is **linux_x86_64-only** (Windows/macOS cannot run it natively). Beyond
WSL2/Docker on your own machine, **free Google Colab works**: the DFC ships as a
`linux_x86_64` wheel that is pip-installed in a plain venv — no Docker needed.
Community end-to-end guides:
https://community.hailo.ai/t/guide-to-using-the-dfc-to-convert-a-modified-yolov11-on-google-colab/7131
(DFC 3.29.0 era) and
https://community.hailo.ai/t/model-zoo-installation-in-google-colab/11928
(DFC 3.30.0). The wheel download needs the (free) Hailo Developer Zone login.
**Unverified:** whether our imgsz-960 yolov8n fits inside Colab's ~12.7 GB free-tier
RAM envelope — assume it may not, and have a local Linux/WSL2 fallback.

## Route B (alternative): Ultralytics native export

Current Ultralytics supports `model.export(format="hailo", name="hailo8",
imgsz=960, data=dataset.yaml)` — the flow was validated on Hailo-8L (DFC
3.33/HailoRT 4.23, now superseded by the pinned stack above); pass
`name="hailo8"` for this hardware target. It still requires DFC on a Linux
x86_64 host (internally: .pt → ONNX → parse → INT8 calibration → HEF, then
deletes the intermediate ONNX). Use Route A when you want explicit control over
calibration images and quantization flags; use Route B for convenience.

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
