# Weather & deployment-condition augmentation — research note (2026-10-02)

What a USV-mounted camera on Bengaluru lakes faces, what our training data already
contains, and which augmentations to trial — **post-gate only** (the locked Run A/B
stay identical for comparability). Verified sources: installed ultralytics 8.4.165
source (`data/augment.py`, `cfg/default.yaml`), Albumentations 2.x docs, WSOD
literature listed at the end.

## 1. Conditions a USV camera faces vs. our training data

| Condition | Cause / appearance | In our data? |
|---|---|---|
| Sun glitter / glare on water | low sun + ripples → bright streaks/specular blobs | PARTIAL — FML river frames have daytime glare; aerial tiles have little |
| Backlit / high dynamic range | shooting toward the sun | PARTIAL |
| Rain streaks + lens droplets | monsoon; droplets defocus regions | MISSING (FML is fair-weather) |
| Fog / haze / mist | winter mornings, humidity | MISSING |
| Strong shadows (banks, boat, self-shadow) | low-angle sun | PARTIAL |
| Turbidity / algal color shift | monsoon churn, blooms | PARTIAL — Saigon/Hagenbeek water color differs from FML already |
| Motion blur / shake | boat motion, ~1 fps frames | PARTIAL (some FML blur) |
| Horizon roll/pitch | boat roll | PARTIAL (USV frames), MISSING in aerial tiles |
| Wake foam / spray | own boat + wind | MISSING |
| Night / dusk | night ops | MISSING — out of scope unless night ops are planned (see Bengaluru capture) |
| Seasonal: bloom density, winter haze | Bengaluru seasonality | MISSING — T9 capture covers it naturally |

Principle: **augment what is missing but photo-physically simple (weather/lens);
capture what is structural (seasonality, night) via T9.**

## 2. Native ultralytics args (verified: cfg/default.yaml @ 8.4.165)

`hsv_h=0.015, hsv_s=0.7, hsv_v=0.4, degrees=0, translate=0.1, scale=0.5,
shear=0, perspective=0, flipud=0.0, fliplr=0.5, mosaic=1.0, mixup=0.0,
copy_paste=0, close_mosaic=10`. Already active: `hsv_v/hsv_s` cover part of the
illumination axis; `scale=0.5` is the downscale risk flagged in RUN_PLAN.

## 3. Albumentations 2.x transforms mapped to conditions

| Condition | Transform (2.x name) | Key params | Priority |
|---|---|---|---|
| Glare / specular blobs | `A.RandomSunFlare` (flare_roi, num_flare_circles_limit) | low p (0.05–0.1), keep flare off image center | P1 |
| Rain streaks | `A.RandomRain` | slant, drop width/length, low p (0.05) | P1 |
| Fog / haze | `A.RandomFog` | fog_coef_lower/upper ≤ 0.4 (deeper kills small objects) | P1 |
| Lens droplets / spray | `A.Spatter` (mode, 2.x) | low intensity; also simulates mud | P2 |
| Shadows | `A.RandomShadow` | shadow_roi limited to lower half (bank shadows) | P2 |
| Motion blur | `A.MotionBlur` (blur_limit ≤ 7) | low p; hurts small boxes if strong | P2 |
| Turbidity / color cast | `A.RandomToneCurve`, `A.HueSaturationValue`, `A.CLAHE` | mild; hue shifts must stay within water-color range | P2 |
| Exposure hunting | `A.RandomBrightnessContrast`, `A.RandomGamma` | already in the built-in default set (p=0 today — see wiring) | P3 |
| Winter | `A.RandomSnow` | LOW priority — Bengaluru winters have no snow; haze is the real signal | skip |

Cautions from the literature + first principles: strong blur/fog destroys small-box
evidence (our p10 box is 15.7 px in-tile); aggressive hue shifts can wash out the
water-color cues separating hyacinth from shadow; synthetic rain texture is not real
rain-on-lens — treat the package as domain-adaptation seasoning, not a substitute for
T9 capture. All photometric-only transforms are bbox-preserving (no label surgery).

## 4. Wiring recipe (no fork needed — verified in installed source)

`ultralytics.data.augment.Albumentations.__init__` accepts a `transforms` list
(objects or serialized dicts) and replaces its defaults entirely (augment.py:2147-2167).
So the trial needs a ~20-line custom dataset wrapper (or monkeypatch) that instantiates
the trainer's dataset, then sets
`dataset.albumentations.transform = A.Compose([...], bbox_params=A.BboxParams("yolo"))`
— spatial transforms stay handled by ultralytics' own pipeline; keep the custom list
photometric-only to avoid bbox-topology pitfalls.

## 5. Recommended ablation order (each its own gate, never folded into Run A/B)

- **A2 "weather-light"**: RandomSunFlare p=0.05 + RandomRain p=0.05 + RandomFog
  (coef ≤ 0.4, p=0.05) + RandomShadow p=0.05 — everything else untouched.
- **A3 "scale fix"**: scale 0.5 → 0.2 (RUN_PLAN already lists it; strongest
  small-object prior).
- **A4 "aerial flips"**: flipud 0 → 0.5 (aerial tiles have no preferred up).
- **A5 "turbidity"**: RandomToneCurve + mild HSV widening (hsv_s 0.7 → 0.8).
Gate every ablation exactly like P1: test-split mAP50 per class AND per source,
with a glare/rain-stratified eyeball set from T9 footage when it exists.

## 6. Literature (and its honest limits)

- Multi-Scale Small-Target Detection for USVs (UCL Discovery) — names horizontal
  ripples, vertical glare, and object reflections as the core feature-confusion
  sources: https://discovery.ucl.ac.uk
- LMS-YOLO (Acadlore, 2025) — small-target detection on dynamic water surfaces.
- YOLOv8 water-surface detection robust to surface reflection (KCI) — reflection
  robustness via physical priors + multi-scale fusion (architecture route).
- Chen et al. 2024 (IOP Conf. Ser.) — small-target water-surface detection in
  complex weather (rain/fog): https://iopscience.iop.org/article/10.1088/1742-6596/2897/1/012043
- WA-YOLO (JMSE 2025): https://www.mdpi.com/2077-1312/14/1/37 ; Improved YOLOv8
  for WSOD (Wang 2024, PMC11314716); Frontiers in Marine Science (2026) maritime
  low-light detection.
- Rip-current deep-learning work (Academia.edu) reports random shadow/rain/fog
  augmentations useful for water-surface glare mitigation — the closest published
  augmentation evidence found.
- **No published all-weather augmentation ablation for floating-debris detection
  was found** — the field solves weather with architecture. Our A2 is therefore a
  measured experiment, not a downloaded recipe.
