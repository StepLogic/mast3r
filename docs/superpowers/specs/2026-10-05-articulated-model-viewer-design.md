# Interactive Articulated Model (Viewer + URDF) — Design

**Date:** 2026-10-05
**Status:** Approved in conversation; awaiting written-spec review
**Builds on:** `docs/superpowers/specs/2026-10-04-articulated-sfm-factorization-design.md` (the `artsfm` pipeline that finds the joint).

## 1. Goal

After the factorization pipeline has found a joint, turn the reconstruction into an **articulated model** that:
- a person can manipulate in a local **gradio app**, moving the part with a joint slider and a Play sweep;
- a simulator can load as a **URDF** (MuJoCo now; Isaac via GaussianArt's conventions).

### Agreed decisions
- **Both viewer and URDF** sit on one shared, saved model folder. Viewer first.
- **Geometry v1 = dense coloured point clouds per part.** Collision and visual shapes in the URDF are convex hulls. Textured meshes are step 2, with no change to the model format.
- **Viewer = local gradio app**, following the pattern of MASt3R's `demo.py`: `gradio.Model3D` showing a `.glb`.
- **Scope = viewer over saved models now.** A "Run pipeline" tab can be added later because the app depends only on the model folder.
- **Dense labelling = motion-hypothesis test**, with a nearest-track vote as the fallback.

### Out of scope (v1)
- Textured meshes.
- The Run tab.
- Multiple moving parts.
- Direct dragging of parts in the 3D view (gradio cannot do this).
- Isaac USD export.

## 2. Architecture

```
run_articulation.py --build_model ──► artsfm/articulate.py ──► <out>/model/
build_model.py <out_dir> ──────────┘                            │
                                          artsfm/urdf.py ◄───────┤──► <out>/model/<object>.urdf, meshes/
                                          app_articulation.py ◄──┘  (gradio viewer)
```

- `artsfm/articulate.py`: dense labelling, fusion, collision hulls, and `model.json`. It is the only new stage that touches MASt3R output, via `extract_geometry`.
- `artsfm/urdf.py`: model folder → URDF and meshes, plus a MuJoCo check.
- `app_articulation.py`: gradio viewer, which reads only model folders.
- `build_model.py <out_dir>`: rebuilds a model from an existing run.
  - It re-runs sparse GA on the cached matches to get the depth maps, which take about 1 minute and are not saved by a run.
  - It re-runs `analyze()` and checks the result against `result.json`.
- `run_articulation.py --build_model`: builds the model right after a successful analysis.

## 3. Model folder (the contract)

```
<out>/model/
  model.json
  base.ply           # body points, RGB, model frame
  part.ply           # part points at joint value 0 (= start state), model frame
  base_hull.obj      # convex hulls (collision/visual proxies)
  part_hull.obj
```

`model.json` fields:
- `object`, `frame` = `"view0_camera_opencv"`, `units` = `"reconstruction"`.
- `joint`:
  - `type`: `revolute | prismatic | free`;
  - `axis_origin`: 3 values, or null for prismatic;
  - `axis_dir`: unit 3-vector;
  - `observed`, `lower` = 0, `upper` = `observed`;
  - `unit`: `rad` or `recon`;
  - `approximate`: true for `free`.
- `motion`: the 4×4 start→end transform T used for labelling, which equals the joint at `observed`.
- `up`: estimated gravity-up unit vector in the model frame (§5).
- `labelling`: `method`, `fallback`, `delta`, `margin`, `n_base`, `n_part`, `n_fallback`, `n_unlabelled`, `mask_source` (`alpha` or `confidence`).
- `warnings`: a list of strings.
- `source`: relative path to `result.json`, and the view list.

Model frame = the view-0 camera frame (OpenCV), the frame in which `analyze()` reports the axis. Joint value 0 is the start state. For a revolute joint, T(θ) is the rotation by θ about (`axis_origin`, `axis_dir`). For a prismatic joint, T(s) is the translation by s·`axis_dir`. The axis sign is chosen so that `observed` > 0. For a `free` joint (screw), T(θ) rotates by θ about the axis and translates by (θ / observed) · `disp_along` along it. `model.json` then also stores `disp_along`. The app animates this full screw motion.

## 4. Dense labelling (`articulate.py`)

**Inputs:** `geom` (`K`, `cam2w`, dense `depths`) from `extract_geometry`, the selected view paths, `states`, and the `analyze()` result (labels, tracks W, axis, angle/disp). T is rebuilt from the recovered joint, not taken from a separate fit.

1. **Foreground mask per view.**
   - If the source image has an alpha channel, use alpha > 0, resized and centre-cropped exactly as dust3r's `load_images` does.
   - Otherwise use MASt3R confidence > 1.5 (`min_conf_thr` default), and record `mask_source`.
2. **Back-project** every foreground pixel with depth > 0 into the model frame through the GA poses: `X0 = inv(cam2w[0]) · cam2w[f] · (D_f(p) K_f⁻¹ p̃)`. Keep the pixel colour.
3. **Hypotheses.**
   - A start-view point X predicts its end-state position as X (body) or T·X (part).
   - An end-view point predicts its start-state position as X (body) or T⁻¹·X (part).
4. **Evidence.** For each view of the opposite state, project the prediction to pixel u and depth z, and read the observed depth D(u):

| Case | Condition | Evidence |
|---|---|---|
| support | u inside the image, foreground, and \|z − D\| < δ·z | support |
| contradiction (free-space violation) | z < D − δ·z | contradiction |
| occlusion | z > D + δ·z | none |
| no observation | outside the image or background | none |

   Score per hypothesis = supports − contradictions.
5. **Decision.** The winning hypothesis labels the point if its score exceeds the other by at least `margin` and it has at least one support. Otherwise the **fallback** applies:
   - majority vote of the k = 7 nearest labelled tracks (`analyze` labels 0/1; view-0 track points);
   - if the nearest such track is farther than 0.1 × the object RMS radius, the point is **unlabelled** and dropped.
6. **Fusion.**
   - base = body points.
   - part = start-view part points ∪ T⁻¹·(end-view part points), so all part points sit at joint 0.
   - Statistical outlier removal (k = 16, 2.5σ), then voxel downsampling to at most 150,000 points per link (voxel size found by bisection).
7. **Hulls:** `trimesh` convex hull of each cleaned cloud, skipped with a warning if it has fewer than 4 non-coplanar points.
8. **Warnings:**
   - part < 500 points → "part sparsely observed";
   - unlabelled > 40% → "many points unlabelled";
   - `free` joint → "screw joint (not representable in URDF); URDF export disabled".
9. **Refusal:** if `result.status != "ok"`, no model is written and the reason is printed.

Defaults are δ = 0.02 and `margin` = 1. Both are calibrated on synthetic data (§8.3), and the calibrated values are recorded here.

## 5. URDF export (`artsfm/urdf.py`)

- Follows GaussianArt `export_isaac.py` conventions:
  - links `base` and `link_1`, joint `joint_1`;
  - files `meshes/<link>.obj` (visual) and `meshes/<link>_collision.obj`, both the convex hull in v1;
  - the child link frame sits at the pivot, with meshes offset by −pivot;
  - masses 10 / 1, damping 0.5, effort 100, velocity 2.
- Prismatic pivot = the part hull centroid.
- **Gravity frame.**
  - Rotate the model so `up` maps to +z, and translate so the base's minimum z is 0 and its x-y centroid is the origin.
  - `up` = the normalised mean of the cameras' −y axes, expressed in the model frame (OpenCV y points down).
  - With `--use_gt_frame` (MPArt-90), `up` and `scale` instead come from the GT Sim(3) alignment, with GT world +z as up.
- **Scale:** `--scale` (default 1.0). Applied to positions, prismatic limits and meshes.
- **Joint limits:** [lower, upper] from `model.json`, with optional `--lower/--upper` overrides.
- **Refusal:** joint type `free` raises a clear error, since URDF has no screw joint.
- **Check:** `python3 -m artsfm.urdf --check <urdf>` loads the file with `mujoco.MjModel.from_xml_path` (MuJoCo 3.3), sets `qpos` to the upper limit, runs `mj_kinematics`, and compares the part body pose with T(upper) (mapped through the export transform). It fails if the difference exceeds 1e-6 (relative).

## 6. Viewer app (`app_articulation.py`)

- gradio 6.20 Blocks, structured like `mast3r/demo.py`.
- **Model picker:** a dropdown of `outputs/artsfm/*/model/model.json`, with a configurable root.
- **3D view:** `gradio.Model3D` showing a `.glb` built by `artsfm.viewer.build_glb(model, value, color_mode, show_axis, show_hulls)`:
  - base points, plus part points transformed by T(value);
  - the axis as a thin cylinder (revolute) or an arrow (prismatic);
  - optional semi-transparent hulls.
- **Controls:**
  - joint slider (degrees for revolute, reconstruction units for prismatic) over [lower, upper];
  - "allow beyond observed range" checkbox, widening to [lower − 0.5·range, upper + 0.5·range];
  - colour mode: RGB or by part;
  - hulls toggle;
  - **Play** button: a generator yielding about 12 frames over 0 → upper → 0.
- **Info panel:** joint type, axis, observed value, labelling statistics, warnings as chips, and GT metrics from `result.json` when present.
- **Export:** writes the URDF with the current slider limits (via `urdf.py`) and returns a zip through `gradio.File`.
- **Build module:** `artsfm/viewer.py` holds the pure glb-building and model-loading functions, so they can be tested without gradio.
- **First task is a probe** of point legibility in `Model3D`. If points are too faint, the glb builder switches to small instanced quads per point (decimated). That is internal to `build_glb` and does not change the model format.

## 7. Error handling

| Situation | Behaviour |
|---|---|
| `result.status` not `ok` | No model; message "no articulation to build: <status> — <reason>". |
| Joint `free` | Model built with `approximate: true` and a warning. The app animates the screw; URDF export refuses. |
| Part < 500 points / < 4 non-coplanar | Warning, and no part hull (URDF export refuses without hulls). |
| Unlabelled > 40% | Warning. |
| No alpha | Confidence mask; `mask_source = "confidence"`. |
| MuJoCo load failure | The exporter prints MuJoCo's error and keeps the files. |
| App: no models found under root | Empty-state text naming the root and how to build a model. |

## 8. Testing

1. **Synthetic dense scenes** (`tests/synth_dense.py`):
   - A box body with a hinged door (or sliding drawer), as trimesh meshes.
   - Pinhole depth maps at 64×48 for 3 start and 3 end views, from trimesh ray casting, with an alpha mask and per-pixel GT labels.
   - Optional relative depth noise.
2. **`articulate` tests:**
   - pixel labelling accuracy > 95% (noise-free) and > 90% at 2% depth noise;
   - near-axis points go to the fallback;
   - occluded predictions give no contradictions;
   - fused end-view part points mapped by T⁻¹ coincide with start-view part points (median distance < 2× noise);
   - `model.json` round-trips;
   - refusal on a non-`ok` result.
3. **Calibration:** sweep δ ∈ {0.01, 0.02, 0.03, 0.05} × `margin` ∈ {1, 2} over noise {0, 0.01, 0.02}. Pick the best mean accuracy and record it in §4.
4. **`urdf` tests:**
   - the MuJoCo forward-kinematics check passes for revolute and prismatic;
   - `up` maps to +z;
   - GaussianArt naming and paths;
   - a `free` joint is refused;
   - scale is applied.
5. **`viewer` tests:**
   - `build_glb` part points at value θ equal T(θ)·part (read back from the glb);
   - the model picker lists only folders with `model.json`;
   - the Blocks construct without error.
6. **Real data:**
   - Build models for 45247, 45166, 45176 and 45855.
   - Report labelling accuracy against the camera-frame GT meshes (`gt/start/start_*_rotate.ply`).
   - Run the MuJoCo check on 45247.
   - Launch the app and take one screenshot as the manual smoke check.
