# Articulated SfM by Factorization on MASt3R-SfM — Design

**Date:** 2026-10-04
**Status:** Draft, awaiting review
**Reference:** P. Tresadern and I. Reid, "Articulated Structure From Motion by Factorization", CVPR 2005 (`tresadern_reid_cvpr2005.pdf` in repo root).

## 1. Goal

Given multiple pictures of an object in two articulation states, detect the articulated part, classify its joint type, and recover the joint parameters (axis, joint state). Use Tresadern & Reid's ideas: rank deficiency of the measurement matrix, RANSAC subspace segmentation, and joint recovery from the subspace intersection. Apply them to the metric 3D geometry from MASt3R-SfM rather than to affine 2D tracks.

### Agreed scope (v1)

- **Input:** images of one object in two articulation states. The state of each image is **known** (MPArt-90 `start/` and `end/`).
- **One moving part** attached to a static base.
- **Joint types output:** `rigid`, `prismatic`, `revolute`, `free`.
- **Data:** MPArt-90 single-joint objects (`/home/robotlab/scratch/GaussianArt/data/MPArt-90/extracted/`):
  - Revolute: StorageFurniture_45166 (primary), 35059, 41086, 45176, 45247, 45937, 47021.
  - Prismatic: StorageFurniture_46107 (primary), 45855.
- **Success:** correct part segmentation, correct joint type, small axis-direction and axis-position error against `gt/trans.json`. v1 reports numbers and sets no pass/fail thresholds on real data.

### Out of scope for v1 (the design leaves room for each)

- Multiple moving parts. Segmentation already peels subspaces one at a time, so N parts becomes a loop.
- Unknown state labels.
- Ball/universal joints. These are unidentifiable from two states; see §4.2.
- Online/SLAM operation.
- A faithful 2D-affine reproduction of the paper (possible later baseline).

## 2. Approach

This keeps the paper's algebra but builds the measurement matrix from **camera-frame metric 3D points**, lifted using MASt3R-SfM's scale-consistent per-view depth:
- Each view contributes 3 rows instead of 2.
- The affine-camera assumption and the self-calibration step (paper §3) are no longer needed.
- Rank-based detection, RANSAC segmentation and nullspace joint recovery carry over unchanged.

Rejected alternatives:
- **Faithful 2D affine.** Affine projection is a poor fit for close-range furniture. It also needs complete 2D tracks across 360° captures and discards MASt3R calibration.
- **Pose/residual + Procrustes.** Simpler, but it drops the factorization ideas the project is about.

## 3. Architecture

A new package `artsfm/` sits beside `mast3r/`. It calls MASt3R without modifying it. The driver is `run_articulation.py`, whose CLI mirrors `run_mast3r_sfm.py`.

```
artsfm/
  views.py        # view selection
  reconstruct.py  # MASt3R-SfM sparse GA on selected views
  tracks.py       # multi-view tracks + measurement matrix W
  factorize.py    # segmentation, classification, joint recovery
  evaluate.py     # GT alignment, metrics, visual outputs
run_articulation.py
tests/
  synth.py        # synthetic articulated scene generator
  test_*.py
```

### 3.1 `views.py`: view selection
- Select K start-state views (default K=5) spread around the object. For each one, select the end-state view with the nearest camera pose. Output: 2K image paths with state labels.
- v1 uses the dataset's camera poses (`transforms_train_{start,end}.json`) **only to choose views**. This sits behind a single function `select_views(...)` so it can later be replaced by a MASt3R-retrieval selector.
- View 0 is always a start-state view.

### 3.2 `reconstruct.py`: reconstruction
- Run `sparse_global_alignment` on the 2K images with a **complete** pair graph. That guarantees cross-state matches and stays cheap at 2K ≈ 10.
- The default optimisation level is `refine`, which keeps per-view depth from being warped to fit a static scene and bending the moving part. `refine+depth` is an ablation option.
- Outputs per view: intrinsics K_f, depthmap D_f, cam-to-world pose, and the cached pairwise correspondences (`corres` files written by `forward_mast3r`).
- The static GA treats the moving part as an outlier. This is acceptable because only per-view depth is consumed downstream, not the fused cloud.

### 3.3 `tracks.py`: tracks and measurement matrix
- Chain pairwise correspondences into multi-view tracks with union-find on (view, pixel) nodes. Drop tracks with conflicting pixels in one view.
- Keep only tracks observed in all 2K views.
- Lift each observation to camera-frame 3D: `X = D_f(p) · K_f^{-1} [p; 1]`.
- Build `W ∈ R^{3F × N}`, F = 2K, with one column per track and 3 rows per view. Also return the per-view state labels and the track pixels (for visualisation and segmentation scoring).

### 3.4 `factorize.py`: articulation analysis
See §4.

### 3.5 `evaluate.py`: evaluation and outputs
- Align the reconstruction to the GT frame with a Sim(3) (Umeyama) fit between estimated and GT camera centres of the selected views. Transform the estimated axis into the GT frame.
- Metrics:
  - joint-type correctness;
  - axis-direction error (degrees, sign-agnostic);
  - axis-position error: the distance between the estimated and GT axis lines, as the closest-point distance for revolute joints only;
  - joint-state error: rotation angle against GT `|l − r|` in degrees, or translation against GT `|l − r|`;
  - segmentation accuracy of base/part track labels against GT part labels. GT labels come from projecting GT part meshes (`gt/start/*.obj`) into view 0 and labelling each track by the mesh it hits.
- Outputs to `out_dir`:
  - `result.json`: status, type, axis, state, metrics, thresholds used;
  - `articulation.ply`: tracks coloured by segment, plus a line for the axis;
  - `W.npz`: the measurement matrix and labels, for debugging.

## 4. Factorization math

### 4.1 Model
- View f has camera rotation R_f and translation t_f.
- A base point s (in the frame of view 0) has camera-f coordinates `R_f s + t_f`. Base columns therefore satisfy `W_base = M [S; 1]`, with `M_f = [R_f | t_f]` (3×4).
- Part points undergo an extra rigid motion `T_σ = [[A_σ, b_σ], [0, 1]]` that depends on the state σ of view f, so `M'_f = M_f T_σ`.
- Because view 0 is a start-state view and both bodies are expressed in its frame, `T_start = I`. Only `T = T_end = (A, b)` is unknown.
- Because depth is metric, `S` is read directly from view 0's rows of W. M is then recovered linearly as `M = W_base [S; 1]^+`, and likewise M′ from the part columns. No self-calibration is needed.

### 4.2 Rank ladder for two states
The intersection of col(M) and col(M′) is the set of `[s; w]` with `(I − A) s = w b`. This is paper Eq. 2 specialised to two states.

| Relative motion between states | Intersection dim d | rank [M ∣ M′] = 8 − d | Type |
|---|---|---|---|
| none | 4 | 4 | `rigid` |
| pure translation (A = I) | 3 | 5 | `prismatic` |
| pure rotation about a line (b ⟂ axis) | 2 | 6 | `revolute` |
| screw (rotation + translation along axis) | 1 | 7 | `free` |

With only two states, a ball joint is indistinguishable from a hinge: any single relative rotation is a rotation about some axis. Identifying a ball joint needs at least 3 states. MPArt-90 contains only rotate/translate joints.

Non-degeneracy requires rank(M_start) = rank(M_end) = 4, which means at least 2 views per state with distinct rotations.

### 4.3 Algorithm
1. **Segmentation** (paper §2.1)
   - RANSAC: sample 4 columns of W, take their 4D column span, and score columns by residual to that span. The inlier threshold is τ = 0.01 × median depth, scaled per view (RMS over 3F rows).
   - The best consensus set is the base. Remove it and repeat for the part.
   - Remaining columns are outliers.
   - Each subspace is refit by SVD on its inliers (rank 4).
2. **Classification**
   - Take orthonormal bases U_b and U_p of the two rank-4 column spaces.
   - Compute the principal angles between them (SVD of U_bᵀ U_p).
   - d = the number of angles below θ_τ, and the type follows from §4.2.
3. **Joint recovery**
   - *Nullspace (paper method):* the d-dimensional approximate nullspace of `[M | −M']` from the smallest singular vectors gives `[s; w]` pairs with `s = s'`.
     - Revolute: two null vectors span the axis line, giving the point (w ≠ 0, normalised) and the direction (w = 0 combination).
     - Prismatic: three null vectors with w = 0. The direction is taken from b in the cross-check.
   - *Cross-check:* fit T least squares across end views (`M'_f = M_f T`), project A to SO(3), and screw-decompose. This gives the axis direction, the angle or displacement, and a point on the axis.
   - If the nullspace and screw axes differ by more than 10°, set `inconsistent`.
4. **Joint state:** the rotation angle of A (revolute) or ‖b‖ (prismatic).

### 4.4 Thresholds
τ, θ_τ and the RANSAC iteration count are calibrated on synthetic data (§6.3). Calibrated values are recorded in this spec and as code defaults. They are **not** tuned on the real evaluation objects.

## 5. Error handling

Every outcome is written to `result.json` as `status` plus `reason`. Expected failures do not raise exceptions.

| Condition | Action / status |
|---|---|
| < 30 complete tracks | retry with K = 3; then `insufficient_tracks` |
| part inliers < 8 or < 5% of tracks | `rigid` (no articulation detected / part not visible) |
| rotation < 3° and translation < 1% of scene scale | `ambiguous_small_motion`, no axis reported |
| nullspace vs screw axis > 10° | output both, `inconsistent: true` |
| rank(M_start) or rank(M_end) < 4, or GA failure | stop with error naming the offending views |

## 6. Testing

Development is test-first, bottom-up.

### 6.1 Synthetic generator (`tests/synth.py`)
- Random base and part point clouds, and 2K cameras on a ring around the object, split into start and end states.
- One relative motion per scene: none, prismatic, revolute or screw.
- Configurable Gaussian noise on the camera-frame 3D points, plus a fraction of outlier tracks.
- Returns W, state labels, GT segmentation, and GT joint.

### 6.2 Unit tests
- `factorize`:
  - correct type for all four motions at zero noise;
  - axis error < 1e-6 at zero noise, degrading smoothly with noise;
  - segmentation recall > 95% with 20% outliers.
- `tracks`: union-find chaining and conflict dropping on hand-built correspondence graphs; correct lifting with known K and depth.
- `evaluate`: Sim(3) recovers a known transform; axis-angle and line-distance metrics are correct on hand-computed cases.

### 6.3 Threshold calibration
Sweep noise levels on synthetic data. Pick τ and θ_τ to maximise type accuracy, and record them in §4.4.

### 6.4 Real-data runs
- End-to-end on StorageFurniture_45166 (revolute) and 46107 (prismatic).
- Then all single-joint objects listed in §1, reported as a table: type accuracy, axis-angle error, axis-position error, state error, segmentation accuracy.
- Reports only; no pass/fail thresholds in v1.
