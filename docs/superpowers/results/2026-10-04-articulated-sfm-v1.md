# Articulated SfM by factorization — v1 results

**Date:** 2026-10-05 · **Branch:** `articulated-sfm` · **Spec:** `docs/superpowers/specs/2026-10-04-articulated-sfm-factorization-design.md`

## Setup
- 9 MPArt-90 single-joint objects: 7 revolute, 2 prismatic.
- K = 5 views per state, clustered around **seed view 0** of the start state. No seed was tuned per object.
- Known intrinsics from `camera_angle_x`. Sparse GA at `refine`. CPU only, because CUDA is unusable with this torch build.
- Thresholds were calibrated on synthetic data only (spec §4.4): `tau_frac = 0.03`, `theta_deg = 3.0`.
- Command: `python3 run_articulation_benchmark.py` (outputs under `outputs/artsfm/`).

## Results

| object | status | gt | pred | axis err ° | axis pos err | state err | seg acc | tracks | tracks on GT part | part tracks found | cam align rmse |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 45166 | ok | revolute | free | 12.5 | 1.02 | 17.6° | 1.00 | 265 | 80 | 21 | 0.52 |
| 35059 | rigid | revolute | rigid | | | | 1.00 | 49 | 0 | 0 | 0.10 |
| 41086 | rigid | revolute | rigid | | | | 1.00 | 246 | 0 | 0 | 0.06 |
| 45176 | ok | revolute | free | 12.4 | 0.56 | 20.6° | 0.84 | 316 | 0 | 0 | 0.15 |
| 45247 | ok | revolute | **revolute** | **2.8** | 0.85 | **0.3°** | 0.86 | 251 | 0 | 0 | 0.64 |
| 45937 | rigid | revolute | rigid | | | | 0.73 | 379 | 95 | 0 | 0.11 |
| 47021 | rigid | revolute | rigid | | | | 0.83 | 33 | 8 | 0 | 0.61 |
| 46107 | rigid | prismatic | rigid | | | | 0.98 | 70 | 69 | 0 | 0.15 |
| 45855 | ok | prismatic | free | 82.2 | | 0.21 | 0.74 | 108 | 13 | 9 | 0.29 |

- Joint type was correct on 1/9 objects. A moving part was detected on 4/9.
- GT-part track counts come from nearest GT surface after Sim(3) alignment. Where `cam_align_rmse` is large (45247, 47021, 45166), those counts and `seg acc` are unreliable. For example, 45247 recovers a 36° door but reports 0 GT-part tracks.

## What the numbers say

1. **The dominant failure is visibility from the seed view, not the factorization.**
   - On 35059 and 41086, view 0 sees no part of the moving door: 0 tracks lie on it.
   - On 46107, 69 of 70 tracks lie on the drawer, so the cabinet body is effectively invisible. RANSAC's "base" is the drawer, and there is nothing to move relative to.
   - On 45937, 95 tracks lie on the part, yet none were segmented. The part's 3D residuals stayed under τ, or its cross-state matches were too noisy. This is the τ lower bound noted in spec §4.4.
2. **When the part is well observed, the axis is recovered.** 45247 gives a 2.8° axis error and a 0.3° angle error. 45166 and 45176 get the direction within about 12°.
3. **Type classification is the weakest step on real data.**
   - Real MASt3R depth noise is about 5% of the object radius (measured on 45166 base tracks).
   - At that noise, the smallest principal angles (3–6°) exceed θ_τ = 3°, so revolute joints are labelled `free`.
   - The rotation angle is also under-estimated (18.7° vs 36.3° on 45166), consistent with biased part tracks: only 21 of 80 door tracks were segmented.
4. **Track yield on the part is low.** Star tracks must be matched in all 2K views, and cross-state matches on the moved part are the scarcest. That is exactly the signal the method needs.

## Engineering findings during the run (all in the ledger and spec)
- MASt3R focal estimates on these renders were about 450 px, against 711 px true. Using known intrinsics cut the median track-to-GT-surface distance from 0.52 to 0.10.
- Union-find track chaining percolated into one 6,881-node component, giving 0 tracks. View-0-anchored star tracks fixed this.
- GT frames: `trans.json` axes need GaussianArt's `R_coord`. The `.obj` meshes are in a different frame from the cameras; the `*_rotate.ply` meshes are camera-frame.
- Review fixes:
  - prismatic translation now uses the centroid shift (Kabsch `b` carried a rotation-noise lever arm);
  - added a part-size guard after reassignment;
  - segmentation accuracy is now invariant to swapped base/part labels.

## Honest limits
- Ball joints are unidentifiable from two states.
- View selection uses dataset poses to choose views (spec §3.1).
- τ scales with depth, so small-travel parts (below about 3% of depth) read as `rigid`.
- Single seed view, single moving part, CPU-only runs: about 30 min per object uncached.

## Suggested next steps (not done; each needs its own decision)
1. **Seed-view selection without GT:** pick the start/end view pair with the most cross-state matches inconsistent with a single rigid motion. Alternatively, run several seeds and keep the result with the largest part consensus.
2. **Use partial tracks:**
   - Use the paper's missing-data view: tracks seen in ≥ 2 views per state, rather than in all 2K views.
   - Fit motions per view on the available subset.
3. **Noise-aware classification:** model selection (e.g. GRIC/BIC over rank hypotheses) instead of a fixed θ_τ, calibrated to MASt3R-like structured noise.
4. **GPU:** install a CUDA build of torch that matches the 12.2 driver, which would make the benchmark about 20× faster.
