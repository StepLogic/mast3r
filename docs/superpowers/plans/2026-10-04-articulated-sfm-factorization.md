# Articulated SfM by Factorization Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Detect the articulated part, the joint type and the joint axis of an object from images in two known articulation states. It applies Tresadern & Reid's rank/subspace factorization to MASt3R-SfM's metric 3D tracks.

**Architecture:** A new package `artsfm/` (views → reconstruct → tracks → factorize → evaluate) is driven by `run_articulation.py`. MASt3R code is called, never modified. The math lives in `factorize.py` and is developed test-first against a synthetic generator before touching real data.

**Tech Stack:** Python 3.10, numpy, scipy, trimesh, torch (CPU: CUDA is unusable on this machine, driver too old), MASt3R/DUSt3R from this repo, pytest 6.2.5.

**Spec:** `docs/superpowers/specs/2026-10-04-articulated-sfm-factorization-design.md`

## Global Constraints

- Do not modify anything under `mast3r/` or `dust3r/`.
- Run tests with `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -m pytest ...` (a broken anyio plugin otherwise crashes pytest).
- Device: default `cpu` when `torch.cuda.is_available()` is False.
- Weights: `checkpoints/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric.pth`.
- Data root: `/home/robotlab/scratch/GaussianArt/data/MPArt-90/extracted/`. Primary objects: StorageFurniture_45166 (revolute), StorageFurniture_46107 (prismatic).
- Joint types output: `rigid`, `prismatic`, `revolute`, `free`.
- Status values: `ok`, `rigid`, `ambiguous_small_motion`, `insufficient_tracks`. Degenerate views raise `DegenerateViewsError`.
- All joint outputs from `factorize.analyze` are in the **view-0 camera frame, metric units of the reconstruction**.
- Thresholds are calibrated on synthetic data only (Task 3), never tuned on the real evaluation objects.

## Deviations from spec (decided while prototyping; spec is updated in Task 3)

1. **RANSAC minimal sample is 3 tracks + per-view Kabsch, not 4 tracks + linear span.** Door panels are near-planar, so 4 coplanar columns span a degenerate 3D subspace. With metric 3D, 3 points fix the 3×4 motion block per view. This is still the paper's rank-4 motion model, solved metrically.
2. **Classification normalises coordinates.** It shifts by the base centroid in view 0 and divides by the base RMS radius before computing principal angles. Normalising by depth instead made prismatic and screw angles too small to separate. Prototype at noise 2% of radius:
   - revolute: angles [0.4, 0.5, 18, 85];
   - screw: [0.36, 6.7, ...];
   - prismatic: [0.33, 0.38, 0.52, 87].
3. **Scene scale = base RMS radius in view 0** (used for the small-motion test). This replaces "median depth", except for the RANSAC τ.
4. **View selection clusters views around a seed view** (needed for complete tracks) rather than spreading them around the object. `--seed_view` chooses the seed.
5. **GT segmentation labels** come from nearest GT mesh (static vs dynamic) to each aligned 3D track, rather than projecting meshes into images. This avoids camera-convention issues.

## Review Focus

1. **Part barely visible / door closed in both states:** expect `status: rigid` with a reason, not a crash or a bogus axis. Tested in Task 2 (`test_rigid_scene`, `test_tiny_part`).
2. **Too few complete tracks on real data (views too far apart):** expect a K=3 retry, then `insufficient_tracks` written to `result.json`. Tested in Task 8 (`test_driver_insufficient_tracks`).
3. **Track pixels landing on invalid depth (0/NaN, background):** expect those columns dropped, not NaNs in W. Tested in Task 4 (`test_lift_drops_invalid_depth`).
4. **Axis sign ambiguity (d vs −d) and arbitrary point on axis:** expect metrics to be sign-agnostic and to use line distance. Tested in Task 7 (`test_axis_angle_sign_agnostic`, `test_line_distance_point_choice`).
5. **Outlier tracks from mismatches:** expect them labelled −1 and joint accuracy kept. Tested in Task 2 (`test_outliers_rejected`).

---

### Task 1: Synthetic generator + rigid-motion RANSAC segmentation

**Files:**
- Create: `artsfm/__init__.py`, `artsfm/factorize.py`, `tests/__init__.py`, `tests/conftest.py`, `tests/synth.py`, `tests/test_segment.py`

**Interfaces:**
- Produces:
  - `tests.synth.make_scene(motion, n_base=300, n_part=150, k_per_state=5, noise=0.0, outlier_frac=0.0, planar_part=True, seed=0) -> (W (3F,N), states (F,) int, labels (N,) int {0 base, 1 part, -1 outlier}, gt dict{type, axis_d, axis_o, angle_deg, disp})`. `gt` is in the view-0 frame. `motion` is one of `none|prismatic|revolute|screw`.
  - `artsfm.factorize.as_views(W) -> (F,N,3)`
  - `kabsch(P, Q) -> (R, t)`
  - `fit_motion(W, idx) -> M (F,3,4)`, where S = view-0 points
  - `residuals(W, M) -> (N,)`, the per-view RMS 3D residual
  - `ransac_rigid(W, cand, tau, n_iter=500, rng=None) -> bool (N,)`

- [ ] **Step 1: Write `tests/conftest.py`, `tests/__init__.py` (empty), `artsfm/__init__.py` (empty), and the synthetic generator `tests/synth.py`**

```python
# tests/conftest.py
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
```

```python
# tests/synth.py
"""Synthetic two-state articulated scenes, as camera-frame metric 3D tracks."""
import numpy as np


def rot(axis, ang):
    axis = np.asarray(axis, float)
    axis = axis / np.linalg.norm(axis)
    K = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
    return np.eye(3) + np.sin(ang) * K + (1 - np.cos(ang)) * K @ K


def look_at(center, target=np.zeros(3), up=np.array([0, 0, 1.0])):
    """World-to-camera (R, t) for a camera at `center` looking at `target` (z forward)."""
    z = target - center
    z = z / np.linalg.norm(z)
    x = np.cross(z, up)
    x = x / np.linalg.norm(x)
    y = np.cross(z, x)
    R_c2w = np.stack([x, y, z], 1)
    return R_c2w.T, -R_c2w.T @ center


def make_scene(motion='revolute', n_base=300, n_part=150, k_per_state=5, noise=0.0,
               outlier_frac=0.0, planar_part=True, seed=0):
    rng = np.random.default_rng(seed)
    base = rng.uniform([-0.5, -0.4, -0.6], [0.5, 0.4, 0.6], (n_base, 3))
    if planar_part:  # a 2cm-thick door panel on the front face
        part = np.c_[rng.uniform(-0.4, 0.4, n_part),
                     -0.42 + rng.uniform(-0.01, 0.01, n_part),
                     rng.uniform(-0.5, 0.5, n_part)]
    else:
        part = rng.uniform([-0.4, -0.7, -0.5], [0.4, -0.42, 0.5], (n_part, 3))

    axis_d = np.array([0, 0, 1.0])
    axis_o = np.array([0.4, -0.42, 0.0])
    angle_deg, disp = 0.0, 0.0
    if motion == 'none':
        A, b = np.eye(3), np.zeros(3)
    elif motion == 'prismatic':
        axis_d = np.array([0, -1.0, 0])
        disp = 0.25
        A, b = np.eye(3), disp * axis_d
    elif motion in ('revolute', 'screw'):
        angle_deg = 40.0
        A = rot(axis_d, np.deg2rad(angle_deg))
        b = axis_o - A @ axis_o + (0.15 * axis_d if motion == 'screw' else 0)
    else:
        raise ValueError(motion)
    part_end = part @ A.T + b

    F = 2 * k_per_state
    states = np.array([0] * k_per_state + [1] * k_per_state)
    az = rng.uniform(-1.2, 1.2, F)
    el = rng.uniform(0.2, 0.7, F)
    centers = 3.5 * np.stack([np.cos(el) * np.sin(az), -np.cos(el) * np.cos(az), np.sin(el)], 1)
    blocks = []
    for f in range(F):
        R, t = look_at(centers[f])
        P = np.vstack([base, part if states[f] == 0 else part_end])
        blocks.append((P @ R.T + t).T)
    W = np.concatenate(blocks, 0)
    W = W + noise * rng.standard_normal(W.shape)
    labels = np.r_[np.zeros(n_base, int), np.ones(n_part, int)]

    n_out = int(outlier_frac * W.shape[1])
    if n_out:
        junk = rng.uniform(-1, 1, (3 * F, n_out)) + np.tile([0, 0, 3.5], F)[:, None]
        W = np.concatenate([W, junk], 1)
        labels = np.r_[labels, -np.ones(n_out, int)]

    R0, t0 = look_at(centers[0])
    gt = dict(type={'none': 'rigid', 'screw': 'free'}.get(motion, motion),
              axis_d=R0 @ axis_d, axis_o=R0 @ axis_o + t0, angle_deg=angle_deg, disp=disp)
    return W, states, labels, gt
```

- [ ] **Step 2: Write the failing segmentation tests `tests/test_segment.py`**

```python
import numpy as np
import pytest

from artsfm.factorize import as_views, kabsch, fit_motion, residuals, ransac_rigid
from tests.synth import make_scene, rot


def test_kabsch_recovers_rigid_motion():
    rng = np.random.default_rng(0)
    P = rng.standard_normal((20, 3))
    R = rot([1, 2, 3], 0.7)
    t = np.array([0.1, -2, 3])
    R_est, t_est = kabsch(P, P @ R.T + t)
    np.testing.assert_allclose(R_est, R, atol=1e-10)
    np.testing.assert_allclose(t_est, t, atol=1e-10)


def test_fit_motion_zero_residual_on_base():
    W, states, labels, gt = make_scene('revolute')
    base = np.flatnonzero(labels == 0)
    M = fit_motion(W, base)
    assert M.shape == (len(states), 3, 4)
    np.testing.assert_allclose(M[0], np.c_[np.eye(3), np.zeros(3)], atol=1e-10)
    assert residuals(W, M)[base].max() < 1e-9


@pytest.mark.parametrize('motion', ['prismatic', 'revolute', 'screw'])
def test_ransac_segments_base_and_planar_part(motion):
    W, states, labels, gt = make_scene(motion, noise=0.01, planar_part=True)
    tau = 0.02 * np.median(as_views(W)[..., 2])
    base = ransac_rigid(W, np.ones(W.shape[1], bool), tau)
    part = ransac_rigid(W, ~base, tau)
    # base may absorb part points lying on/near the hinge axis; part must be clean
    assert (base & (labels == 0)).sum() / (labels == 0).sum() > 0.95
    assert (labels[part] == 1).mean() > 0.95
    assert (part & (labels == 1)).sum() / (labels == 1).sum() > 0.6


def test_ransac_empty_candidates():
    W, *_ = make_scene('none')
    assert not ransac_rigid(W, np.zeros(W.shape[1], bool), 0.05).any()
```

- [ ] **Step 3: Run tests to verify they fail**

Run: `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -m pytest tests/test_segment.py -q`
Expected: FAIL with `ModuleNotFoundError`/`ImportError` for `artsfm.factorize` names.

- [ ] **Step 4: Implement the segmentation half of `artsfm/factorize.py`**

```python
"""Articulated structure from motion by factorization (Tresadern & Reid, CVPR 2005),
on metric camera-frame 3D tracks.

W is (3F, N): rows 3f..3f+2 hold the camera-frame 3D coordinates of N tracks in view f.
View 0 is a start-state view and defines the reference frame for all outputs.
"""
import numpy as np


def as_views(W):
    """(3F, N) -> (F, N, 3)."""
    return W.reshape(-1, 3, W.shape[1]).transpose(0, 2, 1)


def kabsch(P, Q):
    """Rigid (R, t) minimising ||R P_i + t - Q_i||; P, Q are (N, 3)."""
    mp, mq = P.mean(0), Q.mean(0)
    H = (P - mp).T @ (Q - mq)
    U, _, Vt = np.linalg.svd(H)
    D = np.diag([1.0, 1.0, np.sign(np.linalg.det(Vt.T @ U.T))])
    R = Vt.T @ D @ U.T
    return R, mq - R @ mp


def fit_motion(W, idx):
    """Metric motion matrix M (F, 3, 4) of the tracks `idx`, with structure S = view-0 points.

    This is the paper's rank-4 factorization W = M [S; 1], solved per view by Kabsch
    since S is metric. It is robust to near-planar structure, unlike a linear pseudo-inverse.
    """
    X = as_views(W[:, idx])
    M = np.zeros((X.shape[0], 3, 4))
    for f in range(X.shape[0]):
        R, t = kabsch(X[0], X[f])
        M[f] = np.c_[R, t]
    return M


def residuals(W, M):
    """Per-track RMS (over views) 3D distance between W and M [S; 1]."""
    X = as_views(W)
    pred = np.einsum('fij,nj->fni', M[:, :, :3], X[0]) + M[:, None, :, 3]
    return np.sqrt(((pred - X) ** 2).sum(-1).mean(0))


def ransac_rigid(W, cand, tau, n_iter=500, rng=None):
    """Largest set of candidate tracks sharing one rigid motion (paper Sec. 2.1).

    Minimal sample: 3 tracks (metric analogue of the paper's 4-track affine sample).
    """
    rng = np.random.default_rng(0) if rng is None else rng
    cand = np.asarray(cand, bool)
    best = np.zeros(W.shape[1], bool)
    pool = np.flatnonzero(cand)
    if len(pool) < 3:
        return best
    for _ in range(n_iter):
        idx = rng.choice(pool, 3, replace=False)
        inl = (residuals(W, fit_motion(W, idx)) < tau) & cand
        if inl.sum() > best.sum():
            best = inl
    for _ in range(3):  # refit on the consensus set
        if best.sum() < 3:
            break
        new = (residuals(W, fit_motion(W, np.flatnonzero(best))) < tau) & cand
        if new.sum() < 3:
            break
        best = new
    return best
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -m pytest tests/test_segment.py -q`
Expected: 6 passed.

- [ ] **Step 6: Commit**

```bash
git add artsfm/__init__.py artsfm/factorize.py tests/__init__.py tests/conftest.py tests/synth.py tests/test_segment.py
git commit -m "Add synthetic articulated scenes and rigid-motion RANSAC segmentation"
```

---

### Task 2: Classification, joint recovery and `analyze()`

**Files:**
- Modify: `artsfm/factorize.py` (append)
- Create: `tests/test_analyze.py`

**Interfaces:**
- Consumes: Task 1 functions.
- Produces:
  - `Params` dataclass with fields `tau_frac=0.02, theta_deg=3.0, n_iter=500, min_part=8, min_part_frac=0.05, min_angle_deg=3.0, min_disp_frac=0.01, inconsistent_deg=10.0, seed=0`.
  - `class DegenerateViewsError(ValueError)`.
  - `principal_angles(A, B) -> (k,)` in degrees, ascending.
  - `normalize(W, idx) -> (Wn, center (3,), radius float)`.
  - `relative_motion(W, states, base_idx, part_idx) -> (A (3,3), b (3,))`: the part's start→end motion in the view-0 frame.
  - `screw(A, b) -> dict(angle_deg, axis_d, axis_o, disp_along, disp)`.
  - `nullspace_axis(Mb, Mp) -> (axis_o, axis_d)`, in normalized coordinates.
  - `analyze(W, states, params=Params()) -> dict` with keys `status, reason, type, labels (N,), axis_o, axis_d, angle_deg, disp, principal_angles, nullspace_axis_d, inconsistent, n_base, n_part, scale`. The axis and other quantities are in the view-0 frame, in metric units. `type` is set when status is `ok`; for `rigid` it is `'rigid'`; otherwise `None`.

- [ ] **Step 1: Write the failing tests `tests/test_analyze.py`**

```python
import numpy as np
import pytest

from artsfm.factorize import (Params, DegenerateViewsError, analyze, principal_angles,
                              screw, relative_motion)
from tests.synth import make_scene, rot


def axis_err_deg(a, b):
    c = abs(np.dot(a, b)) / np.linalg.norm(a) / np.linalg.norm(b)
    return np.rad2deg(np.arccos(np.clip(c, -1, 1)))


def line_dist(o1, d1, o2, d2):
    n = np.cross(d1, d2)
    if np.linalg.norm(n) < 1e-9:
        return np.linalg.norm(np.cross(o2 - o1, d1)) / np.linalg.norm(d1)
    return abs(np.dot(o2 - o1, n)) / np.linalg.norm(n)


def test_principal_angles_shared_dims():
    rng = np.random.default_rng(0)
    common = rng.standard_normal((30, 2))
    A = np.c_[common, rng.standard_normal((30, 2))]
    B = np.c_[common @ rng.standard_normal((2, 2)), rng.standard_normal((30, 2))]
    ang = principal_angles(A, B)
    assert ang[0] < 1e-5 and ang[1] < 1e-5 and ang[2] > 1


def test_screw_pure_rotation_and_translation():
    A = rot([0, 0, 1], np.deg2rad(30))
    o = np.array([1.0, 2.0, 0.0])
    s = screw(A, o - A @ o)
    assert abs(s['angle_deg'] - 30) < 1e-8
    assert axis_err_deg(s['axis_d'], [0, 0, 1]) < 1e-6
    assert line_dist(s['axis_o'], s['axis_d'], o, np.array([0, 0, 1.0])) < 1e-8
    s = screw(np.eye(3), np.array([0, 0.3, 0]))
    assert s['angle_deg'] < 1e-8 and abs(s['disp'] - 0.3) < 1e-12
    assert axis_err_deg(s['axis_d'], [0, 1, 0]) < 1e-6


@pytest.mark.parametrize('motion', ['prismatic', 'revolute', 'screw'])
@pytest.mark.parametrize('noise', [0.0, 0.01])
def test_analyze_type_and_axis(motion, noise):
    W, states, labels, gt = make_scene(motion, noise=noise)
    r = analyze(W, states)
    assert r['status'] == 'ok', r['reason']
    assert r['type'] == gt['type'], r['principal_angles']
    tol = 1e-6 if noise == 0 else 3.0
    assert axis_err_deg(r['axis_d'], gt['axis_d']) < tol
    if motion == 'revolute':
        assert line_dist(r['axis_o'], r['axis_d'], gt['axis_o'], gt['axis_d']) < (1e-6 if noise == 0 else 0.03)
        assert abs(r['angle_deg'] - gt['angle_deg']) < (1e-6 if noise == 0 else 2.0)
        assert axis_err_deg(r['nullspace_axis_d'], gt['axis_d']) < (1e-6 if noise == 0 else 5.0)
        assert not r['inconsistent']
    if motion == 'prismatic':
        assert abs(r['disp'] - gt['disp']) < (1e-6 if noise == 0 else 0.02)


def test_rigid_scene():
    W, states, labels, gt = make_scene('none', noise=0.01)
    r = analyze(W, states)
    assert r['status'] == 'rigid' and r['type'] == 'rigid'
    assert r['axis_d'] is None


def test_tiny_part():
    W, states, labels, gt = make_scene('revolute', n_part=5, noise=0.005)
    r = analyze(W, states)
    assert r['status'] == 'rigid'


def test_small_motion_is_ambiguous():
    W, states, labels, gt = make_scene('revolute', noise=0.0)
    # rebuild with a 1-degree rotation: scale the part's end-state displacement
    from tests import synth
    old = synth.rot
    synth.rot = lambda axis, ang: old(axis, np.deg2rad(1.0))
    try:
        W, states, labels, gt = synth.make_scene('revolute', noise=0.0)
    finally:
        synth.rot = old
    r = analyze(W, states)
    assert r['status'] in ('ambiguous_small_motion', 'rigid')
    assert r['axis_d'] is None


def test_outliers_rejected():
    W, states, labels, gt = make_scene('revolute', noise=0.01, outlier_frac=0.2)
    r = analyze(W, states)
    assert r['type'] == 'revolute'
    assert (r['labels'][labels == -1] == -1).mean() > 0.9
    assert axis_err_deg(r['axis_d'], gt['axis_d']) < 3.0


def test_degenerate_views_raise():
    W, states, labels, gt = make_scene('revolute', k_per_state=1)
    with pytest.raises(DegenerateViewsError):
        analyze(W, states)


def test_relative_motion_identity_for_base():
    W, states, labels, gt = make_scene('revolute')
    base = np.flatnonzero(labels == 0)
    A, b = relative_motion(W, states, base, base)
    np.testing.assert_allclose(A, np.eye(3), atol=1e-9)
    np.testing.assert_allclose(b, 0, atol=1e-9)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -m pytest tests/test_analyze.py -q`
Expected: FAIL with ImportError on `Params`.

- [ ] **Step 3: Append classification and joint recovery to `artsfm/factorize.py`**

```python
from dataclasses import dataclass

TYPE_BY_INTERSECTION = {4: 'rigid', 3: 'prismatic', 2: 'revolute'}  # else 'free'


@dataclass
class Params:
    tau_frac: float = 0.02         # RANSAC inlier threshold, fraction of median depth
    theta_deg: float = 3.0         # principal angle below which a dimension is shared
    n_iter: int = 500              # RANSAC iterations per subspace
    min_part: int = 8              # minimum part inliers
    min_part_frac: float = 0.05    # minimum part inliers as a fraction of tracks
    min_angle_deg: float = 3.0     # below this rotation AND min_disp_frac -> ambiguous
    min_disp_frac: float = 0.01    # translation threshold, fraction of scene scale
    inconsistent_deg: float = 10.0  # nullspace vs screw axis disagreement flag
    seed: int = 0


class DegenerateViewsError(ValueError):
    pass


def principal_angles(A, B):
    """Principal angles (degrees, ascending) between col(A) and col(B)."""
    Qa, _ = np.linalg.qr(A)
    Qb, _ = np.linalg.qr(B)
    s = np.linalg.svd(Qa.T @ Qb, compute_uv=False).clip(-1, 1)
    return np.rad2deg(np.arccos(s))


def normalize(W, idx):
    """Shift by the view-0 centroid of tracks `idx` and scale by their RMS radius."""
    S = as_views(W)[0][idx]
    center = S.mean(0)
    radius = float(np.sqrt(((S - center) ** 2).sum(1).mean()))
    Wn = ((W.reshape(-1, 3, W.shape[1]) - center[None, :, None]) / radius).reshape(W.shape)
    return Wn, center, radius


def relative_motion(W, states, base_idx, part_idx):
    """Rigid motion (A, b) taking the part from its start to its end pose, in the view-0 frame.

    Base motion per view maps view-0 coordinates into view f; inverting it brings
    the part's observations back into the view-0 (base) frame, averaged per state.
    """
    X = as_views(W)
    Mb = fit_motion(W, base_idx)
    in_base = np.stack([(X[f][part_idx] - Mb[f, :, 3]) @ Mb[f, :, :3] for f in range(len(X))])
    P_start = in_base[states == 0].mean(0)
    P_end = in_base[states == 1].mean(0)
    return kabsch(P_start, P_end)


def screw(A, b):
    """Screw decomposition of x -> A x + b."""
    angle = float(np.arccos(np.clip((np.trace(A) - 1) / 2, -1, 1)))
    disp = float(np.linalg.norm(b))
    if angle < 1e-6:
        d = b / disp if disp > 0 else np.array([1.0, 0, 0])
        return dict(angle_deg=0.0, axis_d=d, axis_o=np.zeros(3), disp_along=disp, disp=disp)
    w = np.array([A[2, 1] - A[1, 2], A[0, 2] - A[2, 0], A[1, 0] - A[0, 1]])
    d = w / np.linalg.norm(w)
    along = float(d @ b)
    o = np.linalg.lstsq(np.eye(3) - A, b - along * d, rcond=None)[0]
    return dict(angle_deg=float(np.rad2deg(angle)), axis_d=d, axis_o=o, disp_along=along, disp=disp)


def nullspace_axis(Mb, Mp):
    """Hinge axis from the 2D right nullspace of [M | -M'] (paper Eq. 2 / Sec. 2.3)."""
    B = np.concatenate([Mb.reshape(-1, 4), -Mp.reshape(-1, 4)], 1)
    V = np.linalg.svd(B)[2][-2:].T             # (8, 2)
    X = (V[:4] + V[4:]) / 2                     # homogeneous points [s; w] on the axis
    c_dir = np.array([X[3, 1], -X[3, 0]])       # combination with w = 0
    d = X[:3] @ c_dir
    d = d / np.linalg.norm(d)
    c_pt = X[3] / (X[3] @ X[3])                 # min-norm combination with w = 1
    return X[:3] @ c_pt, d


def _result(status, reason, labels, **kw):
    r = dict(status=status, reason=reason, type=None, labels=labels, axis_o=None, axis_d=None,
             angle_deg=None, disp=None, principal_angles=None, nullspace_axis_d=None,
             inconsistent=False, n_base=int((labels == 0).sum()), n_part=int((labels == 1).sum()),
             scale=None)
    r.update(kw)
    return r


def _check_views(Mb, states):
    for s in (0, 1):
        views = np.flatnonzero(states == s)
        sv = np.linalg.svd(Mb[views].reshape(-1, 4), compute_uv=False)
        if len(views) < 2 or sv[-1] < 1e-3 * sv[0]:
            raise DegenerateViewsError(f'state {s} views {views.tolist()} do not constrain a rank-4 motion')


def analyze(W, states, params=Params()):
    """Segment base/part, classify the joint by the rank ladder, and recover joint parameters."""
    states = np.asarray(states)
    N = W.shape[1]
    labels = -np.ones(N, int)
    rng = np.random.default_rng(params.seed)
    tau = params.tau_frac * float(np.median(as_views(W)[..., 2]))

    base = ransac_rigid(W, np.ones(N, bool), tau, params.n_iter, rng)
    labels[base] = 0
    if base.sum() < 3:
        return _result('rigid', 'no dominant rigid body found', labels, type='rigid')
    _check_views(fit_motion(W, np.flatnonzero(base)), states)

    part = ransac_rigid(W, ~base, tau, params.n_iter, rng)
    if part.sum() < max(params.min_part, params.min_part_frac * N):
        return _result('rigid', f'part has {int(part.sum())} inliers', labels, type='rigid')
    labels[part] = 1
    bi, pi = np.flatnonzero(base), np.flatnonzero(part)

    Wn, center, radius = normalize(W, bi)
    Mb, Mp = fit_motion(Wn, bi), fit_motion(Wn, pi)
    angles = principal_angles(Mb.reshape(-1, 4), Mp.reshape(-1, 4))
    d = int((angles < params.theta_deg).sum())
    jtype = TYPE_BY_INTERSECTION.get(d, 'free')

    A, b = relative_motion(W, states, bi, pi)
    sc = screw(A, b)
    common = dict(principal_angles=angles.tolist(), scale=radius)
    if sc['angle_deg'] < params.min_angle_deg and sc['disp'] < params.min_disp_frac * radius:
        return _result('ambiguous_small_motion', 'part motion below thresholds', labels, **common)
    if jtype == 'rigid':
        return _result('rigid', 'subspaces coincide', labels, type='rigid', **common)

    r = _result('ok', '', labels, type=jtype, **common)
    if jtype == 'prismatic':
        r.update(axis_d=b / np.linalg.norm(b), axis_o=None, angle_deg=0.0, disp=sc['disp'])
    else:
        r.update(axis_d=sc['axis_d'], axis_o=sc['axis_o'], angle_deg=sc['angle_deg'],
                 disp=sc['disp_along'])
        if jtype == 'revolute':
            no, nd = nullspace_axis(Mb, Mp)
            r['nullspace_axis_d'] = nd
            err = np.rad2deg(np.arccos(np.clip(abs(nd @ sc['axis_d']), -1, 1)))
            r['inconsistent'] = bool(err > params.inconsistent_deg)
    return r
```

Note: the nullspace axis is computed in normalized coordinates. Its direction is unaffected by normalization, so only the direction is compared and reported.

- [ ] **Step 4: Run tests to verify they pass**

Run: `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -m pytest tests/ -q`
Expected: all pass. If a type test fails at noise 0.01, print `r['principal_angles']`. Do not change `theta_deg` here; Task 3 calibrates it.

- [ ] **Step 5: Commit**

```bash
git add artsfm/factorize.py tests/test_analyze.py
git commit -m "Add rank-ladder joint classification and joint recovery"
```

---

### Task 3: Threshold calibration on synthetic data + spec update

**Files:**
- Create: `tests/calibrate.py`
- Modify: `artsfm/factorize.py` (`Params` defaults, only if calibration says so), `docs/superpowers/specs/2026-10-04-articulated-sfm-factorization-design.md` (§3.1, §3.5, §4.3, §4.4)

**Interfaces:**
- Consumes: `analyze`, `Params`, `make_scene`.
- Produces: calibrated `Params` defaults, recorded in spec §4.4.

- [ ] **Step 1: Write `tests/calibrate.py`**

```python
"""Sweep noise x thresholds on synthetic scenes; print type accuracy (not a pytest test)."""
import itertools
import numpy as np

from artsfm.factorize import Params, analyze
from tests.synth import make_scene

MOTIONS = ['none', 'prismatic', 'revolute', 'screw']
NOISES = [0.0, 0.005, 0.01, 0.02, 0.03]
SEEDS = range(5)


def accuracy(params, noise):
    ok = 0
    for motion, seed in itertools.product(MOTIONS, SEEDS):
        W, states, labels, gt = make_scene(motion, noise=noise, outlier_frac=0.1, seed=seed)
        ok += analyze(W, states, params)['type'] == gt['type']
    return ok / (len(MOTIONS) * len(SEEDS))


if __name__ == '__main__':
    print('tau_frac theta_deg | ' + ' '.join(f'n={n}' for n in NOISES))
    for tau_frac, theta in itertools.product([0.01, 0.02, 0.03], [1.5, 3.0, 4.5]):
        p = Params(tau_frac=tau_frac, theta_deg=theta, n_iter=200)
        print(f'{tau_frac:8} {theta:9} | ' + ' '.join(f'{accuracy(p, n):5.2f}' for n in NOISES), flush=True)
```

- [ ] **Step 2: Run calibration**

Run: `cd /home/robotlab/scratch/mast3r && python3 -m tests.calibrate`
Expected: a table. Pick the (tau_frac, theta_deg) maximising mean accuracy over noise ≤ 0.02, breaking ties toward the current defaults.

- [ ] **Step 3: Set the chosen values as `Params` defaults** (if they differ), rerun `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -m pytest tests/ -q`, and expect all to pass.

- [ ] **Step 4: Update the spec**
- Write the calibrated values and the calibration table into §4.4.
- Rewrite §4.3 step 1 for the 3-track Kabsch sample and step 2 for the centroid/radius normalisation.
- Change §3.1 to clustering around a seed view, and §3.5 segmentation labels to nearest-GT-mesh.
- Change §5 "scene scale" to the base RMS radius.

- [ ] **Step 5: Commit**

```bash
git add tests/calibrate.py artsfm/factorize.py docs/superpowers/specs/2026-10-04-articulated-sfm-factorization-design.md
git commit -m "Calibrate factorization thresholds on synthetic data; sync spec"
```

---

### Task 4: Tracks and measurement matrix

**Files:**
- Create: `artsfm/tracks.py`, `tests/test_tracks.py`

**Interfaces:**
- Produces:
  - `build_tracks(pair_corres: dict[(i, j)] -> (xy_i (n,2), xy_j (n,2)), n_views: int, cell: int = 8) -> np.ndarray (N, n_views, 2)`. This returns complete, conflict-free tracks; each pixel is the mean of the observations falling in the track's cell.
  - `lift_tracks(tracks (N,F,2), Ks (F,3,3), depths: list of (H,W) arrays) -> (W (3F, M), keep (N,) bool)`.
  - `load_pair_correspondences(cache_dir, image_paths, subsample=8, desc_conf='desc_conf', min_conf=0.0) -> dict[(i,j)] -> (xy_i, xy_j)`, for i < j.

- [ ] **Step 1: Write the failing tests `tests/test_tracks.py`**

```python
import numpy as np

from artsfm.tracks import build_tracks, lift_tracks


def test_chains_pairwise_matches_into_complete_track():
    p = {0: [10, 10], 1: [50, 20], 2: [90, 30]}
    corres = {(0, 1): (np.array([p[0]]), np.array([p[1]])),
              (1, 2): (np.array([p[1]]) + 1, np.array([p[2]])),   # within the same 8px cell
              (0, 2): (np.array([[200, 200]]), np.array([[300, 300]]))}  # incomplete track
    T = build_tracks(corres, 3, cell=8)
    assert T.shape == (1, 3, 2)
    np.testing.assert_allclose(T[0, 0], p[0])
    np.testing.assert_allclose(T[0, 1], [50.5, 20.5])
    np.testing.assert_allclose(T[0, 2], p[2])


def test_conflicting_track_dropped():
    corres = {(0, 1): (np.array([[10, 10], [10, 10]]), np.array([[50, 20], [150, 120]])),
              (0, 2): (np.array([[10, 10]]), np.array([[90, 30]]))}
    T = build_tracks(corres, 3, cell=8)
    assert T.shape == (0, 3, 2)


def test_lift_tracks_backprojects():
    K = np.array([[100.0, 0, 50], [0, 100, 40], [0, 0, 1]])
    depth = np.full((80, 100), 2.0)
    tracks = np.array([[[60.0, 40.0], [50.0, 50.0]]])
    W, keep = lift_tracks(tracks, np.stack([K, K]), [depth, depth])
    assert keep.tolist() == [True]
    np.testing.assert_allclose(W[:, 0], [0.2, 0.0, 2.0, 0.0, 0.2, 2.0])


def test_lift_drops_invalid_depth():
    K = np.array([[100.0, 0, 50], [0, 100, 40], [0, 0, 1]])
    d0 = np.full((80, 100), 2.0)
    d1 = d0.copy()
    d1[50, 50] = 0.0
    d1[10, 10] = np.nan
    tracks = np.array([[[60.0, 40.0], [50.0, 50.0]],
                       [[60.0, 40.0], [10.0, 10.0]],
                       [[60.0, 40.0], [500.0, 10.0]],   # out of image
                       [[60.0, 40.0], [20.0, 20.0]]])
    W, keep = lift_tracks(tracks, np.stack([K, K]), [d0, d1])
    assert keep.tolist() == [False, False, False, True]
    assert W.shape == (6, 1) and np.isfinite(W).all()
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -m pytest tests/test_tracks.py -q`
Expected: FAIL with ImportError.

- [ ] **Step 3: Implement `artsfm/tracks.py`**

```python
"""Multi-view tracks from MASt3R pairwise matches, lifted to camera-frame metric 3D."""
import os
from collections import defaultdict

import numpy as np
import torch

from mast3r.utils.misc import hash_md5


def load_pair_correspondences(cache_dir, image_paths, subsample=8, desc_conf='desc_conf', min_conf=0.0):
    """Read the pairwise matches cached by mast3r's forward_mast3r. Keys (i, j) with i < j."""
    root = os.path.join(cache_dir, f'corres_conf={desc_conf}_subsample={subsample}')
    hashes = [hash_md5(p) for p in image_paths]
    out = {}
    for i in range(len(image_paths)):
        for j in range(i + 1, len(image_paths)):
            fwd = os.path.join(root, f'{hashes[i]}-{hashes[j]}.pth')
            bwd = os.path.join(root, f'{hashes[j]}-{hashes[i]}.pth')
            if os.path.isfile(fwd):
                _, (xy_i, xy_j, conf) = torch.load(fwd, map_location='cpu')
            elif os.path.isfile(bwd):
                _, (xy_j, xy_i, conf) = torch.load(bwd, map_location='cpu')
            else:
                continue
            xy_i, xy_j, conf = (np.asarray(x.cpu() if torch.is_tensor(x) else x) for x in (xy_i, xy_j, conf))
            valid = conf > min_conf
            out[i, j] = (xy_i[valid].astype(float), xy_j[valid].astype(float))
    return out


def build_tracks(pair_corres, n_views, cell=8):
    """Union-find over (view, pixel cell) nodes; keep tracks seen exactly once in every view."""
    parent = {}

    def find(x):
        parent.setdefault(x, x)
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    obs = defaultdict(list)  # node -> list of pixels
    for (i, j), (xy_i, xy_j) in pair_corres.items():
        for a, b in zip(xy_i, xy_j):
            na = (i, int(a[0] // cell), int(a[1] // cell))
            nb = (j, int(b[0] // cell), int(b[1] // cell))
            obs[na].append(a)
            obs[nb].append(b)
            ra, rb = find(na), find(nb)
            if ra != rb:
                parent[ra] = rb

    groups = defaultdict(list)
    for node in obs:
        groups[find(node)].append(node)
    tracks = []
    for nodes in groups.values():
        views = [n[0] for n in nodes]
        if len(nodes) != n_views or len(set(views)) != n_views:
            continue  # incomplete, or two cells in one view (conflict)
        t = np.zeros((n_views, 2))
        for n in nodes:
            t[n[0]] = np.mean(obs[n], 0)
        tracks.append(t)
    return np.array(tracks).reshape(-1, n_views, 2)


def lift_tracks(tracks, Ks, depths):
    """Back-project track pixels with per-view depth: X = D(p) K^-1 [p; 1]. Drops invalid tracks."""
    N, F, _ = tracks.shape
    X = np.zeros((F, N, 3))
    keep = np.ones(N, bool)
    for f in range(F):
        H, Wd = depths[f].shape
        u = np.round(tracks[:, f, 0]).astype(int)
        v = np.round(tracks[:, f, 1]).astype(int)
        inside = (u >= 0) & (u < Wd) & (v >= 0) & (v < H)
        z = np.zeros(N)
        z[inside] = depths[f][v[inside], u[inside]]
        ok = inside & np.isfinite(z) & (z > 0)
        keep &= ok
        rays = np.c_[tracks[:, f], np.ones(N)] @ np.linalg.inv(Ks[f]).T
        X[f] = rays * np.where(ok, z, 0)[:, None]
    W = X[:, keep].transpose(0, 2, 1).reshape(3 * F, -1)
    return W, keep
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -m pytest tests/test_tracks.py -q`
Expected: 4 passed.

- [ ] **Step 5: Commit**

```bash
git add artsfm/tracks.py tests/test_tracks.py
git commit -m "Add multi-view track chaining and metric lifting"
```

---

### Task 5: View selection

**Files:**
- Create: `artsfm/views.py`, `tests/test_views.py`

**Interfaces:**
- Produces:
  - `load_frames(data_dir, state, split='train') -> list[(path str, c2w (4,4))]`. It reads `transforms_{split}_{state}.json` and appends `.png` when `file_path` has no extension.
  - `select_views(start_frames, end_frames, k=5, seed_view=0) -> (paths list[str] of length 2k, states np.ndarray)`. The first k entries are start views, the seed first; the next k are end views. Each start view is paired with its nearest-direction end view, and end views are distinct.

- [ ] **Step 1: Write the failing tests `tests/test_views.py`**

```python
import json
import numpy as np

from artsfm.views import load_frames, select_views


def _frame(az):
    c2w = np.eye(4)
    c2w[:3, 3] = [np.cos(az), np.sin(az), 0.3]
    return c2w


def test_select_views_clusters_around_seed_and_pairs_states():
    start = [(f's{i}', _frame(a)) for i, a in enumerate(np.linspace(0, 2 * np.pi, 20, endpoint=False))]
    end = [(f'e{i}', _frame(a + 0.05)) for i, a in enumerate(np.linspace(0, 2 * np.pi, 20, endpoint=False))]
    paths, states = select_views(start, end, k=3, seed_view=5)
    assert paths[0] == 's5'
    assert sorted(paths[:3]) == ['s4', 's5', 's6']
    assert sorted(paths[3:]) == ['e4', 'e5', 'e6']
    assert states.tolist() == [0, 0, 0, 1, 1, 1]


def test_load_frames_appends_png(tmp_path):
    (tmp_path / 'start/train/rgba').mkdir(parents=True)
    meta = {'frames': [{'file_path': 'start/train/rgba/0000', 'transform_matrix': np.eye(4).tolist()}]}
    (tmp_path / 'transforms_train_start.json').write_text(json.dumps(meta))
    frames = load_frames(str(tmp_path), 'start')
    assert frames[0][0] == str(tmp_path / 'start/train/rgba/0000.png')
    assert frames[0][1].shape == (4, 4)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -m pytest tests/test_views.py -q`
Expected: FAIL with ImportError.

- [ ] **Step 3: Implement `artsfm/views.py`**

```python
"""Choose a small, overlapping set of views from the two articulation states.

v1 uses the dataset's camera poses only to choose views (spec Sec. 3.1); swap
`select_views` for a retrieval-based selector to drop that dependency.
"""
import json
import os

import numpy as np


def load_frames(data_dir, state, split='train'):
    with open(os.path.join(data_dir, f'transforms_{split}_{state}.json')) as f:
        frames = json.load(f)['frames']
    out = []
    for fr in frames:
        fp = fr['file_path']
        if not os.path.splitext(fp)[1]:
            fp += '.png'
        out.append((os.path.join(data_dir, fp), np.array(fr['transform_matrix'], float)))
    return out


def _dirs(frames):
    c = np.array([T[:3, 3] for _, T in frames])
    return c / np.linalg.norm(c, axis=1, keepdims=True)


def select_views(start_frames, end_frames, k=5, seed_view=0):
    ds, de = _dirs(start_frames), _dirs(end_frames)
    order = np.argsort(-(ds @ ds[seed_view]))       # seed first, then nearest directions
    chosen_s = [seed_view] + [i for i in order if i != seed_view][:k - 1]
    chosen_e, used = [], set()
    for i in chosen_s:
        for j in np.argsort(-(de @ ds[i])):
            if j not in used:
                used.add(j)
                chosen_e.append(j)
                break
    paths = [start_frames[i][0] for i in chosen_s] + [end_frames[j][0] for j in chosen_e]
    return paths, np.array([0] * k + [1] * k)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -m pytest tests/test_views.py -q`
Expected: 2 passed.

- [ ] **Step 5: Commit**

```bash
git add artsfm/views.py tests/test_views.py
git commit -m "Add two-state view selection"
```

---

### Task 6: MASt3R-SfM reconstruction wrapper

**Files:**
- Create: `artsfm/reconstruct.py`, `tests/test_reconstruct.py`

**Interfaces:**
- Consumes: mast3r `sparse_global_alignment`, `make_pairs`, `load_images`, `AsymmetricMASt3R`.
- Produces:
  - `run_sparse_ga(image_paths, cache_dir, weights, device='cpu', optim_level='refine', image_size=512, lr1=0.07, niter1=300, lr2=0.01, niter2=300, matching_conf_thr=5.0) -> scene` (a SparseGA).
  - `extract_geometry(scene) -> dict(K (F,3,3), cam2w (F,4,4), depths list[(H,W)])`, where depths are dense, in the GA's common scale, at model input resolution.

- [ ] **Step 1: Write the slow integration test `tests/test_reconstruct.py`** (2 real images; skipped if weights or data are missing)

```python
import os
import numpy as np
import pytest

WEIGHTS = 'checkpoints/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric.pth'
OBJ = '/home/robotlab/scratch/GaussianArt/data/MPArt-90/extracted/StorageFurniture_45166'


@pytest.mark.skipif(not (os.path.isfile(WEIGHTS) and os.path.isdir(OBJ)), reason='needs weights + data')
def test_two_view_reconstruction(tmp_path):
    from artsfm.reconstruct import run_sparse_ga, extract_geometry
    from artsfm.tracks import load_pair_correspondences
    paths = [f'{OBJ}/start/train/rgba/0000.png', f'{OBJ}/start/train/rgba/0001.png']
    scene = run_sparse_ga(paths, str(tmp_path), WEIGHTS, niter1=20, niter2=0)
    g = extract_geometry(scene)
    assert g['K'].shape == (2, 3, 3) and g['cam2w'].shape == (2, 4, 4)
    H, W = g['depths'][0].shape
    assert (H, W) == scene.imgs[0].shape[:2]
    assert np.isfinite(g['depths'][0]).all() and (g['depths'][0] > 0).mean() > 0.9
    corres = load_pair_correspondences(str(tmp_path), paths)
    assert (0, 1) in corres and len(corres[0, 1][0]) > 100
```

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -m pytest tests/test_reconstruct.py -q`
Expected: FAIL with ImportError on `artsfm.reconstruct`.

- [ ] **Step 3: Implement `artsfm/reconstruct.py`**

```python
"""Thin wrapper around MASt3R-SfM sparse global alignment."""
import os

import numpy as np
import torch

import mast3r.utils.path_to_dust3r  # noqa
from dust3r.utils.image import load_images
from mast3r.cloud_opt.sparse_ga import sparse_global_alignment
from mast3r.image_pairs import make_pairs
from mast3r.model import AsymmetricMASt3R


def run_sparse_ga(image_paths, cache_dir, weights, device='cpu', optim_level='refine', image_size=512,
                  lr1=0.07, niter1=300, lr2=0.01, niter2=300, matching_conf_thr=5.0):
    os.makedirs(cache_dir, exist_ok=True)
    model = AsymmetricMASt3R.from_pretrained(weights).to(device)
    imgs = load_images(list(image_paths), size=image_size, verbose=False)
    pairs = make_pairs(imgs, scene_graph='complete', prefilter=None, symmetrize=True)
    if optim_level == 'coarse':
        niter2 = 0
    return sparse_global_alignment(list(image_paths), pairs, cache_dir, model, lr1=lr1, niter1=niter1,
                                   lr2=lr2, niter2=niter2, device=device,
                                   opt_depth='depth' in optim_level, matching_conf_thr=matching_conf_thr)


def extract_geometry(scene):
    with torch.no_grad():
        _, depthmaps, _ = scene.get_dense_pts3d(clean_depth=False)
    K = np.stack([k.detach().cpu().numpy() for k in scene.intrinsics])
    cam2w = scene.get_im_poses().detach().cpu().numpy()
    depths = [d.detach().cpu().numpy().reshape(im.shape[:2]) for d, im in zip(depthmaps, scene.imgs)]
    return dict(K=K, cam2w=cam2w, depths=depths)
```

- [ ] **Step 4: Run test to verify it passes** (CPU: a few minutes)

Run: `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -m pytest tests/test_reconstruct.py -q`
Expected: 1 passed. If the depth reshape fails, print `depthmaps[0].shape` and `scene.imgs[0].shape`. Dense depth is per pixel at model resolution (`make_pts3d` with all pixels as anchors).

- [ ] **Step 5: Commit**

```bash
git add artsfm/reconstruct.py tests/test_reconstruct.py
git commit -m "Add MASt3R-SfM reconstruction wrapper"
```

---

### Task 7: Evaluation and outputs

**Files:**
- Create: `artsfm/evaluate.py`, `tests/test_evaluate.py`

**Interfaces:**
- Produces:
  - `umeyama(src (n,3), dst (n,3)) -> (s, R, t)` with dst ≈ s R src + t.
  - `axis_angle_err(d1, d2) -> deg` (sign-agnostic).
  - `line_distance(o1, d1, o2, d2) -> float`.
  - `load_gt_joint(data_dir) -> dict(type, axis_o, axis_d, state_change)`. This handles single-joint `gt/trans.json`, mapping `rotate`→`revolute` and `translate`→`prismatic`; `state_change = |l - r|` (degrees or units).
  - `view0_to_gt(points_or_dirs, cam2w0, sim, is_dir=False)`.
  - `gt_part_labels(points_gt (n,3), data_dir) -> (labels (n,) {0 static, 1 dynamic}, dist (n,))`.
  - `evaluate(result, W, geom, view_paths, frames_by_path, data_dir) -> dict metrics`.
  - `write_outputs(out_dir, result, metrics, W, geom)`.

- [ ] **Step 1: Write the failing tests `tests/test_evaluate.py`**

```python
import json
import numpy as np

from artsfm.evaluate import umeyama, axis_angle_err, line_distance, load_gt_joint, view0_to_gt
from tests.synth import rot


def test_umeyama_recovers_similarity():
    rng = np.random.default_rng(0)
    src = rng.standard_normal((10, 3))
    R, s, t = rot([0, 1, 1], 0.4), 2.5, np.array([1, 2, 3.0])
    s_, R_, t_ = umeyama(src, s * src @ R.T + t)
    assert abs(s_ - s) < 1e-9
    np.testing.assert_allclose(R_, R, atol=1e-9)
    np.testing.assert_allclose(t_, t, atol=1e-9)


def test_axis_angle_sign_agnostic():
    assert axis_angle_err([0, 0, 1], [0, 0, -1]) < 1e-9
    assert abs(axis_angle_err([1, 0, 0], [0, 1, 0]) - 90) < 1e-9


def test_line_distance_point_choice():
    d = np.array([0, 0, 1.0])
    assert abs(line_distance(np.array([1, 0, 0.0]), d, np.array([1, 0, 5.0]), -d)) < 1e-12
    assert abs(line_distance(np.array([1, 0, 0.0]), d, np.array([0, 2, 3.0]), d) - np.sqrt(5)) < 1e-12


def test_load_gt_joint(tmp_path):
    (tmp_path / 'gt').mkdir()
    (tmp_path / 'gt/trans.json').write_text(json.dumps({'trans_info': [
        {'axis': {'o': [0, 1, 2], 'd': [1, 0, 0]}, 'rotate': {'l': 68.0, 'r': 32.0}, 'type': 'rotate'}]}))
    g = load_gt_joint(str(tmp_path))
    assert g['type'] == 'revolute' and abs(g['state_change'] - 36.0) < 1e-12
    np.testing.assert_allclose(g['axis_o'], [0, 1, 2])


def test_view0_to_gt_roundtrip():
    cam2w0 = np.eye(4)
    cam2w0[:3, :3] = rot([1, 0, 0], 0.3)
    cam2w0[:3, 3] = [0, 0, 1]
    sim = (2.0, rot([0, 0, 1], 0.5), np.array([1.0, 0, 0]))
    p = np.array([[0.1, 0.2, 0.3]])
    world = p @ cam2w0[:3, :3].T + cam2w0[:3, 3]
    expected = 2.0 * world @ sim[1].T + sim[2]
    np.testing.assert_allclose(view0_to_gt(p, cam2w0, sim), expected)
    d = np.array([[0, 0, 1.0]])
    np.testing.assert_allclose(view0_to_gt(d, cam2w0, sim, is_dir=True),
                               d @ cam2w0[:3, :3].T @ sim[1].T)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -m pytest tests/test_evaluate.py -q`
Expected: FAIL with ImportError.

- [ ] **Step 3: Implement `artsfm/evaluate.py`**

```python
"""Align the reconstruction to MPArt-90 ground truth, score the joint, and write outputs."""
import json
import os

import numpy as np
import trimesh
from scipy.spatial import cKDTree

from artsfm.factorize import as_views

GT_TYPES = {'rotate': 'revolute', 'translate': 'prismatic'}


def umeyama(src, dst):
    ms, md = src.mean(0), dst.mean(0)
    S, D = src - ms, dst - md
    U, sig, Vt = np.linalg.svd(D.T @ S / len(src))
    E = np.diag([1.0, 1.0, np.sign(np.linalg.det(U @ Vt))])
    R = U @ E @ Vt
    s = float(np.trace(np.diag(sig) @ E) / (S ** 2).sum(1).mean())
    return s, R, md - s * R @ ms


def axis_angle_err(d1, d2):
    d1, d2 = np.asarray(d1, float), np.asarray(d2, float)
    c = abs(d1 @ d2) / np.linalg.norm(d1) / np.linalg.norm(d2)
    return float(np.rad2deg(np.arccos(np.clip(c, -1, 1))))


def line_distance(o1, d1, o2, d2):
    o1, d1, o2, d2 = (np.asarray(x, float) for x in (o1, d1, o2, d2))
    n = np.cross(d1, d2)
    if np.linalg.norm(n) < 1e-9 * np.linalg.norm(d1) * np.linalg.norm(d2):
        return float(np.linalg.norm(np.cross(o2 - o1, d1)) / np.linalg.norm(d1))
    return float(abs((o2 - o1) @ n) / np.linalg.norm(n))


def load_gt_joint(data_dir):
    with open(os.path.join(data_dir, 'gt', 'trans.json')) as f:
        info = json.load(f)['trans_info']
    if len(info) != 1:
        raise ValueError(f'v1 handles single-joint objects; {data_dir} has {len(info)} joints')
    j = info[0]
    lr = j[j['type']]
    return dict(type=GT_TYPES[j['type']], axis_o=np.array(j['axis']['o'], float),
                axis_d=np.array(j['axis']['d'], float), state_change=abs(lr['l'] - lr['r']))


def view0_to_gt(x, cam2w0, sim, is_dir=False):
    s, R, t = sim
    x = np.atleast_2d(x)
    if is_dir:
        return x @ cam2w0[:3, :3].T @ R.T
    world = x @ cam2w0[:3, :3].T + cam2w0[:3, 3]
    return s * world @ R.T + t


def gt_part_labels(points_gt, data_dir, n_samples=20000):
    clouds = []
    for name in ('static', 'dynamic'):
        mesh = trimesh.load(os.path.join(data_dir, 'gt', 'start', f'{name}.obj'), force='mesh')
        clouds.append(mesh.sample(n_samples))
    d = np.stack([cKDTree(c).query(points_gt)[0] for c in clouds], 1)
    return d.argmin(1), d.min(1)


def evaluate(result, W, geom, view_paths, frames_by_path, data_dir):
    gt = load_gt_joint(data_dir)
    est_c = geom['cam2w'][:, :3, 3]
    gt_c = np.array([frames_by_path[p][:3, 3] for p in view_paths])
    sim = umeyama(est_c, gt_c)
    cam2w0 = geom['cam2w'][0]
    m = dict(gt_type=gt['type'], pred_type=result['type'], type_correct=result['type'] == gt['type'],
             sim_scale=sim[0],
             cam_align_rmse=float(np.sqrt(((sim[0] * est_c @ sim[1].T + sim[2] - gt_c) ** 2).sum(1).mean())))

    pts_gt = view0_to_gt(as_views(W)[0], cam2w0, sim)
    gt_lab, dist = gt_part_labels(pts_gt, data_dir)
    lab = result['labels']
    seg = lab >= 0
    m['seg_accuracy'] = float((lab[seg] == gt_lab[seg]).mean()) if seg.any() else None
    m['track_to_mesh_median'] = float(np.median(dist))
    m['n_tracks'] = int(W.shape[1])

    if result['axis_d'] is not None:
        d_gt = view0_to_gt(result['axis_d'], cam2w0, sim, is_dir=True)[0]
        m['axis_angle_err_deg'] = axis_angle_err(d_gt, gt['axis_d'])
        m['pred_axis_d_gt'] = d_gt.tolist()
        if result['axis_o'] is not None and gt['type'] == 'revolute':
            o_gt = view0_to_gt(result['axis_o'], cam2w0, sim)[0]
            m['axis_pos_err'] = line_distance(o_gt, d_gt, gt['axis_o'], gt['axis_d'])
            m['pred_axis_o_gt'] = o_gt.tolist()
        if gt['type'] == 'revolute' and result['angle_deg'] is not None:
            m['state_err'] = abs(result['angle_deg'] - gt['state_change'])
        elif gt['type'] == 'prismatic' and result['disp'] is not None:
            m['state_err'] = abs(sim[0] * result['disp'] - gt['state_change'])
    return m


def _jsonable(x):
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, (np.floating, np.integer, np.bool_)):
        return x.item()
    if isinstance(x, dict):
        return {k: _jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_jsonable(v) for v in x]
    return x


def write_outputs(out_dir, result, metrics, W, geom):
    os.makedirs(out_dir, exist_ok=True)
    res = {k: v for k, v in result.items() if k != 'labels'}
    with open(os.path.join(out_dir, 'result.json'), 'w') as f:
        json.dump(_jsonable(dict(result=res, metrics=metrics)), f, indent=2)
    np.savez(os.path.join(out_dir, 'W.npz'), W=W, labels=result['labels'])

    # tracks in view-0 frame coloured by segment, plus the axis as a dense line of points
    pts = as_views(W)[0]
    palette = {0: [160, 160, 160, 255], 1: [230, 60, 40, 255], -1: [40, 40, 40, 255]}
    cols = np.array([palette[int(l)] for l in result['labels']], np.uint8)
    if result['axis_d'] is not None:
        o = result['axis_o'] if result['axis_o'] is not None else pts[result['labels'] == 1].mean(0)
        span = np.linalg.norm(pts - pts.mean(0), axis=1).max()
        line = o + np.linspace(-span, span, 400)[:, None] * result['axis_d']
        pts = np.vstack([pts, line])
        cols = np.vstack([cols, np.tile([40, 90, 230, 255], (len(line), 1)).astype(np.uint8)])
    trimesh.PointCloud(pts, colors=cols).export(os.path.join(out_dir, 'articulation.ply'))
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -m pytest tests/test_evaluate.py -q`
Expected: 5 passed.

- [ ] **Step 5: Commit**

```bash
git add artsfm/evaluate.py tests/test_evaluate.py
git commit -m "Add GT alignment, joint metrics and output writing"
```

---

### Task 8: Driver, real runs, benchmark

**Files:**
- Create: `run_articulation.py`, `run_articulation_benchmark.py`, `tests/test_driver.py`
- Create (results): `docs/superpowers/results/2026-10-04-articulated-sfm-v1.md`

**Interfaces:**
- Consumes: everything above.
- Produces:
  - `run_articulation.run(data_dir, out_dir, weights, k=5, seed_view=0, optim_level='refine', device=None, params=Params(), niter1=300, niter2=300, min_tracks=30) -> dict(result, metrics)`. It writes `out_dir/result.json`.
  - A CLI with the same names.

- [ ] **Step 1: Write the failing driver test `tests/test_driver.py`** (MASt3R is monkeypatched; this tests control flow only)

```python
import json
import numpy as np

import run_articulation as ra


def test_driver_insufficient_tracks(tmp_path, monkeypatch):
    calls = []
    paths = [f'/x/{i}.png' for i in range(10)]
    monkeypatch.setattr(ra, 'load_frames', lambda d, s: [(f'/x/{s}{i}.png', np.eye(4)) for i in range(10)])
    monkeypatch.setattr(ra, 'select_views', lambda s, e, k, seed_view: (paths[:2 * k], np.array([0] * k + [1] * k)))
    monkeypatch.setattr(ra, 'run_sparse_ga', lambda p, c, w, **kw: calls.append(len(p)) or object())
    monkeypatch.setattr(ra, 'extract_geometry', lambda s: dict(K=None, cam2w=None, depths=None))
    monkeypatch.setattr(ra, 'load_pair_correspondences', lambda c, p: {})
    out = ra.run('/data', str(tmp_path), 'w.pth', k=5, device='cpu')
    assert calls == [10, 6]                      # K=5, then retry with K=3
    assert out['result']['status'] == 'insufficient_tracks'
    saved = json.load(open(tmp_path / 'result.json'))
    assert saved['result']['status'] == 'insufficient_tracks'
```

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -m pytest tests/test_driver.py -q`
Expected: FAIL with ModuleNotFoundError `run_articulation`.

- [ ] **Step 3: Implement `run_articulation.py`**

```python
#!/usr/bin/env python3
"""Detect the articulated part and joint of an object from images in two states (MPArt-90 layout).

Tresadern & Reid (CVPR 2005) factorization on MASt3R-SfM metric 3D tracks.
See docs/superpowers/specs/2026-10-04-articulated-sfm-factorization-design.md.
"""
import argparse
import json
import os

import numpy as np
import torch

from artsfm.factorize import Params, analyze
from artsfm.evaluate import evaluate, write_outputs, _jsonable
from artsfm.reconstruct import run_sparse_ga, extract_geometry
from artsfm.tracks import load_pair_correspondences, build_tracks, lift_tracks
from artsfm.views import load_frames, select_views


def _tracks_for(data_dir, out_dir, weights, k, seed_view, optim_level, device, niter1, niter2):
    start, end = load_frames(data_dir, 'start'), load_frames(data_dir, 'end')
    paths, states = select_views(start, end, k=k, seed_view=seed_view)
    cache = os.path.join(out_dir, f'cache_k{k}')
    scene = run_sparse_ga(paths, cache, weights, device=device, optim_level=optim_level,
                          niter1=niter1, niter2=niter2)
    geom = extract_geometry(scene)
    corres = load_pair_correspondences(cache, paths)
    tracks = build_tracks(corres, len(paths))
    W, keep = (lift_tracks(tracks, geom['K'], geom['depths']) if len(tracks)
               else (np.zeros((3 * len(paths), 0)), np.zeros(0, bool)))
    frames_by_path = {p: T for p, T in start + end}
    return paths, states, geom, W, frames_by_path


def run(data_dir, out_dir, weights, k=5, seed_view=0, optim_level='refine', device=None,
        params=Params(), niter1=300, niter2=300, min_tracks=30):
    device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
    os.makedirs(out_dir, exist_ok=True)
    for kk in (k, 3) if k > 3 else (k,):
        paths, states, geom, W, frames_by_path = _tracks_for(
            data_dir, out_dir, weights, kk, seed_view, optim_level, device, niter1, niter2)
        if W.shape[1] >= min_tracks:
            break
    else:
        out = dict(result=dict(status='insufficient_tracks',
                               reason=f'{W.shape[1]} complete tracks < {min_tracks} (k={kk})'),
                   metrics={})
        with open(os.path.join(out_dir, 'result.json'), 'w') as f:
            json.dump(_jsonable(out), f, indent=2)
        return out

    result = analyze(W, states, params)
    metrics = evaluate(result, W, geom, paths, frames_by_path, data_dir)
    metrics['k'] = kk
    metrics['views'] = [os.path.relpath(p, data_dir) for p in paths]
    write_outputs(out_dir, result, metrics, W, geom)
    return dict(result=result, metrics=metrics)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data_dir', required=True)
    p.add_argument('--out_dir', required=True)
    p.add_argument('--weights', default='checkpoints/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric.pth')
    p.add_argument('--k', type=int, default=5, help='views per state')
    p.add_argument('--seed_view', type=int, default=0, help='start-state view the selection clusters around')
    p.add_argument('--optim_level', default='refine', choices=['coarse', 'refine', 'refine+depth'])
    p.add_argument('--device', default=None)
    p.add_argument('--niter1', type=int, default=300)
    p.add_argument('--niter2', type=int, default=300)
    a = p.parse_args()
    out = run(a.data_dir, a.out_dir, a.weights, a.k, a.seed_view, a.optim_level, a.device,
              niter1=a.niter1, niter2=a.niter2)
    r, m = out['result'], out['metrics']
    print(json.dumps(_jsonable(dict(status=r['status'], type=r.get('type'), reason=r.get('reason'),
                                    **{k: v for k, v in m.items() if k != 'views'})), indent=2))


if __name__ == '__main__':
    main()
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -m pytest tests/ -q`
Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add run_articulation.py tests/test_driver.py
git commit -m "Add articulation driver with track-count fallback"
```

- [ ] **Step 6: Real run on the revolute object** (CPU, expect ~10–20 min)

Run: `python3 run_articulation.py --data_dir /home/robotlab/scratch/GaussianArt/data/MPArt-90/extracted/StorageFurniture_45166 --out_dir outputs/artsfm/StorageFurniture_45166`

Expected: printed JSON with `status`, `type`, `axis_angle_err_deg`, etc. Sanity checks before trusting it:
- `cam_align_rmse` should be small relative to the GT camera distance (~4).
- `track_to_mesh_median` should be small (< ~0.05 GT units).

If either is large, the GT frame or alignment is wrong. Debug that (superpowers:systematic-debugging) before reading the joint metrics. If the status is `rigid`, check whether the door is visible from seed view 0 (look at the image) and retry with a different `--seed_view`. Record which seed views were tried.

- [ ] **Step 7: Real run on the prismatic object**

Run: `python3 run_articulation.py --data_dir /home/robotlab/scratch/GaussianArt/data/MPArt-90/extracted/StorageFurniture_46107 --out_dir outputs/artsfm/StorageFurniture_46107`
Expected: same checks.

- [ ] **Step 8: Write `run_articulation_benchmark.py`** to run all single-joint objects and print a markdown table

```python
#!/usr/bin/env python3
"""Run run_articulation over MPArt-90 single-joint objects and print a markdown table."""
import argparse
import json
import os

import run_articulation as ra

ROOT = '/home/robotlab/scratch/GaussianArt/data/MPArt-90/extracted'
OBJECTS = ['StorageFurniture_45166', 'StorageFurniture_35059', 'StorageFurniture_41086',
           'StorageFurniture_45176', 'StorageFurniture_45247', 'StorageFurniture_45937',
           'StorageFurniture_47021', 'StorageFurniture_46107', 'StorageFurniture_45855']
COLS = ['status', 'gt_type', 'pred_type', 'axis_angle_err_deg', 'axis_pos_err', 'state_err',
        'seg_accuracy', 'n_tracks', 'cam_align_rmse']


def fmt(v):
    return f'{v:.3f}' if isinstance(v, float) else ('' if v is None else str(v))


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--out_root', default='outputs/artsfm')
    p.add_argument('--seed_view', type=int, default=0)
    a = p.parse_args()
    print('| object | ' + ' | '.join(COLS) + ' |')
    print('|' + '---|' * (len(COLS) + 1))
    for obj in OBJECTS:
        out_dir = os.path.join(a.out_root, obj)
        res_path = os.path.join(out_dir, 'result.json')
        if not os.path.isfile(res_path):
            try:
                ra.run(os.path.join(ROOT, obj), out_dir, 'checkpoints/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric.pth',
                       seed_view=a.seed_view)
            except Exception as e:  # keep the benchmark going; record the failure
                os.makedirs(out_dir, exist_ok=True)
                json.dump(dict(result=dict(status=f'error: {e}'), metrics={}), open(res_path, 'w'))
        saved = json.load(open(res_path))
        row = dict(saved['metrics'], status=saved['result']['status'])
        print(f'| {obj} | ' + ' | '.join(fmt(row.get(c)) for c in COLS) + ' |', flush=True)


if __name__ == '__main__':
    main()
```

- [ ] **Step 9: Run the benchmark** (long; run in the background)

Run: `python3 run_articulation_benchmark.py | tee outputs/artsfm/benchmark.md`
Expected: a 9-row table.

- [ ] **Step 10: Write `docs/superpowers/results/2026-10-04-articulated-sfm-v1.md`**

Include: the table, the seed views used, the calibrated thresholds, failure analysis per non-`ok` row, and the honest limits (ball joints unidentifiable from 2 states; view selection uses dataset poses).

- [ ] **Step 11: Commit**

```bash
git add run_articulation_benchmark.py docs/superpowers/results/2026-10-04-articulated-sfm-v1.md
git commit -m "Add single-joint MPArt-90 benchmark and v1 results"
```
