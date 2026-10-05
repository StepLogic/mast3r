# Interactive Articulated Model (Viewer + URDF) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Turn a found joint into an articulated model folder (dense per-part point clouds, joint, hulls). It can be manipulated in a local gradio app and exported as a MuJoCo-checked URDF.

**Architecture:**
- `artsfm/articulate.py` labels MASt3R's dense points as body or part. It tests each point against two motion hypotheses using depth and colour from the opposite-state views, with a nearest-track fallback. It then fuses the result into `<run>/model/`.
- `artsfm/model_io.py` reads and writes that folder.
- `artsfm/urdf.py` exports GaussianArt-convention URDFs and checks them in MuJoCo.
- `artsfm/viewer.py` builds `.glb` frames.
- `app_articulation.py` is the gradio UI.
- `build_model.py` and `run_articulation.py --build_model` produce models.

**Tech Stack:** numpy, scipy (cKDTree, ndimage), trimesh 4.12, PIL, gradio 6.20, mujoco 3.3, existing `artsfm` and MASt3R code. CPU only.

**Spec:** `docs/superpowers/specs/2026-10-05-articulated-model-viewer-design.md`

## Global Constraints

- Do not modify anything under `mast3r/` or `dust3r/`.
- Run tests with `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -m pytest ...`.
- **Model frame = view-0 camera frame (OpenCV).** Joint value 0 = start state. Revolute and free joints use radians; prismatic uses reconstruction units.
- **Pixel convention:** integer pixel coordinates are pixel centres. `K` maps camera rays to these coordinates, as in MASt3R/dust3r. Lookups use `np.round`.
- `LabelParams` defaults: `delta=0.08`, `ctol=0.15`, `margin=1`, `k=7`, `fb_radius=0.3`, `edge_frac=0.1`. Task 3 may change them, and only from synthetic calibration.
- At most 150,000 points per link.
- Warnings:
  - part < 500 points → "part sparsely observed";
  - unlabelled > 40% → "many points unlabelled";
  - free joint → "screw joint (not representable in URDF); URDF export disabled".
- The URDF follows GaussianArt naming:
  - links `base` and `link_1`, joint `joint_1`;
  - `meshes/<link>.obj` and `meshes/<link>_collision.obj`;
  - masses 10 / 1, damping 0.5, effort 100, velocity 2;
  - it contains `<mujoco><compiler strippath="false" discardvisual="false"/></mujoco>`.
- Images and colours are float RGB in [0, 1].

## Deviations from spec (decided while planning)

1. **No separate gradio probe task.** MASt3R's own `demo.py` shows point clouds through `gradio.Model3D` (`as_pointcloud=True`), which is precedent that it is legible. Task 9 runs the real check; if points are too faint there, it records a ruling and switches `build_glb` to small quads.
2. **`artsfm/model_io.py`** holds model read/write, plus our own binary PLY reader/writer (trimesh cannot write an empty point cloud). This avoids an `articulate` ↔ `evaluate` import cycle.
3. **URDF CLI** is `python3 -m artsfm.urdf <model_dir> [--scale S] [--use_gt_frame] [--lower L] [--upper U] [--no_check]`. It exports, then runs the MuJoCo check. It writes an `export.json` sidecar (R, s, t, pivot, limits) that the check reads.
4. `run_articulation.run` also stores `data_dir` and `seed_view` in `metrics`, so `build_model.py` can find images without flags.
5. `model.json` gains `gt_frame` (when GT poses exist): `up`, `scale`, `sim_R`, `sim_t`, `cam2w0`. It is used by `--use_gt_frame` and by the real-data label accuracy.

## Review Focus

1. **A model whose part cloud is empty or tiny** (real labelling failure): the app should still load it, showing a warning chip. URDF export should refuse with a clear message. Tests: `test_assemble_tiny_part_warns_and_no_hull` (Task 5) and `test_write_urdf_refuses_without_hull` (Task 6).
2. **Images without alpha** (ordinary photos): masks fall back to MASt3R confidence, and `mask_source` says so. Test: `test_foreground_masks_confidence_fallback` (Task 4).
3. **The slider beyond the observed range or negative:** `joint_transform` must stay a rigid transform, and the glb must still build. Test: `test_joint_transform_is_rigid_any_value` (Task 2).
4. **Prismatic export with scale ≠ 1:** limits and positions must scale together, and the MuJoCo check must pass. Test: `test_urdf_prismatic_scaled_check` (Task 6).
5. **Points exactly on the hinge**, where both hypotheses tie: they must go to the fallback, never be labelled by an arbitrary tie-break. Test: `test_hinge_points_use_fallback` (Task 1).

---

### Task 1: Dense synthetic scenes and the motion-hypothesis labeller

**Files:**
- Create: `tests/synth_dense.py`, `artsfm/articulate.py`, `tests/test_label_dense.py`

**Interfaces:**
- Produces:
  - `tests.synth_dense.make_dense_scene(motion='revolute'|'prismatic', n_per_state=3, H=96, W=128, noise=0.0, seed=0, angle_deg=40.0, disp=0.3) -> dict` with:
    - `K (F,3,3)`, `cam2w (F,4,4)` (world, OpenCV);
    - `depths`, `alphas`, `labels` (each a list of (H,W)), `rgbs` (list of (H,W,3));
    - `states (F,)`, `T_world (4,4)`, `axis_o`, `axis_d` (world).
  - `tests.synth_dense.to_model_frame(scene) -> (geom dict(K, cam2w, depths, imgs), T_model (4,4))`.
  - `tests.synth_dense.sample_tracks(scene, n=150, seed=0) -> (pts (n,3) view-0 frame, labels (n,))`.
  - `artsfm.articulate.LabelParams` (dataclass, defaults above).
  - `reliable_mask(depth, mask, edge_frac) -> bool (H,W)`.
  - `backproject(K, M, depth, mask) -> (pts (n,3), pix (n,))`.
  - `evidence(P, rgb, K, model2cam, depth, rel, img, delta, ctol) -> (support int (n,), contra int (n,))`.
  - `label_dense(geom, masks, states, T, track_pts, track_lab, radius, params=LabelParams()) -> list[dict(pts, rgb, label, fallback, pix)]`. `label` is in {0 body, 1 part, −1 unlabelled}.

- [ ] **Step 1: Write `tests/synth_dense.py`**

```python
"""Synthetic dense two-state scenes: textured oriented boxes ray-cast into pinhole depth maps."""
import numpy as np


def rot(axis, ang):
    axis = np.asarray(axis, float)
    axis = axis / np.linalg.norm(axis)
    K = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
    return np.eye(3) + np.sin(ang) * K + (1 - np.cos(ang)) * K @ K


def look_at(center, target, up=np.array([0, 0, 1.0])):
    """cam2world (4x4, OpenCV: x right, y down, z forward)."""
    z = target - center
    z = z / np.linalg.norm(z)
    x = np.cross(z, up)
    x = x / np.linalg.norm(x)
    y = np.cross(z, x)
    T = np.eye(4)
    T[:3, :3] = np.stack([x, y, z], 1)
    T[:3, 3] = center
    return T


def texture(local):
    """Smooth colour field over a body's own coordinates, so it moves with the body."""
    f = np.array([[7.1, 3.3, 5.2], [2.9, 6.7, 4.1], [5.5, 2.2, 7.9]])
    return 0.5 + 0.45 * np.sin(local @ f.T + np.array([0.3, 1.7, 2.9]))


def ray_boxes(origins, dirs, boxes):
    """Nearest hit of rays with oriented boxes (R, centre, half-extent, label, to_local)."""
    n = len(origins)
    best_t = np.full(n, np.inf)
    best_l = -np.ones(n, int)
    best_rgb = np.zeros((n, 3))
    for R, c, half, label, to_local in boxes:
        o = (origins - c) @ R
        d = dirs @ R
        with np.errstate(divide='ignore', invalid='ignore'):
            t1 = (-half - o) / d
            t2 = (half - o) / d
        tmin = np.nanmax(np.minimum(t1, t2), 1)
        tmax = np.nanmin(np.maximum(t1, t2), 1)
        t = np.where(tmin > 1e-6, tmin, tmax)
        better = (tmax >= tmin) & (tmax > 1e-6) & (t < best_t)
        best_t[better] = t[better]
        best_l[better] = label
        best_rgb[better] = texture(to_local(origins[better] + t[better, None] * dirs[better]))
    return best_t, best_l, best_rgb


def make_dense_scene(motion='revolute', n_per_state=3, H=96, W=128, noise=0.0, seed=0, angle_deg=40.0, disp=0.3):
    rng = np.random.default_rng(seed)
    ident = lambda X: X  # noqa: E731
    body = (np.eye(3), np.array([0, 0, 0.5]), np.array([0.5, 0.35, 0.5]), 0, ident)
    if motion == 'revolute':  # door on the front face (y = -0.35), hinge along z at x = +0.45
        door = (np.eye(3), np.array([0.0, -0.375, 0.5]), np.array([0.45, 0.025, 0.45]), 1, ident)
        o, d = np.array([0.45, -0.375, 0.5]), np.array([0, 0, 1.0])
        A = rot(d, np.deg2rad(angle_deg))
        b = o - A @ o
    elif motion == 'prismatic':  # drawer sliding out along -y
        door = (np.eye(3), np.array([0.0, -0.2, 0.75]), np.array([0.4, 0.2, 0.12]), 1, ident)
        o, d = None, np.array([0, -1.0, 0])
        A, b = np.eye(3), disp * d
    else:
        raise ValueError(motion)
    T = np.eye(4)
    T[:3, :3], T[:3, 3] = A, b
    door_end = (A @ door[0], A @ door[1] + b, door[2], 1, lambda X: (X - b) @ A)

    f = 60.0 * W / 64
    K = np.array([[f, 0, W / 2], [0, f, H / 2], [0, 0, 1]])
    F = 2 * n_per_state
    states = np.array([0] * n_per_state + [1] * n_per_state)
    az = np.deg2rad(rng.uniform(-35, 35, F) + 180)
    el = np.deg2rad(rng.uniform(15, 35, F))
    u, v = np.meshgrid(np.arange(W), np.arange(H))
    pix = np.stack([u.ravel(), v.ravel(), np.ones(H * W)], 1).astype(float)
    target = np.array([0, 0, 0.5])
    out = dict(K=np.stack([K] * F), cam2w=[], depths=[], alphas=[], labels=[], rgbs=[],
               states=states, T_world=T, axis_o=o, axis_d=d)
    for i in range(F):
        c = target + 3.0 * np.array([np.cos(el[i]) * np.sin(az[i]), np.cos(el[i]) * np.cos(az[i]), np.sin(el[i])])
        c2w = look_at(c, target)
        dirs = (pix @ np.linalg.inv(K).T) @ c2w[:3, :3].T     # camera-z component 1, so t == depth
        t, lab, rgb = ray_boxes(np.tile(c, (len(pix), 1)), dirs, [body, door if states[i] == 0 else door_end])
        hit = np.isfinite(t)
        z = np.where(hit, t, 0.0)
        if noise:
            z = z * (1 + noise * rng.standard_normal(z.shape))
        out['cam2w'].append(c2w)
        out['depths'].append(z.reshape(H, W))
        out['alphas'].append(hit.reshape(H, W))
        out['labels'].append(lab.reshape(H, W))
        out['rgbs'].append(rgb.reshape(H, W, 3))
    out['cam2w'] = np.stack(out['cam2w'])
    return out


def to_model_frame(scene):
    """geom dict as extract_geometry returns it, plus the part motion in the view-0 camera frame."""
    c0 = scene['cam2w'][0]
    T_model = np.linalg.inv(c0) @ scene['T_world'] @ c0
    geom = dict(K=scene['K'], cam2w=scene['cam2w'], depths=scene['depths'], imgs=scene['rgbs'])
    return geom, T_model


def sample_tracks(scene, n=150, seed=0):
    """n GT-labelled view-0 points (stand-in for the sparse tracks analyze() labels)."""
    from artsfm.articulate import backproject
    P, idx = backproject(scene['K'][0], np.eye(4), scene['depths'][0], scene['alphas'][0])
    lab = scene['labels'][0].ravel()[idx]
    sel = np.random.default_rng(seed).choice(len(P), min(n, len(P)), replace=False)
    return P[sel], lab[sel]
```

- [ ] **Step 2: Write the failing tests `tests/test_label_dense.py`**

```python
import numpy as np
import pytest

from tests.synth_dense import make_dense_scene, to_model_frame, sample_tracks


def _accuracy(motion, noise, seeds=(0, 1, 2)):
    from artsfm.articulate import label_dense
    accs, unl = [], []
    for seed in seeds:
        s = make_dense_scene(motion, noise=noise, seed=seed)
        geom, T = to_model_frame(s)
        tp, tl = sample_tracks(s, seed=seed)
        radius = float(np.sqrt(((tp - tp.mean(0)) ** 2).sum(1).mean()))
        views = label_dense(geom, s['alphas'], s['states'], T, tp, tl, radius)
        ok = n_lab = n_all = 0
        for f, v in enumerate(views):
            gt = s['labels'][f].ravel()[v['pix']]
            m = v['label'] >= 0
            ok += (v['label'][m] == gt[m]).sum()
            n_lab += m.sum()
            n_all += len(m)
        accs.append(ok / n_lab)
        unl.append(1 - n_lab / n_all)
    return float(np.mean(accs)), float(np.mean(unl))


def test_backproject_roundtrip_through_evidence():
    from artsfm.articulate import backproject, evidence, reliable_mask
    s = make_dense_scene('revolute')
    P, idx = backproject(s['K'][0], np.eye(4), s['depths'][0], s['alphas'][0])
    rgb = s['rgbs'][0].reshape(-1, 3)[idx]
    rel = reliable_mask(s['depths'][0], s['alphas'][0], 0.1)
    sup, con = evidence(P, rgb, s['K'][0], np.eye(4), s['depths'][0], rel, s['rgbs'][0], 0.01, 0.01)
    on_rel = rel.ravel()[idx]
    assert sup[on_rel].all() and not con.any()


def test_reliable_mask_excludes_depth_edges():
    from artsfm.articulate import reliable_mask
    d = np.full((10, 10), 2.0)
    d[:, 5:] = 3.0
    m = reliable_mask(d, np.ones((10, 10), bool), 0.1)
    assert m[:, 0:4].all() and not m[:, 4:6].any() and m[:, 7:].all()


@pytest.mark.parametrize('motion', ['revolute', 'prismatic'])
def test_label_accuracy_noise_free(motion):
    acc, unl = _accuracy(motion, 0.0)
    assert acc > 0.95, acc
    assert unl < 0.10, unl


@pytest.mark.parametrize('motion', ['revolute', 'prismatic'])
def test_label_accuracy_two_percent_noise(motion):
    acc, unl = _accuracy(motion, 0.02)
    assert acc > 0.90, acc


def test_occluded_prediction_is_not_a_contradiction():
    from artsfm.articulate import evidence
    K = np.array([[50.0, 0, 20], [0, 50, 20], [0, 0, 1]])
    depth = np.full((40, 40), 2.0)
    rel = np.ones((40, 40), bool)
    img = np.zeros((40, 40, 3))
    P = np.array([[0.0, 0.0, 3.0], [0.0, 0.0, 1.0]])  # behind the surface / in front of it
    sup, con = evidence(P, np.zeros((2, 3)), K, np.eye(4), depth, rel, img, 0.05, 0.1)
    assert sup.tolist() == [0, 0] and con.tolist() == [0, 1]


def test_hinge_points_use_fallback():
    from artsfm.articulate import label_dense
    s = make_dense_scene('revolute')
    geom, T = to_model_frame(s)
    tp, tl = sample_tracks(s)
    radius = float(np.sqrt(((tp - tp.mean(0)) ** 2).sum(1).mean()))
    views = label_dense(geom, s['alphas'], s['states'], T, tp, tl, radius)
    v = views[0]
    # model-frame hinge line: transform world axis into view-0 frame
    c0inv = np.linalg.inv(s['cam2w'][0])
    o = c0inv[:3, :3] @ s['axis_o'] + c0inv[:3, 3]
    d = c0inv[:3, :3] @ s['axis_d']
    moved = np.linalg.norm(v['pts'] @ T[:3, :3].T + T[:3, 3] - v['pts'], axis=1)
    near = moved < 0.004   # both hypotheses predict the same pixel (1 px is about 0.025 at this depth)
    assert near.sum() > 0
    assert v['fallback'][near].all()
```

- [ ] **Step 3: Run tests to verify they fail**

Run: `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -m pytest tests/test_label_dense.py -q`
Expected: FAIL with ImportError (`artsfm.articulate` missing).

- [ ] **Step 4: Implement the labelling half of `artsfm/articulate.py`**

```python
"""Turn a found joint into an articulated model: dense body/part labelling, fusion, model folder.

Dense labelling tests every foreground pixel against two motion hypotheses (it stayed with
the body, or it moved with the part by T) using depth and colour in the opposite-state views.
Undecided points fall back to a vote of the nearest labelled sparse tracks.
"""
from dataclasses import dataclass

import numpy as np
from scipy.ndimage import maximum_filter, minimum_filter
from scipy.spatial import cKDTree


@dataclass
class LabelParams:
    delta: float = 0.08      # relative depth tolerance for support / contradiction
    ctol: float = 0.15       # max per-channel colour difference for support
    margin: int = 1          # winning score must lead by this much
    k: int = 7               # fallback: nearest tracks voting
    fb_radius: float = 0.3   # fallback: nearest track must be within this x object radius
    edge_frac: float = 0.1   # depth-discontinuity threshold (3x3 range / depth)


def reliable_mask(depth, mask, edge_frac):
    """Foreground pixels away from depth discontinuities."""
    d = np.where(mask & (depth > 0) & np.isfinite(depth), depth, 0.0)
    lo = minimum_filter(np.where(d > 0, d, np.inf), 3, mode='nearest')
    hi = maximum_filter(d, 3, mode='nearest')
    return (d > 0) & np.isfinite(lo) & (hi - lo < edge_frac * d)


def backproject(K, M, depth, mask):
    """Foreground pixels with depth -> points in the frame M maps camera coordinates into."""
    H, W = depth.shape
    v, u = np.nonzero(mask & (depth > 0) & np.isfinite(depth))
    z = depth[v, u]
    X = (np.c_[u, v, np.ones_like(z)] @ np.linalg.inv(K).T) * z[:, None]
    return X @ M[:3, :3].T + M[:3, 3], v * W + u


def evidence(P, rgb, K, model2cam, depth, rel, img, delta, ctol):
    """Support (depth and colour agree) and contradiction (free-space violation) of points P in one view."""
    n = len(P)
    Xc = P @ model2cam[:3, :3].T + model2cam[:3, 3]
    z = Xc[:, 2]
    H, W = depth.shape
    front = z > 1e-6
    u = np.full(n, -1)
    v = np.full(n, -1)
    uvw = Xc[front] @ K.T
    u[front] = np.round(uvw[:, 0] / uvw[:, 2]).astype(int)
    v[front] = np.round(uvw[:, 1] / uvw[:, 2]).astype(int)
    inside = front & (u >= 0) & (u < W) & (v >= 0) & (v < H)
    ok = np.zeros(n, bool)
    ok[inside] = rel[v[inside], u[inside]]
    D = np.zeros(n)
    D[ok] = depth[v[ok], u[ok]]
    colour = np.zeros(n, bool)
    colour[ok] = np.abs(img[v[ok], u[ok]] - rgb[ok]).max(1) < ctol
    tol = delta * z
    support = ok & (np.abs(z - D) < tol) & colour
    contra = ok & (z < D - tol)
    return support.astype(int), contra.astype(int)


def label_dense(geom, masks, states, T, track_pts, track_lab, radius, params=LabelParams()):
    """Label every foreground pixel of every view as body (0), part (1) or unlabelled (-1).

    geom: dict(K, cam2w, depths, imgs). Model frame = view-0 camera frame. T: part motion
    start -> end in the model frame. track_pts/track_lab: sparse tracks (view-0 frame) labelled
    by analyze() (0 body, 1 part, -1 outlier).
    """
    Ks, cam2w, depths, imgs = geom['K'], geom['cam2w'], geom['depths'], geom['imgs']
    states = np.asarray(states)
    F = len(Ks)
    inv0 = np.linalg.inv(cam2w[0])
    model2cam = [np.linalg.inv(cam2w[f]) @ cam2w[0] for f in range(F)]
    rel = [reliable_mask(depths[f], masks[f], params.edge_frac) for f in range(F)]
    Tinv = np.linalg.inv(T)
    track_lab = np.asarray(track_lab)
    good = track_lab >= 0
    tree = cKDTree(track_pts[good]) if good.any() else None
    tl = track_lab[good]

    views = []
    for f in range(F):
        P, pix = backproject(Ks[f], inv0 @ cam2w[f], depths[f], masks[f])
        rgb = imgs[f].reshape(-1, 3)[pix]
        M = T if states[f] == 0 else Tinv
        Q = P @ M[:3, :3].T + M[:3, 3]
        sb = np.zeros(len(P), int)
        cb = np.zeros(len(P), int)
        sp = np.zeros(len(P), int)
        cp = np.zeros(len(P), int)
        for g in np.flatnonzero(states != states[f]):
            s1, c1 = evidence(P, rgb, Ks[g], model2cam[g], depths[g], rel[g], imgs[g], params.delta, params.ctol)
            s2, c2 = evidence(Q, rgb, Ks[g], model2cam[g], depths[g], rel[g], imgs[g], params.delta, params.ctol)
            sb, cb, sp, cp = sb + s1, cb + c1, sp + s2, cp + c2
        score_b, score_p = sb - cb, sp - cp
        is_body = (score_b - score_p >= params.margin) & (sb > 0)
        is_part = (score_p - score_b >= params.margin) & (sp > 0)
        label = np.full(len(P), -1)
        label[is_body] = 0
        label[is_part] = 1
        fallback = ~(is_body | is_part)
        if fallback.any() and tree is not None:
            k = min(params.k, len(tl))
            dist, nn = tree.query(P[fallback], k=k)
            dist = np.asarray(dist).reshape(fallback.sum(), -1)
            nn = np.asarray(nn).reshape(fallback.sum(), -1)
            vote = (tl[nn].mean(1) > 0.5).astype(int)
            vote[dist[:, 0] > params.fb_radius * radius] = -1
            label[fallback] = vote
        views.append(dict(pts=P, rgb=rgb, label=label, fallback=fallback, pix=pix))
    return views
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -m pytest tests/test_label_dense.py -q`
Expected: 7 passed. If an accuracy test misses its target by a small margin, do **not** loosen the test. Record a ruling and let Task 3's calibration decide the defaults.

- [ ] **Step 6: Commit**

```bash
git add tests/synth_dense.py artsfm/articulate.py tests/test_label_dense.py
git commit -m "Add dense synthetic scenes and motion-hypothesis labelling"
```

---

### Task 2: Joint model, fusion, hulls and the model folder

**Files:**
- Create: `artsfm/model_io.py`, `tests/test_model_io.py`
- Modify: `artsfm/articulate.py` (append)

**Interfaces:**
- Consumes: Task 1.
- Produces:
  - `artsfm.articulate.rotation_about(axis, ang) -> (3,3)`.
  - `joint_transform(joint, value) -> (4,4)`.
  - `joint_from_result(result) -> dict` (the `model.json` joint block).
  - `fuse(views, states, T, max_points=150_000) -> dict(base=(P,C), part=(P,C))`.
  - `convex_hull_or_none(P) -> trimesh.Trimesh | None`.
  - `estimate_up(cam2w) -> (3,)`.
  - `artsfm.model_io.write_ply(path, P, C)` and `read_ply(path) -> (P, C)`.
  - `write_model(model_dir, model, clouds, hulls)`.
  - `load_model(model_dir) -> dict`, which is `model.json` plus `dir`, `base`/`part` = dict(pts, rgb), `base_hull`/`part_hull` (trimesh or None), and `motion` as an ndarray.

- [ ] **Step 1: Write failing tests `tests/test_model_io.py`**

```python
import json
import numpy as np
import pytest

from tests.synth import make_scene


def test_ply_roundtrip_including_empty(tmp_path):
    from artsfm.model_io import write_ply, read_ply
    P = np.random.default_rng(0).standard_normal((50, 3))
    C = np.random.default_rng(1).uniform(0, 1, (50, 3))
    write_ply(tmp_path / 'a.ply', P, C)
    P2, C2 = read_ply(tmp_path / 'a.ply')
    np.testing.assert_allclose(P2, P, atol=1e-6)
    np.testing.assert_allclose(C2, C, atol=1 / 255)
    write_ply(tmp_path / 'e.ply', np.zeros((0, 3)), np.zeros((0, 3)))
    assert read_ply(tmp_path / 'e.ply')[0].shape == (0, 3)


@pytest.mark.parametrize('value', [-1.0, 0.0, 0.3, 2.5])
def test_joint_transform_is_rigid_any_value(value):
    from artsfm.articulate import joint_transform
    for joint in (dict(type='revolute', axis_origin=[1, 2, 3], axis_dir=[0, 0.6, 0.8], observed=0.6),
                  dict(type='prismatic', axis_origin=None, axis_dir=[1, 0, 0], observed=0.2),
                  dict(type='free', axis_origin=[0, 0, 1], axis_dir=[0, 1, 0], observed=0.5, disp_along=0.1)):
        T = joint_transform(joint, value)
        np.testing.assert_allclose(T[:3, :3] @ T[:3, :3].T, np.eye(3), atol=1e-12)
        assert abs(np.linalg.det(T[:3, :3]) - 1) < 1e-12
    T = joint_transform(dict(type='revolute', axis_origin=[1, 2, 3], axis_dir=[0, 0.6, 0.8], observed=0.6), value)
    np.testing.assert_allclose(T[:3, :3] @ [1, 2, 3] + T[:3, 3], [1, 2, 3], atol=1e-12)  # axis point fixed


def test_joint_from_result_reproduces_relative_motion():
    from artsfm.articulate import joint_from_result, joint_transform
    from artsfm.factorize import analyze, relative_motion
    for motion in ('revolute', 'prismatic'):
        W, states, labels, gt = make_scene(motion)
        r = analyze(W, states)
        j = joint_from_result(r)
        assert j['lower'] == 0 and j['upper'] == j['observed'] > 0
        T = joint_transform(j, j['observed'])
        A, b = relative_motion(W, states, np.flatnonzero(r['labels'] == 0), np.flatnonzero(r['labels'] == 1))
        np.testing.assert_allclose(T[:3, :3], A, atol=1e-6)
        np.testing.assert_allclose(T[:3, 3], b, atol=1e-6)


def test_fuse_maps_end_view_part_points_back():
    from artsfm.articulate import fuse, joint_transform
    T = joint_transform(dict(type='prismatic', axis_origin=None, axis_dir=[1, 0, 0], observed=0.5), 0.5)
    rng = np.random.default_rng(0)
    lid = rng.uniform(0, 1, (400, 3))
    views = [dict(pts=lid, rgb=np.ones((400, 3)), label=np.ones(400, int)),
             dict(pts=lid + [0.5, 0, 0], rgb=np.ones((400, 3)), label=np.ones(400, int))]
    out = fuse(views, np.array([0, 1]), T, max_points=10_000)
    P = out['part'][0]
    from scipy.spatial import cKDTree
    assert cKDTree(lid).query(P)[0].max() < 0.05
    assert out['base'][0].shape == (0, 3)


def test_fuse_downsamples_to_max_points():
    from artsfm.articulate import fuse
    rng = np.random.default_rng(0)
    P = rng.uniform(0, 1, (20000, 3))
    out = fuse([dict(pts=P, rgb=np.ones((20000, 3)), label=np.zeros(20000, int))], np.array([0]), np.eye(4), max_points=3000)
    assert 0 < len(out['base'][0]) <= 3000


def test_hull_none_for_degenerate():
    from artsfm.articulate import convex_hull_or_none
    assert convex_hull_or_none(np.zeros((3, 3))) is None
    assert convex_hull_or_none(np.c_[np.random.rand(50, 2), np.zeros(50)]) is None
    assert convex_hull_or_none(np.random.rand(50, 3)).is_watertight


def test_estimate_up_from_upright_cameras():
    from artsfm.articulate import estimate_up
    from tests.synth_dense import look_at
    cams = np.stack([look_at(np.array([3 * np.cos(a), 3 * np.sin(a), 1.0]), np.zeros(3)) for a in np.linspace(0, 1, 5)])
    up = estimate_up(cams)
    world_up_in_view0 = cams[0][:3, :3].T @ np.array([0, 0, 1.0])
    assert np.degrees(np.arccos(np.clip(up @ world_up_in_view0, -1, 1))) < 25


def test_model_roundtrip(tmp_path):
    from artsfm.model_io import write_model, load_model
    from artsfm.articulate import convex_hull_or_none
    P = np.random.rand(100, 3)
    clouds = dict(base=(P, np.ones((100, 3))), part=(np.zeros((0, 3)), np.zeros((0, 3))))
    hulls = dict(base=convex_hull_or_none(P), part=None)
    model = dict(object='x', joint=dict(type='revolute', axis_origin=[0, 0, 0], axis_dir=[0, 0, 1], observed=0.5,
                                        lower=0.0, upper=0.5, unit='rad'), motion=np.eye(4), warnings=['w'])
    write_model(tmp_path / 'model', model, clouds, hulls)
    m = load_model(tmp_path / 'model')
    assert m['object'] == 'x' and m['warnings'] == ['w']
    assert m['base']['pts'].shape == (100, 3) and m['part']['pts'].shape == (0, 3)
    assert m['base_hull'] is not None and m['part_hull'] is None
    assert isinstance(m['motion'], np.ndarray)
    assert json.load(open(tmp_path / 'model' / 'model.json'))['links'] == {'base': 'base.ply', 'part': 'part.ply'}
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -m pytest tests/test_model_io.py -q`
Expected: FAIL with ImportError.

- [ ] **Step 3: Write `artsfm/model_io.py`**

```python
"""Articulated model folder: model.json + per-link point clouds (binary PLY) + convex hulls."""
import json
import os

import numpy as np
import trimesh

PLY_DTYPE = np.dtype([('x', '<f4'), ('y', '<f4'), ('z', '<f4'), ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')])


def write_ply(path, P, C):
    P = np.asarray(P, float).reshape(-1, 3)
    C = np.asarray(C, float).reshape(-1, 3)
    data = np.zeros(len(P), PLY_DTYPE)
    data['x'], data['y'], data['z'] = P.T
    rgb = np.clip(np.round(C * 255), 0, 255).astype(np.uint8)
    data['red'], data['green'], data['blue'] = rgb.T
    header = ('ply\nformat binary_little_endian 1.0\n'
              f'element vertex {len(P)}\n'
              'property float x\nproperty float y\nproperty float z\n'
              'property uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n')
    with open(path, 'wb') as f:
        f.write(header.encode('ascii'))
        f.write(data.tobytes())


def read_ply(path):
    with open(path, 'rb') as f:
        n = None
        while True:
            line = f.readline().decode('ascii').strip()
            if line.startswith('element vertex'):
                n = int(line.split()[-1])
            if line == 'end_header':
                break
        data = np.frombuffer(f.read(n * PLY_DTYPE.itemsize), PLY_DTYPE, count=n)
    P = np.stack([data['x'], data['y'], data['z']], 1).astype(float)
    C = np.stack([data['red'], data['green'], data['blue']], 1).astype(float) / 255
    return P, C


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


def write_model(model_dir, model, clouds, hulls):
    model_dir = str(model_dir)
    os.makedirs(model_dir, exist_ok=True)
    for name in ('base', 'part'):
        write_ply(os.path.join(model_dir, f'{name}.ply'), *clouds[name])
        hull_path = os.path.join(model_dir, f'{name}_hull.obj')
        if hulls.get(name) is not None:
            hulls[name].export(hull_path)
        elif os.path.exists(hull_path):
            os.remove(hull_path)
    model = dict(model, links={'base': 'base.ply', 'part': 'part.ply'})
    with open(os.path.join(model_dir, 'model.json'), 'w') as f:
        json.dump(_jsonable(model), f, indent=2)


def load_model(model_dir):
    model_dir = str(model_dir)
    with open(os.path.join(model_dir, 'model.json')) as f:
        m = json.load(f)
    m['dir'] = model_dir
    m['motion'] = np.asarray(m.get('motion', np.eye(4)), float)
    for name in ('base', 'part'):
        P, C = read_ply(os.path.join(model_dir, f'{name}.ply'))
        m[name] = dict(pts=P, rgb=C)
        hull_path = os.path.join(model_dir, f'{name}_hull.obj')
        m[f'{name}_hull'] = trimesh.load(hull_path, force='mesh') if os.path.exists(hull_path) else None
    return m
```

- [ ] **Step 4: Append the joint, fusion, hull and up-vector code to `artsfm/articulate.py`**

```python
def rotation_about(axis, ang):
    d = np.asarray(axis, float)
    d = d / np.linalg.norm(d)
    K = np.array([[0, -d[2], d[1]], [d[2], 0, -d[0]], [-d[1], d[0], 0]])
    return np.eye(3) + np.sin(ang) * K + (1 - np.cos(ang)) * K @ K


def joint_transform(joint, value):
    """Rigid 4x4 motion of the part at joint value `value` (0 = start state), model frame."""
    T = np.eye(4)
    d = np.asarray(joint['axis_dir'], float)
    d = d / np.linalg.norm(d)
    if joint['type'] == 'prismatic':
        T[:3, 3] = value * d
        return T
    o = np.asarray(joint['axis_origin'], float)
    R = rotation_about(d, value)
    T[:3, :3] = R
    T[:3, 3] = o - R @ o
    if joint['type'] == 'free' and joint.get('observed'):
        T[:3, 3] += (value / joint['observed']) * joint.get('disp_along', 0.0) * d
    return T


def joint_from_result(result):
    """model.json joint block from an analyze() result (status 'ok')."""
    jtype = result['type']
    d = np.asarray(result['axis_d'], float)
    d = d / np.linalg.norm(d)
    if jtype == 'prismatic':
        observed = float(result['disp'])
        return dict(type='prismatic', axis_origin=None, axis_dir=d, observed=observed, lower=0.0,
                    upper=observed, unit='recon', approximate=False, disp_along=0.0)
    observed = float(np.deg2rad(result['angle_deg']))
    free = jtype == 'free'
    return dict(type=jtype, axis_origin=np.asarray(result['axis_o'], float), axis_dir=d, observed=observed,
                lower=0.0, upper=observed, unit='rad', approximate=free,
                disp_along=float(result['disp']) if free else 0.0)


def _statistical_inliers(P, k=16, nsig=2.5):
    if len(P) <= k:
        return np.ones(len(P), bool)
    d, _ = cKDTree(P).query(P, k=k + 1)
    m = d[:, 1:].mean(1)
    return m < m.mean() + nsig * m.std()


def _voxel_downsample(P, C, voxel):
    keys = np.floor(P / voxel).astype(np.int64)
    _, inv, cnt = np.unique(keys, axis=0, return_inverse=True, return_counts=True)
    inv = inv.ravel()
    Ps = np.zeros((len(cnt), 3))
    Cs = np.zeros((len(cnt), 3))
    np.add.at(Ps, inv, P)
    np.add.at(Cs, inv, C)
    return Ps / cnt[:, None], Cs / cnt[:, None]


def _downsample_to(P, C, max_points):
    if len(P) <= max_points:
        return P, C
    voxel = float(np.ptp(P, 0).max()) / 2000
    while True:
        Ps, Cs = _voxel_downsample(P, C, voxel)
        if len(Ps) <= max_points:
            return Ps, Cs
        voxel *= 1.25


def fuse(views, states, T, max_points=150_000):
    """Body points as observed; part points from end views mapped back by T^-1 (joint value 0)."""
    Tinv = np.linalg.inv(T)
    groups = {'base': ([], []), 'part': ([], [])}
    for f, v in enumerate(views):
        for name, lab in (('base', 0), ('part', 1)):
            m = v['label'] == lab
            P = v['pts'][m]
            if name == 'part' and states[f] == 1:
                P = P @ Tinv[:3, :3].T + Tinv[:3, 3]
            groups[name][0].append(P)
            groups[name][1].append(v['rgb'][m])
    out = {}
    for name, (Ps, Cs) in groups.items():
        P = np.concatenate(Ps) if Ps else np.zeros((0, 3))
        C = np.concatenate(Cs) if Cs else np.zeros((0, 3))
        keep = _statistical_inliers(P)
        out[name] = _downsample_to(P[keep], C[keep], max_points)
    return out


def convex_hull_or_none(P):
    import trimesh
    P = np.asarray(P, float)
    if len(P) < 4:
        return None
    sv = np.linalg.svd(P - P.mean(0), compute_uv=False)
    if sv[-1] < 1e-6 * sv[0]:
        return None
    return trimesh.convex.convex_hull(P)


def estimate_up(cam2w):
    """Gravity-up guess in the view-0 frame: mean of the cameras' up axes (-y in OpenCV)."""
    R0 = cam2w[0][:3, :3]
    ups = np.stack([R0.T @ c[:3, :3] @ np.array([0, -1.0, 0]) for c in cam2w])
    u = ups.mean(0)
    return u / np.linalg.norm(u)
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -m pytest tests/test_model_io.py tests/test_label_dense.py -q`
Expected: all pass.

- [ ] **Step 6: Commit**

```bash
git add artsfm/model_io.py artsfm/articulate.py tests/test_model_io.py
git commit -m "Add joint model, fusion, hulls and model folder IO"
```

---

### Task 3: Calibrate labelling thresholds on synthetic scenes

**Files:**
- Create: `tests/calibrate_labelling.py`
- Modify: `artsfm/articulate.py` (`LabelParams` defaults only if calibration says so), spec §4 table

**Interfaces:** Consumes `label_dense`, `make_dense_scene`, `to_model_frame`, `sample_tracks`. Produces the calibrated defaults.

- [ ] **Step 1: Write `tests/calibrate_labelling.py`**

```python
"""Sweep LabelParams on synthetic dense scenes (not a pytest test). Prints accuracy on labelled points / unlabelled fraction."""
import itertools
import numpy as np

from artsfm.articulate import LabelParams, label_dense
from tests.synth_dense import make_dense_scene, to_model_frame, sample_tracks

NOISES = [0.0, 0.01, 0.02]


def score(params, motion, noise, seeds=(0, 1, 2)):
    accs, unl = [], []
    for seed in seeds:
        s = make_dense_scene(motion, noise=noise, seed=seed)
        geom, T = to_model_frame(s)
        tp, tl = sample_tracks(s, seed=seed)
        radius = float(np.sqrt(((tp - tp.mean(0)) ** 2).sum(1).mean()))
        views = label_dense(geom, s['alphas'], s['states'], T, tp, tl, radius, params)
        ok = n_lab = n_all = 0
        for f, v in enumerate(views):
            gt = s['labels'][f].ravel()[v['pix']]
            m = v['label'] >= 0
            ok += (v['label'][m] == gt[m]).sum()
            n_lab += m.sum()
            n_all += len(m)
        accs.append(ok / n_lab)
        unl.append(1 - n_lab / n_all)
    return float(np.mean(accs)), float(np.mean(unl))


if __name__ == '__main__':
    print('delta ctol margin | ' + ' '.join(f'{m[:3]}{n}' for m in ('revolute', 'prismatic') for n in NOISES) + ' | mean')
    for delta, ctol, margin in itertools.product([0.03, 0.05, 0.08], [0.1, 0.15, 0.25], [1, 2]):
        p = LabelParams(delta=delta, ctol=ctol, margin=margin)
        accs = [score(p, m, n)[0] for m in ('revolute', 'prismatic') for n in NOISES]
        print(f'{delta:5} {ctol:4} {margin:6} | ' + ' '.join(f'{a:.3f}' for a in accs) + f' | {np.mean(accs):.3f}', flush=True)
```

- [ ] **Step 2: Run calibration** (several minutes; run in the background)

Run: `python3 -m tests.calibrate_labelling | tee .superpowers/sdd/2026-10-05-articulated-model-viewer/calibrate_labelling.txt`
Expected: an 18-row table. Pick the row with the highest mean, breaking ties toward the current defaults.

- [ ] **Step 3: Set the chosen values as `LabelParams` defaults** (if different). Rerun `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -m pytest tests/test_label_dense.py -q` and expect all to pass.

- [ ] **Step 4: Record the table and chosen values in spec §4** (replace the prototype table's caption with "calibrated"), then commit:

```bash
git add tests/calibrate_labelling.py artsfm/articulate.py docs/superpowers/specs/2026-10-05-articulated-model-viewer-design.md
git commit -m "Calibrate dense labelling thresholds on synthetic scenes"
```

---

### Task 4: Foreground masks and richer geometry from MASt3R

**Files:**
- Modify: `artsfm/reconstruct.py` (`extract_geometry` also returns `imgs` and `confs`), `artsfm/articulate.py` (append `crop_like_dust3r`, `foreground_masks`)
- Create: `tests/test_masks.py`
- Modify: `tests/test_reconstruct.py` (assert the new keys)

**Interfaces:**
- Produces:
  - `extract_geometry(scene) -> dict(K, cam2w, depths, imgs (list (H,W,3) float), confs (list (H,W)))`.
  - `artsfm.articulate.crop_like_dust3r(img: PIL.Image, size=512, patch_size=16) -> PIL.Image`.
  - `foreground_masks(paths, model_hw, confs=None, conf_thr=1.5) -> (masks list[(H,W) bool], source str)`.

- [ ] **Step 1: Write failing tests `tests/test_masks.py`**

```python
import numpy as np
from PIL import Image

OBJ = '/home/robotlab/scratch/GaussianArt/data/MPArt-90/extracted/StorageFurniture_45166'


def test_crop_like_dust3r_matches_load_images(tmp_path):
    import mast3r.utils.path_to_dust3r  # noqa
    from dust3r.utils.image import load_images
    from artsfm.articulate import crop_like_dust3r
    rng = np.random.default_rng(0)
    for wh in ((800, 800), (640, 480), (300, 500)):
        arr = rng.integers(0, 255, (wh[1], wh[0], 3), dtype=np.uint8)
        p = tmp_path / f'{wh[0]}x{wh[1]}.png'
        Image.fromarray(arr).save(p)
        ref = load_images([str(p)], size=512, verbose=False)[0]['img'][0].permute(1, 2, 0).numpy() * 0.5 + 0.5
        ours = np.asarray(crop_like_dust3r(Image.open(p).convert('RGB')), float) / 255
        assert ours.shape == ref.shape
        assert np.abs(ours - ref).max() < 1.5 / 255


def test_foreground_masks_from_alpha(tmp_path):
    from artsfm.articulate import foreground_masks
    rgba = np.zeros((800, 800, 4), np.uint8)
    rgba[300:500, 200:600, 3] = 255
    p = tmp_path / 'a.png'
    Image.fromarray(rgba).save(p)
    masks, source = foreground_masks([str(p)], (384, 512))
    assert source == 'alpha' and masks[0].shape == (384, 512)
    # centre is foreground, corners are not
    assert masks[0][192, 256] and not masks[0][5, 5]


def test_foreground_masks_confidence_fallback(tmp_path):
    from artsfm.articulate import foreground_masks
    p = tmp_path / 'rgb.png'
    Image.fromarray(np.zeros((480, 640, 3), np.uint8)).save(p)
    conf = np.ones((384, 512))
    conf[100:200, 100:200] = 3.0
    masks, source = foreground_masks([str(p)], (384, 512), confs=[conf])
    assert source == 'confidence'
    assert masks[0].sum() == 100 * 100
```

Also extend `tests/test_reconstruct.py::test_two_view_reconstruction` with:

```python
    assert len(g['imgs']) == 2 and g['imgs'][0].shape == (H, W, 3)
    assert len(g['confs']) == 2 and g['confs'][0].shape == (H, W)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -m pytest tests/test_masks.py -q`
Expected: FAIL with ImportError on `crop_like_dust3r`.

- [ ] **Step 3: Append to `artsfm/articulate.py`**

```python
def crop_like_dust3r(img, size=512, patch_size=16):
    """Resize + centre-crop exactly as dust3r.utils.image.load_images does (size != 224)."""
    from PIL import Image, ImageOps
    img = ImageOps.exif_transpose(img)
    S = max(img.size)
    interp = Image.LANCZOS if S > size else Image.BICUBIC
    img = img.resize(tuple(int(round(x * size / S)) for x in img.size), interp)
    W, H = img.size
    cx, cy = W // 2, H // 2
    halfw = ((2 * cx) // patch_size) * patch_size / 2
    halfh = ((2 * cy) // patch_size) * patch_size / 2
    if W == H:
        halfh = 3 * halfw / 4
    return img.crop((cx - halfw, cy - halfh, cx + halfw, cy + halfh))


def foreground_masks(paths, model_hw, confs=None, conf_thr=1.5):
    """Object masks at model resolution: image alpha when present, else MASt3R confidence."""
    from PIL import Image
    masks, sources = [], set()
    for i, p in enumerate(paths):
        im = Image.open(p)
        if im.mode in ('RGBA', 'LA') or 'transparency' in im.info:
            a = np.asarray(crop_like_dust3r(im.convert('RGBA').getchannel('A')))
            m = a > 0
            sources.add('alpha')
        elif confs is not None:
            m = np.asarray(confs[i]) > conf_thr
            sources.add('confidence')
        else:
            m = np.ones(model_hw, bool)
            sources.add('none')
        if m.shape != tuple(model_hw):
            raise ValueError(f'mask for {p} has shape {m.shape}, expected {tuple(model_hw)}')
        masks.append(m)
    return masks, '+'.join(sorted(sources))
```

- [ ] **Step 4: Modify `extract_geometry` in `artsfm/reconstruct.py`**

```python
def extract_geometry(scene):
    with torch.no_grad():
        _, depthmaps, confs = scene.get_dense_pts3d(clean_depth=False)
    K = np.stack([k.detach().cpu().numpy() for k in scene.intrinsics])
    cam2w = scene.get_im_poses().detach().cpu().numpy()
    shapes = [im.shape[:2] for im in scene.imgs]
    depths = [d.detach().cpu().numpy().reshape(s) for d, s in zip(depthmaps, shapes)]
    confs = [c.detach().cpu().numpy().reshape(s) for c, s in zip(confs, shapes)]
    imgs = [np.asarray(im, float) for im in scene.imgs]
    return dict(K=K, cam2w=cam2w, depths=depths, imgs=imgs, confs=confs)
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -m pytest tests/test_masks.py tests/test_reconstruct.py -q`
Expected: all pass (the reconstruct tests take about 1 min on CPU).

- [ ] **Step 6: Commit**

```bash
git add artsfm/articulate.py artsfm/reconstruct.py tests/test_masks.py tests/test_reconstruct.py
git commit -m "Add dust3r-consistent foreground masks; expose images and confidences"
```

---

### Task 5: Assemble the model, `--build_model`, and `build_model.py`

**Files:**
- Modify: `artsfm/articulate.py` (append `assemble_model`), `run_articulation.py` (refactor + flag), `tests/test_driver.py`
- Create: `build_model.py`, `tests/test_assemble.py`

**Interfaces:**
- Consumes: Tasks 1–4, `analyze`, `as_views`, `align_poses`.
- Produces:
  - `assemble_model(geom, paths, states, W, result, out_dir, obj_name, params=LabelParams(), frames_by_path=None, max_points=150_000) -> model_dir | None`.
  - `run_articulation.reconstruct_and_track(paths, cache, weights, device, optim_level, niter1, niter2, fov_x) -> (geom, W)`.
  - `run(..., build_model=False)`. `metrics` gains `data_dir` and `seed_view`.
  - CLI: `build_model.py --out_dir D [--data_dir P] [--weights W] [--device cpu] [--estimate_intrinsics]`.

- [ ] **Step 1: Write failing tests `tests/test_assemble.py`**

```python
import json
import numpy as np

from tests.synth_dense import make_dense_scene, to_model_frame, sample_tracks


def _synthetic_inputs(motion='revolute', n_tracks=150):
    s = make_dense_scene(motion)
    geom, T = to_model_frame(s)
    geom['confs'] = [np.full(d.shape, 3.0) for d in s['depths']]
    tp, tl = sample_tracks(s, n=n_tracks)
    W = np.concatenate([tp.T] + [np.zeros((3, len(tp)))] * (len(s['states']) - 1))  # only view-0 rows are read
    c0inv = np.linalg.inv(s['cam2w'][0])
    if motion == 'revolute':
        o = c0inv[:3, :3] @ s['axis_o'] + c0inv[:3, 3]
        result = dict(status='ok', reason='', type='revolute', labels=tl, axis_o=o,
                      axis_d=c0inv[:3, :3] @ s['axis_d'], angle_deg=40.0, disp=0.0)
    else:
        result = dict(status='ok', reason='', type='prismatic', labels=tl, axis_o=None,
                      axis_d=c0inv[:3, :3] @ s['axis_d'], angle_deg=0.0, disp=0.3)
    return s, geom, W, result


def test_assemble_writes_consistent_model(tmp_path, monkeypatch):
    import artsfm.articulate as art
    s, geom, W, result = _synthetic_inputs()
    paths = [str(tmp_path / f'v{i}.png') for i in range(len(s['states']))]
    monkeypatch.setattr(art, 'foreground_masks', lambda p, hw, confs=None: (s['alphas'], 'alpha'))
    model_dir = art.assemble_model(geom, paths, s['states'], W, result, str(tmp_path), 'synthetic')
    from artsfm.model_io import load_model
    m = load_model(model_dir)
    assert m['joint']['type'] == 'revolute' and abs(m['joint']['upper'] - np.deg2rad(40)) < 1e-9
    assert len(m['base']['pts']) > 500 and len(m['part']['pts']) > 500
    assert m['base_hull'] is not None and m['part_hull'] is not None
    lab = m['labelling']
    assert lab['mask_source'] == 'alpha' and lab['n_base'] > lab['n_part'] > 0
    assert not m['warnings']


def test_assemble_refuses_non_ok_result(tmp_path, capsys):
    from artsfm.articulate import assemble_model
    out = assemble_model(None, [], [], None, dict(status='rigid', reason='part has 3 inliers'), str(tmp_path), 'x')
    assert out is None
    assert 'no articulation to build: rigid' in capsys.readouterr().out
    assert not (tmp_path / 'model').exists()


def test_assemble_tiny_part_warns_and_no_hull(tmp_path, monkeypatch):
    import artsfm.articulate as art
    s, geom, W, result = _synthetic_inputs()
    paths = [str(tmp_path / f'v{i}.png') for i in range(len(s['states']))]
    monkeypatch.setattr(art, 'foreground_masks', lambda p, hw, confs=None: (s['alphas'], 'alpha'))
    real = art.label_dense

    def no_part(*a, **k):
        views = real(*a, **k)
        for v in views:
            v['label'][v['label'] == 1] = 0
        return views
    monkeypatch.setattr(art, 'label_dense', no_part)
    from artsfm.model_io import load_model
    m = load_model(art.assemble_model(geom, paths, s['states'], W, result, str(tmp_path), 'x'))
    assert 'part sparsely observed' in m['warnings']
    assert m['part_hull'] is None
```

Update `tests/test_driver.py` so the monkeypatched `run_sparse_ga`/`extract_geometry`/`load_pair_correspondences` still apply (they live in `run_articulation`'s namespace, which `reconstruct_and_track` uses). Add:

```python
def test_driver_records_data_dir_and_seed(tmp_path, monkeypatch):
    import run_articulation as ra
    monkeypatch.setattr(ra, 'load_frames', lambda d, s: [(f'/x/{s}{i}.png', np.eye(4)) for i in range(10)])
    monkeypatch.setattr(ra, 'load_fov', lambda d: 0.7)
    monkeypatch.setattr(ra, 'select_views', lambda s, e, k, seed_view: ([f'/x/{i}.png' for i in range(2 * k)], np.array([0] * k + [1] * k)))
    monkeypatch.setattr(ra, 'reconstruct_and_track', lambda *a, **kw: (dict(K=None, cam2w=None, depths=None), np.zeros((30, 50))))
    monkeypatch.setattr(ra, 'analyze', lambda W, states, params: dict(status='rigid', reason='r', type='rigid', labels=np.zeros(50, int), axis_d=None))
    monkeypatch.setattr(ra, 'evaluate', lambda *a: {})
    monkeypatch.setattr(ra, 'write_outputs', lambda *a: None)
    out = ra.run('/data', str(tmp_path), 'w.pth', k=5, seed_view=3, device='cpu')
    assert out['metrics']['data_dir'] == '/data' and out['metrics']['seed_view'] == 3
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -m pytest tests/test_assemble.py tests/test_driver.py -q`
Expected: FAIL with ImportError on `assemble_model` and AttributeError on `reconstruct_and_track`.

- [ ] **Step 3: Append `assemble_model` to `artsfm/articulate.py`**

```python
def assemble_model(geom, paths, states, W, result, out_dir, obj_name, params=LabelParams(),
                   frames_by_path=None, max_points=150_000):
    """Dense-label, fuse and write <out_dir>/model/. Returns the model dir, or None if nothing to build."""
    import os
    from artsfm.factorize import as_views
    from artsfm.model_io import write_model

    if result.get('status') != 'ok':
        print(f"no articulation to build: {result.get('status')} — {result.get('reason', '')}")
        return None
    states = np.asarray(states)
    joint = joint_from_result(result)
    T = joint_transform(joint, joint['observed'])
    masks, source = foreground_masks(paths, geom['depths'][0].shape, geom.get('confs'))
    labels = np.asarray(result['labels'])
    P0 = as_views(W)[0]
    base_pts = P0[labels == 0]
    radius = float(np.sqrt(((base_pts - base_pts.mean(0)) ** 2).sum(1).mean()))
    views = label_dense(geom, masks, states, T, P0, labels, radius, params)
    clouds = fuse(views, states, T, max_points)
    hulls = {name: convex_hull_or_none(clouds[name][0]) for name in ('base', 'part')}

    n_all = sum(len(v['label']) for v in views)
    n_unl = sum(int((v['label'] == -1).sum()) for v in views)
    warnings = []
    if len(clouds['part'][0]) < 500:
        warnings.append('part sparsely observed')
    if n_all and n_unl > 0.4 * n_all:
        warnings.append('many points unlabelled')
    if joint['type'] == 'free':
        warnings.append('screw joint (not representable in URDF); URDF export disabled')
    for name in ('base', 'part'):
        if hulls[name] is None:
            warnings.append(f'no {name} hull (too few or coplanar points)')

    gt_frame = None
    if frames_by_path is not None:
        from artsfm.evaluate import align_poses
        s, R, t = align_poses(geom['cam2w'], np.array([frames_by_path[p] for p in paths]))
        up = geom['cam2w'][0][:3, :3].T @ R.T @ np.array([0, 0, 1.0])
        gt_frame = dict(up=up / np.linalg.norm(up), scale=s, sim_R=R, sim_t=t, cam2w0=geom['cam2w'][0])

    model = dict(
        object=obj_name, frame='view0_camera_opencv', units='reconstruction', joint=joint, motion=T,
        up=estimate_up(geom['cam2w']), gt_frame=gt_frame,
        labelling=dict(method='motion_hypothesis', fallback='knn_tracks', **vars(params),
                       n_base=int(sum((v['label'] == 0).sum() for v in views)),
                       n_part=int(sum((v['label'] == 1).sum() for v in views)),
                       n_fallback=int(sum(v['fallback'].sum() for v in views)),
                       n_unlabelled=n_unl, mask_source=source),
        warnings=warnings,
        source=dict(result='../result.json', views=[os.path.basename(p) for p in paths]))
    model_dir = os.path.join(out_dir, 'model')
    write_model(model_dir, model, clouds, hulls)
    return model_dir
```

- [ ] **Step 4: Refactor `run_articulation.py`.** Replace `_tracks_for` with `reconstruct_and_track` plus a thin selector, and add the flag:

```python
def reconstruct_and_track(paths, cache, weights, device, optim_level, niter1, niter2, fov_x):
    scene = run_sparse_ga(paths, cache, weights, device=device, optim_level=optim_level,
                          niter1=niter1, niter2=niter2, fov_x=fov_x)
    geom = extract_geometry(scene)
    tracks = build_tracks(load_pair_correspondences(cache, paths), len(paths))
    W = (lift_tracks(tracks, geom['K'], geom['depths'])[0] if len(tracks)
         else np.zeros((3 * len(paths), 0)))
    return geom, W


def _tracks_for(data_dir, out_dir, weights, k, seed_view, optim_level, device, niter1, niter2, fov_x):
    start, end = load_frames(data_dir, 'start'), load_frames(data_dir, 'end')
    paths, states = select_views(start, end, k=k, seed_view=seed_view)
    geom, W = reconstruct_and_track(paths, os.path.join(out_dir, f'cache_k{k}'), weights, device,
                                    optim_level, niter1, niter2, fov_x)
    return paths, states, geom, W, {p: T for p, T in start + end}
```

In `run(...)`, add the parameter `build_model=False`. After `metrics['views'] = ...`, add:

```python
    metrics['data_dir'] = data_dir
    metrics['seed_view'] = seed_view
```

After `write_outputs(...)`, add:

```python
    if build_model:
        from artsfm.articulate import assemble_model
        assemble_model(geom, paths, states, W, result, out_dir, os.path.basename(os.path.normpath(data_dir)),
                       frames_by_path=frames_by_path)
```

In `main()`, add `p.add_argument('--build_model', action='store_true', help='also build the articulated model folder')` and pass `build_model=a.build_model` to `run`.

- [ ] **Step 5: Write `build_model.py`**

```python
#!/usr/bin/env python3
"""Build <out_dir>/model/ (articulated model) from an existing run_articulation.py output directory."""
import argparse
import json
import os
import sys

import numpy as np
import torch

import run_articulation as ra
from artsfm.articulate import assemble_model
from artsfm.factorize import analyze
from artsfm.views import load_fov, load_frames


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out_dir', required=True)
    p.add_argument('--data_dir', default=None, help='defaults to the data_dir recorded in result.json')
    p.add_argument('--weights', default='checkpoints/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric.pth')
    p.add_argument('--device', default=None)
    p.add_argument('--estimate_intrinsics', action='store_true')
    a = p.parse_args()

    saved = json.load(open(os.path.join(a.out_dir, 'result.json')))
    if saved['result']['status'] != 'ok':
        print(f"no articulation to build: {saved['result']['status']} — {saved['result'].get('reason', '')}")
        sys.exit(1)
    m = saved['metrics']
    data_dir = a.data_dir or m.get('data_dir')
    if not data_dir:
        sys.exit('result.json has no data_dir; pass --data_dir')
    k = m['k']
    paths = [os.path.join(data_dir, v) for v in m['views']]
    states = np.array([0] * k + [1] * k)
    device = a.device or ('cuda' if torch.cuda.is_available() else 'cpu')
    fov = None if a.estimate_intrinsics else load_fov(data_dir)
    geom, W = ra.reconstruct_and_track(paths, os.path.join(a.out_dir, f'cache_k{k}'), a.weights, device,
                                       'refine', 300, 300, fov)
    result = analyze(W, states)
    if result['type'] != saved['result']['type']:
        print(f"note: re-analysis gives {result['type']}, result.json says {saved['result']['type']}; using re-analysis")
    frames = {pth: T for pth, T in load_frames(data_dir, 'start') + load_frames(data_dir, 'end')}
    model_dir = assemble_model(geom, paths, states, W, result, a.out_dir,
                               os.path.basename(os.path.normpath(data_dir)), frames_by_path=frames)
    if model_dir:
        mj = json.load(open(os.path.join(model_dir, 'model.json')))
        print(json.dumps(dict(model_dir=model_dir, joint=mj['joint']['type'], labelling=mj['labelling'],
                              warnings=mj['warnings']), indent=2))


if __name__ == '__main__':
    main()
```

- [ ] **Step 6: Run tests to verify they pass**

Run: `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -m pytest tests/ -q --deselect tests/test_reconstruct.py`
Expected: all pass.

- [ ] **Step 7: Commit**

```bash
git add artsfm/articulate.py run_articulation.py build_model.py tests/test_assemble.py tests/test_driver.py
git commit -m "Assemble articulated model folder; --build_model and build_model.py"
```

---

### Task 6: URDF export with MuJoCo check

**Files:**
- Create: `artsfm/urdf.py`, `tests/test_urdf.py`

**Interfaces:**
- Consumes: `load_model`, `joint_transform`, `convex_hull_or_none`, `write_model`.
- Produces:
  - `rotation_between(a, b) -> (3,3)`.
  - `export_transform(model, scale=None, use_gt=False) -> dict(R, s, t)`.
  - `apply_export(E, P) -> P'`.
  - `write_urdf(model_dir, out_dir=None, scale=None, use_gt=False, lower=None, upper=None) -> urdf_path`.
  - `check_urdf(urdf_path) -> float` (max relative error).
  - CLI `python3 -m artsfm.urdf <model_dir> [...]`.

- [ ] **Step 1: Write failing tests `tests/test_urdf.py`**

```python
import os
import numpy as np
import pytest

from artsfm.articulate import convex_hull_or_none, joint_transform
from artsfm.model_io import write_model


def _box(c, half, n=600, seed=0):
    rng = np.random.default_rng(seed)
    return np.asarray(c) + rng.uniform(-1, 1, (n, 3)) * np.asarray(half)


def _make_model(tmp_path, jtype='revolute', up=(0, -1, 0)):
    base = _box([0, 0, 3], [0.5, 0.4, 0.3])
    part = _box([0, 0.45, 2.75], [0.45, 0.05, 0.03], seed=1)
    if jtype == 'revolute':
        joint = dict(type='revolute', axis_origin=[0.45, 0.45, 2.72], axis_dir=[0, -1, 0], observed=0.7,
                     lower=0.0, upper=0.7, unit='rad', approximate=False, disp_along=0.0)
    elif jtype == 'prismatic':
        joint = dict(type='prismatic', axis_origin=None, axis_dir=[0, 0, -1], observed=0.2,
                     lower=0.0, upper=0.2, unit='recon', approximate=False, disp_along=0.0)
    else:
        joint = dict(type='free', axis_origin=[0, 0, 3], axis_dir=[0, -1, 0], observed=0.7,
                     lower=0.0, upper=0.7, unit='rad', approximate=True, disp_along=0.1)
    model = dict(object='toy', joint=joint, motion=joint_transform(joint, joint['observed']),
                 up=np.asarray(up, float), gt_frame=None, warnings=[])
    clouds = dict(base=(base, np.ones_like(base) * 0.5), part=(part, np.ones_like(part) * 0.8))
    hulls = dict(base=convex_hull_or_none(base), part=convex_hull_or_none(part))
    write_model(tmp_path / 'model', model, clouds, hulls)
    return str(tmp_path / 'model')


def test_rotation_between_maps_vectors():
    from artsfm.urdf import rotation_between
    for a, b in (([0, -1, 0], [0, 0, 1]), ([0, 0, -1], [0, 0, 1]), ([1, 2, 3], [0, 0, 1])):
        R = rotation_between(np.array(a, float), np.array(b, float))
        np.testing.assert_allclose(R @ (np.array(a) / np.linalg.norm(a)), b, atol=1e-12)
        np.testing.assert_allclose(R @ R.T, np.eye(3), atol=1e-12)


@pytest.mark.parametrize('jtype', ['revolute', 'prismatic'])
def test_urdf_loads_and_matches_motion_in_mujoco(tmp_path, jtype):
    from artsfm.urdf import write_urdf, check_urdf
    urdf = write_urdf(_make_model(tmp_path, jtype))
    assert check_urdf(urdf) < 1e-6


def test_urdf_prismatic_scaled_check(tmp_path):
    from artsfm.urdf import write_urdf, check_urdf
    urdf = write_urdf(_make_model(tmp_path, 'prismatic'), scale=2.5)
    assert check_urdf(urdf) < 1e-6
    assert 'upper="0.500000"' in open(urdf).read()


def test_export_up_maps_to_z_and_base_on_ground(tmp_path):
    from artsfm.urdf import export_transform, apply_export
    from artsfm.model_io import load_model
    m = load_model(_make_model(tmp_path))
    E = export_transform(m)
    np.testing.assert_allclose(E['R'] @ m['up'], [0, 0, 1], atol=1e-12)
    P = apply_export(E, m['base']['pts'])
    assert abs(P[:, 2].min()) < 1e-9 and np.abs(P[:, :2].mean(0)).max() < 1e-9


def test_gaussianart_names_and_paths(tmp_path):
    from artsfm.urdf import write_urdf
    urdf = write_urdf(_make_model(tmp_path))
    txt = open(urdf).read()
    for s in ('<link name="base">', '<link name="link_1">', '<joint name="joint_1" type="revolute">',
              'meshes/base.obj', 'meshes/base_collision.obj', 'meshes/link_1.obj', 'meshes/link_1_collision.obj',
              'strippath="false"'):
        assert s in txt
    for f in ('base.obj', 'base_collision.obj', 'link_1.obj', 'link_1_collision.obj'):
        assert os.path.isfile(os.path.join(os.path.dirname(urdf), 'meshes', f))


def test_free_joint_refused(tmp_path):
    from artsfm.urdf import write_urdf
    with pytest.raises(ValueError, match='screw joint'):
        write_urdf(_make_model(tmp_path, 'free'))


def test_write_urdf_refuses_without_hull(tmp_path):
    from artsfm.urdf import write_urdf
    d = _make_model(tmp_path)
    os.remove(os.path.join(d, 'part_hull.obj'))
    with pytest.raises(ValueError, match='hull'):
        write_urdf(d)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -m pytest tests/test_urdf.py -q`
Expected: FAIL with ImportError.

- [ ] **Step 3: Write `artsfm/urdf.py`**

```python
"""Export an articulated model folder as a URDF (GaussianArt conventions) and check it in MuJoCo."""
import argparse
import json
import os

import numpy as np

from artsfm.articulate import joint_transform
from artsfm.model_io import load_model


def rotation_between(a, b):
    """Smallest rotation taking unit direction a to unit direction b."""
    a = np.asarray(a, float) / np.linalg.norm(a)
    b = np.asarray(b, float) / np.linalg.norm(b)
    v, c = np.cross(a, b), float(a @ b)
    if c < -1 + 1e-12:  # opposite: rotate pi about any axis orthogonal to a
        w = np.cross(a, [1.0, 0, 0]) if abs(a[0]) < 0.9 else np.cross(a, [0, 1.0, 0])
        w /= np.linalg.norm(w)
        return 2 * np.outer(w, w) - np.eye(3)
    K = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    return np.eye(3) + K + K @ K / (1 + c)


def export_transform(model, scale=None, use_gt=False):
    """Model frame -> gravity-aligned export frame: up -> +z, base on z = 0, base x-y centroid at 0."""
    gt = model.get('gt_frame') if use_gt else None
    if use_gt and not gt:
        raise ValueError('model has no gt_frame; build it with GT poses available or drop --use_gt_frame')
    up = np.asarray(gt['up'] if gt else model['up'], float)
    s = float(scale if scale is not None else (gt['scale'] if gt else 1.0))
    R = rotation_between(up, [0, 0, 1.0])
    B = s * model['base']['pts'] @ R.T
    t = np.array([-B[:, 0].mean(), -B[:, 1].mean(), -B[:, 2].min()])
    return dict(R=R, s=s, t=t)


def apply_export(E, P):
    return E['s'] * np.asarray(P, float) @ E['R'].T + E['t']


def _fmt(v):
    return ' '.join(f'{x:.6f}' for x in v)


def _link_xml(name, mass, com):
    return [f'  <link name="{name}">',
            f'    <inertial><origin xyz="{_fmt(com)}"/><mass value="{mass}"/>'
            f'<inertia ixx="0.01" iyy="0.01" izz="0.01" ixy="0" ixz="0" iyz="0"/></inertial>',
            f'    <visual name="{name}_visual"><origin xyz="0 0 0"/><geometry><mesh filename="meshes/{name}.obj"/></geometry></visual>',
            f'    <collision name="{name}_collision"><origin xyz="0 0 0"/><geometry><mesh filename="meshes/{name}_collision.obj"/></geometry></collision>',
            '  </link>']


def write_urdf(model_dir, out_dir=None, scale=None, use_gt=False, lower=None, upper=None):
    m = load_model(model_dir)
    j = m['joint']
    if j['type'] == 'free':
        raise ValueError('screw joint (not representable in URDF); URDF export disabled')
    if m['base_hull'] is None or m['part_hull'] is None:
        raise ValueError('missing base or part hull (part too sparse or coplanar); cannot export URDF')
    E = export_transform(m, scale, use_gt)
    axis = E['R'] @ np.asarray(j['axis_dir'], float)
    axis /= np.linalg.norm(axis)
    part_c = apply_export(E, m['part']['pts']).mean(0)
    if j['type'] == 'revolute':
        o = apply_export(E, np.asarray(j['axis_origin'], float)[None])[0]
        pivot = o + axis * ((part_c - o) @ axis)
    else:
        pivot = part_c
    lo = float(j['lower'] if lower is None else lower)
    hi = float(j['upper'] if upper is None else upper)
    if j['type'] == 'prismatic':
        lo, hi = lo * E['s'], hi * E['s']

    out_dir = out_dir or model_dir
    os.makedirs(os.path.join(out_dir, 'meshes'), exist_ok=True)
    base = m['base_hull'].copy()
    base.vertices = apply_export(E, base.vertices)
    part = m['part_hull'].copy()
    part.vertices = apply_export(E, part.vertices) - pivot
    for name, mesh in (('base', base), ('link_1', part)):
        mesh.export(os.path.join(out_dir, 'meshes', f'{name}.obj'))
        mesh.export(os.path.join(out_dir, 'meshes', f'{name}_collision.obj'))

    name = m.get('object', 'object')
    xml = ['<?xml version="1.0"?>', f'<robot name="{name}">',
           '  <mujoco><compiler strippath="false" discardvisual="false"/></mujoco>']
    xml += _link_xml('base', 10.0, base.vertices.mean(0))
    xml += _link_xml('link_1', 1.0, part.vertices.mean(0))
    xml += [f'  <joint name="joint_1" type="{j["type"]}">',
            '    <parent link="base"/><child link="link_1"/>',
            f'    <origin xyz="{_fmt(pivot)}" rpy="0 0 0"/><axis xyz="{_fmt(axis)}"/>',
            f'    <limit lower="{lo:.6f}" upper="{hi:.6f}" effort="100" velocity="2"/>',
            '    <dynamics damping="0.5" friction="0"/>',
            '  </joint>', '</robot>']
    urdf = os.path.join(out_dir, f'{name}.urdf')
    with open(urdf, 'w') as f:
        f.write('\n'.join(xml) + '\n')
    with open(os.path.join(out_dir, 'export.json'), 'w') as f:
        json.dump(dict(model_dir=os.path.abspath(model_dir), R=E['R'].tolist(), s=E['s'], t=E['t'].tolist(),
                       pivot=pivot.tolist(), axis=axis.tolist(), lower=lo, upper=hi, type=j['type']), f, indent=2)
    return urdf


def check_urdf(urdf_path):
    """Load in MuJoCo, set the joint to its upper limit, compare the part pose with the model's motion."""
    import mujoco
    ex = json.load(open(os.path.join(os.path.dirname(urdf_path), 'export.json')))
    m = load_model(ex['model_dir'])
    E = dict(R=np.array(ex['R']), s=ex['s'], t=np.array(ex['t']))
    mdl = mujoco.MjModel.from_xml_path(urdf_path)
    d = mujoco.MjData(mdl)
    d.qpos[0] = ex['upper']
    mujoco.mj_kinematics(mdl, d)
    bid = mujoco.mj_name2id(mdl, mujoco.mjtObj.mjOBJ_BODY, 'link_1')
    R_mj, x_mj = d.xmat[bid].reshape(3, 3), d.xpos[bid]
    q_model = ex['upper'] if ex['type'] == 'revolute' else ex['upper'] / ex['s']
    T = joint_transform(m['joint'], q_model)
    P = m['part']['pts'][:500]
    expected = apply_export(E, P @ T[:3, :3].T + T[:3, 3])
    got = (apply_export(E, P) - np.array(ex['pivot'])) @ R_mj.T + x_mj
    extent = float(np.ptp(apply_export(E, m['base']['pts']), 0).max())
    return float(np.linalg.norm(got - expected, axis=1).max() / max(extent, 1e-12))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('model_dir')
    p.add_argument('--out_dir', default=None)
    p.add_argument('--scale', type=float, default=None)
    p.add_argument('--use_gt_frame', action='store_true', help='up and scale from the GT alignment (MPArt-90)')
    p.add_argument('--lower', type=float, default=None)
    p.add_argument('--upper', type=float, default=None)
    p.add_argument('--no_check', action='store_true')
    a = p.parse_args()
    urdf = write_urdf(a.model_dir, a.out_dir, a.scale, a.use_gt_frame, a.lower, a.upper)
    print(f'wrote {urdf}')
    if not a.no_check:
        err = check_urdf(urdf)
        print(f'MuJoCo check: max relative error {err:.2e} ({"PASS" if err < 1e-6 else "FAIL"})')


if __name__ == '__main__':
    main()
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -m pytest tests/test_urdf.py -q`
Expected: 8 passed. MuJoCo may print "duplicate name" warnings on stderr; the named visual and collision elements should prevent them.

- [ ] **Step 5: Commit**

```bash
git add artsfm/urdf.py tests/test_urdf.py
git commit -m "Add URDF export (GaussianArt conventions) with MuJoCo kinematics check"
```

---

### Task 7: glb frames and viewer helpers

**Files:**
- Create: `artsfm/viewer.py`, `tests/test_viewer.py`

**Interfaces:**
- Consumes: `load_model`, `joint_transform`.
- Produces:
  - `OPENCV_TO_GL (3,3)`.
  - `list_models(root) -> list[str]`.
  - `slider_spec(joint, extend=False) -> (lo, hi, label)`, in display units (degrees for rad).
  - `to_joint_value(joint, slider_value) -> float`.
  - `sweep_values(joint, n=12) -> list[float]`, in joint units.
  - `part_points_at(model, value) -> (n,3)`.
  - `build_glb(model, value, color_mode='rgb'|'part', show_axis=True, show_hulls=False, out_path=None) -> path`.
  - `info_markdown(model, result=None) -> str`.

- [ ] **Step 1: Write failing tests `tests/test_viewer.py`**

```python
import numpy as np
import trimesh
from scipy.spatial import cKDTree

from tests.test_urdf import _make_model


def test_build_glb_places_part_at_joint_value(tmp_path):
    from artsfm.viewer import build_glb, OPENCV_TO_GL
    from artsfm.model_io import load_model
    from artsfm.articulate import joint_transform
    m = load_model(_make_model(tmp_path))
    path = build_glb(m, 0.4, out_path=str(tmp_path / 'f.glb'))
    g = trimesh.load(path)
    T = joint_transform(m['joint'], 0.4)
    want = (m['part']['pts'] @ T[:3, :3].T + T[:3, 3]) @ OPENCV_TO_GL.T
    got = np.asarray(g.geometry['part'].vertices)
    assert cKDTree(want).query(got)[0].max() < 1e-5
    assert 'axis' in g.geometry and 'base' in g.geometry


def test_build_glb_with_hulls_and_part_colours(tmp_path):
    from artsfm.viewer import build_glb
    from artsfm.model_io import load_model
    m = load_model(_make_model(tmp_path))
    g = trimesh.load(build_glb(m, 0.0, color_mode='part', show_hulls=True, out_path=str(tmp_path / 'h.glb')))
    assert {'base_hull', 'part_hull'} <= set(g.geometry)


def test_list_models_only_valid(tmp_path):
    from artsfm.viewer import list_models
    _make_model(tmp_path / 'A')
    (tmp_path / 'B' / 'model').mkdir(parents=True)  # no model.json
    assert list_models(str(tmp_path)) == [str(tmp_path / 'A' / 'model')]


def test_slider_and_sweep_units():
    from artsfm.viewer import slider_spec, to_joint_value, sweep_values
    j = dict(type='revolute', lower=0.0, upper=np.deg2rad(40), unit='rad')
    lo, hi, label = slider_spec(j)
    assert (lo, round(hi, 6), label) == (0.0, 40.0, 'degrees')
    assert abs(to_joint_value(j, 20.0) - np.deg2rad(20)) < 1e-12
    lo, hi, _ = slider_spec(j, extend=True)
    assert lo == -20.0 and round(hi, 6) == 60.0
    v = sweep_values(j, n=12)
    assert v[0] == 0 and abs(max(v) - j['upper']) < 1e-12 and v[-1] == 0
    p = dict(type='prismatic', lower=0.0, upper=0.3, unit='recon')
    assert slider_spec(p)[2] == 'reconstruction units' and to_joint_value(p, 0.1) == 0.1


def test_info_markdown_mentions_warnings_and_metrics(tmp_path):
    from artsfm.viewer import info_markdown
    from artsfm.model_io import load_model
    m = load_model(_make_model(tmp_path))
    m['warnings'] = ['part sparsely observed']
    txt = info_markdown(m, dict(metrics=dict(axis_angle_err_deg=2.7, state_err=0.4)))
    assert 'revolute' in txt and 'part sparsely observed' in txt and '2.7' in txt
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -m pytest tests/test_viewer.py -q`
Expected: FAIL with ImportError.

- [ ] **Step 3: Write `artsfm/viewer.py`**

```python
"""Pure helpers for the articulated-model viewer: model discovery, slider units, glb frames, info text."""
import glob
import os
import tempfile

import numpy as np
import trimesh

from artsfm.articulate import joint_transform

OPENCV_TO_GL = np.diag([1.0, -1.0, -1.0])   # three.js / Babylon expect y up, camera looking down -z
BASE_RGB = np.array([29, 127, 166]) / 255
PART_RGB = np.array([217, 72, 43]) / 255
AXIS_RGBA = [242, 182, 48, 255]


def list_models(root):
    return sorted(os.path.dirname(p) for p in glob.glob(os.path.join(root, '*', 'model', 'model.json')))


def slider_spec(joint, extend=False):
    lo, hi = float(joint['lower']), float(joint['upper'])
    if extend:
        r = hi - lo
        lo, hi = lo - 0.5 * r, hi + 0.5 * r
    if joint.get('unit') == 'rad':
        return float(np.rad2deg(lo)), float(np.rad2deg(hi)), 'degrees'
    return lo, hi, 'reconstruction units'


def to_joint_value(joint, slider_value):
    return float(np.deg2rad(slider_value)) if joint.get('unit') == 'rad' else float(slider_value)


def sweep_values(joint, n=12):
    up = list(np.linspace(0.0, float(joint['upper']), n // 2 + 1))
    return up + up[-2::-1]


def part_points_at(model, value):
    T = joint_transform(model['joint'], value)
    return model['part']['pts'] @ T[:3, :3].T + T[:3, 3]


def _cloud(P, C):
    rgba = np.c_[np.clip(np.round(C * 255), 0, 255), np.full(len(P), 255)].astype(np.uint8)
    return trimesh.PointCloud(P @ OPENCV_TO_GL.T, colors=rgba)


def build_glb(model, value, color_mode='rgb', show_axis=True, show_hulls=False, out_path=None):
    scene = trimesh.Scene()
    base_P, base_C = model['base']['pts'], model['base']['rgb']
    part_P = part_points_at(model, value)
    part_C = model['part']['rgb']
    if color_mode == 'part':
        base_C = np.tile(BASE_RGB, (len(base_P), 1))
        part_C = np.tile(PART_RGB, (len(part_P), 1))
    if len(base_P):
        scene.add_geometry(_cloud(base_P, base_C), geom_name='base')
    if len(part_P):
        scene.add_geometry(_cloud(part_P, part_C), geom_name='part')
    allP = np.concatenate([base_P, part_P]) if len(part_P) else base_P
    extent = float(np.ptp(allP, 0).max()) if len(allP) else 1.0
    if show_axis:
        j = model['joint']
        d = np.asarray(j['axis_dir'], float)
        d /= np.linalg.norm(d)
        pc = part_P.mean(0) if len(part_P) else allP.mean(0)
        if j['type'] == 'prismatic':
            o = pc
        else:
            o0 = np.asarray(j['axis_origin'], float)
            o = o0 + d * ((pc - o0) @ d)
        seg = np.stack([o - 0.6 * extent * d, o + 0.6 * extent * d]) @ OPENCV_TO_GL.T
        cyl = trimesh.creation.cylinder(radius=0.004 * extent, segment=seg)
        cyl.visual.face_colors = AXIS_RGBA
        scene.add_geometry(cyl, geom_name='axis')
    if show_hulls:
        T = joint_transform(model['joint'], value)
        for name, rgba in (('base', [29, 127, 166, 70]), ('part', [217, 72, 43, 70])):
            hull = model.get(f'{name}_hull')
            if hull is None:
                continue
            h = hull.copy()
            V = np.asarray(h.vertices)
            if name == 'part':
                V = V @ T[:3, :3].T + T[:3, 3]
            h.vertices = V @ OPENCV_TO_GL.T
            h.visual.face_colors = rgba
            scene.add_geometry(h, geom_name=f'{name}_hull')
    out_path = out_path or tempfile.NamedTemporaryFile(suffix='.glb', delete=False).name
    scene.export(out_path)
    return out_path


def info_markdown(model, result=None):
    j = model['joint']
    unit = 'deg' if j.get('unit') == 'rad' else 'units'
    observed = np.rad2deg(j['observed']) if j.get('unit') == 'rad' else j['observed']
    lines = [f"**{model.get('object', '')}** · joint **{j['type']}**"
             + (' (approximate screw)' if j.get('approximate') else ''),
             f"- observed motion: {observed:.1f} {unit}",
             f"- axis direction (model frame): {np.round(j['axis_dir'], 3).tolist()}"]
    lab = model.get('labelling')
    if lab:
        lines.append(f"- points: body {lab['n_base']:,} · part {lab['n_part']:,} · fallback {lab['n_fallback']:,} · "
                     f"unlabelled {lab['n_unlabelled']:,} (mask: {lab['mask_source']})")
    for w in model.get('warnings', []):
        lines.append(f"- ⚠ {w}")
    m = (result or {}).get('metrics') or {}
    if 'axis_angle_err_deg' in m:
        lines.append(f"- vs ground truth: axis error {m['axis_angle_err_deg']:.1f}°"
                     + (f", state error {m['state_err']:.1f}" if m.get('state_err') is not None else ''))
    return '\n'.join(lines)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -m pytest tests/test_viewer.py -q`
Expected: 5 passed.

- [ ] **Step 5: Commit**

```bash
git add artsfm/viewer.py tests/test_viewer.py
git commit -m "Add glb frame builder and viewer helpers"
```

---

### Task 8: The gradio app

**Files:**
- Create: `app_articulation.py`, `tests/test_app.py`

**Interfaces:**
- Consumes: `artsfm.viewer`, `load_model`, `write_urdf`, `check_urdf`.
- Produces: `app_articulation.build_app(root) -> gradio.Blocks`, and the CLI `python3 app_articulation.py [--root outputs/artsfm] [--port 7861]`.

- [ ] **Step 1: Write failing tests `tests/test_app.py`**

```python
from tests.test_urdf import _make_model


def test_build_app_with_models(tmp_path):
    import gradio as gr
    from app_articulation import build_app
    _make_model(tmp_path / 'toy')
    app = build_app(str(tmp_path))
    assert isinstance(app, gr.Blocks)


def test_build_app_empty_root(tmp_path):
    import gradio as gr
    from app_articulation import build_app
    assert isinstance(build_app(str(tmp_path)), gr.Blocks)


def test_export_fn_produces_zip_and_check(tmp_path):
    from app_articulation import export_package
    from artsfm.model_io import load_model
    m = load_model(_make_model(tmp_path / 'toy'))
    zip_path, msg = export_package(m, extend=False)
    assert zip_path.endswith('.zip') and 'PASS' in msg
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -m pytest tests/test_app.py -q`
Expected: FAIL with ModuleNotFoundError `app_articulation`.

- [ ] **Step 3: Write `app_articulation.py`**

```python
#!/usr/bin/env python3
"""Gradio viewer for articulated models built by build_model.py / run_articulation.py --build_model."""
import argparse
import json
import os
import shutil
import tempfile
import time

import gradio as gr

from artsfm.model_io import load_model
from artsfm.urdf import check_urdf, write_urdf
from artsfm.viewer import build_glb, info_markdown, list_models, slider_spec, sweep_values, to_joint_value


def _result_for(model_dir):
    path = os.path.join(os.path.dirname(model_dir), 'result.json')
    return json.load(open(path)) if os.path.isfile(path) else None


def export_package(model, extend=False):
    """Write the URDF (observed or extended limits), run the MuJoCo check, zip the model folder."""
    j = model['joint']
    lower = upper = None
    if extend:
        r = j['upper'] - j['lower']
        lower, upper = j['lower'] - 0.5 * r, j['upper'] + 0.5 * r
    try:
        urdf = write_urdf(model['dir'], lower=lower, upper=upper)
    except ValueError as e:
        return None, f'URDF export refused: {e}'
    err = check_urdf(urdf)
    msg = f'Wrote {os.path.basename(urdf)}. MuJoCo check: max relative error {err:.1e} ({"PASS" if err < 1e-6 else "FAIL"}).'
    base = os.path.join(tempfile.mkdtemp(), model.get('object', 'model'))
    return shutil.make_archive(base, 'zip', model['dir']), msg


def build_app(root='outputs/artsfm'):
    models = list_models(root)
    with gr.Blocks(title='Articulated model viewer') as app:
        gr.Markdown('## Articulated model viewer\nMove the recovered joint of a reconstructed object, '
                    'then export it as a URDF for simulation.')
        if not models:
            gr.Markdown(f'No articulated models found under `{root}`. Build one with '
                        '`python3 build_model.py --out_dir <run dir>` or '
                        '`python3 run_articulation.py ... --build_model`.')
            return app
        state = gr.State()
        with gr.Row():
            with gr.Column(scale=1):
                picker = gr.Dropdown(choices=models, value=models[0], label='Model')
                slider = gr.Slider(minimum=0, maximum=1, value=0, label='Joint value')
                extend = gr.Checkbox(value=False, label='Allow beyond observed range')
                colour = gr.Radio(['RGB', 'By part'], value='RGB', label='Colour')
                hulls = gr.Checkbox(value=False, label='Show collision hulls')
                play = gr.Button('Play')
                export_btn = gr.Button('Export URDF')
                package = gr.File(label='URDF package')
                export_msg = gr.Markdown()
                info = gr.Markdown()
            with gr.Column(scale=2):
                view = gr.Model3D(label='Model', height=620, clear_color=[0.0, 0.0, 0.0, 0.0])

        def render(m, v, c, h):
            return build_glb(m, to_joint_value(m['joint'], v), 'part' if c == 'By part' else 'rgb', True, h)

        def load(path):
            m = load_model(path)
            lo, hi, unit = slider_spec(m['joint'])
            return (m, gr.update(minimum=lo, maximum=hi, value=0, label=f'Joint value ({unit})'),
                    render(m, 0, 'RGB', False), info_markdown(m, _result_for(path)))

        def set_range(m, e):
            lo, hi, unit = slider_spec(m['joint'], extend=e)
            return gr.update(minimum=lo, maximum=hi)

        def play_fn(m, c, h):
            for v in sweep_values(m['joint']):
                yield build_glb(m, v, 'part' if c == 'By part' else 'rgb', True, h)
                time.sleep(0.15)

        app.load(load, picker, [state, slider, view, info])
        picker.change(load, picker, [state, slider, view, info])
        slider.release(render, [state, slider, colour, hulls], view)
        colour.change(render, [state, slider, colour, hulls], view)
        hulls.change(render, [state, slider, colour, hulls], view)
        extend.change(set_range, [state, extend], slider)
        play.click(play_fn, [state, colour, hulls], view)
        export_btn.click(export_package, [state, extend], [package, export_msg])
    return app


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', default='outputs/artsfm')
    p.add_argument('--port', type=int, default=7861)
    p.add_argument('--server_name', default='127.0.0.1')
    a = p.parse_args()
    build_app(a.root).launch(server_name=a.server_name, server_port=a.port)


if __name__ == '__main__':
    main()
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -m pytest tests/test_app.py -q`
Expected: 3 passed.

- [ ] **Step 5: Commit**

```bash
git add app_articulation.py tests/test_app.py
git commit -m "Add gradio articulated model viewer app"
```

---

### Task 9: Real data — build models, measure labelling, check URDF, launch the app

**Files:**
- Modify: `artsfm/evaluate.py` (append `model_label_accuracy`), `tests/test_evaluate.py`
- Create: `docs/superpowers/results/2026-10-05-articulated-model-viewer.md`

**Interfaces:**
- Produces: `model_label_accuracy(model_dir, data_dir) -> dict(base_acc, part_acc, base_median_dist, part_median_dist)`.

- [ ] **Step 1: Write the failing test** (append to `tests/test_evaluate.py`)

```python
def test_model_label_accuracy_on_toy(tmp_path):
    import trimesh
    from artsfm.evaluate import model_label_accuracy
    from artsfm.model_io import write_model
    (tmp_path / 'data/gt/start').mkdir(parents=True)
    trimesh.creation.box([1, 1, 1]).export(tmp_path / 'data/gt/start/start_static_rotate.ply')
    trimesh.creation.box([1, 1, 1]).apply_translation([3, 0, 0]).export(tmp_path / 'data/gt/start/start_dynamic_rotate.ply')
    base = np.array([[0.5, 0, 0], [0, 0.5, 0.1], [2.6, 0, 0]])   # last one is wrong
    part = np.array([[2.5, 0, 0], [3.5, 0.1, 0]])
    gt_frame = dict(up=[0, 0, 1], scale=1.0, sim_R=np.eye(3), sim_t=np.zeros(3), cam2w0=np.eye(4))
    write_model(tmp_path / 'model', dict(object='t', gt_frame=gt_frame),
                dict(base=(base, np.ones((3, 3))), part=(part, np.ones((2, 3)))), dict(base=None, part=None))
    r = model_label_accuracy(str(tmp_path / 'model'), str(tmp_path / 'data'))
    assert abs(r['base_acc'] - 2 / 3) < 1e-12 and r['part_acc'] == 1.0
```

- [ ] **Step 2: Run it to verify it fails**

Run: `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -m pytest tests/test_evaluate.py -q -k model_label`
Expected: FAIL with ImportError.

- [ ] **Step 3: Append to `artsfm/evaluate.py`**

```python
def model_label_accuracy(model_dir, data_dir):
    """Fraction of base.ply / part.ply points whose nearest camera-frame GT surface agrees (start state)."""
    from artsfm.model_io import load_model
    m = load_model(model_dir)
    g = m.get('gt_frame')
    if not g:
        raise ValueError('model has no gt_frame (built without GT poses)')
    sim = (float(g['scale']), np.asarray(g['sim_R'], float), np.asarray(g['sim_t'], float))
    cam2w0 = np.asarray(g['cam2w0'], float)
    out = {}
    for name, want in (('base', 0), ('part', 1)):
        P = m[name]['pts']
        if not len(P):
            out[f'{name}_acc'] = out[f'{name}_median_dist'] = None
            continue
        lab, dist = gt_part_labels(view0_to_gt(P, cam2w0, sim), data_dir)
        out[f'{name}_acc'] = float((lab == want).mean())
        out[f'{name}_median_dist'] = float(np.median(dist))
    return out
```

- [ ] **Step 4: Run the full suite**

Run: `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -m pytest tests/ -q`
Expected: all pass. Then commit:

```bash
git add artsfm/evaluate.py tests/test_evaluate.py
git commit -m "Add model label accuracy against camera-frame GT meshes"
```

- [ ] **Step 5: Build real models** (CPU, about 2–5 min each, using cached matches)

Run, for each of 45247, 45166, 45176 and 45855:
`python3 build_model.py --out_dir outputs/artsfm/StorageFurniture_<id>`
Expected: a JSON summary with joint type, labelling counts and warnings. 45855 is `free`, so expect the "screw joint" warning.

- [ ] **Step 6: Measure labelling accuracy**

Run: `python3 -c "from artsfm.evaluate import model_label_accuracy as f; import json; [print(i, json.dumps(f(f'outputs/artsfm/StorageFurniture_{i}/model', f'/home/robotlab/scratch/GaussianArt/data/MPArt-90/extracted/StorageFurniture_{i}'))) for i in (45247, 45166, 45176, 45855)]"`
Expected: per-object base and part accuracy, with median distance to the GT surface. Where `cam_align_rmse` is high (45247, 45166), treat these as approximate.

- [ ] **Step 7: Export and check URDF on 45247**

Run: `python3 -m artsfm.urdf outputs/artsfm/StorageFurniture_45247/model --use_gt_frame`
Expected: `MuJoCo check: ... PASS`.

- [ ] **Step 8: Launch the app and confirm it serves**

Run in the background: `python3 app_articulation.py --root outputs/artsfm --port 7861`, then `curl -s -o /dev/null -w '%{http_code}' http://127.0.0.1:7861/`.
Expected: `200`. Report the URL to the user for the manual look. If they report points too faint, record a ruling and switch `_cloud` to decimated small quads.

- [ ] **Step 9: Write `docs/superpowers/results/2026-10-05-articulated-model-viewer.md`** covering:
- the label accuracy table;
- point counts and warnings;
- the URDF check result;
- how to launch the app;
- known limits (convex-hull visuals, labelling only as good as the joint, the scale caveat).

Then commit:

```bash
git add docs/superpowers/results/2026-10-05-articulated-model-viewer.md
git commit -m "Add articulated model viewer v1 results"
```
