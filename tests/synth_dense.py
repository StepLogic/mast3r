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
