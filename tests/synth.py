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
