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
