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
