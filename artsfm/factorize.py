"""Articulated structure from motion by factorization (Tresadern & Reid, CVPR 2005),
on metric camera-frame 3D tracks.

W is (3F, N): rows 3f..3f+2 hold the camera-frame 3D coordinates of N tracks in view f.
View 0 is a start-state view and defines the reference frame for all outputs.
"""
from dataclasses import dataclass

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


TYPE_BY_INTERSECTION = {4: 'rigid', 3: 'prismatic', 2: 'revolute'}  # else 'free'


@dataclass
class Params:
    tau_frac: float = 0.03         # RANSAC inlier threshold, fraction of median depth (calibrated, spec 4.4)
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
    # Points near a hinge axis fit the base within tau; give each track to the better model.
    both = base | part
    for _ in range(3):
        rb = residuals(W, fit_motion(W, np.flatnonzero(base)))
        rp = residuals(W, fit_motion(W, np.flatnonzero(part)))
        new_part = both & (rp < rb)
        if (new_part == part).all():
            break
        part, base = new_part, both & ~new_part
    labels[base], labels[part] = 0, 1
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
