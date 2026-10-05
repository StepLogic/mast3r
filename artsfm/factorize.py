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
