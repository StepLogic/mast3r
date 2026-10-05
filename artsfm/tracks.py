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
