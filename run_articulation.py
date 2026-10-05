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
from artsfm.views import load_frames, load_fov, select_views


def _tracks_for(data_dir, out_dir, weights, k, seed_view, optim_level, device, niter1, niter2, fov_x):
    start, end = load_frames(data_dir, 'start'), load_frames(data_dir, 'end')
    paths, states = select_views(start, end, k=k, seed_view=seed_view)
    cache = os.path.join(out_dir, f'cache_k{k}')
    scene = run_sparse_ga(paths, cache, weights, device=device, optim_level=optim_level,
                          niter1=niter1, niter2=niter2, fov_x=fov_x)
    geom = extract_geometry(scene)
    corres = load_pair_correspondences(cache, paths)
    tracks = build_tracks(corres, len(paths))
    W, keep = (lift_tracks(tracks, geom['K'], geom['depths']) if len(tracks)
               else (np.zeros((3 * len(paths), 0)), np.zeros(0, bool)))
    frames_by_path = {p: T for p, T in start + end}
    return paths, states, geom, W, frames_by_path


def run(data_dir, out_dir, weights, k=5, seed_view=0, optim_level='refine', device=None,
        params=Params(), niter1=300, niter2=300, min_tracks=30, known_intrinsics=True):
    device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
    fov_x = load_fov(data_dir) if known_intrinsics else None
    os.makedirs(out_dir, exist_ok=True)
    for kk in (k, 3) if k > 3 else (k,):
        paths, states, geom, W, frames_by_path = _tracks_for(
            data_dir, out_dir, weights, kk, seed_view, optim_level, device, niter1, niter2, fov_x)
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
    p.add_argument('--estimate_intrinsics', action='store_true',
                   help='let MASt3R estimate focals instead of using the dataset camera_angle_x')
    a = p.parse_args()
    out = run(a.data_dir, a.out_dir, a.weights, a.k, a.seed_view, a.optim_level, a.device,
              niter1=a.niter1, niter2=a.niter2, known_intrinsics=not a.estimate_intrinsics)
    r, m = out['result'], out['metrics']
    print(json.dumps(_jsonable(dict(status=r['status'], type=r.get('type'), reason=r.get('reason'),
                                    **{k: v for k, v in m.items() if k != 'views'})), indent=2))


if __name__ == '__main__':
    main()
