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
    # reproduce the run's own reconstruction settings (older result.json files lack them: defaults)
    known = m.get('known_intrinsics', True) and not a.estimate_intrinsics
    fov = load_fov(data_dir) if known else None
    geom, W = ra.reconstruct_and_track(paths, os.path.join(a.out_dir, f'cache_k{k}'), a.weights, device,
                                       m.get('optim_level', 'refine'), m.get('niter1', 300), m.get('niter2', 300), fov)
    result = analyze(W, states)
    old = saved['result']
    if result.get('type') != old.get('type'):
        print(f"note: re-analysis gives {result.get('type')}, result.json says {old.get('type')}; using re-analysis")
    if result.get('axis_d') is not None and old.get('axis_d') is not None:
        c = abs(np.dot(result['axis_d'], old['axis_d'])) / np.linalg.norm(result['axis_d']) / np.linalg.norm(old['axis_d'])
        drift = float(np.degrees(np.arccos(np.clip(c, -1, 1))))
        if drift > 1.0:
            print(f'note: re-analysis axis differs by {drift:.1f}° from result.json')
    if result.get('angle_deg') is not None and old.get('angle_deg') is not None:
        dang = abs(result['angle_deg'] - old['angle_deg'])
        if dang > 1.0:
            print(f'note: re-analysis angle differs by {dang:.1f}° from result.json')
    frames = {pth: T for pth, T in load_frames(data_dir, 'start') + load_frames(data_dir, 'end')}
    model_dir = assemble_model(geom, paths, states, W, result, a.out_dir,
                               os.path.basename(os.path.normpath(data_dir)), frames_by_path=frames)
    if not model_dir:
        sys.exit(1)
    if model_dir:
        mj = json.load(open(os.path.join(model_dir, 'model.json')))
        print(json.dumps(dict(model_dir=model_dir, joint=mj['joint']['type'], labelling=mj['labelling'],
                              warnings=mj['warnings']), indent=2))


if __name__ == '__main__':
    main()
