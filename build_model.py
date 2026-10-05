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
