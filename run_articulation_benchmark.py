#!/usr/bin/env python3
"""Run run_articulation over MPArt-90 single-joint objects and print a markdown table."""
import argparse
import json
import os

import run_articulation as ra

ROOT = '/home/robotlab/scratch/GaussianArt/data/MPArt-90/extracted'
OBJECTS = ['StorageFurniture_45166', 'StorageFurniture_35059', 'StorageFurniture_41086',
           'StorageFurniture_45176', 'StorageFurniture_45247', 'StorageFurniture_45937',
           'StorageFurniture_47021', 'StorageFurniture_46107', 'StorageFurniture_45855']
COLS = ['status', 'gt_type', 'pred_type', 'axis_angle_err_deg', 'axis_pos_err', 'state_err',
        'seg_accuracy', 'n_tracks', 'n_gt_part_tracks', 'n_pred_part_on_gt_part', 'cam_align_rmse']


def fmt(v):
    return f'{v:.3f}' if isinstance(v, float) else ('' if v is None else str(v))


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--out_root', default='outputs/artsfm')
    p.add_argument('--seed_view', type=int, default=0)
    a = p.parse_args()
    print('| object | ' + ' | '.join(COLS) + ' |')
    print('|' + '---|' * (len(COLS) + 1))
    for obj in OBJECTS:
        out_dir = os.path.join(a.out_root, obj)
        res_path = os.path.join(out_dir, 'result.json')
        if not os.path.isfile(res_path):
            try:
                ra.run(os.path.join(ROOT, obj), out_dir, 'checkpoints/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric.pth',
                       seed_view=a.seed_view)
            except Exception as e:  # keep the benchmark going; record the failure
                os.makedirs(out_dir, exist_ok=True)
                json.dump(dict(result=dict(status=f'error: {e}'), metrics={}), open(res_path, 'w'))
        saved = json.load(open(res_path))
        row = dict(saved['metrics'], status=saved['result']['status'])
        print(f'| {obj} | ' + ' | '.join(fmt(row.get(c)) for c in COLS) + ' |', flush=True)


if __name__ == '__main__':
    main()
