"""Sweep LabelParams on synthetic dense scenes (not a pytest test). Prints accuracy on labelled points / unlabelled fraction."""
import itertools
import numpy as np

from artsfm.articulate import LabelParams, label_dense
from tests.synth_dense import make_dense_scene, to_model_frame, sample_tracks

NOISES = [0.0, 0.01, 0.02]


def score(params, motion, noise, seeds=(0, 1, 2)):
    accs, unl = [], []
    for seed in seeds:
        s = make_dense_scene(motion, noise=noise, seed=seed)
        geom, T = to_model_frame(s)
        tp, tl = sample_tracks(s, seed=seed)
        radius = float(np.sqrt(((tp - tp.mean(0)) ** 2).sum(1).mean()))
        views = label_dense(geom, s['alphas'], s['states'], T, tp, tl, radius, params)
        ok = n_lab = n_all = 0
        for f, v in enumerate(views):
            gt = s['labels'][f].ravel()[v['pix']]
            m = v['label'] >= 0
            ok += (v['label'][m] == gt[m]).sum()
            n_lab += m.sum()
            n_all += len(m)
        accs.append(ok / n_lab)
        unl.append(1 - n_lab / n_all)
    return float(np.mean(accs)), float(np.mean(unl))


if __name__ == '__main__':
    print('delta ctol margin | ' + ' '.join(f'{m[:3]}{n}' for m in ('revolute', 'prismatic') for n in NOISES) + ' | mean')
    for delta, ctol, margin in itertools.product([0.03, 0.05, 0.08], [0.1, 0.15, 0.25], [1, 2]):
        p = LabelParams(delta=delta, ctol=ctol, margin=margin)
        accs = [score(p, m, n)[0] for m in ('revolute', 'prismatic') for n in NOISES]
        print(f'{delta:5} {ctol:4} {margin:6} | ' + ' '.join(f'{a:.3f}' for a in accs) + f' | {np.mean(accs):.3f}', flush=True)
