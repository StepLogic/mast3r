"""Sweep noise x thresholds on synthetic scenes; print type accuracy (not a pytest test)."""
import itertools
import numpy as np

from artsfm.factorize import Params, analyze
from tests.synth import make_scene

MOTIONS = ['none', 'prismatic', 'revolute', 'screw']
NOISES = [0.0, 0.005, 0.01, 0.02, 0.03]
SEEDS = range(5)


def accuracy(params, noise):
    ok = 0
    for motion, seed in itertools.product(MOTIONS, SEEDS):
        W, states, labels, gt = make_scene(motion, noise=noise, outlier_frac=0.1, seed=seed)
        ok += analyze(W, states, params)['type'] == gt['type']
    return ok / (len(MOTIONS) * len(SEEDS))


if __name__ == '__main__':
    print('tau_frac theta_deg | ' + ' '.join(f'n={n}' for n in NOISES))
    for tau_frac, theta in itertools.product([0.01, 0.02, 0.03], [1.5, 3.0, 4.5]):
        p = Params(tau_frac=tau_frac, theta_deg=theta, n_iter=200)
        print(f'{tau_frac:8} {theta:9} | ' + ' '.join(f'{accuracy(p, n):5.2f}' for n in NOISES), flush=True)
