import numpy as np
import pytest

from artsfm.factorize import as_views, kabsch, fit_motion, residuals, ransac_rigid
from tests.synth import make_scene, rot


def test_kabsch_recovers_rigid_motion():
    rng = np.random.default_rng(0)
    P = rng.standard_normal((20, 3))
    R = rot([1, 2, 3], 0.7)
    t = np.array([0.1, -2, 3])
    R_est, t_est = kabsch(P, P @ R.T + t)
    np.testing.assert_allclose(R_est, R, atol=1e-10)
    np.testing.assert_allclose(t_est, t, atol=1e-10)


def test_fit_motion_zero_residual_on_base():
    W, states, labels, gt = make_scene('revolute')
    base = np.flatnonzero(labels == 0)
    M = fit_motion(W, base)
    assert M.shape == (len(states), 3, 4)
    np.testing.assert_allclose(M[0], np.c_[np.eye(3), np.zeros(3)], atol=1e-10)
    assert residuals(W, M)[base].max() < 1e-9


@pytest.mark.parametrize('motion', ['prismatic', 'revolute', 'screw'])
def test_ransac_segments_base_and_planar_part(motion):
    W, states, labels, gt = make_scene(motion, noise=0.01, planar_part=True)
    tau = 0.02 * np.median(as_views(W)[..., 2])
    base = ransac_rigid(W, np.ones(W.shape[1], bool), tau)
    part = ransac_rigid(W, ~base, tau)
    # base may absorb part points lying on/near the hinge axis; part must be clean
    assert (base & (labels == 0)).sum() / (labels == 0).sum() > 0.95
    assert (labels[part] == 1).mean() > 0.95
    assert (part & (labels == 1)).sum() / (labels == 1).sum() > 0.6


def test_ransac_empty_candidates():
    W, *_ = make_scene('none')
    assert not ransac_rigid(W, np.zeros(W.shape[1], bool), 0.05).any()
