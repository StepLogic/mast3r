import numpy as np
import pytest

from artsfm.factorize import (Params, DegenerateViewsError, analyze, principal_angles,
                              screw, relative_motion)
from tests.synth import make_scene, rot


def axis_err_deg(a, b):
    c = abs(np.dot(a, b)) / np.linalg.norm(a) / np.linalg.norm(b)
    return np.rad2deg(np.arccos(np.clip(c, -1, 1)))


def line_dist(o1, d1, o2, d2):
    n = np.cross(d1, d2)
    if np.linalg.norm(n) < 1e-9:
        return np.linalg.norm(np.cross(o2 - o1, d1)) / np.linalg.norm(d1)
    return abs(np.dot(o2 - o1, n)) / np.linalg.norm(n)


def test_principal_angles_shared_dims():
    rng = np.random.default_rng(0)
    common = rng.standard_normal((30, 2))
    A = np.c_[common, rng.standard_normal((30, 2))]
    B = np.c_[common @ rng.standard_normal((2, 2)), rng.standard_normal((30, 2))]
    ang = principal_angles(A, B)
    assert ang[0] < 1e-5 and ang[1] < 1e-5 and ang[2] > 1


def test_screw_pure_rotation_and_translation():
    A = rot([0, 0, 1], np.deg2rad(30))
    o = np.array([1.0, 2.0, 0.0])
    s = screw(A, o - A @ o)
    assert abs(s['angle_deg'] - 30) < 1e-8
    assert axis_err_deg(s['axis_d'], [0, 0, 1]) < 1e-6
    assert line_dist(s['axis_o'], s['axis_d'], o, np.array([0, 0, 1.0])) < 1e-8
    s = screw(np.eye(3), np.array([0, 0.3, 0]))
    assert s['angle_deg'] < 1e-8 and abs(s['disp'] - 0.3) < 1e-12
    assert axis_err_deg(s['axis_d'], [0, 1, 0]) < 1e-6


@pytest.mark.parametrize('motion', ['prismatic', 'revolute', 'screw'])
@pytest.mark.parametrize('noise', [0.0, 0.01])
def test_analyze_type_and_axis(motion, noise):
    W, states, labels, gt = make_scene(motion, noise=noise)
    r = analyze(W, states)
    assert r['status'] == 'ok', r['reason']
    assert r['type'] == gt['type'], r['principal_angles']
    tol = 1e-6 if noise == 0 else 3.0
    assert axis_err_deg(r['axis_d'], gt['axis_d']) < tol
    if motion == 'revolute':
        assert line_dist(r['axis_o'], r['axis_d'], gt['axis_o'], gt['axis_d']) < (1e-6 if noise == 0 else 0.03)
        assert abs(r['angle_deg'] - gt['angle_deg']) < (1e-6 if noise == 0 else 2.0)
        assert axis_err_deg(r['nullspace_axis_d'], gt['axis_d']) < (1e-6 if noise == 0 else 5.0)
        assert not r['inconsistent']
    if motion == 'prismatic':
        assert abs(r['disp'] - gt['disp']) < (1e-6 if noise == 0 else 0.02)


def test_rigid_scene():
    W, states, labels, gt = make_scene('none', noise=0.01)
    r = analyze(W, states)
    assert r['status'] == 'rigid' and r['type'] == 'rigid'
    assert r['axis_d'] is None


def test_tiny_part():
    W, states, labels, gt = make_scene('revolute', n_part=5, noise=0.005)
    r = analyze(W, states)
    assert r['status'] == 'rigid'


def test_small_motion_is_ambiguous():
    W, states, labels, gt = make_scene('revolute', noise=0.0)
    # rebuild with a 1-degree rotation: scale the part's end-state displacement
    from tests import synth
    old = synth.rot
    synth.rot = lambda axis, ang: old(axis, np.deg2rad(1.0))
    try:
        W, states, labels, gt = synth.make_scene('revolute', noise=0.0)
    finally:
        synth.rot = old
    r = analyze(W, states)
    assert r['status'] in ('ambiguous_small_motion', 'rigid')
    assert r['axis_d'] is None


def test_outliers_rejected():
    W, states, labels, gt = make_scene('revolute', noise=0.01, outlier_frac=0.2)
    r = analyze(W, states)
    assert r['type'] == 'revolute'
    assert (r['labels'][labels == -1] == -1).mean() > 0.9
    assert axis_err_deg(r['axis_d'], gt['axis_d']) < 3.0


def test_degenerate_views_raise():
    W, states, labels, gt = make_scene('revolute', k_per_state=1)
    with pytest.raises(DegenerateViewsError):
        analyze(W, states)


def test_relative_motion_identity_for_base():
    W, states, labels, gt = make_scene('revolute')
    base = np.flatnonzero(labels == 0)
    A, b = relative_motion(W, states, base, base)
    np.testing.assert_allclose(A, np.eye(3), atol=1e-9)
    np.testing.assert_allclose(b, 0, atol=1e-9)


def test_tracks_assigned_to_better_model():
    from artsfm.factorize import fit_motion, residuals
    W, states, labels, gt = make_scene('revolute')
    r = analyze(W, states)
    lab = r['labels']
    rb = residuals(W, fit_motion(W, np.flatnonzero(lab == 0)))
    rp = residuals(W, fit_motion(W, np.flatnonzero(lab == 1)))
    assert (rb[lab == 0] <= rp[lab == 0] + 1e-9).all()
    assert (rp[lab == 1] <= rb[lab == 1] + 1e-9).all()
