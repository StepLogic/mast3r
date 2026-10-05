import json
import numpy as np

from artsfm.evaluate import umeyama, axis_angle_err, line_distance, load_gt_joint, view0_to_gt
from tests.synth import rot


def test_umeyama_recovers_similarity():
    rng = np.random.default_rng(0)
    src = rng.standard_normal((10, 3))
    R, s, t = rot([0, 1, 1], 0.4), 2.5, np.array([1, 2, 3.0])
    s_, R_, t_ = umeyama(src, s * src @ R.T + t)
    assert abs(s_ - s) < 1e-9
    np.testing.assert_allclose(R_, R, atol=1e-9)
    np.testing.assert_allclose(t_, t, atol=1e-9)


def test_axis_angle_sign_agnostic():
    assert axis_angle_err([0, 0, 1], [0, 0, -1]) < 1e-9
    assert abs(axis_angle_err([1, 0, 0], [0, 1, 0]) - 90) < 1e-9


def test_line_distance_point_choice():
    d = np.array([0, 0, 1.0])
    assert abs(line_distance(np.array([1, 0, 0.0]), d, np.array([1, 0, 5.0]), -d)) < 1e-12
    assert abs(line_distance(np.array([1, 0, 0.0]), d, np.array([0, 2, 3.0]), d) - np.sqrt(5)) < 1e-12


def test_load_gt_joint(tmp_path):
    (tmp_path / 'gt').mkdir()
    (tmp_path / 'gt/trans.json').write_text(json.dumps({'trans_info': [
        {'axis': {'o': [0, 1, 2], 'd': [1, 0, 0]}, 'rotate': {'l': 68.0, 'r': 32.0}, 'type': 'rotate'}]}))
    g = load_gt_joint(str(tmp_path))
    assert g['type'] == 'revolute' and abs(g['state_change'] - 36.0) < 1e-12
    np.testing.assert_allclose(g['axis_o'], [0, 1, 2])


def test_view0_to_gt_roundtrip():
    cam2w0 = np.eye(4)
    cam2w0[:3, :3] = rot([1, 0, 0], 0.3)
    cam2w0[:3, 3] = [0, 0, 1]
    sim = (2.0, rot([0, 0, 1], 0.5), np.array([1.0, 0, 0]))
    p = np.array([[0.1, 0.2, 0.3]])
    world = p @ cam2w0[:3, :3].T + cam2w0[:3, 3]
    expected = 2.0 * world @ sim[1].T + sim[2]
    np.testing.assert_allclose(view0_to_gt(p, cam2w0, sim), expected)
    d = np.array([[0, 0, 1.0]])
    np.testing.assert_allclose(view0_to_gt(d, cam2w0, sim, is_dir=True),
                               d @ cam2w0[:3, :3].T @ sim[1].T)
