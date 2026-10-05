import json
import numpy as np

from artsfm.evaluate import (umeyama, axis_angle_err, line_distance, load_gt_joint, view0_to_gt,
                             align_poses, gt_part_labels)
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
    # GaussianArt's read_gt maps trans.json axes into the camera frame with R_coord
    np.testing.assert_allclose(g['axis_o'], [1, 0, 2])
    np.testing.assert_allclose(g['axis_d'], [0, -1, 0])


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


def test_align_poses_recovers_similarity_from_opengl_gt():
    rng = np.random.default_rng(1)
    s, R, t = 1.7, rot([1, 2, 0], 0.8), np.array([0.5, -1, 2.0])
    flip = np.diag([1, -1, -1, 1.0])
    est, gt = [], []
    for i in range(6):  # clustered cameras, as in view selection
        c2w = np.eye(4)
        c2w[:3, :3] = rot([0, 0, 1], 0.1 * i) @ rot([1, 0, 0], 0.05 * i)
        c2w[:3, 3] = rng.uniform(-0.2, 0.2, 3) + [0, 0, -4]
        est.append(c2w)
        g = np.eye(4)
        g[:3, :3] = R @ c2w[:3, :3]
        g[:3, 3] = s * R @ c2w[:3, 3] + t
        gt.append(g @ flip)  # GT stored in OpenGL convention
    s_, R_, t_ = align_poses(np.array(est), np.array(gt))
    assert abs(s_ - s) < 1e-9
    np.testing.assert_allclose(R_, R, atol=1e-9)
    np.testing.assert_allclose(t_, t, atol=1e-9)


def test_gt_part_labels_from_camera_frame_ply(tmp_path):
    import trimesh
    (tmp_path / 'gt/start').mkdir(parents=True)
    trimesh.creation.box([1, 1, 1]).export(tmp_path / 'gt/start/start_static_rotate.ply')
    trimesh.creation.box([1, 1, 1]).apply_translation([3, 0, 0]).export(tmp_path / 'gt/start/start_dynamic_rotate.ply')
    lab, dist = gt_part_labels(np.array([[0.5, 0, 0], [2.5, 0, 0], [0, 0.5, 0.1]]), str(tmp_path))
    assert lab.tolist() == [0, 1, 0]
    assert dist.max() < 0.05


def test_segmentation_accuracy_invariant_to_base_part_swap():
    from artsfm.evaluate import segmentation_accuracy
    gt = np.array([0, 0, 0, 1, 1, 0])
    pred = np.array([1, 1, 1, 0, -1, 1])   # base/part swapped; -1 ignored
    acc, flipped = segmentation_accuracy(pred, gt)
    assert acc == 1.0 and flipped
    acc, flipped = segmentation_accuracy(np.array([0, 0, 1, 1, 1, 0]), gt)
    assert abs(acc - 5 / 6) < 1e-12 and not flipped
    assert segmentation_accuracy(np.array([-1, -1]), np.array([0, 1])) == (None, False)


def test_gt_part_track_count_metric():
    from artsfm.evaluate import track_label_counts
    c = track_label_counts(np.array([0, 1, -1, 1]), np.array([1, 1, 1, 0]))
    assert c == dict(n_gt_part_tracks=3, n_pred_part_on_gt_part=1, n_pred_outlier_on_gt_part=1)
