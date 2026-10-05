import numpy as np
import pytest

from tests.synth_dense import make_dense_scene, to_model_frame, sample_tracks


def _accuracy(motion, noise, seeds=(0, 1, 2)):
    from artsfm.articulate import label_dense
    accs, unl = [], []
    for seed in seeds:
        s = make_dense_scene(motion, noise=noise, seed=seed)
        geom, T = to_model_frame(s)
        tp, tl = sample_tracks(s, seed=seed)
        radius = float(np.sqrt(((tp - tp.mean(0)) ** 2).sum(1).mean()))
        views = label_dense(geom, s['alphas'], s['states'], T, tp, tl, radius)
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


def test_backproject_roundtrip_through_evidence():
    from artsfm.articulate import backproject, evidence, reliable_mask
    s = make_dense_scene('revolute')
    P, idx = backproject(s['K'][0], np.eye(4), s['depths'][0], s['alphas'][0])
    rgb = s['rgbs'][0].reshape(-1, 3)[idx]
    rel = reliable_mask(s['depths'][0], s['alphas'][0], 0.1)
    sup, con = evidence(P, rgb, s['K'][0], np.eye(4), s['depths'][0], rel, s['rgbs'][0], 0.01, 0.01)
    on_rel = rel.ravel()[idx]
    assert sup[on_rel].all() and not con.any()


def test_reliable_mask_excludes_depth_edges():
    from artsfm.articulate import reliable_mask
    d = np.full((10, 10), 2.0)
    d[:, 5:] = 3.0
    m = reliable_mask(d, np.ones((10, 10), bool), 0.1)
    assert m[:, 0:4].all() and not m[:, 4:6].any() and m[:, 7:].all()


@pytest.mark.parametrize('motion', ['revolute', 'prismatic'])
def test_label_accuracy_noise_free(motion):
    acc, unl = _accuracy(motion, 0.0)
    assert acc > 0.95, acc
    assert unl < 0.10, unl


@pytest.mark.parametrize('motion', ['revolute', 'prismatic'])
def test_label_accuracy_two_percent_noise(motion):
    acc, unl = _accuracy(motion, 0.02)
    assert acc > 0.90, acc


def test_occluded_prediction_is_not_a_contradiction():
    from artsfm.articulate import evidence
    K = np.array([[50.0, 0, 20], [0, 50, 20], [0, 0, 1]])
    depth = np.full((40, 40), 2.0)
    rel = np.ones((40, 40), bool)
    img = np.zeros((40, 40, 3))
    P = np.array([[0.0, 0.0, 3.0], [0.0, 0.0, 1.0]])  # behind the surface / in front of it
    sup, con = evidence(P, np.zeros((2, 3)), K, np.eye(4), depth, rel, img, 0.05, 0.1)
    assert sup.tolist() == [0, 0] and con.tolist() == [0, 1]


def test_hinge_points_use_fallback():
    """Points on a hinge axis do not move, so both hypotheses predict the same thing.
    With an identity motion every point is such a tie: none may be decided by the hypothesis test."""
    from artsfm.articulate import label_dense
    s = make_dense_scene('revolute')
    geom, _ = to_model_frame(s)
    tp, tl = sample_tracks(s)
    radius = float(np.sqrt(((tp - tp.mean(0)) ** 2).sum(1).mean()))
    views = label_dense(geom, s['alphas'], s['states'], np.eye(4), tp, tl, radius)
    for v in views:
        assert v['fallback'].all()
        assert set(np.unique(v['label'])) <= {-1, 0, 1}
