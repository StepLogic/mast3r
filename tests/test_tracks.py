import numpy as np

from artsfm.tracks import build_tracks, lift_tracks


def test_tracks_anchored_on_view0_matches():
    corres = {(0, 1): (np.array([[10.0, 10], [100, 100]]), np.array([[50.0, 20], [150, 120]])),
              (0, 2): (np.array([[10.0, 10], [100, 100]]), np.array([[90.0, 30], [190, 130]]))}
    T = build_tracks(corres, 3)
    assert T.shape == (2, 3, 2)
    order = np.argsort(T[:, 0, 0])
    np.testing.assert_allclose(T[order[0]], [[10, 10], [50, 20], [90, 30]])
    np.testing.assert_allclose(T[order[1]], [[100, 100], [150, 120], [190, 130]])


def test_reversed_pair_key_is_used():
    corres = {(0, 1): (np.array([[10.0, 10]]), np.array([[50.0, 20]])),
              (2, 0): (np.array([[90.0, 30]]), np.array([[11.0, 11]]))}
    T = build_tracks(corres, 3)
    assert T.shape == (1, 3, 2)
    np.testing.assert_allclose(T[0, 2], [90, 30])


def test_noisy_cross_pair_matches_do_not_merge_tracks():
    # (1, 2) matches link the two tracks' cells; chaining through them must not fuse the tracks
    corres = {(0, 1): (np.array([[10.0, 10], [100, 100]]), np.array([[50.0, 20], [150, 120]])),
              (0, 2): (np.array([[10.0, 10], [100, 100]]), np.array([[90.0, 30], [190, 130]])),
              (1, 2): (np.array([[50.0, 20], [150, 120]]), np.array([[190.0, 130], [90, 30]]))}
    assert build_tracks(corres, 3).shape == (2, 3, 2)


def test_conflicting_track_dropped():
    corres = {(0, 1): (np.array([[10, 10], [10, 10]]), np.array([[50, 20], [150, 120]])),
              (0, 2): (np.array([[10, 10]]), np.array([[90, 30]]))}
    T = build_tracks(corres, 3, cell=8)
    assert T.shape == (0, 3, 2)


def test_lift_tracks_backprojects():
    K = np.array([[100.0, 0, 50], [0, 100, 40], [0, 0, 1]])
    depth = np.full((80, 100), 2.0)
    tracks = np.array([[[60.0, 40.0], [50.0, 50.0]]])
    W, keep = lift_tracks(tracks, np.stack([K, K]), [depth, depth])
    assert keep.tolist() == [True]
    np.testing.assert_allclose(W[:, 0], [0.2, 0.0, 2.0, 0.0, 0.2, 2.0])


def test_lift_drops_invalid_depth():
    K = np.array([[100.0, 0, 50], [0, 100, 40], [0, 0, 1]])
    d0 = np.full((80, 100), 2.0)
    d1 = d0.copy()
    d1[50, 50] = 0.0
    d1[10, 10] = np.nan
    tracks = np.array([[[60.0, 40.0], [50.0, 50.0]],
                       [[60.0, 40.0], [10.0, 10.0]],
                       [[60.0, 40.0], [500.0, 10.0]],   # out of image
                       [[60.0, 40.0], [20.0, 20.0]]])
    W, keep = lift_tracks(tracks, np.stack([K, K]), [d0, d1])
    assert keep.tolist() == [False, False, False, True]
    assert W.shape == (6, 1) and np.isfinite(W).all()
