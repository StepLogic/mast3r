import numpy as np

from artsfm.tracks import build_tracks, lift_tracks


def test_chains_pairwise_matches_into_complete_track():
    p = {0: [10, 10], 1: [50, 20], 2: [90, 30]}
    corres = {(0, 1): (np.array([p[0]]), np.array([p[1]])),
              (1, 2): (np.array([p[1]]) + 1, np.array([p[2]])),   # within the same 8px cell
              (0, 2): (np.array([[200, 200]]), np.array([[300, 300]]))}  # incomplete track
    T = build_tracks(corres, 3, cell=8)
    assert T.shape == (1, 3, 2)
    np.testing.assert_allclose(T[0, 0], p[0])
    np.testing.assert_allclose(T[0, 1], [50.5, 20.5])
    np.testing.assert_allclose(T[0, 2], p[2])


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
