import os
import numpy as np
import pytest

WEIGHTS = 'checkpoints/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric.pth'
OBJ = '/home/robotlab/scratch/GaussianArt/data/MPArt-90/extracted/StorageFurniture_45166'


@pytest.mark.skipif(not (os.path.isfile(WEIGHTS) and os.path.isdir(OBJ)), reason='needs weights + data')
def test_two_view_reconstruction(tmp_path):
    from artsfm.reconstruct import run_sparse_ga, extract_geometry
    from artsfm.tracks import load_pair_correspondences
    paths = [f'{OBJ}/start/train/rgba/0000.png', f'{OBJ}/start/train/rgba/0001.png']
    scene = run_sparse_ga(paths, str(tmp_path), WEIGHTS, niter1=20, niter2=0)
    g = extract_geometry(scene)
    assert g['K'].shape == (2, 3, 3) and g['cam2w'].shape == (2, 4, 4)
    H, W = g['depths'][0].shape
    assert (H, W) == scene.imgs[0].shape[:2]
    assert np.isfinite(g['depths'][0]).all() and (g['depths'][0] > 0).mean() > 0.9
    corres = load_pair_correspondences(str(tmp_path), paths)
    assert (0, 1) in corres and len(corres[0, 1][0]) > 100


@pytest.mark.skipif(not (os.path.isfile(WEIGHTS) and os.path.isdir(OBJ)), reason='needs weights + data')
def test_known_fov_sets_intrinsics(tmp_path):
    from artsfm.reconstruct import run_sparse_ga, extract_geometry
    from artsfm.views import load_fov, model_intrinsics
    paths = [f'{OBJ}/start/train/rgba/0000.png', f'{OBJ}/start/train/rgba/0001.png']
    fov = load_fov(OBJ)
    scene = run_sparse_ga(paths, str(tmp_path), WEIGHTS, niter1=20, niter2=0, fov_x=fov)
    g = extract_geometry(scene)
    H, W = scene.imgs[0].shape[:2]
    np.testing.assert_allclose(g['K'][0], model_intrinsics(fov, (800, 800), (H, W)), rtol=1e-4)
