"""Thin wrapper around MASt3R-SfM sparse global alignment."""
import os

import numpy as np
import torch

import mast3r.utils.path_to_dust3r  # noqa
from dust3r.utils.image import load_images
from mast3r.cloud_opt.sparse_ga import sparse_global_alignment
from mast3r.image_pairs import make_pairs
from mast3r.model import AsymmetricMASt3R
from PIL import Image

from artsfm.views import model_intrinsics


def run_sparse_ga(image_paths, cache_dir, weights, device='cpu', optim_level='refine', image_size=512,
                  lr1=0.07, niter1=300, lr2=0.01, niter2=300, matching_conf_thr=5.0, fov_x=None):
    """fov_x: known horizontal field of view (radians) used to initialise all intrinsics."""
    os.makedirs(cache_dir, exist_ok=True)
    model = AsymmetricMASt3R.from_pretrained(weights).to(device)
    imgs = load_images(list(image_paths), size=image_size, verbose=False)
    pairs = make_pairs(imgs, scene_graph='complete', prefilter=None, symmetrize=True)
    if optim_level == 'coarse':
        niter2 = 0
    init = {}
    if fov_x is not None:
        for path, im in zip(image_paths, imgs):
            K = model_intrinsics(fov_x, Image.open(path).size, tuple(im['true_shape'][0]))
            init[path] = {'intrinsics': torch.tensor(K, dtype=torch.float32, device=device)}
    return sparse_global_alignment(list(image_paths), pairs, cache_dir, model, lr1=lr1, niter1=niter1,
                                   lr2=lr2, niter2=niter2, device=device,
                                   opt_depth='depth' in optim_level, matching_conf_thr=matching_conf_thr,
                                   init=init)


def extract_geometry(scene):
    with torch.no_grad():
        _, depthmaps, _ = scene.get_dense_pts3d(clean_depth=False)
    K = np.stack([k.detach().cpu().numpy() for k in scene.intrinsics])
    cam2w = scene.get_im_poses().detach().cpu().numpy()
    depths = [d.detach().cpu().numpy().reshape(im.shape[:2]) for d, im in zip(depthmaps, scene.imgs)]
    return dict(K=K, cam2w=cam2w, depths=depths)
