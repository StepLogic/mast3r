import numpy as np
from PIL import Image

OBJ = '/home/robotlab/scratch/GaussianArt/data/MPArt-90/extracted/StorageFurniture_45166'


def test_crop_like_dust3r_matches_load_images(tmp_path):
    import mast3r.utils.path_to_dust3r  # noqa
    from dust3r.utils.image import load_images
    from artsfm.articulate import crop_like_dust3r
    rng = np.random.default_rng(0)
    for wh in ((800, 800), (640, 480), (300, 500)):
        arr = rng.integers(0, 255, (wh[1], wh[0], 3), dtype=np.uint8)
        p = tmp_path / f'{wh[0]}x{wh[1]}.png'
        Image.fromarray(arr).save(p)
        ref = load_images([str(p)], size=512, verbose=False)[0]['img'][0].permute(1, 2, 0).numpy() * 0.5 + 0.5
        ours = np.asarray(crop_like_dust3r(Image.open(p).convert('RGB')), float) / 255
        assert ours.shape == ref.shape
        assert np.abs(ours - ref).max() < 1.5 / 255


def test_foreground_masks_from_alpha(tmp_path):
    from artsfm.articulate import foreground_masks
    rgba = np.zeros((800, 800, 4), np.uint8)
    rgba[300:500, 200:600, 3] = 255
    p = tmp_path / 'a.png'
    Image.fromarray(rgba).save(p)
    masks, source = foreground_masks([str(p)], (384, 512))
    assert source == 'alpha' and masks[0].shape == (384, 512)
    # centre is foreground, corners are not
    assert masks[0][192, 256] and not masks[0][5, 5]


def test_foreground_masks_confidence_fallback(tmp_path):
    from artsfm.articulate import foreground_masks
    p = tmp_path / 'rgb.png'
    Image.fromarray(np.zeros((480, 640, 3), np.uint8)).save(p)
    conf = np.ones((384, 512))
    conf[100:200, 100:200] = 3.0
    masks, source = foreground_masks([str(p)], (384, 512), confs=[conf])
    assert source == 'confidence'
    assert masks[0].sum() == 100 * 100
