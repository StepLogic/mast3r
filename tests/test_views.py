import json
import numpy as np

from artsfm.views import load_frames, select_views


def _frame(az):
    c2w = np.eye(4)
    c2w[:3, 3] = [np.cos(az), np.sin(az), 0.3]
    return c2w


def test_select_views_clusters_around_seed_and_pairs_states():
    start = [(f's{i}', _frame(a)) for i, a in enumerate(np.linspace(0, 2 * np.pi, 20, endpoint=False))]
    end = [(f'e{i}', _frame(a + 0.05)) for i, a in enumerate(np.linspace(0, 2 * np.pi, 20, endpoint=False))]
    paths, states = select_views(start, end, k=3, seed_view=5)
    assert paths[0] == 's5'
    assert sorted(paths[:3]) == ['s4', 's5', 's6']
    assert sorted(paths[3:]) == ['e4', 'e5', 'e6']
    assert states.tolist() == [0, 0, 0, 1, 1, 1]


def test_load_frames_appends_png(tmp_path):
    (tmp_path / 'start/train/rgba').mkdir(parents=True)
    meta = {'frames': [{'file_path': 'start/train/rgba/0000', 'transform_matrix': np.eye(4).tolist()}]}
    (tmp_path / 'transforms_train_start.json').write_text(json.dumps(meta))
    frames = load_frames(str(tmp_path), 'start')
    assert frames[0][0] == str(tmp_path / 'start/train/rgba/0000.png')
    assert frames[0][1].shape == (4, 4)


def test_model_intrinsics_from_fov():
    from artsfm.views import model_intrinsics
    fov = 2 * np.arctan(0.5)  # f = 800 px at 800 px width
    K = model_intrinsics(fov, orig_wh=(800, 800), model_hw=(384, 512))
    np.testing.assert_allclose(K, [[512, 0, 256], [0, 512, 192], [0, 0, 1]])


def test_load_fov(tmp_path):
    from artsfm.views import load_fov
    (tmp_path / 'transforms_train_start.json').write_text(json.dumps({'camera_angle_x': 0.7, 'frames': []}))
    assert load_fov(str(tmp_path)) == 0.7
