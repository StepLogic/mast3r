import json
import numpy as np

from tests.synth_dense import make_dense_scene, to_model_frame, sample_tracks


def _synthetic_inputs(motion='revolute', n_tracks=150):
    s = make_dense_scene(motion)
    geom, T = to_model_frame(s)
    geom['confs'] = [np.full(d.shape, 3.0) for d in s['depths']]
    tp, tl = sample_tracks(s, n=n_tracks)
    W = np.concatenate([tp.T] + [np.zeros((3, len(tp)))] * (len(s['states']) - 1))  # only view-0 rows are read
    c0inv = np.linalg.inv(s['cam2w'][0])
    if motion == 'revolute':
        o = c0inv[:3, :3] @ s['axis_o'] + c0inv[:3, 3]
        result = dict(status='ok', reason='', type='revolute', labels=tl, axis_o=o,
                      axis_d=c0inv[:3, :3] @ s['axis_d'], angle_deg=40.0, disp=0.0)
    else:
        result = dict(status='ok', reason='', type='prismatic', labels=tl, axis_o=None,
                      axis_d=c0inv[:3, :3] @ s['axis_d'], angle_deg=0.0, disp=0.3)
    return s, geom, W, result


def test_assemble_writes_consistent_model(tmp_path, monkeypatch):
    import artsfm.articulate as art
    s, geom, W, result = _synthetic_inputs()
    paths = [str(tmp_path / f'v{i}.png') for i in range(len(s['states']))]
    monkeypatch.setattr(art, 'foreground_masks', lambda p, hw, confs=None: (s['alphas'], 'alpha'))
    model_dir = art.assemble_model(geom, paths, s['states'], W, result, str(tmp_path), 'synthetic')
    from artsfm.model_io import load_model
    m = load_model(model_dir)
    assert m['joint']['type'] == 'revolute' and abs(m['joint']['upper'] - np.deg2rad(40)) < 1e-9
    assert len(m['base']['pts']) > 500 and len(m['part']['pts']) > 500
    assert m['base_hull'] is not None and m['part_hull'] is not None
    lab = m['labelling']
    assert lab['mask_source'] == 'alpha' and lab['n_base'] > 1000 and lab['n_part'] > 1000
    assert not m['warnings']


def test_assemble_refuses_non_ok_result(tmp_path, capsys):
    from artsfm.articulate import assemble_model
    out = assemble_model(None, [], [], None, dict(status='rigid', reason='part has 3 inliers'), str(tmp_path), 'x')
    assert out is None
    assert 'no articulation to build: rigid' in capsys.readouterr().out
    assert not (tmp_path / 'model').exists()


def test_assemble_tiny_part_warns_and_no_hull(tmp_path, monkeypatch):
    import artsfm.articulate as art
    s, geom, W, result = _synthetic_inputs()
    paths = [str(tmp_path / f'v{i}.png') for i in range(len(s['states']))]
    monkeypatch.setattr(art, 'foreground_masks', lambda p, hw, confs=None: (s['alphas'], 'alpha'))
    real = art.label_dense

    def no_part(*a, **k):
        views = real(*a, **k)
        for v in views:
            v['label'][v['label'] == 1] = 0
        return views
    monkeypatch.setattr(art, 'label_dense', no_part)
    from artsfm.model_io import load_model
    m = load_model(art.assemble_model(geom, paths, s['states'], W, result, str(tmp_path), 'x'))
    assert 'part sparsely observed' in m['warnings']
    assert m['part_hull'] is None
