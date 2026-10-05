import json
import sys
import numpy as np
import pytest


def _run_dir(tmp_path, metrics):
    d = tmp_path / 'run'
    d.mkdir()
    json.dump(dict(result=dict(status='ok', type='revolute', axis_d=[0, 0, 1], angle_deg=30.0), metrics=metrics),
              open(d / 'result.json', 'w'))
    return d


def test_build_model_reuses_run_settings_and_fails_when_nothing_built(tmp_path, monkeypatch):
    import build_model as bm
    seen = {}
    d = _run_dir(tmp_path, dict(k=2, views=['a.png', 'b.png', 'c.png', 'd.png'], data_dir='/data',
                                optim_level='coarse', niter1=50, niter2=0, known_intrinsics=False))
    monkeypatch.setattr(bm.ra, 'reconstruct_and_track',
                        lambda paths, cache, w, dev, lvl, n1, n2, fov: seen.update(lvl=lvl, n1=n1, n2=n2, fov=fov) or ({}, np.zeros((12, 5))))
    monkeypatch.setattr(bm, 'analyze', lambda W, states: dict(status='rigid', reason='part has 2 inliers', type='rigid'))
    monkeypatch.setattr(bm, 'load_frames', lambda d, s: [])
    monkeypatch.setattr(sys, 'argv', ['build_model.py', '--out_dir', str(d)])
    with pytest.raises(SystemExit) as e:
        bm.main()
    assert e.value.code != 0
    assert seen == dict(lvl='coarse', n1=50, n2=0, fov=None)


def test_build_model_reports_axis_and_angle_drift(tmp_path, monkeypatch, capsys):
    import build_model as bm
    d = _run_dir(tmp_path, dict(k=2, views=['a.png', 'b.png', 'c.png', 'd.png'], data_dir='/data'))
    monkeypatch.setattr(bm.ra, 'reconstruct_and_track', lambda *a: ({}, np.zeros((12, 5))))
    monkeypatch.setattr(bm, 'load_fov', lambda d: 0.7)
    monkeypatch.setattr(bm, 'analyze', lambda W, states: dict(status='ok', type='revolute', axis_d=[0, 1, 0], angle_deg=10.0))
    monkeypatch.setattr(bm, 'load_frames', lambda d, s: [])
    monkeypatch.setattr(bm, 'assemble_model', lambda *a, **k: None)
    monkeypatch.setattr(sys, 'argv', ['build_model.py', '--out_dir', str(d)])
    with pytest.raises(SystemExit):
        bm.main()
    out = capsys.readouterr().out
    assert 'axis differs by 90.0°' in out and 'angle differs by 20.0°' in out
