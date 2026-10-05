import json
import numpy as np

import run_articulation as ra


def test_driver_insufficient_tracks(tmp_path, monkeypatch):
    calls = []
    paths = [f'/x/{i}.png' for i in range(10)]
    monkeypatch.setattr(ra, 'load_frames', lambda d, s: [(f'/x/{s}{i}.png', np.eye(4)) for i in range(10)])
    monkeypatch.setattr(ra, 'load_fov', lambda d: 0.7)
    monkeypatch.setattr(ra, 'select_views', lambda s, e, k, seed_view: (paths[:2 * k], np.array([0] * k + [1] * k)))
    monkeypatch.setattr(ra, 'run_sparse_ga', lambda p, c, w, **kw: calls.append(len(p)) or object())
    monkeypatch.setattr(ra, 'extract_geometry', lambda s: dict(K=None, cam2w=None, depths=None))
    monkeypatch.setattr(ra, 'load_pair_correspondences', lambda c, p: {})
    out = ra.run('/data', str(tmp_path), 'w.pth', k=5, device='cpu')
    assert calls == [10, 6]                      # K=5, then retry with K=3
    assert out['result']['status'] == 'insufficient_tracks'
    saved = json.load(open(tmp_path / 'result.json'))
    assert saved['result']['status'] == 'insufficient_tracks'
