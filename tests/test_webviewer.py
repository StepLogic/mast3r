import base64
import json
import os
import re
import shutil
import subprocess

import numpy as np
import pytest

from tests.test_urdf import _make_model

CHROME = shutil.which('google-chrome') or shutil.which('chromium') or shutil.which('chromium-browser')


def _decode(b64, dtype):
    return np.frombuffer(base64.b64decode(b64), dtype=dtype)


def test_payload_round_trips_clouds_and_hulls(tmp_path):
    from artsfm.model_io import load_model
    from artsfm.webviewer import viewer_payload
    m = load_model(_make_model(tmp_path))
    p = viewer_payload(m)
    for name in ('base', 'part'):
        P = _decode(p[name]['pos'], np.float32).reshape(-1, 3)
        np.testing.assert_allclose(P, m[name]['pts'], atol=1e-5)
        C = _decode(p[name]['rgb'], np.uint8).reshape(-1, 3)
        assert len(C) == len(P)
        hull = p[name]['hull']
        V = _decode(hull['pos'], np.float32).reshape(-1, 3)
        F = _decode(hull['idx'], np.uint32).reshape(-1, 3)
        assert F.max() < len(V)
    assert p['slider']['unit'] == 'degrees' and p['slider']['hi'] == pytest.approx(np.rad2deg(0.7))
    assert p['slider']['ext_lo'] < p['slider']['lo']
    json.dumps(p)


def test_payload_downsamples(tmp_path):
    from artsfm.model_io import load_model
    from artsfm.webviewer import viewer_payload
    m = load_model(_make_model(tmp_path))
    p = viewer_payload(m, max_points=50)
    assert len(_decode(p['base']['pos'], np.float32)) == 150


def test_write_html_is_self_contained(tmp_path):
    from artsfm.webviewer import write_html
    path = write_html(_make_model(tmp_path), str(tmp_path / 'v.html'))
    html = open(path).read()
    assert '__PAYLOAD__' not in html and 'three@0.160.0' in html


@pytest.mark.skipif(CHROME is None, reason='needs headless chrome')
@pytest.mark.parametrize('jtype', ['revolute', 'prismatic', 'free'])
def test_js_joint_matrix_matches_python(tmp_path, jtype):
    """The page's jointMatrix() must reproduce artsfm.articulate.joint_transform exactly."""
    from artsfm.articulate import joint_transform
    from artsfm.model_io import load_model
    from artsfm.webviewer import write_html
    model_dir = _make_model(tmp_path, jtype)
    path = write_html(model_dir, str(tmp_path / 'v.html'))
    value = 0.37
    out = subprocess.run([CHROME, '--headless=new', '--no-sandbox', '--disable-gpu', '--dump-dom',
                          f'file://{os.path.abspath(path)}#check={value}'],
                         capture_output=True, text=True, timeout=60).stdout
    got = np.array(json.loads(re.search(r'<pre id="check">(.*?)</pre>', out, re.S).group(1)))
    want = joint_transform(load_model(model_dir)['joint'], value)
    np.testing.assert_allclose(got, want, atol=1e-9)
