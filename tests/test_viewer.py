import numpy as np
import trimesh
from scipy.spatial import cKDTree

from tests.test_urdf import _make_model


def test_build_glb_places_part_at_joint_value(tmp_path):
    from artsfm.viewer import build_glb, OPENCV_TO_GL
    from artsfm.model_io import load_model
    from artsfm.articulate import joint_transform
    m = load_model(_make_model(tmp_path))
    path = build_glb(m, 0.4, out_path=str(tmp_path / 'f.glb'))
    g = trimesh.load(path)
    T = joint_transform(m['joint'], 0.4)
    want = (m['part']['pts'] @ T[:3, :3].T + T[:3, 3]) @ OPENCV_TO_GL.T
    got = np.asarray(g.geometry['part'].vertices)
    assert cKDTree(want).query(got)[0].max() < 1e-5
    assert 'axis' in g.geometry and 'base' in g.geometry


def test_build_glb_with_hulls_and_part_colours(tmp_path):
    from artsfm.viewer import build_glb
    from artsfm.model_io import load_model
    m = load_model(_make_model(tmp_path))
    g = trimesh.load(build_glb(m, 0.0, color_mode='part', show_hulls=True, out_path=str(tmp_path / 'h.glb')))
    assert {'base_hull', 'part_hull'} <= set(g.geometry)


def test_list_models_only_valid(tmp_path):
    from artsfm.viewer import list_models
    _make_model(tmp_path / 'A')
    (tmp_path / 'B' / 'model').mkdir(parents=True)  # no model.json
    assert list_models(str(tmp_path)) == [str(tmp_path / 'A' / 'model')]


def test_slider_and_sweep_units():
    from artsfm.viewer import slider_spec, to_joint_value, sweep_values
    j = dict(type='revolute', lower=0.0, upper=np.deg2rad(40), unit='rad')
    lo, hi, label = slider_spec(j)
    assert (lo, round(hi, 6), label) == (0.0, 40.0, 'degrees')
    assert abs(to_joint_value(j, 20.0) - np.deg2rad(20)) < 1e-12
    lo, hi, _ = slider_spec(j, extend=True)
    assert lo == -20.0 and round(hi, 6) == 60.0
    v = sweep_values(j, n=12)
    assert v[0] == 0 and abs(max(v) - j['upper']) < 1e-12 and v[-1] == 0
    p = dict(type='prismatic', lower=0.0, upper=0.3, unit='recon')
    assert slider_spec(p)[2] == 'reconstruction units' and to_joint_value(p, 0.1) == 0.1


def test_info_markdown_mentions_warnings_and_metrics(tmp_path):
    from artsfm.viewer import info_markdown
    from artsfm.model_io import load_model
    m = load_model(_make_model(tmp_path))
    m['warnings'] = ['part sparsely observed']
    txt = info_markdown(m, dict(metrics=dict(axis_angle_err_deg=2.7, state_err=0.4)))
    assert 'revolute' in txt and 'part sparsely observed' in txt and '2.7' in txt
