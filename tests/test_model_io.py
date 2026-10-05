import json
import numpy as np
import pytest

from tests.synth import make_scene


def test_ply_roundtrip_including_empty(tmp_path):
    from artsfm.model_io import write_ply, read_ply
    P = np.random.default_rng(0).standard_normal((50, 3))
    C = np.random.default_rng(1).uniform(0, 1, (50, 3))
    write_ply(tmp_path / 'a.ply', P, C)
    P2, C2 = read_ply(tmp_path / 'a.ply')
    np.testing.assert_allclose(P2, P, atol=1e-6)
    np.testing.assert_allclose(C2, C, atol=1 / 255)
    write_ply(tmp_path / 'e.ply', np.zeros((0, 3)), np.zeros((0, 3)))
    assert read_ply(tmp_path / 'e.ply')[0].shape == (0, 3)


@pytest.mark.parametrize('value', [-1.0, 0.0, 0.3, 2.5])
def test_joint_transform_is_rigid_any_value(value):
    from artsfm.articulate import joint_transform
    for joint in (dict(type='revolute', axis_origin=[1, 2, 3], axis_dir=[0, 0.6, 0.8], observed=0.6),
                  dict(type='prismatic', axis_origin=None, axis_dir=[1, 0, 0], observed=0.2),
                  dict(type='free', axis_origin=[0, 0, 1], axis_dir=[0, 1, 0], observed=0.5, disp_along=0.1)):
        T = joint_transform(joint, value)
        np.testing.assert_allclose(T[:3, :3] @ T[:3, :3].T, np.eye(3), atol=1e-12)
        assert abs(np.linalg.det(T[:3, :3]) - 1) < 1e-12
    T = joint_transform(dict(type='revolute', axis_origin=[1, 2, 3], axis_dir=[0, 0.6, 0.8], observed=0.6), value)
    np.testing.assert_allclose(T[:3, :3] @ [1, 2, 3] + T[:3, 3], [1, 2, 3], atol=1e-12)  # axis point fixed


def test_joint_from_result_reproduces_relative_motion():
    from artsfm.articulate import joint_from_result, joint_transform
    from artsfm.factorize import analyze, relative_motion
    for motion in ('revolute', 'prismatic'):
        W, states, labels, gt = make_scene(motion)
        r = analyze(W, states)
        j = joint_from_result(r)
        assert j['lower'] == 0 and j['upper'] == j['observed'] > 0
        T = joint_transform(j, j['observed'])
        A, b = relative_motion(W, states, np.flatnonzero(r['labels'] == 0), np.flatnonzero(r['labels'] == 1))
        np.testing.assert_allclose(T[:3, :3], A, atol=1e-6)
        np.testing.assert_allclose(T[:3, 3], b, atol=1e-6)


def test_fuse_maps_end_view_part_points_back():
    from artsfm.articulate import fuse, joint_transform
    T = joint_transform(dict(type='prismatic', axis_origin=None, axis_dir=[1, 0, 0], observed=0.5), 0.5)
    rng = np.random.default_rng(0)
    lid = rng.uniform(0, 1, (400, 3))
    views = [dict(pts=lid, rgb=np.ones((400, 3)), label=np.ones(400, int)),
             dict(pts=lid + [0.5, 0, 0], rgb=np.ones((400, 3)), label=np.ones(400, int))]
    out = fuse(views, np.array([0, 1]), T, max_points=10_000)
    P = out['part'][0]
    from scipy.spatial import cKDTree
    assert cKDTree(lid).query(P)[0].max() < 0.05
    assert out['base'][0].shape == (0, 3)


def test_fuse_downsamples_to_max_points():
    from artsfm.articulate import fuse
    rng = np.random.default_rng(0)
    P = rng.uniform(0, 1, (20000, 3))
    out = fuse([dict(pts=P, rgb=np.ones((20000, 3)), label=np.zeros(20000, int))], np.array([0]), np.eye(4), max_points=3000)
    assert 0 < len(out['base'][0]) <= 3000


def test_hull_none_for_degenerate():
    from artsfm.articulate import convex_hull_or_none
    assert convex_hull_or_none(np.zeros((3, 3))) is None
    assert convex_hull_or_none(np.c_[np.random.rand(50, 2), np.zeros(50)]) is None
    assert convex_hull_or_none(np.random.rand(50, 3)).is_watertight


def test_estimate_up_from_upright_cameras():
    from artsfm.articulate import estimate_up
    from tests.synth_dense import look_at
    cams = np.stack([look_at(np.array([3 * np.cos(a), 3 * np.sin(a), 1.0]), np.zeros(3)) for a in np.linspace(0, 1, 5)])
    up = estimate_up(cams)
    world_up_in_view0 = cams[0][:3, :3].T @ np.array([0, 0, 1.0])
    assert np.degrees(np.arccos(np.clip(up @ world_up_in_view0, -1, 1))) < 25


def test_model_roundtrip(tmp_path):
    from artsfm.model_io import write_model, load_model
    from artsfm.articulate import convex_hull_or_none
    P = np.random.rand(100, 3)
    clouds = dict(base=(P, np.ones((100, 3))), part=(np.zeros((0, 3)), np.zeros((0, 3))))
    hulls = dict(base=convex_hull_or_none(P), part=None)
    model = dict(object='x', joint=dict(type='revolute', axis_origin=[0, 0, 0], axis_dir=[0, 0, 1], observed=0.5,
                                        lower=0.0, upper=0.5, unit='rad'), motion=np.eye(4), warnings=['w'])
    write_model(tmp_path / 'model', model, clouds, hulls)
    m = load_model(tmp_path / 'model')
    assert m['object'] == 'x' and m['warnings'] == ['w']
    assert m['base']['pts'].shape == (100, 3) and m['part']['pts'].shape == (0, 3)
    assert m['base_hull'] is not None and m['part_hull'] is None
    assert isinstance(m['motion'], np.ndarray)
    assert json.load(open(tmp_path / 'model' / 'model.json'))['links'] == {'base': 'base.ply', 'part': 'part.ply'}
