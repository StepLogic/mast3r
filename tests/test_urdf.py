import os
import numpy as np
import pytest

from artsfm.articulate import convex_hull_or_none, joint_transform
from artsfm.model_io import write_model


def _box(c, half, n=600, seed=0):
    rng = np.random.default_rng(seed)
    return np.asarray(c) + rng.uniform(-1, 1, (n, 3)) * np.asarray(half)


def _make_model(tmp_path, jtype='revolute', up=(0, -1, 0)):
    base = _box([0, 0, 3], [0.5, 0.4, 0.3])
    part = _box([0, 0.45, 2.75], [0.45, 0.05, 0.03], seed=1)
    if jtype == 'revolute':
        joint = dict(type='revolute', axis_origin=[0.45, 0.45, 2.72], axis_dir=[0, -1, 0], observed=0.7,
                     lower=0.0, upper=0.7, unit='rad', approximate=False, disp_along=0.0)
    elif jtype == 'prismatic':
        joint = dict(type='prismatic', axis_origin=None, axis_dir=[0, 0, -1], observed=0.2,
                     lower=0.0, upper=0.2, unit='recon', approximate=False, disp_along=0.0)
    else:
        joint = dict(type='free', axis_origin=[0, 0, 3], axis_dir=[0, -1, 0], observed=0.7,
                     lower=0.0, upper=0.7, unit='rad', approximate=True, disp_along=0.1)
    model = dict(object='toy', joint=joint, motion=joint_transform(joint, joint['observed']),
                 up=np.asarray(up, float), gt_frame=None, warnings=[])
    clouds = dict(base=(base, np.ones_like(base) * 0.5), part=(part, np.ones_like(part) * 0.8))
    hulls = dict(base=convex_hull_or_none(base), part=convex_hull_or_none(part))
    write_model(tmp_path / 'model', model, clouds, hulls)
    return str(tmp_path / 'model')


def test_rotation_between_maps_vectors():
    from artsfm.urdf import rotation_between
    for a, b in (([0, -1, 0], [0, 0, 1]), ([0, 0, -1], [0, 0, 1]), ([1, 2, 3], [0, 0, 1])):
        R = rotation_between(np.array(a, float), np.array(b, float))
        np.testing.assert_allclose(R @ (np.array(a) / np.linalg.norm(a)), b, atol=1e-12)
        np.testing.assert_allclose(R @ R.T, np.eye(3), atol=1e-12)


@pytest.mark.parametrize('jtype', ['revolute', 'prismatic'])
def test_urdf_loads_and_matches_motion_in_mujoco(tmp_path, jtype):
    from artsfm.urdf import write_urdf, check_urdf
    urdf = write_urdf(_make_model(tmp_path, jtype))
    assert check_urdf(urdf) < 1e-6


def test_urdf_prismatic_scaled_check(tmp_path):
    from artsfm.urdf import write_urdf, check_urdf
    urdf = write_urdf(_make_model(tmp_path, 'prismatic'), scale=2.5)
    assert check_urdf(urdf) < 1e-6
    assert 'upper="0.500000"' in open(urdf).read()


def test_export_up_maps_to_z_and_base_on_ground(tmp_path):
    from artsfm.urdf import export_transform, apply_export
    from artsfm.model_io import load_model
    m = load_model(_make_model(tmp_path))
    E = export_transform(m)
    np.testing.assert_allclose(E['R'] @ m['up'], [0, 0, 1], atol=1e-12)
    P = apply_export(E, m['base']['pts'])
    assert abs(P[:, 2].min()) < 1e-9 and np.abs(P[:, :2].mean(0)).max() < 1e-9


def test_gaussianart_names_and_paths(tmp_path):
    from artsfm.urdf import write_urdf
    urdf = write_urdf(_make_model(tmp_path))
    txt = open(urdf).read()
    for s in ('<link name="base">', '<link name="link_1">', '<joint name="joint_1" type="revolute">',
              'meshes/base.obj', 'meshes/base_collision.obj', 'meshes/link_1.obj', 'meshes/link_1_collision.obj',
              'strippath="false"'):
        assert s in txt
    for f in ('base.obj', 'base_collision.obj', 'link_1.obj', 'link_1_collision.obj'):
        assert os.path.isfile(os.path.join(os.path.dirname(urdf), 'meshes', f))


def test_free_joint_refused(tmp_path):
    from artsfm.urdf import write_urdf
    with pytest.raises(ValueError, match='screw joint'):
        write_urdf(_make_model(tmp_path, 'free'))


def test_write_urdf_refuses_without_hull(tmp_path):
    from artsfm.urdf import write_urdf
    d = _make_model(tmp_path)
    os.remove(os.path.join(d, 'part_hull.obj'))
    with pytest.raises(ValueError, match='hull'):
        write_urdf(d)


def test_check_detects_misplaced_exported_mesh(tmp_path):
    import trimesh, json
    from artsfm.urdf import write_urdf, check_urdf
    urdf = write_urdf(_make_model(tmp_path))
    ex = json.load(open(os.path.join(os.path.dirname(urdf), 'export.json')))
    for f in ('link_1.obj', 'link_1_collision.obj'):   # classic bug: mesh not offset by -pivot
        p = os.path.join(os.path.dirname(urdf), 'meshes', f)
        m = trimesh.load(p, force='mesh')
        m.vertices = m.vertices + np.array(ex['pivot'])
        m.export(p)
    assert check_urdf(urdf) > 1e-3


def test_check_detects_misplaced_base_mesh(tmp_path):
    import trimesh
    from artsfm.urdf import write_urdf, check_urdf
    urdf = write_urdf(_make_model(tmp_path))
    for f in ('base.obj', 'base_collision.obj'):
        p = os.path.join(os.path.dirname(urdf), 'meshes', f)
        m = trimesh.load(p, force='mesh')
        m.vertices = m.vertices + np.array([0, 0, 0.5])
        m.export(p)
    assert check_urdf(urdf) > 1e-3
