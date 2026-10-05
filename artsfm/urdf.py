"""Export an articulated model folder as a URDF (GaussianArt conventions) and check it in MuJoCo."""
import argparse
import json
import os

import numpy as np

from artsfm.articulate import joint_transform
from artsfm.model_io import load_model


def rotation_between(a, b):
    """Smallest rotation taking unit direction a to unit direction b."""
    a = np.asarray(a, float) / np.linalg.norm(a)
    b = np.asarray(b, float) / np.linalg.norm(b)
    v, c = np.cross(a, b), float(a @ b)
    if c < -1 + 1e-12:  # opposite: rotate pi about any axis orthogonal to a
        w = np.cross(a, [1.0, 0, 0]) if abs(a[0]) < 0.9 else np.cross(a, [0, 1.0, 0])
        w /= np.linalg.norm(w)
        return 2 * np.outer(w, w) - np.eye(3)
    K = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    return np.eye(3) + K + K @ K / (1 + c)


def export_transform(model, scale=None, use_gt=False):
    """Model frame -> gravity-aligned export frame: up -> +z, base on z = 0, base x-y centroid at 0."""
    gt = model.get('gt_frame') if use_gt else None
    if use_gt and not gt:
        raise ValueError('model has no gt_frame; build it with GT poses available or drop --use_gt_frame')
    up = np.asarray(gt['up'] if gt else model['up'], float)
    s = float(scale if scale is not None else (gt['scale'] if gt else 1.0))
    R = rotation_between(up, [0, 0, 1.0])
    B = s * model['base']['pts'] @ R.T
    t = np.array([-B[:, 0].mean(), -B[:, 1].mean(), -B[:, 2].min()])
    return dict(R=R, s=s, t=t)


def apply_export(E, P):
    return E['s'] * np.asarray(P, float) @ E['R'].T + E['t']


def _fmt(v):
    return ' '.join(f'{x:.6f}' for x in v)


def _link_xml(name, mass, com):
    return [f'  <link name="{name}">',
            f'    <inertial><origin xyz="{_fmt(com)}"/><mass value="{mass}"/>'
            f'<inertia ixx="0.01" iyy="0.01" izz="0.01" ixy="0" ixz="0" iyz="0"/></inertial>',
            f'    <visual name="{name}_visual"><origin xyz="0 0 0"/><geometry><mesh filename="meshes/{name}.obj"/></geometry></visual>',
            f'    <collision name="{name}_collision"><origin xyz="0 0 0"/><geometry><mesh filename="meshes/{name}_collision.obj"/></geometry></collision>',
            '  </link>']


def write_urdf(model_dir, out_dir=None, scale=None, use_gt=False, lower=None, upper=None):
    m = load_model(model_dir)
    j = m['joint']
    if j['type'] == 'free':
        raise ValueError('screw joint (not representable in URDF); URDF export disabled')
    if m['base_hull'] is None or m['part_hull'] is None:
        raise ValueError('missing base or part hull (part too sparse or coplanar); cannot export URDF')
    E = export_transform(m, scale, use_gt)
    axis = E['R'] @ np.asarray(j['axis_dir'], float)
    axis /= np.linalg.norm(axis)
    part_c = apply_export(E, m['part']['pts']).mean(0)
    if j['type'] == 'revolute':
        o = apply_export(E, np.asarray(j['axis_origin'], float)[None])[0]
        pivot = o + axis * ((part_c - o) @ axis)
    else:
        pivot = part_c
    lo = float(j['lower'] if lower is None else lower)
    hi = float(j['upper'] if upper is None else upper)
    if j['type'] == 'prismatic':
        lo, hi = lo * E['s'], hi * E['s']

    out_dir = out_dir or model_dir
    os.makedirs(os.path.join(out_dir, 'meshes'), exist_ok=True)
    base = m['base_hull'].copy()
    base.vertices = apply_export(E, base.vertices)
    part = m['part_hull'].copy()
    part.vertices = apply_export(E, part.vertices) - pivot
    for name, mesh in (('base', base), ('link_1', part)):
        mesh.export(os.path.join(out_dir, 'meshes', f'{name}.obj'))
        mesh.export(os.path.join(out_dir, 'meshes', f'{name}_collision.obj'))

    name = m.get('object', 'object')
    xml = ['<?xml version="1.0"?>', f'<robot name="{name}">',
           '  <mujoco><compiler strippath="false" discardvisual="false"/></mujoco>']
    xml += _link_xml('base', 10.0, base.vertices.mean(0))
    xml += _link_xml('link_1', 1.0, part.vertices.mean(0))
    xml += [f'  <joint name="joint_1" type="{j["type"]}">',
            '    <parent link="base"/><child link="link_1"/>',
            f'    <origin xyz="{_fmt(pivot)}" rpy="0 0 0"/><axis xyz="{_fmt(axis)}"/>',
            f'    <limit lower="{lo:.6f}" upper="{hi:.6f}" effort="100" velocity="2"/>',
            '    <dynamics damping="0.5" friction="0"/>',
            '  </joint>', '</robot>']
    urdf = os.path.join(out_dir, f'{name}.urdf')
    with open(urdf, 'w') as f:
        f.write('\n'.join(xml) + '\n')
    with open(os.path.join(out_dir, 'export.json'), 'w') as f:
        json.dump(dict(model_dir=os.path.abspath(model_dir), R=E['R'].tolist(), s=E['s'], t=E['t'].tolist(),
                       pivot=pivot.tolist(), axis=axis.tolist(), lower=lo, upper=hi, type=j['type']), f, indent=2)
    return urdf


def check_urdf(urdf_path):
    """Load in MuJoCo and compare the meshes MuJoCo actually placed with where the model puts them.

    For q = lower and q = upper: every mesh geom's loaded vertices, transformed by MuJoCo's own
    forward kinematics, must coincide with the exported hulls moved by the model's joint motion
    (base hull fixed, part hull by T(q)). Returns the max relative (to object size) distance.
    """
    import mujoco
    from scipy.spatial import cKDTree
    ex = json.load(open(os.path.join(os.path.dirname(urdf_path), 'export.json')))
    m = load_model(ex['model_dir'])
    E = dict(R=np.array(ex['R']), s=ex['s'], t=np.array(ex['t']))
    mdl = mujoco.MjModel.from_xml_path(urdf_path)
    d = mujoco.MjData(mdl)
    part_body = mujoco.mj_name2id(mdl, mujoco.mjtObj.mjOBJ_BODY, 'link_1')
    base_want = apply_export(E, m['base_hull'].vertices)
    extent = float(np.ptp(base_want, 0).max())
    worst = 0.0
    for q in (ex['lower'], ex['upper']):
        d.qpos[0] = q
        mujoco.mj_kinematics(mdl, d)
        q_model = q if ex['type'] == 'revolute' else q / ex['s']
        T = joint_transform(m['joint'], q_model)
        part_want = apply_export(E, m['part_hull'].vertices @ T[:3, :3].T + T[:3, 3])
        n_mesh = 0
        for g in range(mdl.ngeom):
            if mdl.geom_type[g] != mujoco.mjtGeom.mjGEOM_MESH:
                continue
            n_mesh += 1
            mid = mdl.geom_dataid[g]
            adr, n = mdl.mesh_vertadr[mid], mdl.mesh_vertnum[mid]
            V = mdl.mesh_vert[adr:adr + n] @ d.geom_xmat[g].reshape(3, 3).T + d.geom_xpos[g]
            want = part_want if mdl.geom_bodyid[g] == part_body else base_want
            dist = max(cKDTree(want).query(V)[0].max(), cKDTree(V).query(want)[0].max())
            worst = max(worst, float(dist) / max(extent, 1e-12))
        if n_mesh < 4:
            raise ValueError(f'expected 4 mesh geoms (visual + collision per link), MuJoCo loaded {n_mesh}')
    return worst


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('model_dir')
    p.add_argument('--out_dir', default=None)
    p.add_argument('--scale', type=float, default=None)
    p.add_argument('--use_gt_frame', action='store_true', help='up and scale from the GT alignment (MPArt-90)')
    p.add_argument('--lower', type=float, default=None)
    p.add_argument('--upper', type=float, default=None)
    p.add_argument('--no_check', action='store_true')
    a = p.parse_args()
    urdf = write_urdf(a.model_dir, a.out_dir, a.scale, a.use_gt_frame, a.lower, a.upper)
    print(f'wrote {urdf}')
    if not a.no_check:
        err = check_urdf(urdf)
        print(f'MuJoCo check: max relative error {err:.2e} ({"PASS" if err < 1e-6 else "FAIL"})')


if __name__ == '__main__':
    main()
