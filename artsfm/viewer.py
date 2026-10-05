"""Pure helpers for the articulated-model viewer: model discovery, slider units, glb frames, info text."""
import glob
import os
import tempfile

import numpy as np
import trimesh

from artsfm.articulate import joint_transform

OPENCV_TO_GL = np.diag([1.0, -1.0, -1.0])   # three.js / Babylon expect y up, camera looking down -z
BASE_RGB = np.array([29, 127, 166]) / 255
PART_RGB = np.array([217, 72, 43]) / 255
AXIS_RGBA = [242, 182, 48, 255]


def list_models(root):
    return sorted(os.path.dirname(p) for p in glob.glob(os.path.join(root, '*', 'model', 'model.json')))


def slider_spec(joint, extend=False):
    lo, hi = float(joint['lower']), float(joint['upper'])
    if extend:
        r = hi - lo
        lo, hi = lo - 0.5 * r, hi + 0.5 * r
    if joint.get('unit') == 'rad':
        return float(np.rad2deg(lo)), float(np.rad2deg(hi)), 'degrees'
    return lo, hi, 'reconstruction units'


def to_joint_value(joint, slider_value):
    return float(np.deg2rad(slider_value)) if joint.get('unit') == 'rad' else float(slider_value)


def sweep_values(joint, n=12):
    up = list(np.linspace(0.0, float(joint['upper']), n // 2 + 1))
    return up + up[-2::-1]


def part_points_at(model, value):
    T = joint_transform(model['joint'], value)
    return model['part']['pts'] @ T[:3, :3].T + T[:3, 3]


def _cloud(P, C):
    rgba = np.c_[np.clip(np.round(C * 255), 0, 255), np.full(len(P), 255)].astype(np.uint8)
    return trimesh.PointCloud(P @ OPENCV_TO_GL.T, colors=rgba)


def build_glb(model, value, color_mode='rgb', show_axis=True, show_hulls=False, out_path=None):
    scene = trimesh.Scene()
    base_P, base_C = model['base']['pts'], model['base']['rgb']
    part_P = part_points_at(model, value)
    part_C = model['part']['rgb']
    if color_mode == 'part':
        base_C = np.tile(BASE_RGB, (len(base_P), 1))
        part_C = np.tile(PART_RGB, (len(part_P), 1))
    if len(base_P):
        scene.add_geometry(_cloud(base_P, base_C), geom_name='base')
    if len(part_P):
        scene.add_geometry(_cloud(part_P, part_C), geom_name='part')
    allP = np.concatenate([base_P, part_P]) if len(part_P) else base_P
    extent = float(np.ptp(allP, 0).max()) if len(allP) else 1.0
    if show_axis:
        j = model['joint']
        d = np.asarray(j['axis_dir'], float)
        d /= np.linalg.norm(d)
        pc = part_P.mean(0) if len(part_P) else allP.mean(0)
        if j['type'] == 'prismatic':
            o = pc
        else:
            o0 = np.asarray(j['axis_origin'], float)
            o = o0 + d * ((pc - o0) @ d)
        seg = np.stack([o - 0.6 * extent * d, o + 0.6 * extent * d]) @ OPENCV_TO_GL.T
        cyl = trimesh.creation.cylinder(radius=0.004 * extent, segment=seg)
        cyl.visual.face_colors = AXIS_RGBA
        scene.add_geometry(cyl, geom_name='axis')
    if show_hulls:
        T = joint_transform(model['joint'], value)
        for name, rgba in (('base', [29, 127, 166, 70]), ('part', [217, 72, 43, 70])):
            hull = model.get(f'{name}_hull')
            if hull is None:
                continue
            h = hull.copy()
            V = np.asarray(h.vertices)
            if name == 'part':
                V = V @ T[:3, :3].T + T[:3, 3]
            h.vertices = V @ OPENCV_TO_GL.T
            h.visual.face_colors = rgba
            scene.add_geometry(h, geom_name=f'{name}_hull')
    out_path = out_path or tempfile.NamedTemporaryFile(suffix='.glb', delete=False).name
    scene.export(out_path)
    return out_path


def info_markdown(model, result=None):
    j = model['joint']
    unit = 'deg' if j.get('unit') == 'rad' else 'units'
    observed = np.rad2deg(j['observed']) if j.get('unit') == 'rad' else j['observed']
    lines = [f"**{model.get('object', '')}** · joint **{j['type']}**"
             + (' (approximate screw)' if j.get('approximate') else ''),
             f"- observed motion: {observed:.1f} {unit}",
             f"- axis direction (model frame): {np.round(j['axis_dir'], 3).tolist()}"]
    lab = model.get('labelling')
    if lab:
        lines.append(f"- points: body {lab['n_base']:,} · part {lab['n_part']:,} · fallback {lab['n_fallback']:,} · "
                     f"unlabelled {lab['n_unlabelled']:,} (mask: {lab['mask_source']})")
    for w in model.get('warnings', []):
        lines.append(f"- ⚠ {w}")
    m = (result or {}).get('metrics') or {}
    if 'axis_angle_err_deg' in m:
        lines.append(f"- vs ground truth: axis error {m['axis_angle_err_deg']:.1f}°"
                     + (f", state error {m['state_err']:.1f}" if m.get('state_err') is not None else ''))
    return '\n'.join(lines)
