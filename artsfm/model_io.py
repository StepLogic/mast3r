"""Articulated model folder: model.json + per-link point clouds (binary PLY) + convex hulls."""
import json
import os

import numpy as np
import trimesh

PLY_DTYPE = np.dtype([('x', '<f4'), ('y', '<f4'), ('z', '<f4'), ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')])


def write_ply(path, P, C):
    P = np.asarray(P, float).reshape(-1, 3)
    C = np.asarray(C, float).reshape(-1, 3)
    data = np.zeros(len(P), PLY_DTYPE)
    data['x'], data['y'], data['z'] = P.T
    rgb = np.clip(np.round(C * 255), 0, 255).astype(np.uint8)
    data['red'], data['green'], data['blue'] = rgb.T
    header = ('ply\nformat binary_little_endian 1.0\n'
              f'element vertex {len(P)}\n'
              'property float x\nproperty float y\nproperty float z\n'
              'property uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n')
    with open(path, 'wb') as f:
        f.write(header.encode('ascii'))
        f.write(data.tobytes())


def read_ply(path):
    with open(path, 'rb') as f:
        n = None
        while True:
            line = f.readline().decode('ascii').strip()
            if line.startswith('element vertex'):
                n = int(line.split()[-1])
            if line == 'end_header':
                break
        data = np.frombuffer(f.read(n * PLY_DTYPE.itemsize), PLY_DTYPE, count=n)
    P = np.stack([data['x'], data['y'], data['z']], 1).astype(float)
    C = np.stack([data['red'], data['green'], data['blue']], 1).astype(float) / 255
    return P, C


def _jsonable(x):
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, (np.floating, np.integer, np.bool_)):
        return x.item()
    if isinstance(x, dict):
        return {k: _jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_jsonable(v) for v in x]
    return x


def write_model(model_dir, model, clouds, hulls):
    model_dir = str(model_dir)
    os.makedirs(model_dir, exist_ok=True)
    for name in ('base', 'part'):
        write_ply(os.path.join(model_dir, f'{name}.ply'), *clouds[name])
        hull_path = os.path.join(model_dir, f'{name}_hull.obj')
        if hulls.get(name) is not None:
            hulls[name].export(hull_path)
        elif os.path.exists(hull_path):
            os.remove(hull_path)
    model = dict(model, links={'base': 'base.ply', 'part': 'part.ply'})
    with open(os.path.join(model_dir, 'model.json'), 'w') as f:
        json.dump(_jsonable(model), f, indent=2)


def load_model(model_dir):
    model_dir = str(model_dir)
    with open(os.path.join(model_dir, 'model.json')) as f:
        m = json.load(f)
    m['dir'] = model_dir
    m['motion'] = np.asarray(m.get('motion', np.eye(4)), float)
    for name in ('base', 'part'):
        P, C = read_ply(os.path.join(model_dir, f'{name}.ply'))
        m[name] = dict(pts=P, rgb=C)
        hull_path = os.path.join(model_dir, f'{name}_hull.obj')
        m[f'{name}_hull'] = trimesh.load(hull_path, force='mesh') if os.path.exists(hull_path) else None
    return m
