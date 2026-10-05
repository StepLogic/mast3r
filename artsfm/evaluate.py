"""Align the reconstruction to MPArt-90 ground truth, score the joint, and write outputs."""
import json
import os

import numpy as np
import trimesh
from scipy.spatial import cKDTree

from artsfm.factorize import as_views

GT_TYPES = {'rotate': 'revolute', 'translate': 'prismatic'}
# trans.json axes live in the mesh (.obj) frame; GaussianArt's eval_axis.read_gt maps them
# into the camera / *_rotate.ply frame with this rotation.
R_COORD = np.array([[0., -1., 0.], [1., 0., 0.], [0., 0., 1.]]).T
OPENGL_TO_OPENCV = np.diag([1.0, -1.0, -1.0, 1.0])


def umeyama(src, dst):
    ms, md = src.mean(0), dst.mean(0)
    S, D = src - ms, dst - md
    U, sig, Vt = np.linalg.svd(D.T @ S / len(src))
    E = np.diag([1.0, 1.0, np.sign(np.linalg.det(U @ Vt))])
    R = U @ E @ Vt
    s = float(np.trace(np.diag(sig) @ E) / (S ** 2).sum(1).mean())
    return s, R, md - s * R @ ms


def axis_angle_err(d1, d2):
    d1, d2 = np.asarray(d1, float), np.asarray(d2, float)
    c = abs(d1 @ d2) / np.linalg.norm(d1) / np.linalg.norm(d2)
    return float(np.rad2deg(np.arccos(np.clip(c, -1, 1))))


def line_distance(o1, d1, o2, d2):
    o1, d1, o2, d2 = (np.asarray(x, float) for x in (o1, d1, o2, d2))
    n = np.cross(d1, d2)
    if np.linalg.norm(n) < 1e-9 * np.linalg.norm(d1) * np.linalg.norm(d2):
        return float(np.linalg.norm(np.cross(o2 - o1, d1)) / np.linalg.norm(d1))
    return float(abs((o2 - o1) @ n) / np.linalg.norm(n))


def load_gt_joint(data_dir):
    with open(os.path.join(data_dir, 'gt', 'trans.json')) as f:
        info = json.load(f)['trans_info']
    if len(info) != 1:
        raise ValueError(f'v1 handles single-joint objects; {data_dir} has {len(info)} joints')
    j = info[0]
    lr = j[j['type']]
    return dict(type=GT_TYPES[j['type']], axis_o=R_COORD @ np.array(j['axis']['o'], float),
                axis_d=R_COORD @ np.array(j['axis']['d'], float), state_change=abs(lr['l'] - lr['r']))


def align_poses(est_c2w, gt_c2w_opengl):
    """Sim(3) (s, R, t) mapping the reconstruction's world frame into the GT frame.

    Rotation comes from camera orientations (well conditioned even for clustered views),
    then scale and translation from camera centres. GT poses are OpenGL (-z forward).
    """
    gt = gt_c2w_opengl @ OPENGL_TO_OPENCV
    M = sum(g[:3, :3] @ e[:3, :3].T for g, e in zip(gt, est_c2w))
    U, _, Vt = np.linalg.svd(M)
    R = U @ np.diag([1.0, 1.0, np.sign(np.linalg.det(U @ Vt))]) @ Vt
    ce, cg = est_c2w[:, :3, 3] @ R.T, gt[:, :3, 3]
    s = float(((cg - cg.mean(0)) * (ce - ce.mean(0))).sum() / ((ce - ce.mean(0)) ** 2).sum())
    return s, R, cg.mean(0) - s * ce.mean(0)


def view0_to_gt(x, cam2w0, sim, is_dir=False):
    s, R, t = sim
    x = np.atleast_2d(x)
    if is_dir:
        return x @ cam2w0[:3, :3].T @ R.T
    world = x @ cam2w0[:3, :3].T + cam2w0[:3, 3]
    return s * world @ R.T + t


def gt_part_labels(points_gt, data_dir, n_samples=20000):
    clouds = []
    for name in ('static', 'dynamic'):  # camera-frame meshes (the .obj files use another frame)
        mesh = trimesh.load(os.path.join(data_dir, 'gt', 'start', f'start_{name}_rotate.ply'), force='mesh')
        clouds.append(mesh.sample(n_samples))
    d = np.stack([cKDTree(c).query(points_gt)[0] for c in clouds], 1)
    return d.argmin(1), d.min(1)


def segmentation_accuracy(pred, gt):
    """Accuracy of base/part labels (outliers -1 ignored), invariant to which body is called base."""
    seg = pred >= 0
    if not seg.any():
        return None, False
    acc = float((pred[seg] == gt[seg]).mean())
    return (1.0 - acc, True) if acc < 0.5 else (acc, False)


def track_label_counts(pred, gt):
    """How many tracks lie on the GT moving part, and where the method put them."""
    on_part = gt == 1
    return dict(n_gt_part_tracks=int(on_part.sum()),
                n_pred_part_on_gt_part=int((on_part & (pred == 1)).sum()),
                n_pred_outlier_on_gt_part=int((on_part & (pred == -1)).sum()))


def evaluate(result, W, geom, view_paths, frames_by_path, data_dir):
    gt = load_gt_joint(data_dir)
    est_c = geom['cam2w'][:, :3, 3]
    gt_poses = np.array([frames_by_path[p] for p in view_paths])
    gt_c = gt_poses[:, :3, 3]
    sim = align_poses(geom['cam2w'], gt_poses)
    cam2w0 = geom['cam2w'][0]
    m = dict(gt_type=gt['type'], pred_type=result['type'], type_correct=result['type'] == gt['type'],
             sim_scale=sim[0],
             cam_align_rmse=float(np.sqrt(((sim[0] * est_c @ sim[1].T + sim[2] - gt_c) ** 2).sum(1).mean())))

    pts_gt = view0_to_gt(as_views(W)[0], cam2w0, sim)
    gt_lab, dist = gt_part_labels(pts_gt, data_dir)
    lab = result['labels']
    m['seg_accuracy'], m['seg_labels_swapped'] = segmentation_accuracy(lab, gt_lab)
    m.update(track_label_counts(lab, gt_lab))
    m['track_to_mesh_median'] = float(np.median(dist))
    m['n_tracks'] = int(W.shape[1])

    if result['axis_d'] is not None:
        d_gt = view0_to_gt(result['axis_d'], cam2w0, sim, is_dir=True)[0]
        m['axis_angle_err_deg'] = axis_angle_err(d_gt, gt['axis_d'])
        m['pred_axis_d_gt'] = d_gt.tolist()
        if result['axis_o'] is not None and gt['type'] == 'revolute':
            o_gt = view0_to_gt(result['axis_o'], cam2w0, sim)[0]
            m['axis_pos_err'] = line_distance(o_gt, d_gt, gt['axis_o'], gt['axis_d'])
            m['pred_axis_o_gt'] = o_gt.tolist()
        if gt['type'] == 'revolute' and result['angle_deg'] is not None:
            m['state_err'] = abs(result['angle_deg'] - gt['state_change'])
        elif gt['type'] == 'prismatic' and result['disp'] is not None:
            m['state_err'] = abs(sim[0] * result['disp'] - gt['state_change'])
    return m


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


def write_outputs(out_dir, result, metrics, W, geom):
    os.makedirs(out_dir, exist_ok=True)
    res = {k: v for k, v in result.items() if k != 'labels'}
    with open(os.path.join(out_dir, 'result.json'), 'w') as f:
        json.dump(_jsonable(dict(result=res, metrics=metrics)), f, indent=2)
    np.savez(os.path.join(out_dir, 'W.npz'), W=W, labels=result['labels'])

    # tracks in view-0 frame coloured by segment, plus the axis as a dense line of points
    pts = as_views(W)[0]
    palette = {0: [160, 160, 160, 255], 1: [230, 60, 40, 255], -1: [40, 40, 40, 255]}
    cols = np.array([palette[int(l)] for l in result['labels']], np.uint8)
    if result['axis_d'] is not None:
        o = result['axis_o'] if result['axis_o'] is not None else pts[result['labels'] == 1].mean(0)
        span = np.linalg.norm(pts - pts.mean(0), axis=1).max()
        line = o + np.linspace(-span, span, 400)[:, None] * result['axis_d']
        pts = np.vstack([pts, line])
        cols = np.vstack([cols, np.tile([40, 90, 230, 255], (len(line), 1)).astype(np.uint8)])
    trimesh.PointCloud(pts, colors=cols).export(os.path.join(out_dir, 'articulation.ply'))


def model_label_accuracy(model_dir, data_dir):
    """Fraction of base.ply / part.ply points whose nearest camera-frame GT surface agrees (start state)."""
    from artsfm.model_io import load_model
    m = load_model(model_dir)
    g = m.get('gt_frame')
    if not g:
        raise ValueError('model has no gt_frame (built without GT poses)')
    sim = (float(g['scale']), np.asarray(g['sim_R'], float), np.asarray(g['sim_t'], float))
    cam2w0 = np.asarray(g['cam2w0'], float)
    out = {}
    for name, want in (('base', 0), ('part', 1)):
        P = m[name]['pts']
        if not len(P):
            out[f'{name}_acc'] = out[f'{name}_median_dist'] = None
            continue
        lab, dist = gt_part_labels(view0_to_gt(P, cam2w0, sim), data_dir)
        out[f'{name}_acc'] = float((lab == want).mean())
        out[f'{name}_median_dist'] = float(np.median(dist))
    return out
