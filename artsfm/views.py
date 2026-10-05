"""Choose a small, overlapping set of views from the two articulation states.

v1 uses the dataset's camera poses only to choose views (spec Sec. 3.1); swap
`select_views` for a retrieval-based selector to drop that dependency.
"""
import json
import os

import numpy as np


def load_frames(data_dir, state, split='train'):
    with open(os.path.join(data_dir, f'transforms_{split}_{state}.json')) as f:
        frames = json.load(f)['frames']
    out = []
    for fr in frames:
        fp = fr['file_path']
        if not os.path.splitext(fp)[1]:
            fp += '.png'
        out.append((os.path.join(data_dir, fp), np.array(fr['transform_matrix'], float)))
    return out


def _dirs(frames):
    c = np.array([T[:3, 3] for _, T in frames])
    return c / np.linalg.norm(c, axis=1, keepdims=True)


def select_views(start_frames, end_frames, k=5, seed_view=0):
    ds, de = _dirs(start_frames), _dirs(end_frames)
    order = np.argsort(-(ds @ ds[seed_view]))       # seed first, then nearest directions
    chosen_s = [seed_view] + [i for i in order if i != seed_view][:k - 1]
    chosen_e, used = [], set()
    for i in chosen_s:
        for j in np.argsort(-(de @ ds[i])):
            if j not in used:
                used.add(j)
                chosen_e.append(j)
                break
    paths = [start_frames[i][0] for i in chosen_s] + [end_frames[j][0] for j in chosen_e]
    return paths, np.array([0] * k + [1] * k)


def load_fov(data_dir, split='train'):
    with open(os.path.join(data_dir, f'transforms_{split}_start.json')) as f:
        return float(json.load(f)['camera_angle_x'])


def model_intrinsics(fov_x, orig_wh, model_hw):
    """Pinhole K at MASt3R input resolution: long side resized to the model size, then centre-cropped."""
    W0, H0 = orig_wh
    H, W = model_hw
    f = max(H, W) / max(W0, H0) * (W0 / 2) / np.tan(fov_x / 2)
    return np.array([[f, 0, W / 2], [0, f, H / 2], [0, 0, 1.0]])
