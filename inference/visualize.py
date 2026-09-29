"""
inference/visualize.py

Voxel prediction -> colored .ply export.

VERIFIED: this is a direct port of labeled_voxel2ply / get_xyz / colorMap
from your real test_NYU.py -- not a reimplementation. Two things I changed
on purpose vs. the original:
  - get_xyz's per-axis python for-loops replaced with np.meshgrid (identical
    result, just avoids the slow loop).
  - Takes plain numpy arrays instead of requiring torch tensors already on
    GPU, since our custom pipeline runs single-frame outside the
    dataloader/eval loop.
Everything else (color table, index-based coordinates with no world-space
scaling, weight-masking before export, occupied-only filtering) matches
test_NYU.py exactly.
"""

import os
import numpy as np

# ---- verified: colorMap taken verbatim from test_NYU.py's labeled_voxel2ply default ----
DEFAULT_COLORMAP = np.array([
    [22, 191, 206],   # 0  empty, free space
    [214, 38, 40],    # 1  ceiling
    [43, 160, 4],     # 2  floor
    [158, 216, 229],  # 3  wall
    [114, 158, 206],  # 4  window
    [204, 204, 91],   # 5  chair
    [255, 186, 119],  # 6  bed
    [147, 102, 188],  # 7  sofa
    [30, 119, 181],   # 8  table
    [188, 188, 33],   # 9  tvs
    [255, 127, 12],   # 10 furn
    [196, 175, 214],  # 11 objects
    [153, 153, 153],  # 12 ignore / label==255
], dtype=np.int32)


def get_xyz(size):
    """Verified port of test_NYU.py's get_xyz -- vectorized, identical result.
    size: (60,36,60)-style tuple. Returns three (60,36,60) int32 index grids."""
    x, y, z = np.meshgrid(np.arange(size[0]), np.arange(size[1]), np.arange(size[2]),
                           indexing='ij')
    return x.astype(np.int32), y.astype(np.int32), z.astype(np.int32)


def labeled_voxel2ply(vox_labeled, ply_filename, colorMap=None):
    """Verified port of test_NYU.py's labeled_voxel2ply.
    vox_labeled: numpy int array, shape (60,36,60) -- matches our established
    (z,y,x)-flattened grid convention from frame_loader.py, same shape
    test_NYU.py's own tsdf/label arrays use.
    """
    if colorMap is None:
        colorMap = DEFAULT_COLORMAP

    if not isinstance(vox_labeled, np.ndarray):
        raise Exception(f"Oops! Type of vox_labeled should be 'numpy.ndarray', not {type(vox_labeled)}.")
    if np.amax(vox_labeled) == 0:
        print('Oops! All voxel is labeled empty.')
        return

    size = vox_labeled.shape
    vox_labeled = vox_labeled.flatten()
    _x, _y, _z = get_xyz(size)
    _x = _x.flatten(); _y = _y.flatten(); _z = _z.flatten()

    vox_labeled = vox_labeled.copy()
    vox_labeled[vox_labeled == 255] = 0   # empty
    _rgb = colorMap[vox_labeled[:]]
    xyz_rgb = np.array(list(zip(_x, _y, _z, _rgb[:, 0], _rgb[:, 1], _rgb[:, 2])))
    ply_data = xyz_rgb[np.where(vox_labeled > 0)]

    if len(ply_data) == 0:
        raise Exception("Oops! That was no valid ply data.")

    ply_head = ('ply\nformat ascii 1.0\nelement vertex %d\n'
                'property float x\nproperty float y\nproperty float z\n'
                'property uchar red\nproperty uchar green\nproperty uchar blue\n'
                'end_header' % len(ply_data))
    os.makedirs(os.path.dirname(ply_filename) or '.', exist_ok=True)
    np.savetxt(ply_filename, ply_data, fmt="%d %d %d %d %d %d", header=ply_head, comments='')
    print(f'Saved-->{ply_filename}')


def save_prediction_ply(pred_label, weight, ply_path, colorMap=None):
    """
    pred_label: (60,36,60) int array of predicted class indices (argmax already applied).
    weight: (60,36,60) array/bool -- our TSDF observed-voxel mask (arr_1 from TSDF/{id}.npz).

    Matches test_NYU.py's visualize_3d_predict: unobserved voxels forced to
    'empty' (0) before export, occupied-only filtering happens inside
    labeled_voxel2ply itself.
    """
    pred_label = np.asarray(pred_label).reshape(60, 36, 60).copy()
    weight = np.asarray(weight).reshape(60, 36, 60)
    pred_label[weight == 0] = 0
    labeled_voxel2ply(pred_label, ply_path, colorMap=colorMap)
    return ply_path