"""
reconstruction_GT/show_gt.py

Look at the ground truth you made: the voxels, in the room they came from.

    python -m reconstruction_GT.show_gt captures/room07
    python -m reconstruction_GT.show_gt captures/room07 --cloud     # + the sweep
    python -m reconstruction_GT.show_gt captures/room07 --scored    # only what is scored
    python -m reconstruction_GT.show_gt captures/room07 --export    # write the .ply, no window

It writes `gt/ply/<frame>_gt.ply` through the repo's own `labeled_voxel2ply`,
the same writer `test_NYU.py` uses for NYU's ground truth, so the file is
directly comparable with anything else in `outputs/` and opens in CloudCompare
like the rest.

*** WHAT YOU ARE LOOKING AT ***

One coloured point per OCCUPIED voxel at 8 cm, in the frame's own grid. Empty
and 255 are not drawn -- a filled room would be a solid block with nothing to
see. So a bed reads as a slab, not a surface, which is the point: ground truth
for completion is solid.

`--scored` drops everything the metric ignores (outside `label_weight`, or
255), leaving exactly the voxels a score is computed over. The difference
between the two views is the difference between what you annotated and what
gets measured.

`--cloud` adds the fused sweep in grey, which is the check worth doing: every
coloured voxel should sit on or inside the grey surface it came from.
"""

import os
import sys
import glob
import argparse
import logging

import numpy as np

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_THIS_DIR)
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from inference.visualize import labeled_voxel2ply
from reconstruction_GT.voxelize_gt import CLASSES, VOX_SIZE_LOW, VOX_UNIT_LOW

log = logging.getLogger(__name__)


def parse_args():
    p = argparse.ArgumentParser(description='Show a capture\'s ground truth.')
    p.add_argument('capture_dir')
    p.add_argument('--frame', default=None, help='frame stem (default: the first)')
    p.add_argument('--fov', choices=['all', 'frustum'], default='all',
                   help="'frustum' drops what this frame's camera could not "
                        'see, which is what --fov frustum scores')
    p.add_argument('--scored', action='store_true',
                   help='draw only voxels the metric scores (label_weight, not 255)')
    p.add_argument('--cloud', action='store_true',
                   help='also show the fused sweep, in grey')
    p.add_argument('--export', action='store_true', help='write the .ply and stop')
    return p.parse_args()


def main():
    from inference.utils import setup_logging
    setup_logging()
    args = parse_args()
    cap = args.capture_dir

    lab_dir = os.path.join(cap, 'gt', 'Label')
    files = sorted(glob.glob(os.path.join(lab_dir, '*.npz')))
    if not files:
        raise SystemExit('no ground truth in %s -- run voxelize_gt first' % lab_dir)
    path = (os.path.join(lab_dir, args.frame + '.npz') if args.frame else files[0])
    stem = os.path.splitext(os.path.basename(path))[0]
    lab = np.load(path)['arr_0'].reshape(VOX_SIZE_LOW).astype(np.int64)

    shown = lab.copy()
    if args.fov == 'frustum':
        # the same mask the evaluator applies, on this frame's own camera
        import json
        from inference.frame_loader import cam_pose_from_meta, frame_meta
        from reconstruction_GT.voxelize_gt import Grid, frustum_mask
        meta = json.load(open(os.path.join(cap, 'meta.json')))
        fm = frame_meta(meta, stem)
        g = Grid([-2.4, 0.0, -0.05], VOX_SIZE_LOW, VOX_UNIT_LOW)
        seen = frustum_mask(g, np.asarray(fm['cam_K'], np.float64),
                            cam_pose_from_meta(meta, stem)).reshape(VOX_SIZE_LOW)
        shown[~seen] = 0
    if args.scored:
        tsdf_path = os.path.join(cap, 'gt', 'TSDF', stem + '.npz')
        if not os.path.exists(tsdf_path):
            raise SystemExit('--scored needs %s' % tsdf_path)
        lw = np.load(tsdf_path)['arr_1'].reshape(VOX_SIZE_LOW) > 0
        shown[~lw] = 0

    occupied = (shown > 0) & (shown != 255)
    counts = [(CLASSES[c], int((shown == c).sum())) for c in range(1, 12)
              if (shown == c).sum()]
    log.info('%s: %d occupied voxels%s', stem, int(occupied.sum()),
             ' (scored only)' if args.scored else
             " (inside this frame's view)" if args.fov == 'frustum' else '')
    for name, n in counts:
        log.info('   %-8s %6d', name, n)

    name = stem + ('_gt_scored' if args.scored else
                   '_gt_fov' if args.fov == 'frustum' else '_gt') + '.ply'
    ply = os.path.join(cap, 'gt', 'ply', name)
    labeled_voxel2ply(shown, ply)
    # a copy beside the predictions, where display_overlay.py resolves the
    # capture on its own (outputs/<room>/ply/<frame>_gt.ply -> captures/<room>/)
    out_ply = os.path.join('outputs', os.path.basename(cap.rstrip('/')), 'ply', name)
    os.makedirs(os.path.dirname(out_ply), exist_ok=True)
    import shutil
    shutil.copyfile(ply, out_ply)
    print('\n  compare it with the prediction:')
    print('    python inference/display_overlay.py %s' % out_ply)
    pred = out_ply.replace(name, stem + '.ply')
    if os.path.exists(pred):
        print('    python inference/display_overlay.py %s' % pred)
    if args.export:
        print('open it with:  CloudCompare -O %s' % os.path.abspath(ply))
        return

    import open3d as o3d
    gt = o3d.io.read_point_cloud(ply)
    if not gt.has_points():
        raise SystemExit('the .ply came out empty')

    # labeled_voxel2ply writes voxel INDICES as (axis0, axis1, axis2), which in
    # this grid is (world X, world Z, world Y) -- height in the middle. Put it
    # back into world metres so it can sit beside the sweep.
    vox_origin = np.array([-2.4, 0.0, -0.05])      # from_live_camera's grid
    idx = np.asarray(gt.points)
    world = np.stack([vox_origin[0] + (idx[:, 0] + 0.5) * VOX_UNIT_LOW,
                      vox_origin[1] + (idx[:, 2] + 0.5) * VOX_UNIT_LOW,
                      vox_origin[2] + (idx[:, 1] + 0.5) * VOX_UNIT_LOW], axis=1)
    gt.points = o3d.utility.Vector3dVector(world)
    geoms = [gt]
    log.info('ground truth spans X %.2f..%.2f  Y %.2f..%.2f  Z %.2f..%.2f m',
             world[:, 0].min(), world[:, 0].max(), world[:, 1].min(),
             world[:, 1].max(), world[:, 2].min(), world[:, 2].max())
    if args.cloud:
        cloud_path = os.path.join(cap, 'scan', 'room_cloud.ply')
        if os.path.exists(cloud_path):
            c = o3d.io.read_point_cloud(cloud_path).voxel_down_sample(0.03)
            c.paint_uniform_color([0.72, 0.72, 0.72])
            geoms.append(c)                        # already in world metres
        else:
            log.warning('no fused cloud at %s', cloud_path)
    geoms.append(o3d.geometry.TriangleMesh.create_coordinate_frame(0.4))
    o3d.visualization.draw_geometries(
        geoms, window_name='%s ground truth%s' % (stem, ' (scored)' if args.scored else ''),
        width=1400, height=900)


if __name__ == '__main__':
    main()
