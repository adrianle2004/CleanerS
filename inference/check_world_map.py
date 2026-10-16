"""
inference/check_world_map.py

The checks that keep --accumulate honest. Each one measures the world map
against something this repo already committed, so none of them needs new
ground truth and none of them can be made to pass by agreeing with itself.

    python -m inference.check_world_map algebra  captures/room07 \
        --pretrained_path ./checkpoint/CleanerS_ckpt.pth
    python -m inference.check_world_map tracker  captures/room07
    python -m inference.check_world_map measured captures/room07 \
        --world outputs/room07_world/world.npz

algebra   The frame placement, against run_inference's own output.
          fuse_scan anchors the world on the first fused frame BECAUSE
          that frame's pose is live_cam_pose(camera_height, yaw, tilt) --
          so with world_T_cam = cam_pose the composition collapses to the
          identity and the accumulated map of one frame must land
          voxel-for-voxel where the single-frame tool puts it. A
          transposed rotation, an inverted pose or a permuted axis shows
          up here as a translation or a tipped-over room. This is the
          check that caught the grid being indexed (X, Y, Z) when
          frame_loader builds it as [z_g, y_g, x_g].

tracker   The in-loop poses against captures/*/scan/trajectory.txt.
          They cannot be identical -- different reader, different skip,
          different surviving frames -- so they are compared as
          TRAJECTORIES, by nearest pose, and must agree to within the
          drift document/DRIFT_AND_ALIGNMENT.md already measured: 66-102
          mm at the ground-truth frames, about one voxel at 8 cm.

measured  The measured layer against scan/room.ply, the same sensor fused
          by a mature implementation. Reported as the fraction of the
          map's measured voxels that the fused mesh also occupies.
"""

import argparse
import json
import logging
import os
import sys

import numpy as np

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_THIS_DIR)
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

try:
    from .world_map import WorldMap, VOX, gather_prediction
    from .frame_loader import FrameLoader, frame_meta
    from .utils import setup_logging
except ImportError:
    from world_map import WorldMap, VOX, gather_prediction
    from frame_loader import FrameLoader, frame_meta
    from utils import setup_logging

log = logging.getLogger('check_world_map')


def _meta(capture):
    with open(os.path.join(capture, 'meta.json')) as f:
        return json.load(f)


def _still_frame(capture):
    """The capture's evaluation frame: the first one meta.json lists."""
    names = _meta(capture).get('frames')
    if names:
        return names[0]
    d = os.path.join(capture, 'depth')
    names = sorted(n for n in os.listdir(d) if n.endswith('.png'))
    if not names:
        raise SystemExit('no depth frames in %s' % d)
    return os.path.splitext(names[0])[0]


# ---------------------------------------------------------------- algebra

def check_algebra(args):
    import cv2
    try:
        from . import model as model_mod
        from .visualize import save_prediction_ply          # noqa: F401
    except ImportError:
        import model as model_mod
    from cleaner.utils import EasyConfig
    import torch

    cap = args.capture
    stem = args.frame or _still_frame(cap)
    meta = _meta(cap)
    depth = cv2.imread(os.path.join(cap, 'depth', stem + '.png'),
                       -1).astype(np.float32) / 1000.0
    rgb = cv2.imread(os.path.join(cap, 'rgb', stem + '.png'))
    fm = frame_meta(meta, stem)      # the frame's own record, not the room's
    loader = FrameLoader.from_live_camera(
        depth, rgb=rgb,
        cam_K=np.asarray(fm['cam_K'], np.float32).reshape(3, 3),
        camera_height=fm['camera_height'], yaw=fm.get('yaw', 0.0),
        up_camera=fm.get('up_camera'), encoder='auto')
    sample = loader.build_tsdf_and_mapping()

    cfg = EasyConfig()
    cfg.load(args.cfg, recursive=True)
    cfg.pretrained_path = args.pretrained_path
    cfg.rank = 0 if torch.cuda.is_available() else 'cpu'
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = model_mod.load_model(cfg, device=device)
    pred, _ = model_mod.predict(model, sample, device=device, amp='auto')

    tsdf = np.asarray(sample['tsdf']).reshape(60, 36, 60)
    weight = np.asarray(sample['weight']).reshape(60, 36, 60).astype(bool)
    keep = (tsdf < -0.5) | (weight & (np.abs(tsdf) < 0.25))

    # the first frame: world_T_cam IS this frame's cam_pose
    cam_pose = loader.cam_pose.astype(np.float64)
    world = WorldMap.around((0.0, 0.0))
    idx, cls = gather_prediction(world, pred, keep, loader.vox_origin,
                                 cam_pose, cam_pose)
    world.add_prediction(idx, cls, 0)
    got = set(map(tuple, np.argwhere(world.predicted()[0])))

    # where run_inference would put the same voxels: straight from the
    # frame grid, with no pose arithmetic at all
    fidx = np.argwhere(keep & (pred > 0))
    centres = (loader.vox_origin.astype(np.float64)
               + (fidx[:, [0, 2, 1]] + 0.5) * VOX)
    want = set(map(tuple, world.index_of(centres)[0]))

    missing, extra = want - got, got - want
    log.info('frame %s: %d occupied voxels in the frame grid, %d placed '
             'in the world', stem, len(want), len(got))
    log.info('missing %d, extra %d', len(missing), len(extra))
    ok = not missing and not extra
    log.info('ALGEBRA %s', 'OK -- the composition collapses to the identity'
             if ok else 'FAILED -- the placement does not match')
    return 0 if ok else 1


# ---------------------------------------------------------------- tracker

def _read_trajectory(path):
    idx, T = [], []
    with open(path) as f:
        for line in f:
            if line.startswith('#') or not line.strip():
                continue
            v = line.split()
            idx.append(int(v[0]))
            T.append(np.array([float(x) for x in v[1:17]]).reshape(4, 4))
    return np.array(idx), np.array(T)


def check_tracker(args):
    cap = args.capture
    ref_idx, ref_T = _read_trajectory(
        os.path.join(cap, 'scan', 'trajectory.txt'))
    mine = np.load(args.poses)['poses'] if args.poses else None
    if mine is None:
        raise SystemExit(
            'pass --poses <npz with a "poses" array>; run_live writes one '
            'beside world.npz when --accumulate is on')
    a = ref_T[:, :3, 3]
    b = mine[:, :3, 3]
    log.info('reference %d poses, mine %d', len(a), len(b))
    # compare as trajectories: for each of mine, the nearest reference
    # pose. The two cannot share an index -- see tracker.py.
    d = np.linalg.norm(b[:, None, :] - a[None, :, :], axis=2)
    near = d.min(axis=1)
    log.info('nearest-pose distance: median %.1f mm, p90 %.1f mm, max '
             '%.1f mm', 1000 * np.median(near),
             1000 * np.percentile(near, 90), 1000 * near.max())
    log.info('path length: reference %.2f m, mine %.2f m',
             np.linalg.norm(np.diff(a, axis=0), axis=1).sum(),
             np.linalg.norm(np.diff(b, axis=0), axis=1).sum())
    ok = np.median(near) < args.tol
    log.info('TRACKER %s (tolerance %.0f mm, the documented drift is '
             '66-102 mm)', 'OK' if ok else 'FAILED', 1000 * args.tol)
    return 0 if ok else 1


# --------------------------------------------------------------- measured

def check_measured(args):
    import open3d as o3d
    world = WorldMap.load(args.world)
    mesh = o3d.io.read_point_cloud(
        os.path.join(args.capture, 'scan', 'room_cloud.ply'))
    pts = np.asarray(mesh.points)
    if not len(pts):
        raise SystemExit('room_cloud.ply has no points')
    fused = np.zeros(world.shape, bool)
    idx, inside = world.index_of(pts)
    idx = idx[inside]
    fused[idx[:, 0], idx[:, 1], idx[:, 2]] = True

    meas = world.verdicts()['measured']
    both = int((meas & fused).sum())
    log.info('map measured %d voxels, the fused sweep occupies %d',
             int(meas.sum()), int(fused.sum()))
    frac = both / max(1, int(meas.sum()))
    log.info('%d of the map\'s measured voxels are in the fused sweep '
             '(%.1f%%)', both, 100 * frac)
    ok = frac >= args.tol
    log.info('MEASURED %s (wanted %.0f%%)', 'OK' if ok else 'FAILED',
             100 * args.tol)
    return 0 if ok else 1


def main():
    setup_logging()
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('check', choices=['algebra', 'tracker', 'measured'])
    p.add_argument('capture')
    p.add_argument('--frame', default=None)
    p.add_argument('--cfg', default='./cfgs/NYU/voxelSSC.yaml')
    p.add_argument('--pretrained_path',
                   default='./checkpoint/CleanerS_ckpt.pth')
    p.add_argument('--world', default=None, help='a world.npz')
    p.add_argument('--poses', default=None, help='a poses.npz')
    p.add_argument('--tol', type=float, default=None)
    a = p.parse_args()
    if a.check == 'algebra':
        return check_algebra(a)
    if a.check == 'tracker':
        a.tol = 0.15 if a.tol is None else a.tol
        return check_tracker(a)
    a.tol = 0.80 if a.tol is None else a.tol
    if not a.world:
        raise SystemExit('measured needs --world <world.npz>')
    return check_measured(a)


if __name__ == '__main__':
    sys.exit(main())
