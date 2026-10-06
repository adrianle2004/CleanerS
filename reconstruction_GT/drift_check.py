"""
reconstruction_GT/drift_check.py

Does Open3D's SLAM accumulate significant error over one of our sweeps?

    python -m reconstruction_GT.drift_check captures/room08

*** WHY THIS EXISTS ***

The grid of a frame lifted out of the sweep is placed by that frame's pose in
`scan/trajectory.txt`, which comes from `o3d.t.pipelines.slam`'s frame-to-model
tracking in fuse_scan.py. If those poses drift, a late frame's ground truth is
annotated in the wrong place and nothing in the pipeline notices: the mesh and
the annotation drift together, so they stay mutually consistent while both
slide away from the room.

The checks already in `fuse_report.json` do not bound this:

  still_match   compares frame 0 against the separately captured still, so it
                validates the ORIGIN of the pose chain, not its accumulation
  floor         the fused floor is flat to 0.02-0.15 deg and sits at z = 0,
                which bounds ROTATIONAL drift (a late frame tilted against an
                early one could not fuse into a plane that flat) and VERTICAL
                drift -- but is completely blind to a horizontal slide
  imu_tilt      the accelerometer is a cross-check on gravity, never a pose
                input: nothing integrates it, so it cannot accumulate into
                the trajectory at all (see imu_check.py for what it IS worth)

So horizontal drift was unmeasured. This measures it by running the *same*
geometry through the *same* tracker twice, once forwards and once backwards.
Identical input, opposite direction of accumulation, so the two trajectories
can only disagree by drift:

    forward  frame 0 .. N-1, origin at frame 0
    reverse  frame N-1 .. 0, origin at frame N-1

Put them in one frame by re-referencing the forward run to its last frame --
`P_fwd(N-1)^-1 @ P_fwd(i)` is then directly comparable with `P_rev(i)` -- and
the residual at frame 0 is the accumulated disagreement over the whole sweep.

Full write-up, including the two ways this is easy to get wrong:
document/DRIFT_AND_ALIGNMENT.md.

That residual is a LOWER bound on drift, not the drift itself: a bias the
tracker commits in both directions cancels. A small residual therefore does
not prove the trajectory is right, but a large one proves it is not.

Frames are cached to a scratch memmap first, because a bag can only be read
forwards (bag_reader.py) and the reverse pass needs them in reverse.
"""

import os
import sys
import json
import time
import shutil
import argparse
import logging
import tempfile

import numpy as np

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_THIS_DIR)
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from reconstruction_GT.bag_reader import iter_bag, bag_metadata      # noqa: E402
from reconstruction_GT.fuse_scan import (                            # noqa: E402
    MAX_SPEED_M_S, MAX_TURN_DEG_S, MIN_FITNESS, LOST_AFTER_S)

log = logging.getLogger(__name__)


def cache_frames(bag, cache_dir, skip):
    """Write every frame's depth and colour to a memmap. -> (depth, rgb, idx)

    `idx[k]` is the bag frame index of row k, so a pose can still be reported
    against the numbering trajectory.txt uses.
    """
    os.makedirs(cache_dir, exist_ok=True)
    dpath = os.path.join(cache_dir, 'depth.npy')
    cpath = os.path.join(cache_dir, 'rgb.npy')
    md = bag_metadata(bag)
    H, W = md['height'], md['width']
    # two passes would mean decoding the bag twice; instead grow a list of
    # indices and write straight into generously sized memmaps
    cap = 1200          # our longest sweep is 1050 frames; 848x480 x (2+3) B
    depth = np.lib.format.open_memmap(dpath, 'w+', np.uint16, (cap, H, W))
    rgb = np.lib.format.open_memmap(cpath, 'w+', np.uint8, (cap, H, W, 3))
    idx, n, t0 = [], 0, time.time()
    for bf in iter_bag(bag):
        if bf.index < skip:
            continue
        if n >= cap:
            log.warning('cache full at %d frames', cap)
            break
        depth[n] = bf.depth_raw
        rgb[n] = bf.rgb
        idx.append(bf.index)
        n += 1
        if n % 200 == 0:
            log.info('  cached %d frames (%.0f/s)', n, n / (time.time() - t0))
    depth.flush()
    rgb.flush()
    log.info('cached %d frames in %.0f s', n, time.time() - t0)
    return depth[:n], rgb[:n], np.asarray(idx)


def run_slam(depth, rgb, order, K, md, args, label):
    """fuse_scan.py's tracking loop over `order`. -> {row: 4x4}, [fitness]

    Deliberately the same calls, constants and reject rules as fuse_scan, so a
    difference between two runs is the direction and nothing else.
    """
    import open3d as o3d
    import open3d.core as o3c
    device = o3c.Device('CPU:0' if args.cpu else 'CUDA:0')
    scale = md['depth_scale_o3d']
    Kt = o3c.Tensor(K, o3c.Dtype.Float64)
    T = np.identity(4)
    model = o3d.t.pipelines.slam.Model(args.voxel, 16, args.block_count,
                                       o3c.Tensor(T), device)
    frame = o3d.t.pipelines.slam.Frame(md['height'], md['width'], Kt, device)
    raycast = o3d.t.pipelines.slam.Frame(md['height'], md['width'], Kt, device)
    fps = md['fps']
    max_step, max_turn = MAX_SPEED_M_S / fps, MAX_TURN_DEG_S / fps
    lost_after = max(1, int(round(LOST_AFTER_S * fps)))

    poses, fitness, dropped, run_lost, fused = {}, [], 0, 0, 0
    t0 = time.time()
    for row in order:
        frame.set_data_from_image('depth', o3d.t.geometry.Image(
            o3c.Tensor(depth[row][:, :, None])).to(device))
        frame.set_data_from_image('color', o3d.t.geometry.Image(
            o3c.Tensor(np.ascontiguousarray(rgb[row]))).to(device))
        if fused > 0:
            reason = None
            try:
                res = model.track_frame_to_model(frame, raycast, scale,
                                                 args.depth_max, args.depth_diff)
                step = res.transformation.cpu().numpy()
                turn = np.degrees(np.arccos(np.clip(
                    (np.trace(step[:3, :3]) - 1.0) / 2.0, -1.0, 1.0)))
                if res.fitness < MIN_FITNESS:
                    reason = 'fitness %.3f' % res.fitness
                elif np.linalg.norm(step[:3, 3]) > max_step or turn > max_turn:
                    reason = 'jump'
            except RuntimeError as e:
                reason = str(e)
            if reason is not None:
                dropped += 1
                run_lost += 1
                if run_lost >= lost_after:
                    log.warning('%s: tracking lost after %d frames', label, fused)
                    break
                continue
            run_lost = 0
            T = T @ step
            fitness.append(float(res.fitness))
        model.update_frame_pose(fused, o3c.Tensor(T))
        model.integrate(frame, scale, args.depth_max, args.trunc)
        model.synthesize_model_frame(raycast, scale, 0.1, args.depth_max,
                                     args.trunc, False)
        poses[int(row)] = T.copy()
        fused += 1
        if fused % 100 == 0:
            log.info('  %s: %d frames (%.1f fps)', label, fused,
                     fused / (time.time() - t0))
    log.info('%s: %d fused, %d dropped, fitness median %.3f', label, fused,
             dropped, float(np.median(fitness)) if fitness else float('nan'))
    return poses, fitness


def rot_deg(R):
    return float(np.degrees(np.arccos(np.clip(
        (np.trace(R[:3, :3]) - 1.0) / 2.0, -1.0, 1.0))))


def compare(fwd, rev, idx):
    """Forward and reverse re-referenced to a common frame. -> rows, summary"""
    common = sorted(set(fwd) & set(rev))
    if len(common) < 10:
        raise SystemExit('only %d frames tracked by both runs' % len(common))
    anchor = common[-1]                     # the reverse run's own origin
    A = np.linalg.inv(fwd[anchor])
    rows = []
    for row in common:
        d = np.linalg.inv(rev[row]) @ (A @ fwd[row])
        rows.append((int(idx[row]),
                     float(np.linalg.norm(d[:3, 3])),      # metres apart
                     rot_deg(d),                           # degrees apart
                     float(np.linalg.norm(
                         (A @ fwd[row])[:3, 3] - fwd[anchor][:3, 3] * 0))))
    rows.sort()
    t = np.array([r[1] for r in rows])
    a = np.array([r[2] for r in rows])
    # how far the camera travelled, so the residual can be read as a rate
    path = sum(float(np.linalg.norm(fwd[b][:3, 3] - fwd[a_][:3, 3]))
               for a_, b in zip(common, common[1:]))
    return rows, {'n': len(rows), 'path_m': path,
                  'trans_median': float(np.median(t)),
                  'trans_max': float(t.max()),
                  'trans_at_far_end': rows[0][1],
                  'rot_median': float(np.median(a)),
                  'rot_max': float(a.max()),
                  'rot_at_far_end': rows[0][2]}


def main():
    from inference.utils import setup_logging
    p = argparse.ArgumentParser(description=__doc__.strip().splitlines()[2])
    p.add_argument('capture_dir')
    p.add_argument('--bag', default=None)
    p.add_argument('--cache', default=None, help='where to hold the frames')
    p.add_argument('--keep_cache', action='store_true')
    p.add_argument('--every', type=int, default=1,
                   help='use every Nth cached frame in BOTH runs (2 halves the '
                        'work and still accumulates over the same sweep)')
    # identical to fuse_scan.py's, so the tracker sees what it saw
    p.add_argument('--voxel', type=float, default=0.01)
    p.add_argument('--depth_max', type=float, default=3.0)
    p.add_argument('--depth_diff', type=float, default=0.07)
    p.add_argument('--trunc', type=float, default=8.0)
    p.add_argument('--block_count', type=int, default=30000)
    p.add_argument('--skip_s', type=float, default=1.0)
    p.add_argument('--cpu', action='store_true')
    args = p.parse_args()
    setup_logging()

    cap = args.capture_dir.rstrip('/')
    bag = args.bag or os.path.join(cap, 'scan', 'scan.bag')
    md = bag_metadata(bag)
    K = md['K']
    skip = int(round(args.skip_s * md['fps']))
    cache = args.cache or os.path.join(
        tempfile.gettempdir(), 'drift_%s' % os.path.basename(cap))
    log.info('%s: %s, %.0f fps, cache %s', os.path.basename(cap),
             md['device_name'], md['fps'], cache)

    depth, rgb, idx = cache_frames(bag, cache, skip)
    order = np.arange(0, len(idx), args.every)
    log.info('running %d frames forwards, then the same backwards', len(order))
    try:
        fwd, f_fit = run_slam(depth, rgb, order, K, md, args, 'forward')
        rev, r_fit = run_slam(depth, rgb, order[::-1], K, md, args, 'reverse')
    finally:
        del depth, rgb
        if not args.keep_cache:
            shutil.rmtree(cache, ignore_errors=True)

    rows, s = compare(fwd, rev, idx)
    print('\n=== %s: forward vs reverse, %d frames tracked by both ==='
          % (os.path.basename(cap), s['n']))
    print('camera path length %.2f m\n' % s['path_m'])
    print('%-10s %14s %12s' % ('bag frame', 'apart (mm)', 'apart (deg)'))
    step = max(1, len(rows) // 12)
    for r in rows[::step]:
        print('%-10d %13.1f %12.3f' % (r[0], 1000 * r[1], r[2]))
    print('\nresidual over the whole sweep (at the far end from the shared'
          ' origin):')
    print('  translation %.1f mm   rotation %.3f deg'
          % (1000 * s['trans_at_far_end'], s['rot_at_far_end']))
    print('  median %.1f mm / %.3f deg      worst %.1f mm / %.3f deg'
          % (1000 * s['trans_median'], s['rot_median'],
             1000 * s['trans_max'], s['rot_max']))
    print('  as a fraction of path length: %.2f%%'
          % (100 * s['trans_at_far_end'] / max(s['path_m'], 1e-9)))
    print('\nFor scale: one low-res voxel is 80 mm, one high-res voxel 20 mm.')
    print('This residual is a LOWER bound -- a bias committed in both'
          ' directions cancels.')
    out = os.path.join(_REPO_ROOT, 'outputs', os.path.basename(cap),
                       'drift_check.json')
    os.makedirs(os.path.dirname(out), exist_ok=True)
    json.dump({'summary': s,
               'fitness': {'forward_median': float(np.median(f_fit)),
                           'reverse_median': float(np.median(r_fit))},
               'per_frame': [{'frame': r[0], 'trans_m': r[1], 'rot_deg': r[2]}
                             for r in rows]},
              open(out, 'w'), indent=2)
    print('\nwrote %s' % os.path.relpath(out, _REPO_ROOT))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
