"""
reconstruction_GT/play_bag.py

Play a recorded sweep: the photo and the depth side by side, and the room
building up in 3D as it tracks.

    python -m reconstruction_GT.play_bag captures/room07
    python -m reconstruction_GT.play_bag captures/room07 --speed 2
    python -m reconstruction_GT.play_bag captures/room07 --headless   # no windows

    SPACE pause/resume      Q or ESC  stop

This is the same tracking and fusion `fuse_scan.py` runs, with the model drawn
every few frames instead of only at the end. Use it to see WHY a sweep came out
the way it did: where tracking got thin, which surfaces never got a second
look, when you walked through your own shot.

It writes nothing. `fuse_scan.py` remains the thing that produces `room.ply`;
this only looks.

*** WHAT THE TWO PANES TELL YOU ***

The depth pane matters more than the photo. Dark, shiny and translucent
surfaces look fine in colour and are missing in depth, and a sweep that lost
tracking almost always lost it over a stretch where depth was sparse. The
title bar carries the live frame rate, the tracking fitness and how many
frames have been dropped, which is the same fitness `fuse_scan` thresholds at
0.1.
"""

import os
import sys
import time
import json
import logging
import argparse

import numpy as np

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_THIS_DIR)
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from inference.colorize_depth import INVALID_COLORS, colorize
from reconstruction_GT.fuse_scan import (MAX_SPEED_M_S, MAX_TURN_DEG_S,
                                         MIN_FITNESS)

log = logging.getLogger(__name__)


def parse_args():
    p = argparse.ArgumentParser(description='Play a recorded sweep.')
    p.add_argument('capture_dir')
    p.add_argument('--bag', default=None)
    p.add_argument('--voxel', type=float, default=0.02)
    p.add_argument('--depth_max', type=float, default=3.0)
    p.add_argument('--every', type=int, default=10,
                   help='redraw the 3D model every N frames')
    p.add_argument('--speed', type=float, default=1.0,
                   help='playback speed; 0 = as fast as it will go')
    p.add_argument('--max_frames', type=int, default=0)
    p.add_argument('--headless', action='store_true',
                   help='track and fuse without opening windows')
    return p.parse_args()


def main():
    from inference.utils import setup_logging
    setup_logging()
    args = parse_args()
    import cv2
    import open3d as o3d
    import open3d.core as o3c

    bag = args.bag or os.path.join(args.capture_dir, 'scan', 'scan.bag')
    if not os.path.exists(bag):
        raise SystemExit('no bag at %s' % bag)
    rec = {}
    rec_path = os.path.join(os.path.dirname(bag), 'record.json')
    if os.path.exists(rec_path):
        rec = json.load(open(rec_path))

    reader = o3d.t.io.RSBagReader()
    reader.open(bag)
    md = reader.metadata
    fps = float(md.fps)
    scale = float(md.depth_scale)
    device = o3c.Device('CUDA:0' if o3c.cuda.is_available() else 'CPU:0')
    K = o3c.Tensor(np.asarray(md.intrinsics.intrinsic_matrix), o3c.Dtype.Float64)
    log.info('%s: %dx%d @ %g fps, %.1f s, on %s', os.path.basename(bag),
             md.width, md.height, fps, md.stream_length_usec / 1e6, device)

    model = o3d.t.pipelines.slam.Model(args.voxel, 16, 30000,
                                       o3c.Tensor(np.eye(4)), device)
    frame = o3d.t.pipelines.slam.Frame(md.height, md.width, K, device)
    raycast = o3d.t.pipelines.slam.Frame(md.height, md.width, K, device)

    # the same limits fuse_scan uses, so what you see dropped here is what it
    # would drop
    measured = (rec.get('frames_seen', 0) / rec['duration_s']
                if rec.get('duration_s') else fps)
    rate = measured if measured < 0.8 * fps else fps
    max_step, max_turn = MAX_SPEED_M_S / rate, MAX_TURN_DEG_S / rate

    vis = None
    if not args.headless:
        vis = o3d.visualization.Visualizer()
        vis.create_window('the room, building up', 1100, 800)
    cloud = o3d.geometry.PointCloud()
    added = False

    T = np.eye(4)
    n = fused = dropped = 0
    fitness = 0.0
    paused = False
    t0 = time.time()
    try:
        while not reader.is_eof():
            if paused:
                if vis is not None:
                    vis.poll_events(); vis.update_renderer()
                k = cv2.waitKey(30) & 0xFF
                if k == 32:
                    paused = False
                elif k in (ord('q'), 27):
                    break
                continue
            rgbd = reader.next_frame()
            n += 1
            if rgbd.is_empty():
                continue

            frame.set_data_from_image('depth', rgbd.depth.to(device))
            frame.set_data_from_image('color', rgbd.color.to(device))
            keep = True
            if fused > 0:
                try:
                    res = model.track_frame_to_model(frame, raycast, scale,
                                                     args.depth_max)
                    step = res.transformation.cpu().numpy()
                    turn = np.degrees(np.arccos(np.clip(
                        (np.trace(step[:3, :3]) - 1.0) / 2.0, -1.0, 1.0)))
                    fitness = float(res.fitness)
                    keep = (fitness >= MIN_FITNESS and
                            np.linalg.norm(step[:3, 3]) <= max_step and
                            turn <= max_turn)
                    if keep:
                        T = T @ step
                except RuntimeError:
                    keep = False
            if not keep:
                dropped += 1
            else:
                model.update_frame_pose(fused, o3c.Tensor(T))
                model.integrate(frame, scale, args.depth_max)
                model.synthesize_model_frame(raycast, scale, 0.1, args.depth_max)
                fused += 1

            if not args.headless:
                rgb = rgbd.color.as_tensor().numpy()
                depth = rgbd.depth.as_tensor().numpy()[..., 0]
                vis_d = colorize(np.clip(depth, 0, 65535).astype(np.uint16),
                                 'rs-jet', INVALID_COLORS['black'], 'equalize')
                pane = np.hstack([cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), vis_d])
                txt = ('frame %d/%d   fused %d   dropped %d   fitness %.2f   %s'
                       % (n, int(md.stream_length_usec / 1e6 * fps), fused,
                          dropped, fitness, 'DROPPED' if not keep else ''))
                cv2.putText(pane, txt, (14, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                            (0, 0, 0), 4, cv2.LINE_AA)
                cv2.putText(pane, txt, (14, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                            (90, 255, 90) if keep else (60, 60, 255), 2, cv2.LINE_AA)
                cv2.imshow('photo  |  depth      SPACE pause, Q quit', pane)

                if fused and fused % args.every == 0:
                    pcd = model.extract_pointcloud(3.0).to_legacy()
                    cloud.points = pcd.points
                    cloud.colors = pcd.colors
                    if not added:
                        vis.add_geometry(cloud)
                        added = True
                    else:
                        vis.update_geometry(cloud)
                vis.poll_events()
                vis.update_renderer()

                k = cv2.waitKey(1) & 0xFF
                if k in (ord('q'), 27):
                    break
                if k == 32:
                    paused = True
            if args.speed:
                due = t0 + n / (fps * args.speed)
                if due > time.time():
                    time.sleep(due - time.time())
            if args.max_frames and n >= args.max_frames:
                break
    finally:
        reader.close()
        if not args.headless:
            cv2.destroyAllWindows()
            cv2.waitKey(1)
            if vis is not None:
                vis.destroy_window()

    log.info('played %d frames: %d fused, %d dropped, %.1f s',
             n, fused, dropped, time.time() - t0)
    if dropped > 0.05 * max(n, 1):
        log.warning('%.0f%% of frames were dropped -- that sweep would fuse '
                    'badly. Move more slowly, and keep furniture or a corner '
                    'in view rather than a blank wall.', 100 * dropped / n)


if __name__ == '__main__':
    main()
