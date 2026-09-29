"""
reconstruction_GT/measure_floor.py

Grab one extra frame, aimed so the floor IS in it, and measure how far the lens
sits above the floor -- from that single frame, with no sweep and no tracking.

    python -m reconstruction_GT.measure_floor --capture captures/room07

This is a DIAGNOSTIC frame, not an evaluation frame. Tilt the camera down as
much as you need: the measurement uses the accelerometer's gravity vector, so a
tilted camera is fine, and nothing here touches the capture's meta.json or its
`depth/live_000000.png`. The frame is saved under <capture>/floor_check/ so the
number stays auditable.

*** WHY IT EXISTS ***

A level camera at desk height usually cannot see its own floor -- in room07 the
bed blocks it, and the still frame's deepest point is 0.99 m below the lens.
So the floor distance has only ever come from the fused sweep, through the
tracking chain. That chain disagrees with the tape by ~7 cm (mesh 1.076,
tape 1.150), and there is no way to tell drift from a depth-scale error while
both numbers come from the same source.

One tilted frame settles it: it sees the floor directly, at close range, with
no tracking in between.

*** HOW ***

    up          mean accelerometer vector while still -- at rest it reads the
                reaction to gravity, i.e. it points UP, in camera coordinates
    height      -(P . up) for every unprojected depth point: metres below the
                lens along gravity
    floor       the largest peak in that histogram below --min_drop, refined by
                a least-squares plane fit over its inliers
    answer      perpendicular distance from the lens to that plane

The plane's tilt against gravity is reported too: a real floor comes out within
a degree or two, and anything larger means the peak is a bed or a desk.
"""

import os
import sys
import json
import time
import logging
import argparse

import numpy as np

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_THIS_DIR)
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from inference.utils import setup_logging

log = logging.getLogger(__name__)


def parse_args():
    p = argparse.ArgumentParser(
        description='Measure lens height above the floor from one tilted frame.')
    p.add_argument('--capture', default=None,
                   help='capture folder to save into and compare against '
                        '(e.g. captures/room07)')
    p.add_argument('--out_dir', default=None,
                   help='where to save the frame (default <capture>/floor_check)')
    p.add_argument('--width', type=int, default=848)
    p.add_argument('--height', type=int, default=480)
    p.add_argument('--fps', type=int, default=15)
    p.add_argument('--frames', type=int, default=20,
                   help='depth frames to median together')
    p.add_argument('--warmup', type=int, default=30)
    p.add_argument('--min_drop', type=float, default=0.40,
                   help='ignore surfaces less than this far below the lens, so '
                        'a bed or a desk cannot be mistaken for the floor')
    p.add_argument('--max_drop', type=float, default=2.50)
    return p.parse_args()


def grab(args):
    """-> (depth metres HxW, colour BGR, K 3x3, up vector in camera coords)."""
    import pyrealsense2 as rs
    pipe, cfg = rs.pipeline(), rs.config()
    cfg.enable_stream(rs.stream.depth, args.width, args.height, rs.format.z16, args.fps)
    cfg.enable_stream(rs.stream.color, args.width, args.height, rs.format.bgr8, args.fps)
    accel = True
    try:
        cfg.enable_stream(rs.stream.accel)
        prof = pipe.start(cfg)
    except RuntimeError:
        accel = False
        cfg = rs.config()
        cfg.enable_stream(rs.stream.depth, args.width, args.height, rs.format.z16, args.fps)
        cfg.enable_stream(rs.stream.color, args.width, args.height, rs.format.bgr8, args.fps)
        prof = pipe.start(cfg)
        log.warning('no accelerometer: assuming the camera is LEVEL, which '
                    'defeats the point of tilting it. Install the udev rules '
                    '(1_check_camera.sh says how).')
    try:
        scale = prof.get_device().first_depth_sensor().get_depth_scale()
        di = prof.get_stream(rs.stream.depth).as_video_stream_profile().get_intrinsics()
        K = np.array([[di.fx, 0, di.ppx], [0, di.fy, di.ppy], [0, 0, 1]], np.float64)
        R = np.eye(3)
        if accel:
            ext = prof.get_stream(rs.stream.accel).get_extrinsics_to(
                prof.get_stream(rs.stream.depth))
            R = np.asarray(ext.rotation, np.float64).reshape(3, 3).T
        for _ in range(args.warmup):
            pipe.wait_for_frames()
        depths, colour, acc = [], None, []
        t0 = time.time()
        while len(depths) < args.frames and time.time() - t0 < 20:
            fs = pipe.wait_for_frames(5000)
            d, c = fs.get_depth_frame(), fs.get_color_frame()
            if d:
                depths.append(np.asanyarray(d.get_data()).astype(np.float32) * scale)
            if c is not None and c:
                colour = np.asanyarray(c.get_data()).copy()
            for f in fs:
                if f.is_motion_frame():
                    m = f.as_motion_frame().get_motion_data()
                    acc.append((m.x, m.y, m.z))
    finally:
        pipe.stop()

    if not depths:
        raise SystemExit('no depth frames arrived')
    depth = np.median(np.stack(depths), axis=0)
    if accel and len(acc) >= 5:
        a = np.asarray(acc)
        mag = np.linalg.norm(a, axis=1)
        if mag.std() > 0.05 * 9.81:
            log.warning('the camera moved while measuring (|a| std %.2f m/s^2); '
                        'hold it still', mag.std())
        up = R @ (a.mean(axis=0) / np.linalg.norm(a.mean(axis=0)))
    else:
        up = np.array([0.0, -1.0, 0.0])          # camera -Y, i.e. assume level
    return depth, colour, K, up / np.linalg.norm(up)


def fit_floor(depth, K, up, min_drop, max_drop):
    v, u = np.nonzero(depth > 0)
    z = depth[v, u].astype(np.float64)
    P = np.stack([(u - K[0, 2]) * z / K[0, 0],
                  (v - K[1, 2]) * z / K[1, 1], z], axis=1)
    h = -(P @ up)                                # metres below the lens
    band = (h > min_drop) & (h < max_drop)
    if band.sum() < 2000:
        raise SystemExit('only %d points between %.2f and %.2f m below the lens '
                         '-- tilt the camera further down so the floor is in '
                         'view' % (int(band.sum()), min_drop, max_drop))
    hist, edges = np.histogram(h[band], bins=np.arange(min_drop, max_drop, 0.02))
    peak = edges[int(np.argmax(hist))]
    # two refinement passes: take a slab around the peak, fit a plane, re-select
    sel = band & (np.abs(h - (peak + 0.01)) < 0.06)
    # plane as n.P + d = 0. Start from "level at the peak": h = -(P.up), so a
    # plane at height h below the lens is up.P + h = 0.
    n, d = up.copy(), float(np.median(h[sel]))
    for _ in range(2):
        Q = P[sel]
        c = Q.mean(axis=0)
        n = np.linalg.svd(Q - c, full_matrices=False)[2][2]
        if n @ up < 0:                           # make the normal point UP
            n = -n
        d = -(n @ c)
        dist = P @ n + d                         # signed distance to the plane
        sel = band & (np.abs(dist) < 0.03)
        if sel.sum() < 1000:
            break
    height = abs(d)                              # lens is at the origin
    tilt = float(np.degrees(np.arccos(np.clip(abs(n @ up), -1.0, 1.0))))
    return {'lens_height_m': round(float(height), 4),
            'plane_tilt_vs_gravity_deg': round(tilt, 2),
            'inliers': int(sel.sum()),
            'peak_below_lens_m': round(float(peak), 3),
            'points_in_band': int(band.sum()),
            'normal_camera': [round(float(x), 4) for x in n],
            'up_camera': [round(float(x), 4) for x in up]}


def main():
    setup_logging()
    args = parse_args()
    import cv2

    out_dir = args.out_dir or (os.path.join(args.capture, 'floor_check')
                               if args.capture else 'floor_check')
    os.makedirs(out_dir, exist_ok=True)

    log.info('aim the camera so the FLOOR fills a good part of the frame -- '
             'tilt down as far as you like, and keep the rig at its capture '
             'height. Hold still.')
    depth, colour, K, up = grab(args)
    res = fit_floor(depth, K, up, args.min_drop, args.max_drop)

    stamp = time.strftime('%Y%m%d_%H%M%S')
    cv2.imwrite(os.path.join(out_dir, 'floor_%s_depth.png' % stamp),
                np.clip(depth * 1000.0, 0, 65535).astype(np.uint16))
    if colour is not None:
        cv2.imwrite(os.path.join(out_dir, 'floor_%s_rgb.png' % stamp), colour)

    res['camera_pitch_down_deg'] = round(float(np.degrees(np.arctan2(-up[2], -up[1]))), 2)
    res['camera_roll_deg'] = round(float(np.degrees(np.arctan2(up[0], -up[1]))), 2)
    res['cam_K_depth'] = K.tolist()
    res['frames'] = args.frames
    res['note'] = ('diagnostic frame, NOT an evaluation frame: the camera is '
                   'deliberately tilted and the voxel grid assumes level')

    print('\n  lens above floor : %.3f m' % res['lens_height_m'])
    print('  floor plane tilt : %.2f deg vs gravity (%d inliers)'
          % (res['plane_tilt_vs_gravity_deg'], res['inliers']))
    print('  camera aimed     : %.1f deg down, %.1f deg roll'
          % (res['camera_pitch_down_deg'], res['camera_roll_deg']))
    if res['plane_tilt_vs_gravity_deg'] > 3.0:
        print('  WARNING: that surface is not level -- it is probably a bed or '
              'a desk, not the floor. Tilt further down.')

    if args.capture:
        meta_path = os.path.join(args.capture, 'meta.json')
        if os.path.exists(meta_path):
            rec = json.load(open(meta_path))['camera_height']
            res['capture_camera_height'] = rec
            res['difference_m'] = round(rec - res['lens_height_m'], 4)
            print('\n  %s meta.json    : %.3f m' % (os.path.basename(args.capture), rec))
            print('  this measurement : %.3f m   (difference %+.3f m)'
                  % (res['lens_height_m'], res['difference_m']))
        rep = os.path.join(args.capture, 'scan', 'fuse_report.json')
        if os.path.exists(rep):
            r = json.load(open(rep))
            fl = r['checks'].get('floor', {}).get('lowest_up_surface_z_m')
            if fl is not None and 'capture_camera_height' in res:
                mesh_h = res['capture_camera_height'] - fl
                res['fused_mesh_height_m'] = round(mesh_h, 4)
                print('  fused mesh says  : %.3f m' % mesh_h)
                d_mesh = abs(mesh_h - res['lens_height_m'])
                d_tape = abs(res['capture_camera_height'] - res['lens_height_m'])
                # Only a reading taken at the capture height can arbitrate. A
                # camera resting somewhere else matches neither, and saying it
                # "agrees" with whichever is nearer would be nonsense.
                if min(d_mesh, d_tape) > 0.15:
                    res['verdict'] = 'inconclusive'
                    print('\n  -> matches NEITHER (%.2f m from the mesh, %.2f m '
                          'from the tape).\n     Is the rig at its capture '
                          'height? Re-run with the camera where it was.'
                          % (d_mesh, d_tape))
                elif d_mesh < d_tape:
                    res['verdict'] = 'mesh'
                    print('\n  -> agrees with the MESH (%.3f vs %.3f): the tape '
                          'was off, or\n     taken at a different height. Put '
                          'this number in meta.json.' % (mesh_h, res['capture_camera_height']))
                else:
                    res['verdict'] = 'tape'
                    print('\n  -> agrees with the TAPE (%.3f vs %.3f): the fused '
                          'floor is biased,\n     by drift or by depth scale. '
                          'Keep meta.json and tell me.'
                          % (res['capture_camera_height'], mesh_h))

    with open(os.path.join(out_dir, 'floor_%s.json' % stamp), 'w') as f:
        json.dump(res, f, indent=2)
    log.info('saved the frame and the numbers to %s', out_dir)
    print('\n  Nothing was written to meta.json. If you decide this is the '
          'height,\n  put it there yourself and re-run 5_fuse_scan.sh.')


if __name__ == '__main__':
    main()
