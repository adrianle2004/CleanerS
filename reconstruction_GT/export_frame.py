"""
reconstruction_GT/export_frame.py

Add a frame from the SWEEP to the room capture, so the annotation you already
made scores it too.

    python -m reconstruction_GT.export_frame captures/room08 --list
    python -m reconstruction_GT.export_frame captures/room08 --frame 445

Then the usual loop, on the room itself:

    python -m reconstruction_GT.voxelize_gt captures/room08 --frame live_000445
    python -m inference.run_inference ... --live_dir captures/room08 \
        --out_dir ./outputs/room08
    python -m reconstruction_GT.evaluate_gt captures/room08 --report

*** WHY ***

The still frame is one viewpoint, chosen before anything was known about it.
The sweep holds a thousand more, already tracked, and some of them are far
better tests: they see the floor, and they put furniture between the camera and
empty space, which is the only way the SC metric has anything to measure. On
room08 the still frame sees 0% floor; frame 445 sees 14%.

It also multiplies the annotation. The solids describe the ROOM, so every frame
of that room can be scored against them -- MAKING_GT.md, "Annotate the room,
not the frame". One afternoon of boxes, thirty evaluation frames.

*** WHAT IT WRITES ***

Nothing new beside the room; everything lands in the room's own folders, under
the frame's stem:

    depth/live_000445.png   the frame, resized and cropped to the 640x480 NYU
    rgb/live_000445.png     geometry the model is trained on (`match_nyu_fov`)
    meta.json           "frames" gains live_000445, and "frame_meta" gains its
                        camera: cam_K, camera_height and up_camera from the
                        tracked pose, and `world_from_room`, the 4x4 that
                        carries the room's solids into this frame's world

So there is still ONE annotation (`gt/solids.json`, in the room's world) and
one set of measurements per frame. `voxelize_gt` applies the matrix.

*** WHAT IT CANNOT DO ***

The bag holds raw frames: `capture.py` undistorts before cropping, and there is
no distortion model in the bag's metadata, so an exported frame keeps the
lens's own distortion -- up to ~3 cm at the frame corners, nothing at the
centre. And the pose comes from tracking rather than from a tripod, so it
carries whatever drift the sweep accumulated by that frame. Both are fine for
an extra evaluation frame; neither is as clean as the still.

Frames come through reconstruction_GT/bag_reader.py, the same reader
fuse_scan.py writes trajectory.txt with, so an index here is the index there --
MAKING_GT.md, gotcha 10.
"""

import os
import sys
import json
import glob
import logging
import argparse

import numpy as np

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_THIS_DIR)
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from inference.frame_loader import live_cam_pose

log = logging.getLogger(__name__)


def parse_args():
    p = argparse.ArgumentParser(
        description='Export a swept frame as an evaluation capture.')
    p.add_argument('capture_dir')
    p.add_argument('--frame', type=int, default=None,
                   help='bag frame index, as --list prints them')
    p.add_argument('--name', default=None,
                   help='frame stem (default live_<frame>, zero-padded to 6 '
                        'digits like capture.py names its own)')
    p.add_argument('--list', action='store_true',
                   help='rank candidate frames and exit')
    p.add_argument('--every', type=int, default=5, help='--list: sampling step')
    p.add_argument('--top', type=int, default=12, help='--list: how many to show')
    p.add_argument('--rank', choices=['both', 'floor', 'occlusion'],
                   default='both', help='--list: what to sort by (default both)')
    return p.parse_args()


def load_poses(cap):
    path = os.path.join(cap, 'scan', 'trajectory.txt')
    if not os.path.exists(path):
        raise SystemExit('no %s -- fuse the sweep first' % path)
    out = {}
    for line in open(path):
        if line.startswith('#'):
            continue
        f = line.split()
        out[int(f[0])] = np.array([float(x) for x in f[1:]]).reshape(4, 4)
    return out


def frame_stats(cap, poses, every):
    """-> [(frame, cam_z, valid%, floor%, near%, far%)] for sampled frames."""
    from reconstruction_GT.bag_reader import iter_bag
    out = []
    for bf in iter_bag(os.path.join(cap, 'scan', 'scan.bag'), want_color=False):
        i = bf.index
        if i not in poses or i % every:
            continue
        d = bf.depth
        ok = d > 0
        if ok.sum() < 5000:
            continue
        uu, vv = np.meshgrid(np.arange(d.shape[1]), np.arange(d.shape[0]))
        K, z = bf.K, d[ok]
        P = np.stack([(uu[ok] - K[0, 2]) * z / K[0, 0],
                      (vv[ok] - K[1, 2]) * z / K[1, 1], z], axis=1)
        W = P @ poses[i][:3, :3].T + poses[i][:3, 3]
        out.append((i, float(poses[i][2, 3]), 100 * float(ok.mean()),
                    100 * float((W[:, 2] < 0.12).mean()),
                    100 * float((z < 1.6).mean()), 100 * float((z > 2.5).mean())))
    return out


def read_frame(cap, index, align='color'):
    """-> the BagFrame for one bag frame, at native resolution.

    align='color' -- the depth is projected into the COLOUR camera and the
    colour image is left alone, which is what NYU is and what
    inference/camera.py does for the still frame. It is the arrangement the
    model was trained under: complete RGB, and depth that carries the
    reprojection's own losses as part of its noise. Matching it matters more
    than any individual property of it; a capture that half-matches is a
    capture whose scores cannot be compared with anything.
    """
    from reconstruction_GT.bag_reader import read_frame as read_bag_frame
    return read_bag_frame(os.path.join(cap, 'scan', 'scan.bag'), index,
                          align=align)


def main():
    from inference.utils import setup_logging
    setup_logging()
    args = parse_args()
    import cv2
    from inference.camera import match_nyu_fov

    cap = args.capture_dir.rstrip('/')
    poses = load_poses(cap)
    log.info('%d tracked frames, %d..%d', len(poses), min(poses), max(poses))

    if args.list or args.frame is None:
        rows = frame_stats(cap, poses, args.every)
        rows.sort(key=lambda r: -rank_score(r, args.rank))
        print('\n%5s %6s %7s %7s %7s %7s %7s   best %s first'
              % ('frame', 'cam z', 'valid%', 'floor%', 'near%', 'far%',
                 'score', args.rank))
        for row in rows[:args.top]:
            print('%5d %6.2f %6.0f%% %6.1f%% %6.1f%% %6.1f%% %7.2f'
                  % (row + (rank_score(row, args.rank),)))
        print('\n--frame N to export one (--rank floor|occlusion|both).\n'
              'floor% is how much of the frame is floor, which is the class a '
              'level\ncamera never sees; near/far are the near and far halves, '
              'whose overlap\nis where occlusion -- and therefore anything for '
              'SC to measure -- lives.')
        return

    idx = args.frame
    if idx not in poses:
        raise SystemExit('frame %d was not tracked (fused frames: %d..%d)'
                         % (idx, min(poses), max(poses)))
    stem = args.name or 'live_%06d' % idx
    meta_path = os.path.join(cap, 'meta.json')
    meta = json.load(open(meta_path))
    if stem in meta.get('frames', []) and stem not in meta.get('frame_meta', {}):
        raise SystemExit(
            '%s already has a frame called %s, and it is not an exported one '
            '-- capture.py numbers its own frames live_000000 upward, so bag '
            'frame %d collides with it. Pass --name to choose another stem.'
            % (cap, stem, idx))
    for sub in ('depth', 'rgb'):
        os.makedirs(os.path.join(cap, sub), exist_ok=True)

    bf = read_frame(cap, idx)
    bgr = cv2.cvtColor(bf.rgb, cv2.COLOR_RGB2BGR)
    # one frame, one K: after align='color' both images live in the colour
    # camera, exactly as NYU's pair does
    depth_c, rgb_c, K_eff = match_nyu_fov(bf.depth, bgr, bf.K)
    cv2.imwrite(os.path.join(cap, 'depth', stem + '.png'),
                np.clip(depth_c * 1000.0, 0, 65535).astype(np.uint16))
    cv2.imwrite(os.path.join(cap, 'rgb', stem + '.png'), rgb_c)

    # This frame's own camera record. The room's meta.json keeps describing the
    # still frame; everything that differs here goes under "frame_meta", and
    # `world_from_room` is how the room's solids reach this frame's world --
    # frame_loader.frame_meta, voxelize_gt.transform_solids.
    # trajectory.txt holds DEPTH-camera poses (fuse_scan reads the bag with
    # align='depth'), and this frame is now expressed in the colour camera, so
    # the pose has to move with it: p_world = T_depth . A^-1 . p_colour
    T = poses[idx] @ np.linalg.inv(np.asarray(bf.color_from_depth, np.float64))
    height = float(T[2, 3])
    up_cam = (T[:3, :3].T @ np.array([0.0, 0.0, 1.0])).tolist()
    A = live_cam_pose(height, 0.0, up_cam).astype(np.float64) @ np.linalg.inv(T)
    entry = {
        'source': 'sweep frame %d of scan/scan.bag' % idx,
        'cam_K': np.asarray(K_eff, np.float64).tolist(),
        'cam_K_native': bf.K.tolist(),
        'registration': 'depth projected into the colour camera (rs.align to '
                        'colour), colour left untouched, unprojected with the '
                        'colour intrinsics -- NYU\'s arrangement, and the '
                        'same one inference/camera.py uses for the still '
                        'frame, so every frame of this capture matches what '
                        'the model was trained on',
        'camera_height': round(height, 4),
        'camera_height_source': 'tracked pose of bag frame %d' % idx,
        'yaw': 0.0,
        'up_camera': [round(float(v), 6) for v in up_cam],
        'up_camera_source': 'tracked pose of bag frame %d (gravity in its own '
                            'camera frame)' % idx,
        'world_from_room': [[round(float(v), 6) for v in row] for row in A],
        'undistorted': False,
        'note': 'exported from the sweep: raw (distorted) frame, pose from '
                'tracking. See reconstruction_GT/export_frame.py.',
    }
    meta.setdefault('frame_meta', {})[stem] = entry
    if stem not in meta.get('frames', []):
        meta.setdefault('frames', []).append(stem)
    with open(meta_path, 'w') as f:
        json.dump(meta, f, indent=2)

    out_dir = cap
    print('\nadded %s to %s' % (stem, cap))
    print('  NYU arrangement: %.2f%% black RGB, %.0f%% of the frame has depth'
          % (100 * float((rgb_c.sum(axis=2) == 0).mean()),
             100 * float((depth_c > 0).mean())))
    print('  camera %.3f m up, tilt %.1f deg, %.0f%% of the frame has depth'
          % (height,
             np.degrees(np.arccos(min(1.0, -up_cam[1] / np.linalg.norm(up_cam)))),
             100 * float((depth_c > 0).mean())))
    print('  python -m reconstruction_GT.voxelize_gt %s --frame %s' % (cap, stem))


if __name__ == '__main__':
    main()
