"""
reconstruction_GT/record_scan.py

Record a hand-held D455 sweep of a room to <out_dir>/scan/scan.bag, for
fuse_scan.py to turn into a mesh to place annotation solids against
(MAKING_GT.md, step 2). Run it straight after the still capture, into the same
folder, without moving the camera:

    python -m inference.capture --preview --out_dir captures/room02
    python -m reconstruction_GT.record_scan --preview --out_dir captures/room02
    python -m reconstruction_GT.fuse_scan captures/room02

The still capture (inference/capture.py) is the evaluation frame and writes
meta.json. This writes only under scan/, and never touches meta.json.
"""

import os
import sys
import json
import time
import logging
import argparse

import cv2
import numpy as np

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_THIS_DIR)
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from inference.capture import DEPTH_MAX_MM
from inference.colorize_depth import INVALID_COLORS, colorize
from inference.utils import ensure_dir, setup_logging

log = logging.getLogger(__name__)

# A sweep is minutes of frames, not one: at 848x480 -- the D455's native depth
# resolution -- and 15 fps the bag grows ~30 MB/s instead of ~135 MB/s for the
# still capture's 1280x720 @ 30, while slow hand motion still tracks.
RECORD_STREAM = (848, 480, 15)

# The first fused frame must be the still frame's pose, so the camera is held
# there this long before the sweep. fuse_scan skips ~1 s of it for
# auto-exposure and anchors the mesh on the frame after.
DEFAULT_HOLD_S = 3.0


def parse_args():
    p = argparse.ArgumentParser(
        description='Record a hand-held RealSense sweep for fuse_scan.py.')
    p.add_argument('--out_dir', required=True,
                   help='the capture folder the still frame went into; the '
                        'bag is written to <out_dir>/scan/scan.bag')
    p.add_argument('--width', type=int, default=RECORD_STREAM[0])
    p.add_argument('--height', type=int, default=RECORD_STREAM[1])
    p.add_argument('--fps', type=int, default=RECORD_STREAM[2])
    p.add_argument('--seconds', type=float, default=0.0,
                   help='stop after this long (0 = until q in the preview, or '
                        'Ctrl-C)')
    p.add_argument('--hold', type=float, default=DEFAULT_HOLD_S,
                   help='seconds to keep the camera exactly at the still pose '
                        'before sweeping (default %.1f)' % DEFAULT_HOLD_S)
    p.add_argument('--preview', action='store_true',
                   help='live RGB + colourised depth with the hold countdown')
    return p.parse_args()


def record(args):
    """Write a RealSense .bag of a hand-held sweep for fuse_scan.py.

    The bag holds raw sensor data -- depth, colour and the accelerometer --
    with none of the still path's crop, undistortion or filtering: fusion wants
    the full field of view, and fuse_scan clips depth itself. It writes nothing
    to meta.json. A scan is scaffolding for placing annotation solids
    (MAKING_GT.md), not an input to the network.

    The first --hold seconds must be taken with the camera exactly where the
    still capture left it. fuse_scan anchors the whole mesh on the first frame
    it fuses, so that pose IS the evaluation frame's pose; move before the hold
    ends and the mesh is offset from the grid by however far you moved.
    fuse_scan checks this against the still depth frame and warns.
    """
    import pyrealsense2 as rs

    bag = os.path.join(args.out_dir, 'scan', 'scan.bag')
    if os.path.exists(bag):
        raise SystemExit('%s already exists -- move it away first; a '
                         'recording is not something to overwrite by accident'
                         % bag)
    if not rs.context().query_devices():
        raise SystemExit('no RealSense device found -- check the USB 3 cable '
                         'and the udev rules (99-realsense-libusb.rules)')
    scan_dir = ensure_dir(os.path.dirname(bag))
    if not os.path.exists(os.path.join(args.out_dir, 'meta.json')):
        log.warning('%s has no meta.json: record the still frame first, or '
                    'fuse_scan can only give you a mesh in camera coordinates',
                    args.out_dir)

    def _config(accel):
        cfg = rs.config()
        cfg.enable_stream(rs.stream.depth, args.width, args.height,
                          rs.format.z16, args.fps)
        # rgb8, not bgr8: Open3D's bag reader takes the channels as RGB
        cfg.enable_stream(rs.stream.color, args.width, args.height,
                          rs.format.rgb8, args.fps)
        if accel:
            cfg.enable_stream(rs.stream.accel)
        cfg.enable_record_to_file(bag)
        return cfg

    pipe = rs.pipeline()
    try:
        pipe.start(_config(True))
        accel = True
    except RuntimeError as e:
        log.warning('accelerometer unavailable (%s); recording without it -- '
                    'fuse_scan will skip its tilt check', e)
        if os.path.exists(bag):
            os.remove(bag)              # the empty file the failed start made
        pipe.start(_config(False))
        accel = False

    preview = None
    if args.preview:
        if os.environ.get('DISPLAY') or os.environ.get('WAYLAND_DISPLAY'):
            preview = 'CleanerS record'
            cv2.namedWindow(preview, cv2.WINDOW_AUTOSIZE)
        else:
            log.warning('--preview ignored: no DISPLAY')
    log.info('recording to %s: HOLD STILL for %.1f s, then sweep slowly -- '
             '%s', bag, args.hold,
             'q in the window to stop' if preview else
             ('%.0f s total' % args.seconds if args.seconds else 'Ctrl-C to stop'))

    frames, told, t0 = 0, False, time.time()
    try:
        while True:
            el = time.time() - t0
            if args.seconds and el >= args.seconds:
                break
            if not told and el >= args.hold:
                log.info('hold done -- sweep now: slowly, within ~3 m of '
                         'surfaces, floor and furniture in view')
                told = True
            fs = pipe.wait_for_frames(5000)
            d, c = fs.get_depth_frame(), fs.get_color_frame()
            if not (d and c):
                continue                         # an accelerometer-only set
            frames += 1
            if preview is None:
                continue
            rgb = cv2.cvtColor(np.asanyarray(c.get_data()), cv2.COLOR_RGB2BGR)
            depth_mm = np.asanyarray(d.get_data()).astype(np.float32) * (
                d.get_units() * 1000.0)
            vis = colorize(np.clip(depth_mm, 0, DEPTH_MAX_MM).astype(np.uint16),
                           'rs-jet', INVALID_COLORS['black'], 'equalize')
            img = np.hstack([rgb, vis])
            if el < args.hold:
                text, color = 'HOLD STILL  %.1f s' % (args.hold - el), (60, 60, 255)
            else:
                text, color = ('RECORDING  %.0f s  --  sweep slowly, q to stop'
                               % el), (120, 255, 120)
            cv2.putText(img, text, (16, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.9,
                        (0, 0, 0), 4, cv2.LINE_AA)
            cv2.putText(img, text, (16, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.9,
                        color, 2, cv2.LINE_AA)
            cv2.imshow(preview, img)
            k = cv2.waitKey(1) & 0xFF
            if k in (ord('q'), 27):
                break
            try:
                if cv2.getWindowProperty(preview, cv2.WND_PROP_VISIBLE) < 1:
                    break
            except cv2.error:
                break
    except KeyboardInterrupt:
        log.info('stopped')
    finally:
        # stop() is what finalises the bag; a bag left open is unreadable
        pipe.stop()
        if preview is not None:
            cv2.destroyWindow(preview)
            cv2.waitKey(1)

    duration = time.time() - t0
    rec = {
        'stream': {'width': args.width, 'height': args.height, 'fps': args.fps},
        'accel': accel,
        'hold_s': args.hold,
        'duration_s': round(duration, 2),
        'frames_seen': frames,
    }
    with open(os.path.join(scan_dir, 'record.json'), 'w') as f:
        json.dump(rec, f, indent=2)
    if duration < args.hold + 5.0:
        log.warning('only %.1f s recorded -- that is barely past the hold, not '
                    'a sweep of the room', duration)
    log.info('wrote %s (%.0f MB, %.1f s, %d frames)', bag,
             os.path.getsize(bag) / 1e6, duration, frames)
    log.info('now run:  python -m reconstruction_GT.fuse_scan %s', args.out_dir)


def main():
    setup_logging()
    record(parse_args())


if __name__ == '__main__':
    main()
