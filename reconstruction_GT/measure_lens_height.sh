#!/usr/bin/env bash
# Let the sensor measure its own height above the floor.
#
#   ./reconstruction_GT/measure_lens_height.sh            # just measure
#   ./reconstruction_GT/measure_lens_height.sh room07     # and compare to that capture
#
# Point the camera STRAIGHT DOWN at bare floor, at the height you want to
# record, and run this. It reports the distance from the depth module's front
# glass -- which is where the voxel grid's world origin sits -- to the floor.
#
# It exists because a tape measure needs you to pick a point on the camera, and
# picking the tripod head or the top of the body instead of the lens is a ~12 cm
# error that lands in meta.json as a measured fact and shifts every GT voxel.
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"

ROOM="${1:-}"
say "measuring -- point the camera straight down at bare floor"
"$PY" - "$([ -n "$ROOM" ] && capdir "$ROOM")" <<'PY'
import sys, json, os
import numpy as np
import pyrealsense2 as rs

pipe, cfg = rs.pipeline(), rs.config()
cfg.enable_stream(rs.stream.depth, 848, 480, rs.format.z16, 15)
try:
    prof = pipe.start(cfg)
except RuntimeError as e:
    raise SystemExit('cannot start the camera: %s' % e)
try:
    scale = prof.get_device().first_depth_sensor().get_depth_scale()
    for _ in range(30):                      # let auto-exposure settle
        pipe.wait_for_frames()
    med, valid = [], []
    for _ in range(30):
        d = np.asanyarray(pipe.wait_for_frames().get_depth_frame().get_data())
        c = d[220:260, 404:444].astype(np.float64) * scale     # central patch
        ok = c[c > 0]
        valid.append(ok.size / c.size)
        if ok.size:
            med.append(np.median(ok))
finally:
    pipe.stop()

if not med:
    raise SystemExit('no depth in the centre of the frame: too close (min ~0.5 m), '
                     'too dark, or aimed at something the sensor cannot see')
med = np.array(med)
h, spread, v = float(np.median(med)), float(med.max() - med.min()), float(np.mean(valid))
print('\n  lens to floor : %.3f m' % h)
print('  frame spread  : %.3f m over %d frames' % (spread, med.size))
print('  valid pixels  : %.0f%% of the centre patch' % (100 * v))
if v < 0.5:
    print('  WARNING: less than half the patch has depth -- aim at bare floor, '
          'not a rug edge or a shiny surface')
if spread > 0.02:
    print('  WARNING: the reading moves between frames -- hold the camera still')

cap = sys.argv[1] if len(sys.argv) > 1 else ''
if cap:
    meta = os.path.join(cap, 'meta.json')
    if os.path.exists(meta):
        rec = json.load(open(meta))['camera_height']
        print('\n  %s records camera_height = %.3f m' % (os.path.basename(cap), rec))
        print('  difference             = %+.3f m' % (rec - h))
        if abs(rec - h) > 0.03:
            print('\n  These disagree. The grid is floor-anchored, so this offset moves')
            print('  every ground-truth voxel. If the camera has NOT moved since that')
            print('  capture, this reading is the measurement -- put it in meta.json')
            print('  and re-run 5_fuse_scan.sh. If it HAS moved, re-record instead.')
        else:
            print('\n  Agrees with the capture. Nothing to change.')
    else:
        print('\n  no meta.json at %s' % meta)
print('\n  Put this number in the preview\'s height field for the next capture.')
PY
