#!/usr/bin/env bash
# Put the capture's MEASURED tilt into meta.json, then re-fuse.
#
#   ./reconstruction_GT/refine_tilt.sh room08
#   ./reconstruction_GT/refine_tilt.sh room08 --from-floor
#   ./reconstruction_GT/refine_tilt.sh room08 --no-refuse
#
# --from-floor takes the tilt from the FITTED FLOOR instead of the
# accelerometer, which is the better instrument once a sweep exists: tens of
# thousands of points on a real plane against an IMU's bias. Run it after the
# accelerometer pass and the floor comes out level.
#
# The grid is built level unless meta.json carries `up_camera`, the gravity
# direction the accelerometer read while the camera sat still at the start of
# the sweep (fuse_scan records it as the imu_tilt check). With it, a camera
# that was pitched or rolled still gets a level, floor-anchored grid.
#
# It is worth doing whenever imu_tilt warns: 4.2 degrees of pitch moves a
# surface 3 m away by 22 cm, which is about three low-res voxels, and no
# annotation can compensate for it afterwards.
#
# Like refine_height.sh this writes a measured value, records where it came
# from, and re-fuses so the mesh matches the new frame.
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"
need_room "${1:-}"
ROOM="$1"; shift || true
DIR="$(capdir "$ROOM")"
REFUSE=1
export TILT_FROM=imu
for a in "$@"; do
    [ "$a" = "--no-refuse" ] && REFUSE=0
    [ "$a" = "--from-floor" ] && TILT_FROM=floor
done

REPO="$REPO" "$PY" - "$DIR" <<'PY'
import json, os, sys, time
import numpy as np
cap = sys.argv[1]
meta_path = os.path.join(cap, 'meta.json')
rep_path = os.path.join(cap, 'scan', 'fuse_report.json')
for p in (meta_path, rep_path):
    if not os.path.exists(p):
        raise SystemExit('missing %s -- capture and fuse first' % p)
meta, rep = json.load(open(meta_path)), json.load(open(rep_path))
imu = rep['checks'].get('imu_tilt', {})
if imu.get('status') == 'skipped':
    raise SystemExit('no IMU measurement in this capture (%s).\n'
                     'Without gravity the grid can only assume level.'
                     % imu.get('reason'))
source_of = os.environ.get('TILT_FROM', 'imu')
if source_of == 'floor':
    fl = rep['checks'].get('floor', {})
    n_world = fl.get('normal_world')
    if n_world is None:
        raise SystemExit('the floor check has no normal_world -- re-fuse first')
    if fl.get('status') == 'skipped':
        raise SystemExit('no floor was found in this sweep: %s' % fl.get('reason'))
    # the floor's normal IS up. Express it in camera coordinates and that is
    # the gravity the grid should use.
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(cap))))
    sys.path.insert(0, os.environ['REPO'])
    from inference.frame_loader import cam_pose_from_meta
    R = cam_pose_from_meta(meta)[:3, :3].astype(np.float64)
    up = (R.T @ np.asarray(n_world, np.float64)).tolist()
    tilt = float(np.degrees(np.arccos(min(1.0, abs(np.asarray(n_world)[2])))))
    print('  floor plane       : %d points, %.1f m2, %.2f deg off level'
          % (fl.get('vertices', 0), fl.get('area_m2', 0), tilt))
else:
    up = imu.get('up_in_camera')
    if up is None:
        raise SystemExit('the fuse report has no up_in_camera -- re-fuse first')
    tilt = imu.get('tilt_deg', 0.0)
print('  measured gravity  : %s   tilt %.2f deg (%.0f cm at 3 m)'
      % (np.round(up, 4).tolist(), tilt, imu.get('offset_at_3m_cm', 0)))
if meta.get('up_camera') == up:
    print('  meta.json already carries it -- nothing to do')
    sys.exit(3)
if tilt > 25:
    raise SystemExit('%.0f deg is not a camera someone aimed at a room -- '
                     'refusing' % tilt)
# Changing `up_camera` re-defines the world: the mesh will be re-fused into a
# frame rotated by the correction, and ANY ANNOTATION ALREADY MADE is expressed
# in the old one. Carry it across with the same 4x4 the frames use, or an
# afternoon of boxes silently ends up tilted against its own room.
sol_path = os.path.join(cap, 'gt', 'solids.json')
if os.path.exists(sol_path):
    sys.path.insert(0, os.environ.get('REPO', '.'))
    from inference.frame_loader import live_cam_pose
    from reconstruction_GT.voxelize_gt import transform_solids
    h = float(meta['camera_height'])
    yaw_old = float(meta.get('yaw', 0.0))
    P_old = live_cam_pose(h, yaw_old, meta.get('up_camera')).astype(np.float64)
    P_new = live_cam_pose(h, yaw_old, up).astype(np.float64)
    A = P_new @ np.linalg.inv(P_old)                 # new world <- old world
    spec = json.load(open(sol_path))
    backup = sol_path + '.pretilt'
    if not os.path.exists(backup):
        json.dump(spec, open(backup, 'w'), indent=2)
    moved = transform_solids(spec, A)
    moved['_moved'] = ('rotated into the corrected frame on %s (was %.2f deg '
                       'off level); the original is beside this file as '
                       'solids.json.pretilt' % (time.strftime('%Y-%m-%d'), tilt))
    json.dump(moved, open(sol_path, 'w'), indent=2)
    print('  moved %d solids into the corrected frame (backup: %s)'
          % (len(moved.get('solids', [])), os.path.basename(backup)))

meta['up_camera'] = [round(float(v), 6) for v in up]
meta['up_camera_source'] = (
    ('floor plane, %d points, via scan/fuse_report.json on %s; it was %.2f deg '
     'off level' % (rep['checks']['floor'].get('vertices', 0),
                    time.strftime('%Y-%m-%d'), tilt))
    if source_of == 'floor' else
    ('accelerometer, %d samples while still at the start of the sweep, via '
     'scan/fuse_report.json on %s; tilt %.2f deg'
     % (imu.get('samples', 0), time.strftime('%Y-%m-%d'), tilt)))
json.dump(meta, open(meta_path, 'w'), indent=2)
print('  meta.json updated: the grid now follows gravity, not the assumption')
PY
RC=$?
# `[ test ] && action` under `set -e` aborts the script when the test is FALSE,
# which silently skipped the re-fuse below and left meta.json describing a
# world the mesh was not in. Same trap as the three guards in _common.sh.
if [ "$RC" -eq 3 ]; then
    exit 0
fi
if [ "$RC" -ne 0 ]; then
    exit "$RC"
fi

if [ "$REFUSE" -eq 1 ]; then
    say "re-fusing in the corrected frame"
    "$(dirname "${BASH_SOURCE[0]}")/5_fuse_scan.sh" "$ROOM"
else
    warn "not re-fusing (--no-refuse): the mesh still uses the level frame"
fi
