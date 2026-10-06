#!/usr/bin/env bash
# Re-estimate camera_height from the fused floor, then re-fuse.
#
#   ./reconstruction_GT/refine_height.sh room07
#   ./reconstruction_GT/refine_height.sh room07 --no-refuse
#
# The workflow this belongs to:
#   1. measure the lens height with a tape and type it at capture time
#   2. fuse -- the floor check says where that assumption put the floor
#   3. run this: camera_height := camera_height - floor_z, so the floor lands
#      at z = 0, and re-fuse
#
# Step 1 stays a real measurement and is kept in meta.json as camera_height_tape;
# a tape is measured to some point on a table-tripod-camera stack and has been
# out by up to 14 cm, while the floor fit uses tens of thousands of points from
# a sensor verified against a tape to 1 cm. Both are recorded, with which one is
# in force, so no number in meta.json is silently a guess.
#
# It refuses when the floor check cannot support the estimate: no floor seen, a
# tilted plane (so the surface is furniture, not floor), or too few points.
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"
need_room "${1:-}"
ROOM="$1"; shift || true
DIR="$(capdir "$ROOM")"
REFUSE=1
[ "${1:-}" = "--no-refuse" ] && REFUSE=0

REPO="$REPO" "$PY" - "$DIR" <<'PY'
import json, os, sys, time
cap = sys.argv[1]
meta_path = os.path.join(cap, 'meta.json')
rep_path = os.path.join(cap, 'scan', 'fuse_report.json')
for p in (meta_path, rep_path):
    if not os.path.exists(p):
        raise SystemExit('missing %s -- capture and fuse first' % p)
meta = json.load(open(meta_path))
rep = json.load(open(rep_path))
fl = rep['checks'].get('floor', {})
if fl.get('status') == 'skipped':
    raise SystemExit('the floor check was skipped (%s): nothing to estimate from.\n'
                     'Sweep with the floor in view, or use measure_floor.sh.'
                     % fl.get('reason'))
tilt, verts = fl.get('surface_tilt_deg', 99), fl.get('vertices', 0)
if tilt > 3.0:
    raise SystemExit('that surface is %.1f deg off level -- furniture, not floor. '
                     'Refusing.' % tilt)
if verts < 2000:
    raise SystemExit('only %d points in the floor plane -- too few to trust. '
                     'Refusing.' % verts)

old = float(meta['camera_height'])
new = round(old - float(fl['lowest_up_surface_z_m']), 4)
print('  floor sits at z = %+.4f with camera_height = %.4f' % (fl['lowest_up_surface_z_m'], old))
print('  plane: %d points, %.2f deg off level' % (verts, tilt))
print('  camera_height %.4f -> %.4f  (%+.1f cm)' % (old, new, 100 * (new - old)))
if abs(new - old) < 0.01:
    print('\n  within 1 cm already -- leaving meta.json alone')
    sys.exit(3)
if not 0.2 <= new <= 3.0:
    raise SystemExit('%.3f m is not a plausible camera height -- refusing' % new)

meta.setdefault('camera_height_tape', old)      # keep the original measurement
# A height change moves the world vertically, so an annotation made in the old
# one no longer sits on the room. Carry it across, same as refine_tilt.sh.
sol_path = os.path.join(cap, 'gt', 'solids.json')
if os.path.exists(sol_path) and abs(new - float(meta['camera_height'])) > 1e-6:
    import numpy as np
    sys.path.insert(0, os.environ.get('REPO', '.'))
    from inference.frame_loader import live_cam_pose
    from reconstruction_GT.voxelize_gt import transform_solids
    yaw0 = float(meta.get('yaw', 0.0))
    up0 = meta.get('up_camera')
    A = (live_cam_pose(new, yaw0, up0).astype(np.float64)
         @ np.linalg.inv(live_cam_pose(float(meta['camera_height']), yaw0, up0)
                         .astype(np.float64)))
    spec = json.load(open(sol_path))
    backup = sol_path + '.preheight'
    if not os.path.exists(backup):
        json.dump(spec, open(backup, 'w'), indent=2)
    moved = transform_solids(spec, A)
    moved['_moved'] = ('shifted %+.4f m on %s when camera_height was refined; '
                       'the original is beside this file as '
                       'solids.json.preheight'
                       % (new - float(meta['camera_height']),
                          time.strftime('%Y-%m-%d')))
    json.dump(moved, open(sol_path, 'w'), indent=2)
    print('  moved %d solids by %+.4f m to follow the new height'
          % (len(moved.get('solids', [])), new - float(meta['camera_height'])))

meta['camera_height'] = new
meta['camera_height_source'] = (
    'floor plane fit, %d points, %.2f deg, from scan/fuse_report.json on %s; '
    'tape reading kept as camera_height_tape'
    % (verts, tilt, time.strftime('%Y-%m-%d')))
json.dump(meta, open(meta_path, 'w'), indent=2)
print('\n  meta.json updated (tape value kept as camera_height_tape)')
PY
RC=$?
[ "$RC" -eq 3 ] && exit 0
[ "$RC" -ne 0 ] && exit "$RC"

if [ "$REFUSE" -eq 1 ]; then
    say "re-fusing with the new height"
    "$(dirname "${BASH_SOURCE[0]}")/5_fuse_scan.sh" "$ROOM"
else
    warn "not re-fusing (--no-refuse): the mesh still uses the OLD height"
fi
