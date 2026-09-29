#!/usr/bin/env bash
# Step 5: fuse the sweep into scan/room.ply, then report the checks.
#
#   ./reconstruction_GT/5_fuse_scan.sh room06 [--voxel 0.01]
#
# Exits 2 if any check warns, because a warning means the mesh does not line up
# with the grid and annotating against it would waste the session. Extra
# arguments go to fuse_scan.py.
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"
need_room "${1:-}"
ROOM="$1"; shift || true
DIR="$(capdir "$ROOM")"
[ -f "$DIR/scan/scan.bag" ] || die "no $DIR/scan/scan.bag -- run 4_record_sweep.sh first"

cd "$REPO"
say "fusing $DIR"
"$PY" -m reconstruction_GT.fuse_scan "$DIR" "$@"

say "checks"
"$PY" - "$DIR/scan/fuse_report.json" <<'PY'
import json, sys
r = json.load(open(sys.argv[1]))
print('%d of %d frames fused, %d dropped (%s)' % (
    r['frames_fused'], r['frames_read'], r['frames_dropped'], r['stop_reason']))
print('mesh: %d vertices, bounds %s .. %s' % (
    r['mesh']['vertices'], r['bounds_m']['min'], r['bounds_m']['max']))
warned = False
for name, c in r['checks'].items():
    print('  %-12s %-7s %s' % (name, c['status'],
          {k: v for k, v in c.items() if k != 'status'}))
    warned |= c['status'] == 'warn'
if warned:
    print('\nA warning means the mesh may not line up with the voxel grid:')
    print('  still_match -> the camera moved between step 3 and step 4:')
    print('                 redo both, the recording must start from the still pose')
    print('  imu_tilt    -> the camera was not level: re-level and redo step 3')
    print('  floor       -> if the floor was in view, camera_height is wrong:')
    print('                 re-measure and redo step 3. Never edit meta.json.')
sys.exit(2 if warned else 0)
PY

say "mesh ready: $DIR/scan/room.ply"
echo "Open it in CloudCompare and place solids (MAKING_GT.md step 5)."
echo "Note: the voxelizer that turns solids into GT files is not written yet."
