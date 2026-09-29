#!/usr/bin/env bash
# Step 2: throwaway 15 s run, to prove the rig works before a real room.
#
#   ./reconstruction_GT/2_rig_test.sh
#
# record_scan.py has never run against a camera, only against a sample file, so
# spend one short recording finding out here rather than halfway through a room
# you cared about. Nothing this writes is kept: the height is a made-up 1.0 m
# and the folder is deleted at the end.
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"

DIR="$(capdir _rigtest)"
if [ -e "$DIR" ]; then warn "removing previous $DIR"; rm -rf "$DIR"; fi

cd "$REPO"
say "still frame -- point at your desk, press SPACE (height is a throwaway 1.0)"
"$PY" -m inference.capture --preview --camera_height 1.0 --out_dir "$DIR"

say "15 s sweep -- hold still for the countdown, then move slowly around the desk"
"$PY" -m reconstruction_GT.record_scan --preview --seconds 15 --out_dir "$DIR"

say "fuse"
"$PY" -m reconstruction_GT.fuse_scan "$DIR" || warn "fuse_scan failed -- that is what this test is for"

say "report"
cat "$DIR/scan/fuse_report.json" 2>/dev/null || warn "no fuse_report.json"

cat <<TXT

What should have happened:
  - both previews opened, HOLD STILL counted down, then RECORDING
  - fuse_scan fused most frames and dropped few
  - $DIR/scan/room.ply looks like your desk (open it in CloudCompare)

Send me that fuse_report.json -- the IMU tilt check is running for the first
time here and its sign convention is unverified.

Then throw this away:   rm -rf $DIR
and record a real room:  ./reconstruction_GT/3_capture_still.sh room06
TXT
