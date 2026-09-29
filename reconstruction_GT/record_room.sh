#!/usr/bin/env bash
# All of it, one room, one command: check -> still -> sweep -> fuse.
#
#   ./reconstruction_GT/record_room.sh room07
#   ./reconstruction_GT/record_room.sh room07 --seconds 60    # -> the sweep
#
# Runs steps 1, 3, 4 and 5 in order and stops at the first one that fails.
# Extra arguments go to the SWEEP step (4), which is the only one that usually
# needs them.
#
# It pauses before the still frame and before the sweep, because each is a
# physical act you have to be ready for -- levelling and measuring, then
# keeping your hands off the camera. With no terminal attached it runs straight
# through without pausing.
#
# Use a NEW room name each time: step 3 refuses to overwrite a capture that
# already has an evaluation frame, since ground truth may already be built
# against it.
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"
need_room "${1:-}"
ROOM="$1"; shift || true
HERE="$(dirname "${BASH_SOURCE[0]}")"

pause() {
    [ -t 0 ] || return 0
    printf '\n\033[1m%s\033[0m' "$1"
    read -r _ || true
}

say "room $ROOM -- step 1/4: camera"
"$HERE/1_check_camera.sh"

pause "Level the camera and measure the lens height with a tape.
A 7 degree roll puts a surface 3 m away 37 cm from where the grid draws it,
and the grid has no roll term -- so this is the difference between ground
truth that lines up and ground truth that silently does not.
Press Enter when the camera is level and you know the height: "

say "room $ROOM -- step 2/4: still evaluation frame"
"$HERE/3_capture_still.sh" "$ROOM"

pause "DO NOT MOVE THE CAMERA. The sweep must start from this exact pose.
Stand where you will not be in shot, and plan to stay behind the camera.
Press Enter to start recording: "

say "room $ROOM -- step 3/4: sweep"
"$HERE/4_record_sweep.sh" "$ROOM" "$@"

say "room $ROOM -- step 4/4: fuse"
set +e
"$HERE/5_fuse_scan.sh" "$ROOM"
FUSE=$?
set -e

DIR="$(capdir "$ROOM")"
if [ "$FUSE" -eq 0 ]; then
    say "done -- $DIR/scan/room.ply is ready to annotate against"
else
    warn "fusion finished with warnings (exit $FUSE) -- read the checks above
before annotating: a warning means the mesh may not line up with the voxel
grid, and re-recording is cheaper than annotating twice."
fi
echo "report: $DIR/scan/fuse_report.json"
exit "$FUSE"
