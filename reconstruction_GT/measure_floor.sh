#!/usr/bin/env bash
# Measure the lens height above the floor from ONE tilted frame.
#
#   ./reconstruction_GT/measure_floor.sh room07
#
# Aim the camera DOWN so the floor fills much of the view, at the rig's capture
# height. Uses the accelerometer, so the tilt does not matter. Saves the frame
# under <capture>/floor_check/ and writes nothing to meta.json.
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"
ARGS=()
[ -n "${1:-}" ] && { ARGS=(--capture "$(capdir "$1")"); shift || true; }
cd "$REPO"
exec "$PY" -m reconstruction_GT.measure_floor "${ARGS[@]}" "$@"
