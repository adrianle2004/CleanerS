#!/usr/bin/env bash
# Step 3: the still evaluation frame -- the one the model is scored on.
#
#   ./reconstruction_GT/3_capture_still.sh room06
#
# LEVEL THE TRIPOD and MEASURE the lens height with a tape first. The grid is
# floor-anchored and has no pitch/roll term, so both are the difference between
# ground truth that lines up and ground truth that silently does not.
# Extra arguments are passed through to capture.py.
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"
need_room "${1:-}"
ROOM="$1"; shift || true
DIR="$(capdir "$ROOM")"

if [ -f "$DIR/meta.json" ]; then
    die "$DIR already has a still frame (meta.json).
Use a new room name, or move that folder away. Overwriting it would orphan any
ground truth already built against that frame."
fi

cat <<'TXT'
Before you press SPACE:
  - tripod levelled
  - lens height measured with a tape, and typed into the preview field
  - nothing closer than ~0.6 m (min-Z dropout)
  - check the DEPTH pane, not the RGB one: dark, shiny and translucent
    surfaces look fine in colour and are missing in depth
TXT

cd "$REPO"
say "still capture -> $DIR"
"$PY" -m inference.capture --preview --out_dir "$DIR" "$@"

say "DO NOT MOVE THE CAMERA"
cat <<TXT
The sweep has to start from this exact pose -- that is what puts the fused mesh
in the same world frame as the voxel grid.

Next, without touching the camera:
  ./reconstruction_GT/4_record_sweep.sh $ROOM
TXT
