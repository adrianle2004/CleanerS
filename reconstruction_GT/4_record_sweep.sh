#!/usr/bin/env bash
# Step 4: record the room sweep -- scaffolding to place annotation solids on.
#
#   ./reconstruction_GT/4_record_sweep.sh room06 [--seconds 60]
#
# Run straight after step 3 WITHOUT having moved the camera. Extra arguments go
# to record_scan.py.
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"
need_room "${1:-}"
ROOM="$1"; shift || true
DIR="$(capdir "$ROOM")"

[ -f "$DIR/meta.json" ] || warn "no $DIR/meta.json -- run 3_capture_still.sh first,
  or the mesh comes out in camera coordinates instead of the grid's world frame."
if [ -f "$DIR/scan/scan.bag" ]; then die "$DIR/scan/scan.bag exists -- move it away first."; fi

FREE_GB=$(df -BG --output=avail "$REPO" | tail -1 | tr -dc '0-9')
if [ "${FREE_GB:-0}" -lt 5 ]; then warn "only ${FREE_GB} GB free; a sweep is ~30 MB/s (~1.8 GB per minute)"; fi

cat <<'TXT'
The sweep:
  - HOLD STILL through the countdown, THEN lift the camera
  - move slowly, 30-60 s is plenty
  - stay within ~3 m of surfaces (D455 error is ~2% at 4 m)
  - see each object from several sides
  - point at the FLOOR at some point (fuse_scan's floor check needs it)
  - keep furniture or a corner in view; a blank wall lets tracking slide
  - q in the window stops it
TXT

cd "$REPO"
say "recording -> $DIR/scan/scan.bag"
"$PY" -m reconstruction_GT.record_scan --preview --out_dir "$DIR" "$@"

say "next:  ./reconstruction_GT/5_fuse_scan.sh $ROOM"
