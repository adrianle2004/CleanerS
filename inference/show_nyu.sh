#!/usr/bin/env bash
# Open an NYU frame's CleanerS prediction and NYU's own GT (data/NYU/Label, not
# ISO's) in two display_overlay.py windows at the same time.
#
#   inference/show_nyu.sh 0001
#   inference/show_nyu.sh NYU0015 --show voxels     # extra flags go to both windows
#   PYTHON=python3 inference/show_nyu.sh 0001        # pick the interpreter
#   NO_RGB=1 inference/show_nyu.sh 0001              # skip the colour image
#   MASK=none inference/show_nyu.sh 0001             # the full prediction, not label_weight-masked
#
# Also opens the frame's colour image (data/NYU/RGB/NYU<id>_colors.png) in eog.
# Both plys are masked like test_NYU.py:visualize_3d_predict (outside
# label_weight -> empty) and written once to custom_visual_pred/CleanerS/compare/.
# --frame-grid makes the two windows open at exactly the same view.
set -euo pipefail

if [[ $# -lt 1 ]]; then
    sed -n '2,14p' "$0" | sed 's/^# \{0,1\}//'
    exit 1
fi

cd "$(dirname "$0")/.."                         # repo root
PY=${PYTHON:-python}
ID=${1#NYU}; ID=${ID%%_*}; shift
ID=$(printf '%04d' "$((10#$ID))")
MASK=${MASK:-label_weight}
DIR=custom_visual_pred/CleanerS/compare
if [[ $MASK == none ]]; then PRED=$DIR/NYU${ID}_pred_full.ply; else PRED=$DIR/NYU${ID}_pred.ply; fi
GT=$DIR/NYU${ID}_gt.ply
RGB=../data/NYU/RGB/NYU${ID}_colors.png

if [[ ! -f $PRED || ! -f $GT ]]; then
    "$PY" inference/show_nyu.py "$ID" --mask "$MASK" --export_only | grep -v '^Saved-->'
fi

pids=()
for ply in "$PRED" "$GT"; do
    if [[ -f $ply ]]; then
        "$PY" inference/display_overlay.py "$ply" --frame-grid "$@" &
        pids+=($!)
    else
        echo "skip: $ply (nothing inside label_weight)"
    fi
done

# the colour image too, unless NO_RGB=1
if [[ ${NO_RGB:-0} != 1 ]]; then
    if [[ -f $RGB ]]; then
        if command -v eog >/dev/null; then
            eog --new-instance "$RGB" 2>/dev/null &
            pids+=($!)
        else
            xdg-open "$RGB" >/dev/null 2>&1 &
        fi
    else
        echo "no colour image at $RGB"
    fi
fi
trap 'kill "${pids[@]}" 2>/dev/null' INT TERM
wait
