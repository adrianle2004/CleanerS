#!/usr/bin/env bash
# Open an Occ-ScanNet frame's CleanerS prediction and its GT in two
# display_overlay.py windows at the same time.
#
#   inference/show_scannet.sh scene0702_00              # the scene's best frame
#   inference/show_scannet.sh scene0702_00 worst
#   inference/show_scannet.sh scene0702_00 00004        # any frame
#   inference/show_scannet.sh scene0702_00 best --show voxels   # extra flags go to both windows
#   PRED_DIR=outputs/scannet inference/show_scannet.sh scene0702_00   # a different run
#   NO_RGB=1 inference/show_scannet.sh scene0702_00     # skip the colour image
#   MASK=none inference/show_scannet.sh scene0702_00    # the full prediction, not label_weight-masked
#
# Also opens the colour image in eog: <frame>_rgb.png, the 1296x968 photo
# registered onto the 640x480 depth camera -- exactly what the model saw.
# The original photo is data/Scannet/posed_images/<scene>/<frame>.jpg.
# Plys come from outputs/<run>/ply/<scene>/ (export_scannet_ply.py); a frame not
# exported yet is written first. Masked like NYU's (outside label_weight ->
# empty). --frame-grid makes the two windows open at exactly the same view.
set -euo pipefail

if [[ $# -lt 1 ]]; then
    sed -n '2,18p' "$0" | sed 's/^# \{0,1\}//'
    exit 1
fi

cd "$(dirname "$0")/.."                         # repo root
PY=${PYTHON:-python}
PRED_DIR=${PRED_DIR:-outputs/scannet}
MASK=${MASK:-label_weight}
SCENE=$1; shift
PICK=best
FRAME=
if [[ $# -gt 0 && $1 != -* ]]; then
    case $1 in
        best|worst) PICK=$1 ;;
        *) FRAME=$(printf '%05d' "$((10#$1))") ;;
    esac
    shift
fi

args=("$SCENE")
[[ -n $FRAME ]] && args+=("$FRAME") || args+=(--pick "$PICK")
out=$("$PY" inference/show_scannet.py "${args[@]}" --pred_dir "$PRED_DIR" --mask "$MASK" --export_only | grep -v '^Saved-->')
echo "$out" | grep -v '_ply='
PRED=$(echo "$out" | sed -n 's/^pred_ply=//p')
GT=$(echo "$out" | sed -n 's/^gt_ply=//p')
RGB=${PRED%_pred*.ply}_rgb.png

pids=()
for ply in "$PRED" "$GT"; do
    if [[ -n $ply && -f $ply ]]; then
        "$PY" inference/display_overlay.py "$ply" --frame-grid "$@" &
        pids+=($!)
    fi
done
[[ ${#pids[@]} -gt 0 ]] || { echo "nothing to display"; exit 1; }

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
