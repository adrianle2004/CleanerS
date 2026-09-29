# Shared by the numbered step scripts. Not a step -- source it, do not run it.
#
#   REPO      repo root, from this file's location
#   PY        an interpreter that can import pyrealsense2 / open3d
#   capdir X  -> "X" if it looks like a path, else "$REPO/captures/X"

set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# Order: an explicit CLEANERS_PY, then whatever is already active (so a
# `conda activate CleanerS` shell just works), then the known env.
if [ -n "${CLEANERS_PY:-}" ]; then
    PY="$CLEANERS_PY"
elif command -v python >/dev/null 2>&1 && python -c 'import pyrealsense2' 2>/dev/null; then
    PY="$(command -v python)"
elif [ -x "$HOME/miniconda3/envs/CleanerS/bin/python" ]; then
    PY="$HOME/miniconda3/envs/CleanerS/bin/python"
else
    echo "ERROR: no interpreter with pyrealsense2." >&2
    echo "  conda activate CleanerS   (or set CLEANERS_PY=/path/to/python)" >&2
    exit 1
fi

say()  { printf '\n\033[1m== %s\033[0m\n' "$*"; }
warn() { printf '\033[33m!! %s\033[0m\n' "$*" >&2; }
die()  { printf '\033[31mERROR: %s\033[0m\n' "$*" >&2; exit 1; }

# A bare name means a room under captures/; anything with a slash is a path.
capdir() {
    case "$1" in
        */*|.|..) printf '%s' "$1" ;;
        *)        printf '%s' "$REPO/captures/$1" ;;
    esac
}

need_room() {
    [ $# -ge 1 ] && [ -n "${1:-}" ] || die "usage: $(basename "$0") <room>   e.g. $(basename "$0") room06"
}
