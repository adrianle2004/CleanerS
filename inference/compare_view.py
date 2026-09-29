"""
inference/compare_view.py

Shared by show_nyu.py and show_scannet.py: open a prediction .ply and its GT
.ply in display_overlay.py side by side, or render both to one PNG.

Both windows get --frame-grid, so they reflect about the grid and frame the
whole 60x36x60 volume: the same voxel sits at the same place on screen in both,
whatever each file happens to contain.
"""

import os
import subprocess
import sys
import tempfile

import numpy as np

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
OVERLAY = os.path.join(_THIS_DIR, 'display_overlay.py')


def overlay_cmd(ply, extra):
    return [sys.executable, OVERLAY, ply, '--frame-grid'] + list(extra)


def open_windows(plys, extra=()):
    """One display_overlay window per ply, all at once; returns when all close."""
    procs = [subprocess.Popen(overlay_cmd(p, extra)) for p in plys]
    for p in procs:
        p.wait()


def screenshot_pair(plys, labels, out_png, extra=()):
    """Render each ply offscreen and put them side by side with a caption."""
    import cv2
    tiles = []
    with tempfile.TemporaryDirectory() as tmp:
        for i, (ply, label) in enumerate(zip(plys, labels)):
            png = os.path.join(tmp, '%d.png' % i)
            subprocess.run(overlay_cmd(ply, extra) + ['--screenshot', png], check=True,
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            img = cv2.imread(png)
            cv2.putText(img, label, (24, 56), cv2.FONT_HERSHEY_SIMPLEX, 1.6, (255, 255, 255), 3, cv2.LINE_AA)
            tiles.append(img)
    os.makedirs(os.path.dirname(os.path.abspath(out_png)), exist_ok=True)
    cv2.imwrite(out_png, np.hstack(tiles))
    print('wrote %s' % out_png)


def add_view_args(ap):
    ap.add_argument('--only', choices=['pred', 'gt'], default=None,
                    help='open just one of the two (default: both)')
    ap.add_argument('--screenshot', metavar='PNG',
                    help='render prediction and GT side by side into this PNG instead of opening windows')
