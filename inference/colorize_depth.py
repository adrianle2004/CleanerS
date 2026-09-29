"""
inference/colorize_depth.py

Turn 16-bit depth PNGs into human-readable colour images, matching what the
RealSense Viewer draws.

    python -m inference.colorize_depth captures/room01/depth \
        --out captures/room01/depth_vis

Depth PNGs are unreadable by eye: a 2.3 m wall and a 2.9 m wall differ by 600
out of 65535, so the file looks uniformly black. This maps depth to colour and
draws a scale bar, so a glance tells you whether the scene was framed sensibly
and where the sensor returned nothing.

REALSENSE PARITY
The default (`--cmap rs-jet --scaling equalize --invalid black`) is the Viewer's
own "Dynamic" preset, reimplemented from librealsense `src/proc/colorizer.{h,cpp}`
-- the ten `rs-*` ramps, the cumulative-histogram equalisation, and the float32
truncation in the LUT lookup. It is a port, not an approximation: every one of
the 10 schemes x {equalize, linear} was checked pixel-for-pixel against
`pyrealsense2`'s own `rs.colorizer()` on three captures and matched 100%.
`--preset {dynamic,fixed,near,far}` reproduces RS2_OPTION_VISUAL_PRESET.

Note that RealSense's "Jet" is NOT OpenCV's COLORMAP_JET -- it is a five-stop
ramp ending in dark maroon (50,0,0), which is why far surfaces go near-black in
the Viewer and bright red under `--cmap jet`. Both are available.

HISTOGRAM EQUALISATION IS PER-FRAME AND NON-LINEAR
It spends colour where the pixels are, which is why the Viewer looks so vivid,
but a colour means a different depth in every frame, and equal colour steps are
not equal metre steps. The scale bar is drawn through the inverse CDF so its
labels stay truthful, but they will be unevenly spaced. Use `--scaling linear`
when you need to compare frames, or `--fixed` to share one ramp across a folder.

INVALID PIXELS (depth == 0) ARE DRAWN IN A FLAT COLOUR, not as "0 metres".
They are missing measurements, not near ones, and colouring them as the near
end of the ramp would paint every hole as a surface against the lens -- which
is the same mistake the uint16 wraparound makes in capture.py. The default is
black to match the Viewer; `--invalid magenta` is the better choice when you are
hunting holes, because no ramp here produces magenta but several go near-black.

Reads both encodings:
    --encoding mm    plain uint16 millimetres, what capture.py writes (default)
    --encoding nyu   NYU's bit-rotated PNGs, ((d << 13) | (d >> 3)) / 1000
Using the wrong one is silent -- the image just looks like noise -- so the
value range is printed for every file as a sanity check.
"""

import argparse
import glob
import logging
import os
import sys

import cv2
import numpy as np

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_THIS_DIR)
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

try:
    from .utils import ensure_dir, setup_logging
except ImportError:
    from utils import ensure_dir, setup_logging

log = logging.getLogger(__name__)

# Depth PNGs written by capture.py are always millimetres: camera.py has already
# multiplied by the device's own depth_scale, so this is not the device setting.
DEPTH_UNITS = 0.001

# --------------------------------------------------------------------------
# RealSense colour maps
#
# Control points copied verbatim from librealsense src/proc/colorizer.cpp, in
# source order, so RS_MAP_ORDER[i] is RS2_OPTION_COLOR_SCHEME == i. RGB, and
# evenly spaced over [0,1] -- color_map's vector constructor assigns key
# i/(N-1) to entry i.
# --------------------------------------------------------------------------
RS_COLOR_MAPS = {
    'rs-jet': [(0, 0, 255), (0, 255, 255), (255, 255, 0), (255, 0, 0),
               (50, 0, 0)],
    'rs-classic': [(30, 77, 203), (25, 60, 192), (45, 117, 220),
                   (204, 108, 191), (196, 57, 178), (198, 33, 24)],
    'rs-white-to-black': [(255, 255, 255), (0, 0, 0)],
    'rs-black-to-white': [(0, 0, 0), (255, 255, 255)],
    'rs-bio': [(0, 0, 204), (204, 230, 255), (255, 255, 153), (170, 255, 128),
               (0, 153, 0), (230, 242, 255)],
    'rs-cold': [(230, 247, 255), (0, 92, 230), (0, 179, 179), (0, 51, 153),
                (0, 5, 15)],
    'rs-warm': [(255, 255, 230), (255, 204, 0), (255, 136, 77), (255, 51, 0),
                (128, 0, 0), (10, 0, 0)],
    'rs-quantized': [(255, 255, 255), (0, 0, 0)],
    'rs-pattern': [(255, 255, 255), (0, 0, 0)] * 25,
    'rs-hue': [(255, 0, 0), (255, 255, 0), (0, 255, 0), (0, 255, 255),
               (0, 0, 255), (255, 0, 255), (255, 0, 0)],
}
RS_MAP_ORDER = ['rs-jet', 'rs-classic', 'rs-white-to-black', 'rs-black-to-white',
                'rs-bio', 'rs-cold', 'rs-warm', 'rs-quantized', 'rs-pattern',
                'rs-hue']
# color_map's second ctor argument; 4000 everywhere but the posterised ramp
RS_STEPS = {'rs-quantized': 6}
RS_DEFAULT_STEPS = 4000

CMAPS = {
    'turbo': getattr(cv2, 'COLORMAP_TURBO', cv2.COLORMAP_JET),
    'jet': cv2.COLORMAP_JET,
    'viridis': getattr(cv2, 'COLORMAP_VIRIDIS', cv2.COLORMAP_PARULA),
    'magma': getattr(cv2, 'COLORMAP_MAGMA', cv2.COLORMAP_HOT),
    'inferno': getattr(cv2, 'COLORMAP_INFERNO', cv2.COLORMAP_HOT),
    'plasma': getattr(cv2, 'COLORMAP_PLASMA', cv2.COLORMAP_HOT),
    'bone': cv2.COLORMAP_BONE,
}
ALL_CMAPS = sorted(RS_COLOR_MAPS) + sorted(CMAPS)

INVALID_COLORS = {'black': (0, 0, 0), 'magenta': (255, 0, 255),
                  'white': (255, 255, 255), 'grey': (40, 40, 40)}

# RS2_OPTION_VISUAL_PRESET: (cmap, scaling, min_m, max_m)
RS_PRESETS = {
    'dynamic': ('rs-jet', 'equalize', None, None),
    'fixed': ('rs-jet', 'linear', 0.0, 6.0),
    'near': ('rs-classic', 'linear', 0.3, 1.5),
    'far': ('rs-jet', 'linear', 1.0, 16.0),
}

_CACHE = {}


def rs_color_cache(name):
    """(steps+1, 3) float32 RGB -- librealsense color_map::initialize().

    Reproduced in float32 rather than numpy's default float64 on purpose. The
    lookup truncates to uint8, so a value landing on x.99999 in one precision
    and x.00001 in the other differs by a whole level; doing the lerp in float64
    put `rs-quantized` off by one across entire plateaus.
    """
    if name in _CACHE:
        return _CACHE[name]
    f32 = np.float32
    pts = np.asarray(RS_COLOR_MAPS[name], f32)
    steps = RS_STEPS.get(name, RS_DEFAULT_STEPS)
    keys = (np.arange(len(pts), dtype=f32) / f32(len(pts) - 1)).astype(f32)
    x = (np.arange(steps + 1, dtype=f32) / f32(steps)).astype(f32)

    # calc(): segment is [last key < x, first key > x], then lerp
    j = np.clip(np.searchsorted(keys, x, side='left') - 1, 0, len(pts) - 2)
    t = ((x - keys[j]) / (keys[j + 1] - keys[j])).astype(f32)
    out = (pts[j + 1] * t[:, None] + pts[j] * (f32(1.0) - t)[:, None]).astype(f32)
    # ...but an exact key match short-circuits the interpolation
    at = np.searchsorted(keys, x, side='left')
    hit = (at < len(keys)) & (keys[np.minimum(at, len(keys) - 1)] == x)
    out[hit] = pts[at[hit]]

    _CACHE[name] = (out, steps)
    return _CACHE[name]


def rs_apply(f, name):
    """f in [0,1] -> BGR uint8, via color_map::get()."""
    cache, steps = rs_color_cache(name)
    t = np.clip(np.asarray(f, np.float32), 0.0, 1.0)
    idx = (t * np.float32(steps)).astype(np.int32)
    return cache[idx].astype(np.uint8)[..., ::-1]


def rs_histogram(raw):
    """Cumulative histogram over raw uint16 depth -- colorizer::update_histogram.

    Entry 0 is left as the raw count of invalid pixels and the running sum
    starts at index 2, so holes never shift the ramp. Invalid pixels are painted
    flat anyway, so cum[0] is never read.
    """
    hist = np.bincount(np.asarray(raw, np.uint16).ravel(),
                       minlength=65536).astype(np.int64)
    cum = hist.copy()
    cum[1:] = np.cumsum(hist[1:])
    return cum


def colorize(raw, cmap, invalid_bgr, scaling='equalize', lo=0.0, hi=6.0,
             cum=None):
    """Raw uint16 millimetre depth -> BGR uint8. 0 is painted invalid_bgr.

    `cum` lets a whole sequence share one equalisation curve; pass None to
    equalise each frame against itself, as the Viewer does.
    """
    raw = np.asarray(raw, np.uint16)
    if scaling == 'equalize':
        if cum is None:
            cum = rs_histogram(raw)
        total = np.float32(cum[65535])
        f = (cum[raw].astype(np.float32) / total if total > 0
             else np.zeros(raw.shape, np.float32))
    else:
        f = (np.zeros(raw.shape, np.float32) if lo >= hi else
             ((raw.astype(np.float32) * np.float32(DEPTH_UNITS) - np.float32(lo))
              / np.float32(hi - lo)))

    if cmap in RS_COLOR_MAPS:
        out = rs_apply(f, cmap)
    else:
        # 1..255, leaving 0 free so a clipped near surface never collides with
        # the invalid colour after the LUT is applied
        idx = (1 + np.clip(f, 0.0, 1.0) * 254).astype(np.uint8)
        out = cv2.applyColorMap(idx, CMAPS[cmap])
    out = np.ascontiguousarray(out)
    out[raw == 0] = invalid_bgr
    return out


def depth_at(f, scaling, lo, hi, cum):
    """Metres shown at fraction f along the ramp -- inverse of the scaling."""
    if scaling != 'equalize':
        return lo + f * (hi - lo)
    total = cum[65535]
    if total <= 0:
        return 0.0
    # cum[0] is a raw count, not part of the running sum, so it can exceed
    # cum[1] and break the search -- slice it off. side='right' is what makes
    # f=0 report the nearest measured surface rather than 1 mm.
    d = 1 + int(np.searchsorted(cum[1:], f * total, side='right'))
    # nothing is strictly greater than `total`, so f=1 would otherwise run off
    # the end of the histogram and report 65.5 m instead of the far surface
    far = 1 + int(np.searchsorted(cum[1:], total, side='left'))
    return min(d, far) * DEPTH_UNITS


def draw_bar(img, cmap, height, invalid_bgr, scaling, lo, hi, cum, n_ticks=5):
    """Append a horizontal colour ramp with metre labels."""
    h, w = img.shape[:2]
    t = np.linspace(0.0, 1.0, w)
    if cmap in RS_COLOR_MAPS:
        row = rs_apply(t, cmap)[None, :, :]
    else:
        ramp = (1 + t * 254).astype(np.uint8)[None, :]
        row = cv2.applyColorMap(ramp, CMAPS[cmap])
    bar = np.repeat(row, height // 2, axis=0)
    strip = np.full((height, w, 3), 24, np.uint8)
    strip[:height // 2] = bar
    for i in range(n_ticks):
        f = i / (n_ticks - 1.0)
        x = int(round(f * (w - 1)))
        label = '%.2fm' % depth_at(f, scaling, lo, hi, cum)
        (tw, _), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.42, 1)
        x = min(max(x - tw // 2, 2), w - tw - 2)
        cv2.putText(strip, label, (x, height - 6), cv2.FONT_HERSHEY_SIMPLEX,
                    0.42, (235, 235, 235), 1, cv2.LINE_AA)
    # swatch + label so the hole colour is self-documenting
    cv2.rectangle(strip, (w - 78, 2), (w - 62, height // 2 - 2), invalid_bgr, -1)
    cv2.putText(strip, 'no data', (w - 58, height // 2 - 4),
                cv2.FONT_HERSHEY_SIMPLEX, 0.38, (235, 235, 235), 1, cv2.LINE_AA)
    return np.vstack([img, strip])


def load_depth_mm(path, encoding):
    """-> (H,W) uint16 millimetres, 0 == invalid."""
    raw = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if raw is None:
        raise FileNotFoundError('cv2.imread returned None for %r' % path)
    if raw.ndim != 2:
        raise ValueError('%r has %d channels; expected a single-channel depth '
                         'image. Already colourised?' % (path, raw.shape[2]))
    if raw.dtype != np.uint16:
        raise ValueError('%r is %s, expected uint16.' % (path, raw.dtype))
    if encoding == 'nyu':
        raw = ((raw << 13) | (raw >> 3)).astype(np.uint16)
    return raw


def parse_args():
    p = argparse.ArgumentParser(
        description='Colourise 16-bit depth PNGs the way the RealSense Viewer '
                    'does.')
    p.add_argument('src', help='a depth .png, or a folder of them')
    p.add_argument('--out', default=None,
                   help='output file or folder (default: <src>_vis alongside)')
    p.add_argument('--encoding', choices=['mm', 'nyu'], default='mm',
                   help="'mm' = plain uint16 millimetres (capture.py); "
                        "'nyu' = bit-rotated NYU PNGs")
    p.add_argument('--preset', choices=sorted(RS_PRESETS), default=None,
                   help='RealSense Viewer visual preset; sets --cmap, '
                        '--scaling, --min and --max together, overriding them')
    p.add_argument('--cmap', choices=ALL_CMAPS, default='rs-jet',
                   help="rs-* are the Viewer's own ramps (default rs-jet); the "
                        'rest are OpenCV LUTs')
    p.add_argument('--scaling', choices=['equalize', 'linear'],
                   default='equalize',
                   help="'equalize' = the Viewer's per-frame cumulative "
                        'histogram, vivid but frame-dependent; '
                        "'linear' = plain metres, comparable across frames")
    p.add_argument('--min', type=float, default=None,
                   help='linear only: metres at the near end '
                        '(default: 1st percentile)')
    p.add_argument('--max', type=float, default=None,
                   help='linear only: metres at the far end '
                        '(default: 99th percentile)')
    p.add_argument('--fixed', action='store_true',
                   help='use one ramp across every file, so a sequence is '
                        'comparable frame to frame; otherwise each is scaled '
                        'to its own content and colours mean different things '
                        'in different frames')
    p.add_argument('--invalid', choices=sorted(INVALID_COLORS), default='black',
                   help='colour for pixels the sensor returned nothing for; '
                        "black matches the Viewer, magenta is easier to spot")
    p.add_argument('--no_bar', action='store_true', help='omit the scale bar')
    p.add_argument('--bar_h', type=int, default=46)
    args = p.parse_args()
    if args.preset:
        args.cmap, args.scaling, args.min, args.max = RS_PRESETS[args.preset]
    return args


def main():
    setup_logging()
    args = parse_args()

    if os.path.isdir(args.src):
        paths = sorted(glob.glob(os.path.join(args.src, '*.png')))
        if not paths:
            raise FileNotFoundError('no .png files under %r' % args.src)
        out_dir = args.out or (os.path.normpath(args.src) + '_vis')
        ensure_dir(out_dir)
    else:
        paths = [args.src]
        out_dir = None

    equalize = args.scaling == 'equalize'
    inv_bgr = INVALID_COLORS[args.invalid]

    # one shared ramp keeps colours comparable across a sequence
    shared_cum, shared_range = None, None
    if args.fixed or (not equalize and args.min is not None
                      and args.max is not None):
        if equalize:
            shared_cum = np.zeros(65536, np.int64)
            for p in paths:
                shared_cum += np.bincount(load_depth_mm(p, args.encoding).ravel(),
                                          minlength=65536).astype(np.int64)
            shared_cum[1:] = np.cumsum(shared_cum[1:])
            log.info('shared equalisation over %d frame(s)', len(paths))
        else:
            vals = None
            if args.min is None or args.max is None:
                vals = np.concatenate([
                    (d[d > 0].ravel() * DEPTH_UNITS) for d in
                    (load_depth_mm(p, args.encoding).astype(np.float32)
                     for p in paths)])
            lo = args.min if args.min is not None else float(np.percentile(vals, 1))
            hi = args.max if args.max is not None else float(np.percentile(vals, 99))
            shared_range = (lo, hi)
            log.info('shared range %.2f-%.2f m', lo, hi)

    for p in paths:
        raw = load_depth_mm(p, args.encoding)
        valid = raw > 0
        if not valid.any():
            log.warning('%s: no valid depth, skipping', p)
            continue
        v = raw[valid].astype(np.float32) * DEPTH_UNITS

        cum, lo, hi = None, 0.0, 6.0
        if equalize:
            cum = shared_cum if shared_cum is not None else rs_histogram(raw)
        elif shared_range is not None:
            lo, hi = shared_range
        else:
            lo = args.min if args.min is not None else float(np.percentile(v, 1))
            hi = args.max if args.max is not None else float(np.percentile(v, 99))
        if not equalize and hi <= lo:
            hi = lo + 1e-3

        img = colorize(raw, args.cmap, inv_bgr, args.scaling, lo, hi, cum)
        if not args.no_bar:
            img = draw_bar(img, args.cmap, args.bar_h, inv_bgr, args.scaling,
                           lo, hi, cum)

        if out_dir is None:
            dst = args.out or (os.path.splitext(p)[0] + '_color.png')
        else:
            dst = os.path.join(out_dir, os.path.basename(p))
        # --out pointing at the source (the file, or its own folder) would
        # replace measured depth with a picture of it, and the capture has no
        # other copy at this resolution. Refuse rather than destroy data.
        if os.path.exists(dst) and os.path.samefile(dst, p):
            raise SystemExit('refusing to overwrite the source depth %r with '
                             'its colourised image; pass a different --out'
                             % p)
        cv2.imwrite(dst, img)

        ramp = ('equalised %.2f-%.2f m (p50 %.2f)'
                % (depth_at(0.0, 'equalize', lo, hi, cum),
                   depth_at(1.0, 'equalize', lo, hi, cum),
                   depth_at(0.5, 'equalize', lo, hi, cum))
                if equalize else 'ramp %.2f-%.2f m' % (lo, hi))
        log.info('%s -> %s', os.path.basename(p), dst)
        log.info('    valid %.1f%%  actual %.2f-%.2f m (median %.2f)  %s',
                 100 * valid.mean(), v.min(), v.max(), float(np.median(v)), ramp)


if __name__ == '__main__':
    main()
