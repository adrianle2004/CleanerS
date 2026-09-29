"""
inference/capture.py

Grab N frames from a RealSense D4xx and write them to a folder that
run_inference.py can consume directly (`--live_dir`).

    python -m inference.capture --out_dir ./captures/scene01 \
        --camera_height 1.25 --filters

Add --preview to frame the shot first: a live window showing RGB beside the
colourised depth, with a CAPTURE button (or spacebar) that writes the frame
you are looking at. In preview mode the camera height is typed into the window
rather than passed on the command line -- you measure it while standing at the
camera, which is exactly when the window is in front of you, and CAPTURE stays
disabled until a plausible value is there. The depth pane matters more than the RGB one -- min-Z
dropout, a dark sofa and a translucent bottle are all invisible in colour and
obvious in depth, and every capture this repo has had to throw away failed for
one of those reasons.

Layout written:

    captures/scene01/
      depth/live_000000.png     640x480 uint16, MILLIMETRES  <- inference input
      rgb/live_000000.png       640x480 BGR
      native/live_000000_depth.png   sensor resolution, uint16 mm
      native/live_000000_rgb.png     sensor resolution, BGR
      meta.json

Why the depth PNG is plain millimetres and NOT NYU's encoding: NYU ships
bit-rotated Kinect PNGs that need ((d << 13) | (d >> 3)) before dividing by
1000 (frame_loader._load_depth). That rotation is an artefact of how the
original dataset was dumped, not a format anyone should reproduce. Anything
reading these files must use the plain path -- which is what --live_dir does.

meta.json carries the EFFECTIVE intrinsics, measured after match_nyu_fov did
its resize and crop, not the sensor's catalogue values. Feeding the network
NYU's fx/fy for a D455 frame misplaces every unprojected point by tens of
centimetres at room scale, so the sidecar is not optional bookkeeping -- it
is what makes the saved frame reconstructable.
"""

import os
import sys
import json
import time
import logging
import argparse

import cv2
import numpy as np

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_THIS_DIR)
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

try:
    from .camera import RealSenseSource
    from .colorize_depth import INVALID_COLORS, colorize
    from .utils import ensure_dir, setup_logging
except ImportError:
    from camera import RealSenseSource
    from colorize_depth import INVALID_COLORS, colorize
    from utils import ensure_dir, setup_logging

log = logging.getLogger(__name__)

DEPTH_MAX_MM = 65535        # uint16 ceiling

# Used only when --preview is off and no --camera_height was given. There is no
# defensible default for this number -- the grid is floor-anchored, so it shifts
# the whole scene -- hence the warning that goes with it.
DEFAULT_CAMERA_HEIGHT = 1.25


def parse_args():
    p = argparse.ArgumentParser(
        description='Capture depth+RGB frames from a RealSense into a folder.')
    p.add_argument('--out_dir', required=True,
                   help='folder to create; run_inference.py --live_dir reads it')
    p.add_argument('--frames', type=int, default=1,
                   help='how many frames to save (default 1)')
    p.add_argument('--interval', type=float, default=0.0,
                   help='seconds between saved frames, when --frames > 1')

    p.add_argument('--width', type=int, default=1280)
    p.add_argument('--height', type=int, default=720)
    p.add_argument('--fps', type=int, default=30)
    p.add_argument('--fov', choices=['nyu', 'native'], default='nyu',
                   help="'nyu' resizes+crops to 640x480 at NYU's intrinsics, "
                        "which is what the network was trained on")
    p.add_argument('--warmup', type=int, default=30,
                   help='frames discarded before capture, for auto-exposure')
    p.add_argument('--filters', action='store_true',
                   help='SDK spatial + temporal depth filters')
    p.add_argument('--hole_fill', action='store_true',
                   help='also fill depth holes -- INVENTS geometry the sensor '
                        'never saw, and the TSDF encoder will treat it as a '
                        'real observed surface. Off unless you mean it.')
    p.add_argument('--depth_max', type=float, default=10.0,
                   help='mark depth beyond this many metres invalid (0 = keep '
                        'everything). The disparity->depth conversion maps '
                        'near-zero disparity to tens of metres, so a filtered '
                        'frame carries a few hundred nonsense pixels; a D455 '
                        'is only trustworthy to about 6 m anyway.')
    p.add_argument('--no_undistort', action='store_true',
                   help='skip the pinhole resampling. The D455 colour stream '
                        'has real distortion and the encoder assumes a pinhole '
                        'camera, so leaving this on costs up to 3.3 cm of '
                        'geometric error at the frame corners.')
    p.add_argument('--no_native', action='store_true',
                   help='skip writing the pre-crop sensor frames')
    p.add_argument('--preview', action='store_true',
                   help='open a live RGB + colourised-depth window, take the '
                        'camera height typed into it, and wait for you to '
                        'press space (or click CAPTURE) before '
                        'saving each frame. Exits once --frames have been '
                        'saved, or on q. Falls back to capturing immediately '
                        'if there is no display.')

    # recorded into meta.json for from_live_camera's grid placement
    p.add_argument('--camera_height', type=float, default=None,
                   help='metres from floor to camera. The voxel grid is '
                        'floor-anchored, so this shifts the whole scene '
                        'vertically -- measure it, do not guess. Under '
                        '--preview this only pre-fills the on-screen field, '
                        'which is where the value is actually entered; without '
                        '--preview, omitting it records %.2f m and warns.'
                        % DEFAULT_CAMERA_HEIGHT)
    p.add_argument('--yaw', type=float, default=0.0,
                   help='radians about the vertical axis; only affects how the '
                        'volume aligns to walls')
    return p.parse_args()


def _to_uint16_mm(depth_m, where, depth_max=0.0, quiet=False):
    """metres float -> millimetres uint16, saturating rather than wrapping.

    `quiet` suppresses the log lines only. The preview calls this 30 times a
    second on frames it is not saving, and the messages would bury the ones
    that refer to a frame actually written to disk.
    """
    mm = np.round(np.nan_to_num(depth_m, nan=0.0, posinf=0.0, neginf=0.0) * 1000.0)
    if depth_max and depth_max > 0:
        far = mm > depth_max * 1000.0
        if far.any():
            if not quiet:
                log.info('%s: %d pixels beyond %.1f m marked invalid',
                         where, int(far.sum()), depth_max)
            mm[far] = 0
    over = int((mm > DEPTH_MAX_MM).sum())
    if over:
        # uint16 wraps silently, turning a 70 m reading into a 4 mm one -- a
        # phantom surface pressed against the lens. Clamp to 0 (== invalid).
        if not quiet:
            log.warning('%s: %d pixels beyond %.1f m, marking them invalid',
                        where, over, DEPTH_MAX_MM / 1000.0)
        mm[mm > DEPTH_MAX_MM] = 0
    return mm.astype(np.uint16)


# Just above the D455's ~0.5 m min-Z (document/GEOMETRY.md S1). Surviving depth down
# here means whatever is there straddles the limit, so part of it has already
# dropped out -- the surface you can still see is the rim of the hole.
NEAR_EDGE_M = 0.6


def _frame_stats(depth_m, depth_max):
    """What the preview reports and what decides the warnings."""
    valid = depth_m > 0
    if depth_max and depth_max > 0:
        valid &= depth_m <= depth_max
    n = int(valid.sum())
    v = depth_m[valid] if n else None
    h = depth_m.shape[0]
    return {
        'n': n,
        'pct': 100.0 * valid.mean(),
        'lo': float(v.min()) if n else 0.0,
        'hi': float(v.max()) if n else 0.0,
        'med': float(np.median(v)) if n else 0.0,
        'near': 100.0 * float((v < NEAR_EDGE_M).mean()) if n else 0.0,
        # a too-close camera is usually a surface it is standing on
        'bottom_holes': 100.0 * float((~valid[2 * h // 3:]).mean()),
    }


def _stats_warning(st):
    """-> (text, is_error) or (None, False).

    The failures this repo has actually shipped a bad capture over. All of them
    are invisible in the RGB feed, which is why the preview shows depth beside
    it.

    Thresholds are calibrated, not guessed. Across the captures on disk and the
    NYU frames the model was trained on:

        capture      valid%   <0.6 m %   lower-third holes %
        room01  bad    86.6       8.87                  30.5
        test01  bad    63.1      10.82                  62.3
        NYU0001 good   86.3       0.00                   7.1
        NYU0002 good   86.3       0.00                   5.5
        NYU0003 good   93.8       0.00                   1.7

    `<0.6 m` separates them outright -- no NYU frame has a single pixel there,
    both bad D455 captures have ~10%. That is the signal worth alarming on;
    hole fraction alone is much weaker (30.5% on a bad frame vs 7.1% on a good
    one).
    """
    if st['pct'] < 20.0:
        return ('very sparse depth -- aim at a wall 1-4 m away, check the light',
                True)
    if st['near'] > 2.0:
        return ('%.0f%% of the depth is under %.1f m, at the ~0.5 m min-Z edge '
                '-- anything closer is already dropping out. Move back.'
                % (st['near'], NEAR_EDGE_M), True)
    if st['bottom_holes'] > 20.0:
        return ('%.0f%% of the lower third has no depth' % st['bottom_holes'],
                True)
    if st['med'] > 0 and st['med'] < 1.0:
        return ('scene is very close (median %.2f m); the grid is 4.8 m deep'
                % st['med'], False)
    return (None, False)


# ---------------------------------------------------------------------------
class Preview(object):
    """RGB | colorised depth, with a clickable capture button.

    The depth pane uses the same colouriser as `colorize_depth.py` -- the
    RealSense Viewer's own rs-jet ramp with per-frame histogram equalisation --
    so what you frame here is what the Viewer would have shown you.

    It renders the 640x480 frame that will actually be SAVED, not the sensor's
    native one. Framing decisions are about what the network sees, and
    match_nyu_fov throws away most of the D455's width before that point.

    The footer also holds the camera-height field. Digits, '.' and backspace
    edit it directly -- none of those keys is bound to anything else here, so
    there is no focus mode to enter or get stuck in -- and clicking the field
    clears it. CAPTURE refuses to fire while the value is missing or outside
    HEIGHT_MIN..HEIGHT_MAX, because a height is not a rendering preference: it
    places the voxel grid, and a frame saved with the wrong one is silently
    wrong rather than obviously broken.
    """

    FOOTER = 128
    _BTN_BG, _BTN_HOT = (62, 62, 62), (92, 92, 92)
    # loose sanity bounds, not a claim about your room: NYU's own camera
    # heights span 0.749-1.553 m, so anything outside this is a typo.
    HEIGHT_MIN, HEIGHT_MAX = 0.2, 3.0
    _EDIT_KEYS = frozenset(ord(c) for c in '0123456789.')

    def __init__(self, window='CleanerS capture', total=1, height=None):
        self.window, self.total = window, total
        self.want_capture = self.want_quit = False
        self._hover = None
        self._flash_until, self._flash_name = 0.0, ''
        self._msg_until, self._msg = 0.0, ''
        self.height_text = '' if height is None else ('%g' % height)
        row = 480 + 58
        self.btn_capture = (16, row, 196, row + 40)
        self.btn_quit = (212, row, 312, row + 40)
        self.field = (1010, 480 + 10, 1150, 480 + 42)
        cv2.namedWindow(window, cv2.WINDOW_AUTOSIZE)
        cv2.setMouseCallback(window, self._on_mouse)

    def height_value(self):
        """-> float metres, or None while the field is empty or implausible."""
        try:
            h = float(self.height_text)
        except ValueError:
            return None
        return h if self.HEIGHT_MIN <= h <= self.HEIGHT_MAX else None

    @staticmethod
    def _inside(r, x, y):
        return r[0] <= x <= r[2] and r[1] <= y <= r[3]

    def _on_mouse(self, event, x, y, flags, param):
        if event == cv2.EVENT_MOUSEMOVE:
            self._hover = ('capture' if self._inside(self.btn_capture, x, y)
                           else 'quit' if self._inside(self.btn_quit, x, y)
                           else 'field' if self._inside(self.field, x, y)
                           else None)
        elif event == cv2.EVENT_LBUTTONDOWN:
            if self._inside(self.btn_capture, x, y):
                self.want_capture = True
            elif self._inside(self.btn_quit, x, y):
                self.want_quit = True
            elif self._inside(self.field, x, y):
                self.height_text = ''

    def flash(self, name):
        self._flash_until, self._flash_name = time.time() + 0.7, name

    def message(self, text, secs=2.0):
        """Transient footer line -- why a keypress did nothing."""
        self._msg_until, self._msg = time.time() + secs, text

    def _button(self, img, rect, label, hot, accent):
        x0, y0, x1, y1 = rect
        cv2.rectangle(img, (x0, y0), (x1, y1),
                      self._BTN_HOT if hot else self._BTN_BG, -1)
        cv2.rectangle(img, (x0, y0), (x1, y1), accent, 1)
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.putText(img, label, (x0 + (x1 - x0 - tw) // 2,
                                 y0 + (y1 - y0 + th) // 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, accent, 1, cv2.LINE_AA)

    def _height_field(self, img, ok):
        F, GREY = cv2.FONT_HERSHEY_SIMPLEX, (160, 160, 160)
        x0, y0, x1, y1 = self.field
        accent = (140, 240, 140) if ok else (70, 170, 255)
        label = 'camera height'
        (tw, _th), _ = cv2.getTextSize(label, F, 0.5, 1)
        cv2.putText(img, label, (x0 - tw - 14, y1 - 10), F, 0.5, GREY, 1,
                    cv2.LINE_AA)
        cv2.rectangle(img, (x0, y0), (x1, y1),
                      (48, 48, 48) if self._hover == 'field' else (16, 16, 16), -1)
        cv2.rectangle(img, (x0, y0), (x1, y1), accent, 1)
        # a caret that blinks only while the value is unusable, so a finished
        # field reads as a value rather than as something still being typed
        caret = '' if ok else ('_' if int(time.time() * 2) % 2 else ' ')
        cv2.putText(img, self.height_text + caret, (x0 + 10, y1 - 10), F, 0.55,
                    (240, 240, 240), 1, cv2.LINE_AA)
        cv2.putText(img, 'm', (x1 + 8, y1 - 10), F, 0.5, GREY, 1, cv2.LINE_AA)
        if not ok:
            cv2.putText(img, 'type it (%.1f-%.1f m) -- measure, do not guess'
                        % (self.HEIGHT_MIN, self.HEIGHT_MAX),
                        (x0 - 300, y1 + 26), F, 0.44, (70, 170, 255), 1,
                        cv2.LINE_AA)

    def draw(self, rgb, depth_vis, st, saved):
        top = np.hstack([rgb, depth_vis])
        img = np.vstack([top, np.full((self.FOOTER, top.shape[1], 3), 26,
                                      np.uint8)])
        F, WHITE, GREY = cv2.FONT_HERSHEY_SIMPLEX, (240, 240, 240), (160, 160, 160)
        for x, t in ((10, 'RGB'), (650, 'DEPTH  rs-jet, equalised')):
            cv2.putText(img, t, (x, 22), F, 0.5, (0, 0, 0), 3, cv2.LINE_AA)
            cv2.putText(img, t, (x, 22), F, 0.5, WHITE, 1, cv2.LINE_AA)
        tail = 'saved %d / %d' % (saved, self.total)
        (tw, _th), _ = cv2.getTextSize(tail, F, 0.5, 1)
        cv2.putText(img, tail, (img.shape[1] - tw - 12, 22), F, 0.5, (0, 0, 0),
                    3, cv2.LINE_AA)
        cv2.putText(img, tail, (img.shape[1] - tw - 12, 22), F, 0.5, WHITE, 1,
                    cv2.LINE_AA)

        y = 480
        cv2.putText(img, 'valid %5.1f%%   %.2f-%.2f m   median %.2f m'
                    % (st['pct'], st['lo'], st['hi'], st['med']),
                    (16, y + 24), F, 0.52, WHITE, 1, cv2.LINE_AA)
        warn, is_err = _stats_warning(st)
        if warn:
            cv2.putText(img, warn, (16, y + 46), F, 0.44,
                        (60, 60, 255) if is_err else (0, 190, 255), 1, cv2.LINE_AA)

        ok = self.height_value() is not None
        self._height_field(img, ok)
        self._button(img, self.btn_capture, 'CAPTURE  (space)',
                     self._hover == 'capture', (140, 240, 140) if ok
                     else (110, 110, 110))
        self._button(img, self.btn_quit, 'QUIT  (q)',
                     self._hover == 'quit', (200, 200, 200))

        if time.time() < self._msg_until:
            cv2.putText(img, self._msg, (330, self.btn_capture[1] + 26), F,
                        0.5, (70, 170, 255), 1, cv2.LINE_AA)
        elif time.time() < self._flash_until:
            cv2.rectangle(img, (0, 0), (img.shape[1] - 1, img.shape[0] - 1),
                          (120, 255, 120), 4)
            cv2.putText(img, 'SAVED  ' + self._flash_name,
                        (330, self.btn_capture[1] + 26), F, 0.55,
                        (140, 255, 140), 1, cv2.LINE_AA)
        cv2.imshow(self.window, img)

    def poll(self):
        """-> 'capture' | 'quit' | None. Merges mouse clicks and key presses.

        Digit / '.' / backspace go to the height field. They are checked first
        and never fall through, but nothing else is bound to them either, so
        typing a height can never be mistaken for a capture or a quit.
        """
        k = cv2.waitKey(1) & 0xFF
        if k in self._EDIT_KEYS:
            if len(self.height_text) < 6:
                self.height_text += chr(k)
            return None
        if k in (8, 127):                                   # backspace / delete
            self.height_text = self.height_text[:-1]
            return None
        if self.want_capture or k in (32, 13, 10):          # space / enter
            self.want_capture = False
            if self.height_value() is None:
                self.message('type the camera height first (%.1f-%.1f m)'
                             % (self.HEIGHT_MIN, self.HEIGHT_MAX))
                return None
            return 'capture'
        if self.want_quit or k in (ord('q'), 27):           # q / esc
            self.want_quit = False
            return 'quit'
        # the window manager's X button; getWindowProperty goes <1 once closed
        try:
            if cv2.getWindowProperty(self.window, cv2.WND_PROP_VISIBLE) < 1:
                return 'quit'
        except cv2.error:
            return 'quit'
        return None

    def close(self):
        try:
            cv2.destroyWindow(self.window)
            cv2.waitKey(1)
        except cv2.error:
            pass


def _open_preview(total, height=None):
    """-> Preview, or None if there is no display to draw on."""
    if not (os.environ.get('DISPLAY') or os.environ.get('WAYLAND_DISPLAY')):
        log.warning('--preview ignored: no DISPLAY. Capturing straight away.')
        return None
    try:
        return Preview(total=total, height=height)
    except cv2.error as e:
        log.warning('--preview ignored: OpenCV has no GUI backend here (%s). '
                    'Capturing straight away.', e)
        return None


def _save_frame(frame, name, dirs, args):
    """Write one frame's PNGs. Returns nothing; raises on I/O failure."""
    depth_dir, rgb_dir, native_dir = dirs
    cv2.imwrite(os.path.join(depth_dir, name + '.png'),
                _to_uint16_mm(frame.depth, name, args.depth_max))
    cv2.imwrite(os.path.join(rgb_dir, name + '.png'), frame.rgb)
    if native_dir is not None and frame.depth_native is not None:
        cv2.imwrite(os.path.join(native_dir, name + '_depth.png'),
                    _to_uint16_mm(frame.depth_native, name + ' native',
                                  args.depth_max))
        cv2.imwrite(os.path.join(native_dir, name + '_rgb.png'),
                    frame.rgb_native)


def main():
    setup_logging()
    args = parse_args()

    depth_dir = ensure_dir(os.path.join(args.out_dir, 'depth'))
    rgb_dir = ensure_dir(os.path.join(args.out_dir, 'rgb'))
    native_dir = None if args.no_native else ensure_dir(
        os.path.join(args.out_dir, 'native'))

    src = RealSenseSource(width=args.width, height=args.height, fps=args.fps,
                          fov=args.fov, warmup=args.warmup,
                          filters=args.filters, hole_fill=args.hole_fill,
                          keep_native=not args.no_native,
                          undistort=not args.no_undistort)

    saved = []
    cam_K_eff = None
    dirs = (depth_dir, rgb_dir, native_dir)
    preview = (_open_preview(args.frames, args.camera_height)
               if args.preview else None)
    # With a preview the height is typed in the window and read back at save
    # time; without one there is nowhere to type it, so the flag stands in.
    camera_height = args.camera_height
    if preview is not None:
        log.info('preview: type the camera height, then SPACE (or click '
                 'CAPTURE) to save a frame, q to stop')
    elif camera_height is None:
        camera_height = DEFAULT_CAMERA_HEIGHT
        log.warning('no --camera_height given: recording the %.2f m default. '
                    'The voxel grid is floor-anchored, so this shifts the '
                    'whole scene vertically -- measure it and pass it, or use '
                    '--preview and type it.', camera_height)
    try:
        for frame in src.frames():
            if len(saved) >= args.frames:
                break
            # Numbering follows what was SAVED, not what the camera yielded, so
            # a preview session that ran for 400 frames before the first
            # keypress still writes live_000000. run_inference globs these in
            # sorted order and meta.json lists them.
            name = 'live_%06d' % len(saved)
            st = _frame_stats(frame.depth, args.depth_max)

            if preview is not None:
                depth_vis = colorize(
                    _to_uint16_mm(frame.depth, name, args.depth_max, quiet=True),
                    'rs-jet', INVALID_COLORS['black'], 'equalize')
                preview.draw(frame.rgb, depth_vis, st, len(saved))
                action = preview.poll()
                if action == 'quit':
                    log.info('preview closed after %d frame(s)', len(saved))
                    break
                if action != 'capture':
                    continue
                camera_height = preview.height_value()

            # match_nyu_fov recomputes intrinsics from the resize scale and the
            # crop offset it actually achieved, so take them off the frame
            # rather than assuming NYU's values -- a clamped crop moves the
            # principal point, and only the measured K matches the saved pixels.
            cam_K_eff = frame.cam_K
            _save_frame(frame, name, dirs, args)
            saved.append(name)

            rng = ('%.2f-%.2f m (median %.2f)'
                   % (st['lo'], st['hi'], st['med'])) if st['n'] \
                else 'NO VALID DEPTH'
            log.info('%s: %.1f%% of pixels have depth, range %s',
                     name, st['pct'], rng)
            warn, _ = _stats_warning(st)
            if warn:
                log.warning('%s: %s', name, warn)

            if preview is not None:
                preview.flash(name)
            elif args.interval and len(saved) < args.frames:
                time.sleep(args.interval)
    finally:
        if preview is not None:
            preview.close()
        src.close()

    if not saved:
        log.warning('no frames saved; leaving %s without a meta.json',
                    args.out_dir)
        return

    meta = {
        'source': 'realsense',
        'stream': {'width': args.width, 'height': args.height, 'fps': args.fps},
        'depth_scale': float(src.depth_scale),
        'depth_units': 'uint16 millimetres (plain; NOT the NYU bit-rotated encoding)',
        'fov': args.fov,
        # what the saved 640x480 depth/ and rgb/ frames actually have
        'cam_K': np.asarray(cam_K_eff if cam_K_eff is not None
                            else src.cam_K).tolist(),
        'cam_K_native': np.asarray(src.cam_K).tolist(),
        'camera_height': camera_height,
        'yaw': args.yaw,
        'depth_max': args.depth_max,
        'undistorted': not args.no_undistort,
        'filters': {'spatial_temporal': bool(args.filters),
                    'hole_fill': bool(args.hole_fill)},
        'frames': saved,
    }
    meta_path = os.path.join(args.out_dir, 'meta.json')
    with open(meta_path, 'w') as f:
        json.dump(meta, f, indent=2)

    log.info('wrote %d frame(s) to %s', len(saved), args.out_dir)
    log.info('now run:')
    log.info('  python -m inference.run_inference --cfg ./cfgs/NYU/voxelSSC.yaml \\')
    log.info('      --pretrained_path ./checkpoint/CleanerS_ckpt.pth \\')
    log.info('      --live_dir %s --out_dir ./outputs', args.out_dir)


if __name__ == '__main__':
    main()
