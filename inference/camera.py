"""
inference/camera.py

Frame sources for the streaming entrypoint. Built for the Intel RealSense
D455, with two hardware-free replay sources for testing without hardware.

    realsense                 D455 (or any D4xx) over USB
    realsense:1280x720@30     ... with an explicit stream profile
    nyu:/path/to/depth        replay NYU .bin/.png pairs (real vox_origin/pose)
    folder:/path/to/pngs      replay plain 16-bit depth PNGs in millimetres

*** WHY THE FOV MATCHING EXISTS ***
The pretrained CleanerS checkpoint was trained on NYU Depth v2, captured with
a Kinect v1: fx=518.86, fy=519.47 at 640x480, i.e. a ~63 x 49 degree field of
view. A D455 is far wider -- roughly 87x58 (depth) and 90x65 (RGB).

Feeding the D455's native frame straight in is geometrically consistent (the
encoder uses whatever cam_K it is handed), but it puts the RGB branch far
outside its training distribution: the same wall occupies a very different
pixel footprint, and SegFormer features do not transfer for free.

So the default is fov='nyu': resize and centre-crop each frame so the result
is 640x480 with intrinsics matching NYU's. Stream at 1280x720 and the match
is a DOWNSCALE, which loses no detail; at 640x480 it would have to upscale,
which is why 1280x720 is the default profile.

Pass fov='native' to skip it and hand the model the raw wide-FOV frame with
true device intrinsics. Geometry stays correct either way -- the effective
cam_K is always recomputed from what was actually done to the pixels, never
assumed.
"""

import glob
import logging
import os
import time
from dataclasses import dataclass
from typing import Iterator, Optional

import json

import cv2
import numpy as np

try:
    from .frame_loader import CAM_K, IMG_H, IMG_W
except ImportError:
    from frame_loader import CAM_K, IMG_H, IMG_W

log = logging.getLogger(__name__)


@dataclass
class Frame:
    depth: np.ndarray                       # (480,640) float32, METRES
    name: str
    rgb: Optional[np.ndarray] = None        # (480,640,3) uint8 BGR
    cam_K: Optional[np.ndarray] = None      # 3x3; None -> NYU default
    # Only NYU replay knows the dataset's room-canonical grid placement.
    # Live sensors leave these None; from_live_camera then builds a
    # gravity-aligned floor-anchored grid.
    vox_origin: Optional[np.ndarray] = None
    cam_pose: Optional[np.ndarray] = None
    # The sensor's own frame before match_nyu_fov resized/cropped it. Only
    # populated when the source is asked for it (keep_native=True), so a
    # capture can be re-cropped later without going back to the hardware.
    # Recorded by capture.py in meta.json, so replaying a capture reproduces
    # the single-frame result instead of falling back to CLI defaults.
    camera_height: Optional[float] = None
    yaw: Optional[float] = None
    depth_native: Optional[np.ndarray] = None
    rgb_native: Optional[np.ndarray] = None
    cam_K_native: Optional[np.ndarray] = None


class CameraSource:
    name = 'base'

    def frames(self) -> Iterator[Frame]:
        raise NotImplementedError

    def close(self):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


def _require_shape(depth, where):
    if depth.shape != (IMG_H, IMG_W):
        raise ValueError(
            '{}: depth is {}, but the model requires ({}, {}). MAPPING_SENTINEL '
            '== {} is baked into the encoder and into test_NYU.py\'s SC mask, so '
            'the resolution is not negotiable.'.format(
                where, depth.shape, IMG_H, IMG_W, IMG_H * IMG_W))
    return depth


def match_nyu_fov(depth, rgb, K):
    """Resize + centre-crop a frame so the result is 640x480 with intrinsics
    as close to NYU's as the source allows.

    Returns (depth, rgb, K_effective). K_effective is measured from what was
    actually done, not assumed -- if the crop had to be clamped or padded, the
    returned principal point reflects that, so the unprojection stays exact.

    Depth is resampled with INTER_NEAREST on purpose: bilinear interpolation
    across a depth discontinuity invents surfaces that were never observed,
    halfway between the foreground and the background.
    """
    fx_s, fy_s, cx_s, cy_s = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    fx_t, fy_t = CAM_K[0, 0], CAM_K[1, 1]
    cx_t, cy_t = CAM_K[0, 2], CAM_K[1, 2]

    s = float(fx_t / fx_s)
    H_s, W_s = depth.shape
    W_r, H_r = int(round(W_s * s)), int(round(H_s * s))

    depth_r = cv2.resize(depth, (W_r, H_r), interpolation=cv2.INTER_NEAREST)
    rgb_r = cv2.resize(rgb, (W_r, H_r), interpolation=cv2.INTER_AREA) \
        if rgb is not None else None

    # place the principal point at NYU's (cx_t, cy_t)
    x0 = int(round(cx_s * s - cx_t))
    y0 = int(round(cy_s * s - cy_t))

    def _take(img, fill):
        out = np.full((IMG_H, IMG_W) + img.shape[2:], fill, dtype=img.dtype)
        sx0, sy0 = max(0, x0), max(0, y0)
        sx1, sy1 = min(W_r, x0 + IMG_W), min(H_r, y0 + IMG_H)
        if sx1 <= sx0 or sy1 <= sy0:
            raise ValueError('FOV match produced an empty crop; check intrinsics')
        out[sy0 - y0:sy1 - y0, sx0 - x0:sx1 - x0] = img[sy0:sy1, sx0:sx1]
        return out

    # depth fill value 0 == "invalid", which the encoder already handles
    depth_c = _take(depth_r, 0)
    rgb_c = _take(rgb_r, 0) if rgb_r is not None else None

    K_eff = np.array([[fx_s * s, 0, cx_s * s - x0],
                      [0, fy_s * s, cy_s * s - y0],
                      [0, 0, 1]], dtype=np.float32)
    return depth_c, rgb_c, K_eff


# ---------------------------------------------------------------------------
class RealSenseSource(CameraSource):
    """Intel RealSense D455 (works for any D4xx), depth aligned to colour.

    Reads real intrinsics and depth scale off the device. A D455's RGB fx at
    1280x720 is roughly 640 against NYU's 518.86 -- assuming NYU's values
    would misplace every point by tens of centimetres at room scale.
    """

    name = 'realsense'

    def __init__(self, width=1280, height=720, fps=30, fov='nyu',
                 emitter=True, align_to_color=True, warmup=30,
                 filters=False, hole_fill=False, keep_native=False,
                 undistort=True):
        """warmup: frames to pull and discard before yielding. A D4xx opens
            with auto-exposure and white-balance unconverged -- the first
            frames come out dark, and the depth is correspondingly sparse.
            30 frames is 1 s at the default rate. Never set this to 0 for a
            single-shot capture; that one frame IS the unconverged one.
        filters: run the SDK's spatial + temporal edge-preserving filters.
            Temporal needs history, so it is fed during warmup too -- which is
            the other reason warmup must not be 0 when this is on.
        hole_fill: additionally run hole_filling_filter. Off by default and
            deliberately separate: it INVENTS depth for pixels the sensor
            never measured, and the TSDF encoder cannot tell an invented
            surface from a real one -- it stamps both as observed geometry.
            Completing unseen space is the network's job, not a filter's.
        keep_native: also attach the pre-FOV-match frame (see Frame).
        undistort: resample into an ideal pinhole frame. The D455's COLOR
            stream is inverse_brown_conrady with real coefficients, and after
            align-to-color the depth inherits them -- but frame_loader
            unprojects with a plain pinhole model, which on this unit is off
            by up to 3.3 cm at the crop corners (1.7 high-res voxels), zero at
            centre. Measured, not assumed; see _build_undistort_maps.
        """
        try:
            import pyrealsense2 as rs
        except ImportError as e:
            raise RuntimeError(
                'pyrealsense2 not installed: `pip install pyrealsense2`. Use '
                '`--source nyu:/data/NYU/depth` to test without hardware.') from e
        self._rs = rs
        self.fov = fov

        ctx = rs.context()
        devs = ctx.query_devices()
        if len(devs) == 0:
            raise RuntimeError(
                'no RealSense device found. Check: (1) the camera is plugged '
                'into a USB 3 port, (2) udev rules are installed '
                '(99-realsense-libusb.rules).')
        dev = devs[0]
        log.info('device: %s  fw=%s  serial=%s',
                 dev.get_info(rs.camera_info.name),
                 dev.get_info(rs.camera_info.firmware_version),
                 dev.get_info(rs.camera_info.serial_number))

        self._check_profile(rs, dev, width, height, fps)

        cfg = rs.config()
        cfg.enable_stream(rs.stream.depth, width, height, rs.format.z16, fps)
        cfg.enable_stream(rs.stream.color, width, height, rs.format.bgr8, fps)
        self.pipeline = rs.pipeline(ctx)
        self.profile = self.pipeline.start(cfg)
        self.align = rs.align(rs.stream.color) if align_to_color else None

        depth_sensor = self.profile.get_device().first_depth_sensor()
        self.depth_scale = depth_sensor.get_depth_scale()
        if depth_sensor.supports(rs.option.emitter_enabled):
            depth_sensor.set_option(rs.option.emitter_enabled, 1 if emitter else 0)

        intr = (self.profile.get_stream(rs.stream.color)
                .as_video_stream_profile().get_intrinsics())
        self.cam_K = np.array([[intr.fx, 0, intr.ppx],
                               [0, intr.fy, intr.ppy],
                               [0, 0, 1]], dtype=np.float32)

        hfov = 2 * np.degrees(np.arctan(intr.width / (2 * intr.fx)))
        vfov = 2 * np.degrees(np.arctan(intr.height / (2 * intr.fy)))
        log.info('stream %dx%d@%d  depth_scale=%.6f', width, height, fps,
                 self.depth_scale)
        log.info('device intrinsics fx=%.2f fy=%.2f cx=%.2f cy=%.2f  '
                 '(FOV %.1f x %.1f deg)', intr.fx, intr.fy, intr.ppx, intr.ppy,
                 hfov, vfov)

        if fov == 'nyu':
            nyu_h = 2 * np.degrees(np.arctan(IMG_W / (2 * CAM_K[0, 0])))
            nyu_v = 2 * np.degrees(np.arctan(IMG_H / (2 * CAM_K[1, 1])))
            scale = CAM_K[0, 0] / intr.fx
            log.info('fov=nyu -> %s to %.3f then crop to 640x480, targeting '
                     'NYU %.1f x %.1f deg', 'DOWNSCALE' if scale <= 1 else
                     'UPSCALE (prefer a higher stream resolution)', scale,
                     nyu_h, nyu_v)
        else:
            log.warning('fov=native -> the RGB branch sees a %.1f deg frame '
                        'against %.1f deg in training; expect degraded '
                        'semantics', hfov,
                        2 * np.degrees(np.arctan(IMG_W / (2 * CAM_K[0, 0]))))

        self.keep_native = keep_native
        self._umap = None
        if undistort:
            self._umap = self._build_undistort_maps(intr)
            if self._umap is None:
                log.info('undistort: coefficients are negligible, skipping')
            else:
                log.info('undistort: resampling to an ideal pinhole frame '
                         '(coeffs %s)',
                         ' '.join('%.4f' % c for c in intr.coeffs))
        self._filters = None
        if filters:
            # librealsense's documented order: work in disparity space for the
            # spatial/temporal pass, since disparity is where the sensor's
            # noise is actually uniform, then convert back.
            self._filters = [rs.disparity_transform(True),
                             rs.spatial_filter(),
                             rs.temporal_filter(),
                             rs.disparity_transform(False)]
            if hole_fill:
                self._filters.append(rs.hole_filling_filter())
            log.info('depth filters: spatial + temporal%s',
                     ' + hole_filling' if hole_fill else '')

        if warmup > 0:
            log.info('warming up (%d frames) for auto-exposure', warmup)
            for _ in range(warmup):
                try:
                    fs = self.pipeline.wait_for_frames()
                except Exception:
                    break
                # push warmup frames through the temporal filter so it has
                # history by the time the first real frame arrives
                if self._filters is not None:
                    d = fs.get_depth_frame()
                    if d:
                        for f in self._filters:
                            d = f.process(d)

    @staticmethod
    def _build_undistort_maps(intr, eps=1e-6):
        """cv2.remap tables sending an ideal pinhole grid to the source pixels.

        For an output pixel q, the pinhole ray is ((qx-cx)/fx, (qy-cy)/fy).
        The source pixel observing that ray is found by applying the
        Brown-Conrady polynomial FORWARD -- despite the model being named
        `inverse_brown_conrady`, which means the stored coefficients are
        already fitted for deprojection, so inverting them here doubles the
        error instead of removing it. Verified against
        rs2_deproject_pixel_to_point: 0.0025 cm residual against 3.34 cm
        uncorrected, over the region match_nyu_fov actually keeps.

        Intrinsics are unchanged by construction -- the output is the same
        camera with zero distortion -- so cam_K stays valid.
        """
        k1, k2, p1, p2, k3 = [float(c) for c in intr.coeffs[:5]]
        if max(abs(k1), abs(k2), abs(k3), abs(p1), abs(p2)) < eps:
            return None
        u, v = np.meshgrid(np.arange(intr.width, dtype=np.float32),
                           np.arange(intr.height, dtype=np.float32))
        x = (u - intr.ppx) / intr.fx
        y = (v - intr.ppy) / intr.fy
        r2 = x * x + y * y
        f = 1 + k1 * r2 + k2 * r2 * r2 + k3 * r2 * r2 * r2
        xd = x * f + 2 * p1 * x * y + p2 * (r2 + 2 * x * x)
        yd = y * f + p1 * (r2 + 2 * y * y) + 2 * p2 * x * y
        return ((xd * intr.fx + intr.ppx).astype(np.float32),
                (yd * intr.fy + intr.ppy).astype(np.float32))

    @staticmethod
    def _check_profile(rs, dev, width, height, fps):
        """Fail with the reason, not with 'Couldn\'t resolve requests'.

        The usual cause is a USB 2.1 link: the camera enumerates and opens
        fine, but the bandwidth ceiling silently removes most high-rate modes,
        so a request that works on every USB 3 machine is simply absent here.
        """
        want = {rs.stream.depth: rs.format.z16, rs.stream.color: rs.format.bgr8}
        avail = {k: set() for k in want}
        for sensor in dev.sensors:
            for prof in sensor.get_stream_profiles():
                v = prof.as_video_stream_profile()
                if not v:
                    continue
                st = prof.stream_type()
                if st in want and prof.format() == want[st]:
                    avail[st].add((v.width(), v.height(), prof.fps()))

        missing = [st for st in want if (width, height, fps) not in avail[st]]
        if not missing:
            return

        try:
            usb = dev.get_info(rs.camera_info.usb_type_descriptor)
        except Exception:
            usb = 'unknown'

        def _fmt(st):
            both = sorted(avail[rs.stream.depth] & avail[rs.stream.color])
            return ', '.join('%dx%d@%d' % p for p in both) or '(none in common)'

        msg = ["%dx%d@%d is not available on this device." % (width, height, fps)]
        msg.append("USB link: %s" % usb)
        if str(usb).startswith('2'):
            msg.append(
                "That is the cause. A D4xx on USB 2.1 loses most high-rate "
                "modes to the bandwidth ceiling. Plug into a blue/SS USB 3 "
                "port using the cable that shipped with the camera -- many "
                "generic USB-C cables are charge-only or USB 2 and will "
                "enumerate at 2.1 exactly like this.")
        msg.append("Resolutions supported for BOTH depth(z16) and color(bgr8) "
                   "on the current link: %s" % _fmt(None))
        raise RuntimeError('\n  '.join(msg))

    def frames(self):
        i = 0
        while True:
            fs = self.pipeline.wait_for_frames()
            if self.align is not None:
                fs = self.align.process(fs)
            d, c = fs.get_depth_frame(), fs.get_color_frame()
            if not d or not c:
                continue
            if self._filters is not None:
                for f in self._filters:
                    d = f.process(d)

            depth = np.asanyarray(d.get_data()).astype(np.float32) * self.depth_scale
            rgb = np.asanyarray(c.get_data())

            if self._umap is not None:
                mx, my = self._umap
                # NEAREST for depth: interpolating across a depth
                # discontinuity invents a surface halfway between the
                # foreground and the background, and the TSDF encoder cannot
                # tell that from a real observation.
                depth = cv2.remap(depth, mx, my, cv2.INTER_NEAREST,
                                  borderMode=cv2.BORDER_CONSTANT, borderValue=0)
                rgb = cv2.remap(rgb, mx, my, cv2.INTER_LINEAR,
                                borderMode=cv2.BORDER_CONSTANT, borderValue=0)

            depth_n, rgb_n = (depth.copy(), rgb.copy()) if self.keep_native \
                else (None, None)

            if self.fov == 'nyu':
                depth, rgb, K = match_nyu_fov(depth, rgb, self.cam_K)
            else:
                K = self.cam_K
            yield Frame(depth=_require_shape(depth, 'realsense'), rgb=rgb,
                        cam_K=K, name='live_{:06d}'.format(i),
                        depth_native=depth_n, rgb_native=rgb_n,
                        cam_K_native=self.cam_K if self.keep_native else None)
            i += 1

    def close(self):
        try:
            self.pipeline.stop()
        except Exception:
            pass


# ---------------------------------------------------------------------------
class NYUSource(CameraSource):
    """Replay NYU .bin/.png pairs with each frame's real vox_origin/cam_pose.

    The only source whose output is directly comparable to the reference
    results, because it reproduces the dataset's room-canonical grid placement.
    Use it to prove the pipeline works before trusting live output.
    """

    name = 'nyu'

    def __init__(self, path, rgb_dir=None, loop=False, fps=None):
        try:
            from .frame_loader import FrameLoader
        except ImportError:
            from frame_loader import FrameLoader
        self._FL = FrameLoader
        bins = sorted(glob.glob(os.path.join(path, '*.bin')))
        self.pairs = [(b, b[:-4] + '.png') for b in bins
                      if os.path.exists(b[:-4] + '.png')]
        if not self.pairs:
            raise FileNotFoundError('no matched .bin/.png pairs under %r' % path)
        self.rgb_dir, self.loop = rgb_dir, loop
        self.period = (1.0 / fps) if fps else 0.0
        log.info('NYUSource: %d pairs from %s', len(self.pairs), path)

    def frames(self):
        while True:
            for bin_path, png_path in self.pairs:
                t0 = time.time()
                stem = os.path.splitext(os.path.basename(bin_path))[0]
                vox_origin, cam_pose = self._FL._load_bin_header(bin_path)
                depth = self._FL._load_depth(png_path)
                rgb = None
                if self.rgb_dir:
                    # this dataset ships NYU0001_rgb.png; other CleanerS-derived
                    # dumps use _colors.png, so try both rather than silently
                    # yielding rgb=None and skipping every frame
                    for cand in ('%s_rgb.png' % stem[:7],
                                 '%s_colors.png' % stem[:7]):
                        fp = os.path.join(self.rgb_dir, cand)
                        if os.path.exists(fp):
                            rgb = cv2.imread(fp)
                            break
                yield Frame(depth=_require_shape(depth, png_path), rgb=rgb,
                            cam_K=CAM_K, vox_origin=vox_origin,
                            cam_pose=cam_pose, name=stem)
                if self.period:
                    time.sleep(max(0.0, self.period - (time.time() - t0)))
            if not self.loop:
                return


class FolderSource(CameraSource):
    """Replay 16-bit depth PNGs holding raw MILLIMETRES -- the plain sensor
    encoding, e.g. what `rs-convert` writes.

    NOT the NYU encoding: NYU's PNGs are bit-rotated Kinect data needing
    ((d << 13) | (d >> 3)) before the /1000. Use NYUSource for those.
    """

    name = 'folder'

    @staticmethod
    def _find_meta(path):
        """Locate capture.py's meta.json for `path`, which may be either the
        capture root or its depth/ subfolder. Returns (meta, root)."""
        for root in (path, os.path.dirname(os.path.normpath(path))):
            mp = os.path.join(root, 'meta.json')
            if root and os.path.exists(mp):
                try:
                    with open(mp) as f:
                        return json.load(f), root
                except (ValueError, OSError) as e:
                    log.warning('ignoring unreadable %s: %s', mp, e)
        return None, path

    def __init__(self, path, rgb_dir=None, cam_K=None, loop=False, fps=None,
                 fov='native'):
        meta, root = self._find_meta(path)

        depth_dir = path
        if meta is not None and os.path.isdir(os.path.join(root, 'depth')):
            depth_dir = os.path.join(root, 'depth')
            if rgb_dir is None and os.path.isdir(os.path.join(root, 'rgb')):
                rgb_dir = os.path.join(root, 'rgb')

        self.paths = sorted(glob.glob(os.path.join(depth_dir, '*.png')))
        if not self.paths:
            raise FileNotFoundError('no .png files under %r' % depth_dir)
        self.rgb_dir, self.loop, self.fov = rgb_dir, loop, fov
        self.cam_K = np.asarray(cam_K, np.float32) if cam_K is not None else CAM_K
        self.camera_height = self.yaw = None
        self.period = (1.0 / fps) if fps else 0.0

        if meta is not None:
            if cam_K is None and 'cam_K' in meta:
                # the measured intrinsics of these exact files -- the sensor's
                # fy differs from NYU's by ~0.24% and drifts with temperature,
                # so the recorded value beats the hardcoded default
                self.cam_K = np.asarray(meta['cam_K'], np.float32)
            self.camera_height = meta.get('camera_height')
            self.yaw = meta.get('yaw')
            if meta.get('fov') == 'nyu' and fov == 'nyu':
                # capture.py already resized and cropped these; running
                # match_nyu_fov a second time would crop an already-cropped
                # frame against intrinsics that describe the result, not the
                # input
                self.fov = 'none'
                log.info('FolderSource: frames already FOV-matched at capture '
                         'time, skipping the second match')
            log.info('FolderSource: meta.json -> fx=%.2f fy=%.2f cx=%.2f '
                     'cy=%.2f, camera_height=%s, yaw=%s',
                     self.cam_K[0, 0], self.cam_K[1, 1], self.cam_K[0, 2],
                     self.cam_K[1, 2], self.camera_height, self.yaw)
        else:
            log.warning('FolderSource: no meta.json under %r -- falling back to '
                        "NYU's intrinsics, which are only right if these frames "
                        'came from NYU', root)

        log.info('FolderSource: %d frames from %s (loop=%s)',
                 len(self.paths), depth_dir, loop)

    def _rgb_for(self, depth_path):
        if not self.rgb_dir:
            return None
        stem = os.path.splitext(os.path.basename(depth_path))[0]
        for cand in ('%s.png' % stem, '%s_colors.png' % stem,
                     '%s_colors.png' % stem.replace('_0000', '')):
            p = os.path.join(self.rgb_dir, cand)
            if os.path.exists(p):
                return cv2.imread(p)
        return None

    def frames(self):
        while True:
            for p in self.paths:
                t0 = time.time()
                raw = cv2.imread(p, cv2.IMREAD_UNCHANGED)
                if raw is None:
                    log.warning('unreadable, skipping: %s', p)
                    continue
                depth = raw.astype(np.float32) / 1000.0        # mm -> m
                rgb = self._rgb_for(p)
                K = self.cam_K
                if self.fov == 'nyu':
                    depth, rgb, K = match_nyu_fov(depth, rgb, K)
                yield Frame(depth=_require_shape(depth, p), rgb=rgb, cam_K=K,
                            name=os.path.splitext(os.path.basename(p))[0],
                            camera_height=self.camera_height, yaw=self.yaw)
                if self.period:
                    time.sleep(max(0.0, self.period - (time.time() - t0)))
            if not self.loop:
                return


# ---------------------------------------------------------------------------
def open_source(spec, rgb_dir=None, loop=False, fps=None, fov='nyu'):
    """Build a CameraSource from a spec string. See the module docstring."""
    kind, _, arg = spec.partition(':')
    kind = kind.strip().lower()

    if kind == 'realsense':
        w, h, f = 1280, 720, 30
        if arg:                                    # "1280x720@30"
            res, _, rate = arg.partition('@')
            if res:
                w, h = (int(v) for v in res.lower().split('x'))
            if rate:
                f = int(rate)
        return RealSenseSource(width=w, height=h, fps=f, fov=fov)

    if kind == 'nyu':
        return NYUSource(arg, rgb_dir=rgb_dir, loop=loop, fps=fps)

    if kind == 'folder':
        return FolderSource(arg, rgb_dir=rgb_dir, loop=loop, fps=fps, fov=fov)

    raise ValueError(
        'unknown source %r. Expected realsense[:WxH@FPS], nyu:/path, or '
        'folder:/path' % spec)
