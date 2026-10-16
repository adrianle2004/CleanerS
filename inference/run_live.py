"""
inference/run_live.py

Streaming entrypoint: D455 (or a replay source) -> TSDF -> CleanerS -> .ply.
Single process, model loaded once, one frame at a time.

    python -m inference.run_live \
        --cfg ./cfgs/NYU/voxelSSC.yaml \
        --pretrained_path ./checkpoint/CleanerS_ckpt.pth \
        --source realsense --camera_height 1.25

Test without hardware -- a recording of the real sensor:

    python -m inference.run_live ... --source bag:captures/room07/scan/scan.bag

or NYU frames, which reproduce reference-comparable output:

    python -m inference.run_live ... --source nyu:/data/NYU/depth \
        --rgb_dir /data/NYU/test/RGB --max_frames 20

*** WHAT LIVE OUTPUT IS AND IS NOT ***
NYU frames carry a per-scene vox_origin/cam_pose from a room-canonical
registration step. A live sensor has none, so from_live_camera builds a
gravity-aligned, floor-anchored grid: 4.8 m wide x 4.8 m deep x 2.88 m tall,
world Z=0 at the floor, camera at `--camera_height`. That reproduces NYU's
axis convention, which the encoder depends on -- the 144-cell axis must be
vertical.

One limit remains, and one was removed:

1. REMOVED, by `--accumulate`. The per-frame volume is still fixed to the
   camera, but it no longer has to be the thing you look at. With the flag,
   a tracker in this loop (inference/tracker.py, the tracking half of
   reconstruction_GT/fuse_scan.py) supplies world_T_cam, and every frame is
   gathered into ONE room-sized 8 cm grid that fills as the camera moves
   (inference/world_map.py). The display then separates what the sensor has
   MEASURED from what the model COMPLETED, and says which completions the
   sweep went on to prove. Without the flag nothing about a run changes.

   It needs a measured --camera_height: the world is anchored on the first
   frame and cannot be moved afterwards, so `auto` is refused.

2. Level mounting is no longer assumed, IF the source supplies gravity.
   Both `bag:` and `realsense` request the D455's accelerometer and build
   the rotation around measured up, per frame (`--no_imu` to compare against
   assuming level). Verified on the device: a D455 (fw 5.17.0.10) reports
   gravity on every frameset -- accel 60 of 60 samples used -- reading 1.4
   deg of pitch on a rig that looked level.

   `--camera_height` is still a number you supply, or `auto`, which measures
   the lowest level surface in view and keeps the lowest one the run has
   seen. It needs gravity for the same reason the grid does: without it a
   level bed is indistinguishable from a level floor. See AutoHeight.
"""

import argparse
import logging
import os
import signal
import sys
import time

import numpy as np
import torch

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_THIS_DIR)
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from cleaner.utils import EasyConfig

try:
    from .camera import open_source
    from .frame_loader import (FrameLoader, torch_cuda_available,
                               camera_height_from_depth)
    from . import model as model_mod
    from .visualize import save_prediction_ply, save_world_ply
    from .world_map import WorldMap, observe, gather_prediction
    from .tracker import Tracker
    from .utils import get_device, setup_logging, ensure_dir
except ImportError:
    from camera import open_source
    from frame_loader import (FrameLoader, torch_cuda_available,
                              camera_height_from_depth)
    import model as model_mod
    from visualize import save_prediction_ply, save_world_ply
    from world_map import WorldMap, observe, gather_prediction
    from tracker import Tracker
    from utils import get_device, setup_logging, ensure_dir

log = logging.getLogger('run_live')
_STOP = False


def _restore_log_levels(level=logging.INFO):
    """mmcv's get_logger (mmcv/utils/logging.py, 1.5.0) walks
    `logger.root.handlers` and forces every StreamHandler to ERROR -- it does
    this to mute duplicate output on non-zero ranks during distributed
    training. Importing the model triggers it, so without this call every log
    line after load_model() silently vanishes, including the throughput
    numbers and the 'no frames processed' warning.
    """
    for h in logging.root.handlers:
        if isinstance(h, logging.StreamHandler):
            h.setLevel(level)
    logging.root.setLevel(level)


def _handle_signal(signum, _frame):
    """SIGTERM/SIGINT: finish the current frame and close the sensor cleanly
    -- a RealSense left running can need a USB replug before it will open
    again."""
    global _STOP
    log.info('%s received; finishing current frame then shutting down',
             signal.Signals(signum).name)
    _STOP = True


def parse_args():
    p = argparse.ArgumentParser('CleanerS streaming inference')
    p.add_argument('--cfg', default='./cfgs/NYU/voxelSSC.yaml')
    p.add_argument('--pretrained_path', required=True)

    p.add_argument('--source', default='realsense',
                   help='realsense[:WxH@FPS] | bag:/path/scan.bag | '
                        'nyu:/path | folder:/path')
    p.add_argument('--fov', default='nyu', choices=['nyu', 'native'],
                   help="'nyu' resizes+crops to NYU's 63x49 deg intrinsics "
                        '(recommended: the checkpoint was trained there). '
                        "'native' keeps the D455's ~90 deg frame.")
    p.add_argument('--camera_height', type=_height, default=None,
                   metavar='M|auto',
                   help='metres above the floor. NYU sits at 1.18-1.32; '
                        'measure yours, it shifts the whole volume. Defaults '
                        "to the capture's meta.json when replaying one, "
                        "otherwise 1.25. 'auto' measures it from the frames "
                        'themselves -- the lowest level surface in view, '
                        'needs the accelerometer, costs 17 ms a frame, and '
                        'keeps looking for something lower as the run goes '
                        'on.')
    p.add_argument('--yaw', type=float, default=None,
                   help='radians about the vertical axis; defaults to '
                        "meta.json's value, otherwise 0")
    p.add_argument('--no_imu', dest='imu', action='store_false',
                   help='bag replay: ignore the recorded accelerometer and '
                        'assume the camera was level, as every other source '
                        'has to. Only for comparing against it -- a handheld '
                        'sweep is not level, and the grid is built around up.')
    p.add_argument('--rgb_dir', default=None, help='replay sources only')
    p.add_argument('--loop', action='store_true', help='replay sources only')
    p.add_argument('--fps', type=float, default=None, help='throttle replay')
    p.add_argument('--realtime', action='store_true',
                   help='bag replay: play at the recorded rate and DROP the '
                        'frames we are too slow to take, which is what a live '
                        'camera does. Without it a recording waits for us and '
                        'every frame is processed.')

    p.add_argument('--encoder', default='auto',
                   choices=['auto', 'torch', 'numpy', 'cuda'],
                   help="which arithmetic builds the TSDF, the stage that "
                        "dominates a frame: 'torch' ~90 ms, 'cuda' "
                        "(SSCNet's kernels) ~1.1 s, 'numpy' ~1.0 s. 'auto' is "
                        'torch when a GPU is visible, else numpy. See '
                        'FrameLoader and document/HARDWARE.md.')
    p.add_argument('--amp', default='auto', choices=['auto', 'on', 'off'],
                   help='mixed precision on the forward pass: ~350 -> '
                        '~205 ms, for 1 voxel in 500 and the fourth decimal '
                        "of the NYU score. 'auto' is on whenever the model is "
                        'on a GPU.')

    p.add_argument('--display', action='store_true',
                   help='open a live 3D window (inference/live_view.py) and '
                        'replace the voxels in it every frame. Orbit and zoom '
                        'with the mouse; the viewpoint survives the updates. '
                        'Closing the window stops the run.')
    p.add_argument('--zoom', type=float, default=2.0,
                   help='how much of the window the volume fills. 1.0 is '
                        "open3d's own fit, which frames the bounding SPHERE "
                        'and so sits about twice as far back as it needs to; '
                        '1.7 fills roughly 90%% of the height. Applied to the '
                        'geometry, not the camera -- see live_view.py.')
    p.add_argument('--window', default='1280x800', metavar='WxH',
                   help='size of the 3D window (default 1280x800)')
    p.add_argument('--panel_width', type=int, default=420,
                   help='width of the colour/depth window beside it. It is '
                        'resizable -- this only sets where it starts. 420 + '
                        "the 3D window's 1280 fits a 1728-wide desktop.")
    p.add_argument('--sync_display', action='store_true',
                   help='draw each frame in its own iteration instead of one '
                        'behind. Costs the ~75-130 ms of painting that would '
                        'otherwise hide behind the next frame\'s inference '
                        '(~20%% with --colour all), and buys back a frame of '
                        'display lag.')
    p.add_argument('--no_panels', dest='panels', action='store_false',
                   help='--display only: drop the second window showing the '
                        'colour and depth the model was given')
    p.add_argument('--colour', '--color', dest='colour', default='zbuffer',
                   choices=['class', 'zbuffer', 'all'],
                   help="what --display paints the voxels with: 'class' the "
                        "semantic colour (free), 'zbuffer' the photo pixel "
                        'each voxel is front-most on (17 ms), '
                        "'all' the three photo rules stacked, best evidence "
                        'first (98 ms). `c` in the window cycles them.')

    p.add_argument('--out_dir', default='./outputs/live_session',
                   help='predictions go to <out_dir>/prediction, voxel plys to '
                        '<out_dir>/ply, window screenshots to <out_dir>. '
                        'Nothing is created unless something is written. NOTE '
                        'that live frames are named live_000000 upward from '
                        'each run, so a second run into the same folder '
                        'overwrites the first.')
    p.add_argument('--save', default='none',
                   choices=['ply', 'npy', 'both', 'none'],
                   help="default 'none': a live run is usually watched, not "
                        'kept, and writing a .ply costs 83 ms of a 320 ms '
                        'frame. Ask for it when you want it.')
    p.add_argument('--mask', default='surface',
                   choices=['surface', 'occluded', 'frustum'],
                   help="which voxels the .ply shows. The GT-derived "
                        "label_weight mask that test_NYU.py uses is NOT "
                        "available live -- there is no ground truth. 'surface' "
                        'is the closest GT-free stand-in (0.9748 agreement on '
                        'NYU).')
    p.add_argument('--accumulate', action='store_true',
                   help='fill ONE room-sized grid as the camera moves, '
                        'instead of replacing the voxels every frame. The '
                        'pose comes from a tracker in this loop, the world '
                        'is anchored on the first frame, and the display '
                        'then separates what the sensor has MEASURED from '
                        'what the model COMPLETED. Needs a measured '
                        '--camera_height.')
    p.add_argument('--world_extent', default='16x16x3.04', metavar='XxYxZ',
                   help='the accumulated box in metres (default '
                        '16x16x3.04, i.e. 200x200x38 voxels, ~80 MB)')
    p.add_argument('--view_extent', default='8x8x3.04', metavar='XxYxZ',
                   help='--accumulate: what the OPENING view is framed on, '
                        'metres. Separate from --world_extent, which only '
                        'has to be big enough for the camera to walk '
                        'around in; this only has to show the room. The '
                        'orbit pivot stays the map centre either way.')
    p.add_argument('--meas_range', type=float, default=3.5,
                   help='how far the depth frame is believed, metres. The '
                        'prediction reaches 4.8 m, so beyond this a '
                        'completion stays untested until you walk to it.')
    p.add_argument('--track_voxel', type=float, default=0.02,
                   help='the tracker only has to track, not make a mesh')
    p.add_argument('--track_blocks', type=int, default=8000)
    p.add_argument('--track_fps', type=float, default=None,
                   help='the rate frames really REACH the tracker, if you '
                        'know it. It only sets how big a jump is judged '
                        'impossible; without it a generous per-frame '
                        'allowance is used, because a guessed rate is '
                        'worse than none -- see tracker.DT_FALLBACK_S')
    p.add_argument('--world_colour', '--world_color', dest='world_colour',
                   default='photo', choices=['photo', 'verdict', 'class'],
                   help="--accumulate: which question the room's colours "
                        "answer. 'photo' (default) is what it LOOKS like "
                        "-- every voxel in the camera's own colour, the "
                        "same rule as --colour all; 'verdict' is whether "
                        "the model got it right; 'class' is what the "
                        "model thinks each thing is. `c` cycles them in "
                        "the window.")
    p.add_argument('--world_shrink', type=_shrink, default=None,
                   metavar='auto|FRACTION',
                   help='--accumulate: how big the voxels only the MODEL '
                        'claims are drawn, as a fraction of full size. '
                        "'auto' (default) decides per colour mode: whole "
                        'in photo, so the room looks like a room, and '
                        'smaller in verdict, where whole cubes hide the '
                        'comparison -- culling shows only the outermost '
                        "shell and that shell is the model's unverified "
                        'claims wrapped around the real room.')
    p.add_argument('--mesh_every', type=int, default=3,
                   help='--accumulate: rebuild the room mesh every N '
                        'frames. The room changes by a few voxels a frame '
                        'and the window stays responsive in between.')
    p.add_argument('--record', action='store_true',
                   help='--accumulate: write every drawn frame to '
                        '<out_dir>/frames/, to make the fill-in video')
    p.add_argument('--track_device', default='auto',
                   choices=['auto', 'cpu', 'cuda'],
                   help='cpu if the card cannot hold the tracker and the '
                        'model at once')
    p.add_argument('--max_frames', type=int, default=0, help='0 = unlimited')
    p.add_argument('--log_every', type=int, default=30)
    return p.parse_known_args()


def _shrink(v):
    """--world_shrink takes a fraction or the word auto."""
    if str(v).strip().lower() == 'auto':
        return None
    try:
        f = float(v)
    except ValueError:
        raise argparse.ArgumentTypeError(
            'world shrink must be a fraction or "auto", not %r' % (v,))
    if not 0.05 <= f <= 1.0:
        raise argparse.ArgumentTypeError(
            'world shrink must be between 0.05 and 1.0, not %g' % f)
    return f


def _height(v):
    """--camera_height takes metres or the word auto."""
    if str(v).strip().lower() == 'auto':
        return 'auto'
    try:
        return float(v)
    except ValueError:
        raise argparse.ArgumentTypeError(
            'camera height must be metres or "auto", not %r' % (v,))


class AutoHeight:
    """--camera_height auto: the lowest level surface the run has seen.

    The floor is the lowest level thing in a room, so the estimate is a
    running MAXIMUM of per-frame fits rather than an average -- which is also
    why every fit is gated hard before it counts (`camera_height_from_depth`:
    level against measured gravity, 3000+ inliers, within 2.5 m of the lens).
    On room07's recorded sweep exactly one frame of 808 sees the floor
    properly, and it answers 1.111 m against the fused sweep's 1.086 and the
    tape's 1.150. Taking an average, or asking for two frames to agree, gives
    1.006 -- 8 cm low -- because what the other 500 frames can see is the bed.

    It needs the accelerometer. Without gravity there is no way to tell a
    level floor from a level bed, and the same room07 frames return the bed at
    0.52 m, which is worse than assuming 1.25. So without it this refuses and
    says so, rather than quietly handing the encoder a number half a metre out.
    """

    def __init__(self):
        self.value = None
        self.frames = 0
        self.fits = 0
        self.at = None
        self.info = None
        self.said_no_imu = False
        self.said_waiting = False

    def update(self, frame):
        """-> the best height so far, or None while there is none."""
        self.frames += 1
        if frame.up_camera is None:
            if not self.said_no_imu:
                self.said_no_imu = True
                log.warning('--camera_height auto needs measured gravity and '
                            'this source has none, so it cannot tell a floor '
                            'from a bed. Falling back; pass the height you '
                            'measured instead.')
            return self.value
        h, info = camera_height_from_depth(
            frame.depth, frame.cam_K, frame.up_camera)
        if h is None:
            if self.value is None and not self.said_waiting:
                self.said_waiting = True
                log.info('--camera_height auto: no floor yet (%s). Aim lower; '
                         'using the fallback until then.',
                         info.get('error', 'no fit'))
            return self.value
        self.fits += 1
        if self.value is None or h > self.value + 0.02:
            self.value, self.at, self.info = h, self.frames, info
            log.info('--camera_height auto: %.3f m (frame %d, floor plane '
                     '%.1f deg off level, %d inliers)', h, self.frames,
                     info['plane_tilt_vs_gravity_deg'], info['inliers'])
        return self.value

    def report(self):
        if self.value is None:
            if self.frames and not self.said_no_imu:
                return ('camera height auto: never found a floor in %d frames '
                        '-- measure it and pass --camera_height' % self.frames)
            return None
        return ('camera height auto: %.3f m, from frame %d of %d (%d frames '
                'gave a usable floor). Record it in meta.json rather than '
                'measuring it again.'
                % (self.value, self.at, self.frames, self.fits))


def load_cfg(args, opts):
    cfg = EasyConfig()
    cfg.load(args.cfg, recursive=True)
    cfg.update(opts)
    cfg.pretrained_path = args.pretrained_path
    cfg.rank = 0 if torch.cuda.is_available() else 'cpu'
    return cfg


def build_mask(sample, mode):
    """GT-free display masks only -- see --mask help."""
    tsdf = np.asarray(sample['tsdf']).reshape(60, 36, 60)
    observed = np.asarray(sample['weight']).reshape(60, 36, 60).astype(bool)
    occluded = tsdf < -0.5
    if mode == 'occluded':
        return occluded
    if mode == 'frustum':
        return observed
    return occluded | (observed & (np.abs(tsdf) < 0.25))


def main():
    setup_logging()
    args, opts = parse_args()
    cfg = load_cfg(args, opts)
    device = get_device()
    encoder = args.encoder
    if encoder == 'auto':
        encoder = 'torch' if torch_cuda_available() else 'numpy'

    if args.accumulate and args.camera_height == 'auto':
        raise SystemExit(
            '--accumulate cannot use --camera_height auto. The world frame '
            'is anchored on the FIRST frame and everything is fused into '
            'it, and auto has no answer yet at that point -- on room07 '
            'exactly one frame in 808 ever sees the floor properly, so the '
            'height would be revised later and move the world out from '
            'under everything already in it. Measure the height and pass '
            'it, or replay a capture whose meta.json carries one.')

    signal.signal(signal.SIGTERM, _handle_signal)
    signal.signal(signal.SIGINT, _handle_signal)

    # created on first write, not now: a --save none run should leave nothing
    # behind, and the screenshot key makes its own
    ply_dir = os.path.join(args.out_dir, 'ply')
    pred_dir = os.path.join(args.out_dir, 'prediction')

    # The window is built AFTER the model and the source, just before the
    # loop. A window only answers the compositor while poll_events is being
    # called, and nothing calls it until the first frame: opening it here
    # would leave it mute through a ~4 s model load, the camera's warm-up and
    # a first frame's cudnn autotune, which is how you get "not responding --
    # force quit or wait". The size is still parsed now, so a typo fails
    # before any of that rather than after.
    view = None
    LiveView = w3d = h3d = None
    if args.display:
        try:
            from .live_view import LiveView
        except ImportError:
            from live_view import LiveView
        try:
            w3d, h3d = (int(v) for v in args.window.lower().split('x'))
        except ValueError:
            raise SystemExit('--window wants WxH, e.g. 1280x800, not %r'
                             % args.window)

    log.info('loading model on %s', device)
    model = model_mod.load_model(cfg, device=device)
    _restore_log_levels()

    n = skipped = 0
    t_start = t_window = time.time()
    enc_s = inf_s = 0.0
    said_height = False        # the 1.25 m fallback is worth saying out loud
    auto_height = AutoHeight()
    world = tracker = None
    track_s = fuse_s = 0.0
    untracked = 0
    poses = []

    with open_source(args.source, rgb_dir=args.rgb_dir, loop=args.loop,
                     fps=args.fps, fov=args.fov, imu=args.imu,
                     realtime=args.realtime) as src:
        if LiveView is not None:
            view = LiveView(colour=args.colour, out_dir=args.out_dir,
                            panels=args.panels, zoom=args.zoom,
                            width=w3d, height=h3d,
                            panel_width=args.panel_width,
                            overlap=not args.sync_display)
            if args.accumulate:
                view.mesh_every = args.mesh_every
                view.world_colour = args.world_colour
                view.shrink = args.world_shrink
                view.view_extent = tuple(
                    float(v) for v in args.view_extent.lower().split('x'))
                if args.record:
                    view.record_dir = os.path.join(args.out_dir, 'frames')
            view.pump()               # show it before the first frame's work
        log.info('source=%s fov=%s save=%s mask=%s | encoder=%s amp=%s%s',
                 args.source, args.fov, args.save, args.mask, encoder,
                 'on' if model_mod.use_amp(device, args.amp) else 'off',
                 ' | display colour=%s%s' % (
                     args.colour,
                     '' if args.sync_display or args.colour == 'class'
                     else ', one frame behind') if view else '')
        for frame in src.frames():
            if _STOP:
                break
            if frame.rgb is None:
                skipped += 1
                if skipped <= 3:
                    log.warning('[%s] no colour frame; the model consumes RGB, '
                                'skipping', frame.name)
                continue

            t0 = time.time()
            if frame.vox_origin is not None:
                # replay of a dataset frame that carries its own registration
                loader = FrameLoader(frame.depth, frame.vox_origin,
                                     frame.cam_pose, rgb=frame.rgb,
                                     cam_K=frame.cam_K, encoder=encoder,
                                     device=device)
            else:
                # explicit CLI beats the capture's record, which beats the
                # default -- so replaying a capture reproduces the
                # single-frame run_inference result without re-typing anything
                height = args.camera_height
                if height == 'auto':
                    height = auto_height.update(frame)
                if height is None:
                    height = frame.camera_height
                if height is None:
                    height = 1.25
                    if not said_height:
                        said_height = True
                        # world Z=0 is the floor and the volume is only
                        # 2.88 m tall, so this number is the only thing
                        # saying where the floor is. 15 cm out moves ~14%
                        # of predicted voxels -- document/HARDWARE.md
                        log.warning('no camera height given and the source '
                                    'has none: ASSUMING %.2f m. Measure it '
                                    'and pass --camera_height; 15 cm out '
                                    'changes about 14%% of the prediction.',
                                    height)
                yaw = args.yaw
                if yaw is None:
                    yaw = frame.yaw
                if yaw is None:
                    yaw = 0.0
                # measured gravity when the source has it (a .bag carries the
                # accel stream); without it the grid assumes a level camera,
                # and a pitched one puts a surface 3 m away 21 cm from where
                # the grid draws it
                loader = FrameLoader.from_live_camera(
                    frame.depth, rgb=frame.rgb, cam_K=frame.cam_K,
                    camera_height=height, yaw=yaw, up_camera=frame.up_camera,
                    encoder=encoder, device=device)
            world_T_cam = None
            if args.accumulate:
                if world is None:
                    ex = tuple(float(v) for v in
                               args.world_extent.lower().split('x'))
                    world = WorldMap.around((0.0, 0.0), extent=ex)
                    tracker = Tracker(
                        frame.cam_K, frame.depth.shape[1],
                        frame.depth.shape[0],
                        initial_pose=loader.cam_pose.astype(np.float64),
                        fps=args.track_fps, voxel=args.track_voxel,
                        block_count=args.track_blocks,
                        device=args.track_device)
                    log.info('accumulating into %dx%dx%d voxels of %.0f cm '
                             '(%.1f x %.1f x %.1f m), world anchored on '
                             'this frame', world.shape[0], world.shape[1],
                             world.shape[2], 100 * world.vox, ex[0], ex[1],
                             ex[2])
                ta = time.time()
                world_T_cam = tracker.step(frame.depth, frame.rgb)
                track_s += time.time() - ta
                if world_T_cam is None:
                    # a bad pose does not degrade the map, it permanently
                    # corrupts it -- so an unsure frame touches nothing
                    untracked += 1
                    continue
                poses.append(world_T_cam.copy())
                ta = time.time()
                hit, hit_rgb, free = observe(
                    world, frame.depth, frame.cam_K, world_T_cam,
                    max_range=args.meas_range, rgb=frame.rgb)
                world.add_measurement(hit, free, n, rgb=hit_rgb)
                fuse_s += time.time() - ta

            sample = loader.build_tsdf_and_mapping()
            t1 = time.time()

            if view is not None:
                view.pump()           # the encode just took ~100-900 ms
            t1b = time.time()         # the pump is display cost, not infer
            pred_label, _ = model_mod.predict(model, sample, device=device,
                                              amp=args.amp)
            t2 = time.time()
            enc_s += t1 - t0
            inf_s += t2 - t1b

            mask = None
            if view is not None or args.save in ('ply', 'both') \
                    or args.accumulate:
                mask = build_mask(sample, args.mask)
            if args.accumulate:
                ta = time.time()
                widx, wcls = gather_prediction(
                    world, pred_label, mask, loader.vox_origin,
                    loader.cam_pose.astype(np.float64), world_T_cam)
                world.add_prediction(widx, wcls, n)
                fuse_s += time.time() - ta
            if args.save in ('npy', 'both'):
                np.save(os.path.join(ensure_dir(pred_dir),
                                     '%s.npy' % frame.name), pred_label)
            if args.save in ('ply', 'both'):
                save_prediction_ply(pred_label, mask,
                                    os.path.join(ensure_dir(ply_dir),
                                                 '%s.ply' % frame.name))
            if view is not None:
                hz = 1.0 / max(1e-6, time.time() - t0)
                if args.accumulate:
                    alive = view.update_world(
                        world, frame.depth, frame.rgb, name=frame.name,
                        fps=hz,
                        note='TRACKING LOST -- the map stopped growing'
                             if tracker.lost else None)
                else:
                    alive = view.update(pred_label, mask, loader, frame.rgb,
                                        name=frame.name, fps=hz)
                if not alive:
                    log.info('display window closed; stopping')
                    n += 1
                    break
                # SPACE holds the room exactly as it is. The wait sits
                # here, AFTER a frame rather than before one, so nothing
                # is pulled from the source while it holds: a camera's
                # frames are dropped by the driver as they always are,
                # and a replay carries on at the frame it stopped at
                # instead of skipping ahead. The window keeps answering
                # the compositor throughout, so the state can be orbited.
                if view.frozen:
                    log.info('frozen at %s -- press space in the window '
                             'to carry on, or close it to stop here',
                             frame.name)
                    view.update_world(world, frame.depth, frame.rgb,
                                      name=frame.name, force=True,
                                      note='FROZEN -- space to carry on')
                    while view.frozen and not _STOP:
                        if not view.pump():
                            break
                        time.sleep(0.03)
                    if view.closed or _STOP:
                        n += 1
                        break

            n += 1
            if args.log_every and n % args.log_every == 0:
                dt = time.time() - t_window
                log.info('%d frames | %.1f fps | encode %.0f ms | infer %.0f ms',
                         n, args.log_every / dt if dt > 0 else 0.0,
                         1000 * enc_s / n, 1000 * inf_s / n)
                t_window = time.time()
            if args.max_frames and n >= args.max_frames:
                break

    total = time.time() - t_start
    said = src.report()
    if said:
        log.info('%s', said)
    said = auto_height.report()
    if said:
        log.info('%s', said)
    if world is not None:
        log.info('%s', tracker.report())
        if untracked:
            log.info('%d frames contributed nothing: the tracker was not '
                     'sure enough of the pose to risk the map', untracked)
        c = world.counts()
        settled = c['confirmed'] + c['contradicted']
        log.info('world: %d voxels measured, %d predicted occupied',
                 c['measured'], c['predicted'])
        log.info('blind completions -- filled in where NOTHING had been '
                 'measured yet: %d. The sweep later reached %d of them: '
                 '%d confirmed, %d contradicted%s. %d are still untested '
                 '(inside furniture, behind walls -- the sensor can never '
                 'reach most of them, and that is not a win).',
                 c['confirmed'] + c['contradicted'] + c['untested'],
                 settled, c['confirmed'], c['contradicted'],
                 ' (%.1f%% right)' % (100.0 * c['confirmed'] / settled)
                 if settled else '', c['untested'])
        log.info('%d voxels the sensor measured were never predicted',
                 c['missed'])
        if n:
            log.info('accumulate cost: track %.0f ms/frame, fuse %.0f '
                     'ms/frame', 1000 * track_s / n, 1000 * fuse_s / n)
        out = ensure_dir(args.out_dir)
        # the poses too: inference/check_world_map.py compares them with
        # the trajectory fuse_scan already committed for this capture
        np.savez_compressed(os.path.join(out, 'poses.npz'),
                            poses=np.asarray(poses))
        world.save(os.path.join(out, 'world.npz'))
        save_world_ply(world, os.path.join(out, 'world.ply'))
        log.info('wrote %s/world.npz, world.ply and poses.npz', out)

    if view is not None:
        log.info('%s', view.report())
        view.close()
    if n:
        log.info('done: %d frames in %.1fs (%.2f fps) | encode %.0f ms/frame | '
                 'infer %.0f ms/frame%s', n, total, n / total,
                 1000 * enc_s / n, 1000 * inf_s / n,
                 ' | %d skipped (no RGB)' % skipped if skipped else '')
    else:
        log.warning('no frames processed (%d skipped for missing RGB)', skipped)


if __name__ == '__main__':
    main()
