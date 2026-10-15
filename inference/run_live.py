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

Two limits remain:

1. The volume is fixed relative to the camera, so each frame is an
   independent single-shot completion, not an accumulating reconstruction.
   Fusing across frames needs real localisation (SLAM/ICP) driving cam_pose.

2. `--camera_height` is assumed; level mounting no longer is, IF the source
   supplies gravity. A .bag carries the D455's accelerometer, so `--source
   bag:` reads it per frame and builds the rotation around measured up
   (`--no_imu` to compare against assuming level). A live camera still does
   not: RealSenseSource requests depth and colour only, and wiring the accel
   stream into it is the remaining half of this.
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
    from .frame_loader import FrameLoader, torch_cuda_available
    from . import model as model_mod
    from .visualize import save_prediction_ply
    from .utils import get_device, setup_logging, ensure_dir
except ImportError:
    from camera import open_source
    from frame_loader import FrameLoader, torch_cuda_available
    import model as model_mod
    from visualize import save_prediction_ply
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
    p.add_argument('--camera_height', type=float, default=None,
                   help='metres above the floor. NYU sits at 1.18-1.32; '
                        'measure yours, it shifts the whole volume. Defaults '
                        "to the capture's meta.json when replaying one, "
                        'otherwise 1.25.')
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
                        "dominates a frame: 'torch' 116 ms, 'cuda' (SSCNet's "
                        "kernels) 1101, 'numpy' 1026. 'auto' is torch when a "
                        'GPU is visible, else numpy. See FrameLoader.')
    p.add_argument('--amp', default='auto', choices=['auto', 'on', 'off'],
                   help='mixed precision on the forward pass: 346 -> 201 ms, '
                        'for 1 voxel in 500 and the fourth decimal of the NYU '
                        "score. 'auto' is on whenever the model is on a GPU.")

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
    p.add_argument('--max_frames', type=int, default=0, help='0 = unlimited')
    p.add_argument('--log_every', type=int, default=30)
    return p.parse_known_args()


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

    signal.signal(signal.SIGTERM, _handle_signal)
    signal.signal(signal.SIGINT, _handle_signal)

    # created on first write, not now: a --save none run should leave nothing
    # behind, and the screenshot key makes its own
    ply_dir = os.path.join(args.out_dir, 'ply')
    pred_dir = os.path.join(args.out_dir, 'prediction')

    view = None
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
        view = LiveView(colour=args.colour, out_dir=args.out_dir,
                        panels=args.panels, zoom=args.zoom,
                        width=w3d, height=h3d, panel_width=args.panel_width)

    log.info('loading model on %s', device)
    model = model_mod.load_model(cfg, device=device)
    _restore_log_levels()

    n = skipped = 0
    t_start = t_window = time.time()
    enc_s = inf_s = 0.0

    with open_source(args.source, rgb_dir=args.rgb_dir, loop=args.loop,
                     fps=args.fps, fov=args.fov, imu=args.imu,
                     realtime=args.realtime) as src:
        log.info('source=%s fov=%s save=%s mask=%s | encoder=%s amp=%s%s',
                 args.source, args.fov, args.save, args.mask, encoder,
                 'on' if model_mod.use_amp(device, args.amp) else 'off',
                 ' | display colour=%s' % args.colour if view else '')
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
                if height is None:
                    height = frame.camera_height
                if height is None:
                    height = 1.25
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
            sample = loader.build_tsdf_and_mapping()
            t1 = time.time()

            pred_label, _ = model_mod.predict(model, sample, device=device,
                                              amp=args.amp)
            t2 = time.time()
            enc_s += t1 - t0
            inf_s += t2 - t1

            mask = None
            if view is not None or args.save in ('ply', 'both'):
                mask = build_mask(sample, args.mask)
            if args.save in ('npy', 'both'):
                np.save(os.path.join(ensure_dir(pred_dir),
                                     '%s.npy' % frame.name), pred_label)
            if args.save in ('ply', 'both'):
                save_prediction_ply(pred_label, mask,
                                    os.path.join(ensure_dir(ply_dir),
                                                 '%s.ply' % frame.name))
            if view is not None and not view.update(
                    pred_label, mask, loader, frame.rgb, name=frame.name,
                    fps=1.0 / max(1e-6, time.time() - t0)):
                log.info('display window closed; stopping')
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
