"""
inference/run_live.py

Streaming entrypoint: D455 (or a replay source) -> TSDF -> CleanerS -> .ply.
Single process, model loaded once, one frame at a time.

    python -m inference.run_live \
        --cfg ./cfgs/NYU/voxelSSC.yaml \
        --pretrained_path ./checkpoint/CleanerS_ckpt.pth \
        --source realsense --camera_height 1.25

Test without hardware (this reproduces reference-comparable output):

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

2. `--camera_height` and level mounting are assumed. A handheld camera
   pitched down presents the scene tilted relative to the grid, and floor /
   ceiling degrade first. The D455 has a built-in IMU, so the principled fix
   is to read its accelerometer, derive the gravity vector, and build the
   rotation from that instead of assuming level. That is the single highest
   -value improvement to this path and is NOT implemented here.
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
    from .frame_loader import FrameLoader
    from . import model as model_mod
    from .visualize import save_prediction_ply
    from .utils import get_device, setup_logging, ensure_dir
except ImportError:
    from camera import open_source
    from frame_loader import FrameLoader
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
                   help='realsense[:WxH@FPS] | nyu:/path | folder:/path')
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
    p.add_argument('--rgb_dir', default=None, help='replay sources only')
    p.add_argument('--loop', action='store_true', help='replay sources only')
    p.add_argument('--fps', type=float, default=None, help='throttle replay')

    p.add_argument('--out_dir', default='./outputs')
    p.add_argument('--save', default='ply', choices=['ply', 'npy', 'both', 'none'],
                   help="'none' still reports throughput; use it to benchmark")
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

    signal.signal(signal.SIGTERM, _handle_signal)
    signal.signal(signal.SIGINT, _handle_signal)

    ply_dir = ensure_dir(os.path.join(args.out_dir, 'ply'))
    pred_dir = ensure_dir(os.path.join(args.out_dir, 'prediction'))

    log.info('loading model on %s', device)
    model = model_mod.load_model(cfg, device=device)
    _restore_log_levels()

    n = skipped = 0
    t_start = t_window = time.time()
    enc_s = inf_s = 0.0

    with open_source(args.source, rgb_dir=args.rgb_dir, loop=args.loop,
                     fps=args.fps, fov=args.fov) as src:
        log.info('source=%s fov=%s save=%s mask=%s',
                 args.source, args.fov, args.save, args.mask)
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
                                     cam_K=frame.cam_K)
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
                loader = FrameLoader.from_live_camera(
                    frame.depth, rgb=frame.rgb, cam_K=frame.cam_K,
                    camera_height=height, yaw=yaw)
            sample = loader.build_tsdf_and_mapping()
            t1 = time.time()

            pred_label, _ = model_mod.predict(model, sample, device=device)
            t2 = time.time()
            enc_s += t1 - t0
            inf_s += t2 - t1

            if args.save in ('npy', 'both'):
                np.save(os.path.join(pred_dir, '%s.npy' % frame.name), pred_label)
            if args.save in ('ply', 'both'):
                save_prediction_ply(pred_label, build_mask(sample, args.mask),
                                    os.path.join(ply_dir, '%s.ply' % frame.name))

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
    if n:
        log.info('done: %d frames in %.1fs (%.2f fps) | encode %.0f ms/frame | '
                 'infer %.0f ms/frame%s', n, total, n / total,
                 1000 * enc_s / n, 1000 * inf_s / n,
                 ' | %d skipped (no RGB)' % skipped if skipped else '')
    else:
        log.warning('no frames processed (%d skipped for missing RGB)', skipped)


if __name__ == '__main__':
    main()
