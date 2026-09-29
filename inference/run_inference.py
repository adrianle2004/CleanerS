"""
inference/run_inference.py

Main entry point: depth (+RGB) -> TSDF/mapping -> model prediction -> .ply export.

Config loading mirrors test_NYU.py's __main__ block:
    cfg = EasyConfig()
    cfg.load(args.cfg, recursive=True)
    cfg.update(opts)
but DROPS everything that's irrelevant to single-frame inference: distributed
setup, experiment-directory/log generation, and the whole dataloader/dataset
config -- we replace that entirely with FrameLoader, since we're not reading
from their NYU Dataset class at all.

Usage (single frame):
    cd /path/to/CleanerS   # so `cleaner` package is importable
    python -m inference.run_inference \
        --cfg ./cfgs/NYU/voxelSSC.yaml \
        --pretrained_path ./checkpoint/CleanerS_ckpt.pth \
        --bin_path ./data/NYU/testset/depth/NYU0001_0000.bin \
        --png_path ./data/NYU/testset/depth/NYU0001_0000.png \
        --rgb_path ./data/NYU/testset/RGB/NYU0001_colors.png \
        --out_dir ./outputs

Usage (batch -- every matched .bin/.png pair in a folder, paired by their
shared 7-char frame id, NOT by sorted-list-position):
    python -m inference.run_inference \
        --cfg ./cfgs/NYU/voxelSSC.yaml \
        --pretrained_path ./checkpoint/CleanerS_ckpt.pth \
        --data_dir ./data/NYU/testset/depth \
        --rgb_dir ./data/NYU/testset/RGB \
        --out_dir ./outputs

*** STATUS ***
frame_loader.py: validated geometry, ready to use.
model.py: uses your real build_model_from_cfg/load_checkpoint calls verbatim;
the one open question is whether forward() truly ignores the label3d/
label_weight/tsdf_CAD placeholders we pass for a no-ground-truth custom scene
(see model.py's docstring).
visualize.py: verified direct port of labeled_voxel2ply/get_xyz.
"""

import os
import sys
import json
import logging
import argparse
import cv2
import numpy as np
import torch

# Make this runnable both as `python -m inference.run_inference` (from the
# repo root) AND as `python run_inference.py` (from inside inference/) --
# the latter needs the repo root added to sys.path manually, since Python
# only searches the script's own folder by default.
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_THIS_DIR)
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from cleaner.utils import EasyConfig   # same import test_NYU.py uses

try:
    from .frame_loader import FrameLoader, frame_meta
    from . import model as model_mod
    from .visualize import save_prediction_ply
    from .utils import get_device, setup_logging, ensure_dir, find_frame_pairs
except ImportError:
    # fallback when run directly (no parent package context)
    from frame_loader import FrameLoader, frame_meta
    import model as model_mod
    from visualize import save_prediction_ply
    from utils import get_device, setup_logging, ensure_dir, find_frame_pairs


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--cfg', type=str, default='./cfgs/NYU/voxelSSC.yaml',
                    help="same config file test_NYU.py uses")
    p.add_argument('--pretrained_path', type=str, required=True,
                    help="e.g. ./checkpoint/CleanerS_ckpt.pth")

    # --- mode 1: a single frame ---
    p.add_argument('--bin_path', type=str, default=None)
    p.add_argument('--png_path', type=str, default=None)
    p.add_argument('--rgb_path', type=str, default=None,
                    help="required if the model actually consumes RGB features")

    # --- mode 2: batch over a folder, paired by frame id (same logic as
    # generate_cleaners_data.py's find_frame_pairs -- NOT sorted-list-position) ---
    p.add_argument('--data_dir', type=str, default=None,
                    help="folder containing NYUxxxx_0000.bin / .png pairs; "
                         "processes every matched pair found. Mutually "
                         "exclusive with --bin_path/--png_path.")
    p.add_argument('--rgb_dir', type=str, default=None,
                    help="optional folder of {frame_id}_colors.png files, used "
                         "with --data_dir batch mode")

    # --- mode 3: a folder written by inference/capture.py (live camera) ---
    p.add_argument('--live_dir', type=str, default=None,
                    help="folder produced by `python -m inference.capture`: "
                         "depth/*.png (plain uint16 mm), rgb/*.png, meta.json. "
                         "There is no .bin header for a live frame, so the "
                         "grid comes from from_live_camera instead.")
    p.add_argument('--camera_height', type=float, default=None,
                    help="override meta.json's camera_height (metres to floor)")
    p.add_argument('--yaw', type=float, default=None,
                    help="override meta.json's yaw (radians)")

    p.add_argument('--out_dir', type=str, default='./outputs')
    p.add_argument('--overwrite', action='store_true',
                   help='redo frames that already have a prediction; without '
                        'it they are skipped, so an interrupted run resumes')
    p.add_argument('--mask', type=str, default='auto',
                    choices=['auto', 'label_weight', 'occluded', 'surface', 'frustum'],
                    help="which voxels the .ply shows; see build_vis_mask. "
                         "'auto' uses the GT-derived label_weight when the TSDF "
                         "npz is available (exact match to the reference plys), "
                         "otherwise falls back to 'surface'.")
    p.add_argument('--tsdf_dir', type=str, default='../data/NYU/Custom_TSDF',
                    help="where to read arr_1 (label_weight) for --mask label_weight/auto")
    args, opts = p.parse_known_args()

    modes = [args.live_dir is not None,
             args.data_dir is not None,
             args.bin_path is not None or args.png_path is not None]
    if sum(modes) == 0:
        p.error("give one of: --live_dir, --data_dir, or both --bin_path and --png_path")
    if sum(modes) > 1:
        p.error("--live_dir, --data_dir and --bin_path/--png_path are mutually exclusive")
    if args.live_dir is None and args.data_dir is None and (
            args.bin_path is None or args.png_path is None):
        p.error("--bin_path and --png_path must be given together")

    return args, opts


def load_cfg(args, opts):
    cfg = EasyConfig()
    cfg.load(args.cfg, recursive=True)
    cfg.update(opts)
    cfg.pretrained_path = args.pretrained_path
    # cfg.rank is used as a device index in test_NYU.py (`.to(cfg.rank)`);
    # for single-process inference just resolve a real device instead.
    cfg.rank = 0 if torch.cuda.is_available() else 'cpu'
    return cfg


def build_vis_mask(sample, frame_name, mode, tsdf_dir):
    """Choose which voxels the .ply shows. This is ONLY a display choice -- it
    never touches the prediction, which is always saved in full to prediction/.

    test_NYU.py masks with `label_weight` (visualize_3d_predict, test_NYU.py:116),
    which is ground-truth-derived: GT-occupied | (tsdf < -0.5). There is no GT for
    a live camera, hence the modes below.

      label_weight : read arr_1 from {tsdf_dir}/{id}.npz -- reproduces the
                     reference .ply exactly (verified: 1.0000 class agreement,
                     33301/33302 occupied voxels over 5 frames).
      occluded     : tsdf < -0.5, the GT-free half of label_weight. Strict
                     subset of it, so no false positives, but it drops the
                     visible surface (~half the voxels).
      surface      : occluded | (observed & |tsdf| < 0.25) -- adds the visible
                     shell back. Closest GT-free match at 0.9748 agreement and
                     32246/33302 occupied.
      frustum      : every observed voxel. Shows ~2.7x too much; kept only for
                     debugging what the model does outside the scored region.
      auto         : label_weight when the npz exists, else surface.
    """
    tsdf = np.asarray(sample['tsdf']).reshape(60, 36, 60)
    observed = np.asarray(sample['weight']).reshape(60, 36, 60).astype(bool)
    occluded = tsdf < -0.5

    if mode in ('auto', 'label_weight'):
        # 'NYU0001_0000' -> '0001'; matches CleanerS's item[3:] convention
        lw_path = os.path.join(tsdf_dir, f'{frame_name[3:7]}.npz')
        if os.path.exists(lw_path):
            npz = np.load(lw_path)
            if 'arr_1' in npz.files:
                return npz['arr_1'].reshape(60, 36, 60).astype(bool), 'label_weight'
        if mode == 'label_weight':
            raise FileNotFoundError(
                f"--mask label_weight needs arr_1 from {lw_path!r}, which is missing. "
                f"Use --mask surface for frames without ground truth."
            )

    if mode == 'occluded':
        return occluded, 'occluded'
    if mode == 'frustum':
        return observed, 'frustum'
    return occluded | (observed & (np.abs(tsdf) < 0.25)), 'surface'


def process_one_frame(bin_path, png_path, rgb_path, model, device, pred_dir, ply_dir,
                      mask_mode='auto', tsdf_dir=None):
    frame_name = os.path.splitext(os.path.basename(bin_path))[0]

    loader = FrameLoader.from_nyu_bin(bin_path, png_path, rgb_path=rgb_path)
    sample = loader.build_tsdf_and_mapping()

    pred_label, _ = model_mod.predict(model, sample, device=device)
    mask, used = build_vis_mask(sample, frame_name, mask_mode, tsdf_dir)

    np.save(os.path.join(pred_dir, f'{frame_name}.npy'), pred_label)
    ply_path = os.path.join(ply_dir, f'{frame_name}.ply')
    save_prediction_ply(pred_label, mask, ply_path)
    print(f"[{frame_name}] mask={used} ({100 * mask.mean():.1f}% of grid), "
          f"occupied={int(((np.asarray(pred_label).reshape(60,36,60) > 0) & mask).sum())} "
          f"-> {ply_path}")


def load_live_meta(live_dir, args):
    """Read capture.py's meta.json. Per-frame values come from
    resolve_frame_camera: a capture may hold frames shot from elsewhere."""
    meta_path = os.path.join(live_dir, 'meta.json')
    if not os.path.exists(meta_path):
        raise FileNotFoundError(
            "%r has no meta.json. --live_dir expects a folder written by "
            "`python -m inference.capture`; without the sidecar there is no "
            "record of the camera's intrinsics, and NYU's defaults would "
            "misplace every unprojected point." % live_dir)
    with open(meta_path) as f:
        return json.load(f)


def resolve_frame_camera(meta, stem, args):
    """The intrinsics and pose for ONE frame: the capture's, overridden by
    whatever that frame states (a frame lifted out of the sweep was shot from
    its own height and tilt -- frame_loader.frame_meta), then by the flags."""
    m = frame_meta(meta, stem)
    cam_K = np.asarray(m['cam_K'], dtype=np.float32)
    height = args.camera_height if args.camera_height is not None \
        else m.get('camera_height', 1.25)
    yaw = args.yaw if args.yaw is not None else m.get('yaw', 0.0)
    # present only on frames whose colour was left in its own camera
    extra = {}
    if m.get('cam_K_color') is not None and m.get('color_from_depth') is not None:
        extra = {'cam_K_color': np.asarray(m['cam_K_color'], np.float32),
                 'color_from_depth': np.asarray(m['color_from_depth'], np.float64)}
    # measured gravity, when the capture has it: the grid then follows a
    # camera that was not level instead of assuming it was
    up = m.get('up_camera')
    return cam_K, float(height), float(yaw), up, extra


def process_one_live_frame(depth_path, rgb_path, cam_K, camera_height, yaw,
                            model, device, pred_dir, ply_dir, mask_mode,
                            up_camera=None, extra=None):
    frame_name = os.path.splitext(os.path.basename(depth_path))[0]

    raw = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
    if raw is None:
        raise FileNotFoundError('cv2.imread returned None for %r' % depth_path)
    if raw.dtype != np.uint16:
        raise ValueError(
            '%r is %s, expected uint16. capture.py writes 16-bit millimetres; '
            'an 8-bit file has already been quantised to 256 depth levels and '
            'is unusable for TSDF.' % (depth_path, raw.dtype))
    # plain millimetres -- NOT NYU's ((d << 13) | (d >> 3)) bit rotation
    depth = raw.astype(np.float32) / 1000.0

    rgb = cv2.imread(rgb_path)
    if rgb is None:
        raise FileNotFoundError('cv2.imread returned None for %r' % rgb_path)

    loader = FrameLoader.from_live_camera(
        depth, rgb=rgb.astype(np.float32), cam_K=cam_K,
        camera_height=camera_height, yaw=yaw, up_camera=up_camera,
        **(extra or {}))
    sample = loader.build_tsdf_and_mapping()

    pred_label, _ = model_mod.predict(model, sample, device=device)
    # no ground truth for a live frame, so label_weight/auto cannot apply
    mask, used = build_vis_mask(sample, frame_name,
                                'surface' if mask_mode in ('auto', 'label_weight')
                                else mask_mode, None)

    np.save(os.path.join(pred_dir, f'{frame_name}.npy'), pred_label)
    ply_path = os.path.join(ply_dir, f'{frame_name}.ply')
    save_prediction_ply(pred_label, mask, ply_path)

    observed = np.asarray(sample['weight']).reshape(60, 36, 60).astype(bool)
    print(f"[{frame_name}] mask={used} ({100 * mask.mean():.1f}% of grid), "
          f"observed={100 * observed.mean():.1f}%, "
          f"occupied={int(((np.asarray(pred_label).reshape(60,36,60) > 0) & mask).sum())} "
          f"-> {ply_path}")


def main():
    setup_logging()
    args, opts = parse_args()
    cfg = load_cfg(args, opts)
    device = get_device()

    pred_dir = ensure_dir(os.path.join(args.out_dir, 'prediction'))
    ply_dir = ensure_dir(os.path.join(args.out_dir, 'ply'))

    # ---- Load model once (verified build_model_from_cfg/load_checkpoint calls) ----
    model = model_mod.load_model(cfg, device=device)

    if args.live_dir is not None:
        # ---- Live mode: frames captured from a real camera ----
        import glob
        meta = load_live_meta(args.live_dir, args)
        depth_paths = sorted(glob.glob(os.path.join(args.live_dir, 'depth', '*.png')))
        if not depth_paths:
            raise FileNotFoundError(
                'no depth PNGs under %s' % os.path.join(args.live_dir, 'depth'))
        print(f'live: {len(depth_paths)} frame(s) from {args.live_dir}')

        for dp in depth_paths:
            stem = os.path.splitext(os.path.basename(dp))[0]
            rp = os.path.join(args.live_dir, 'rgb', stem + '.png')
            if not os.path.exists(rp):
                logging.warning('[%s] no RGB at %s, skipping (the model needs '
                                'img -- it is not optional)', stem, rp)
                continue
            done = os.path.join(pred_dir, stem + '.npy')
            if os.path.exists(done) and not args.overwrite:
                print(f'  [{stem}] already predicted, skipping '
                      f'(--overwrite to redo)')
                continue
            cam_K, camera_height, yaw, up_camera, extra = \
                resolve_frame_camera(meta, stem, args)
            print(f'  [{stem}] cam_K fx={cam_K[0,0]:.2f} fy={cam_K[1,1]:.2f} '
                  f'cx={cam_K[0,2]:.2f} cy={cam_K[1,2]:.2f}')
            print(f'  [{stem}] camera_height={camera_height:.2f} m  '
                  f'yaw={yaw:.3f} rad' + ('' if up_camera is None else
                  '  tilt %.1f deg' % np.degrees(np.arccos(min(1.0,
                      -up_camera[1] / np.linalg.norm(up_camera))))))
            if extra:
                print(f'  [{stem}] colour kept in its own camera; registered '
                      f'per point')
            process_one_live_frame(dp, rp, cam_K, camera_height, yaw, model,
                                    device, pred_dir, ply_dir, args.mask,
                                    up_camera, extra)

    elif args.data_dir is not None:
        # ---- Batch mode: every matched .bin/.png pair in the folder ----
        pairs = find_frame_pairs(args.data_dir)
        print(f'found {len(pairs)} matched .bin/.png pairs in {args.data_dir}')

        succeeded, skipped = [], []
        for frame_id, bin_path, png_path in pairs:
            rgb_path = None
            if args.rgb_dir is not None:
                candidate = os.path.join(args.rgb_dir, f'{frame_id}_colors.png')
                if not os.path.exists(candidate):
                    logging.warning(f'[{frame_id}] no RGB found at {candidate}, skipping frame '
                                     f'(the model needs img -- see model.py: TSDFNet/SegFormerHead '
                                     f'both take real inputs, this is not optional)')
                    skipped.append(frame_id)
                    continue
                rgb_path = candidate
            try:
                process_one_frame(bin_path, png_path, rgb_path, model, device, pred_dir, ply_dir,
                                  mask_mode=args.mask, tsdf_dir=args.tsdf_dir)
                succeeded.append(frame_id)
            except Exception as e:
                logging.warning(f'[{frame_id}] failed, skipping: {e}')
                skipped.append(frame_id)

        print(f'\nbatch done: {len(succeeded)} succeeded, {len(skipped)} skipped.')
        if skipped:
            print(f'skipped frame ids: {skipped[:20]}{"..." if len(skipped) > 20 else ""}')
    else:
        # ---- Single-pair mode ----
        process_one_frame(args.bin_path, args.png_path, args.rgb_path, model, device, pred_dir, ply_dir,
                          mask_mode=args.mask, tsdf_dir=args.tsdf_dir)


if __name__ == "__main__":
    main()