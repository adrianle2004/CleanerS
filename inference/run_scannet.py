"""
inference/run_scannet.py

Runs CleanerS on Occ-ScanNet frames: depth (+RGB) -> FrameLoader -> model, and
saves the prediction plus the encoded tsdf/mapping that evaluate_scannet.py
needs for the NYU-style scored set.

Inputs, all under --scannet_root (default ../data/Scannet):
    <scene>/<frame>.npz               GT from data/occscannet_gt.py:
                                      target, target_fov, voxel_origin, cam_pose,
                                      intrinsic_color
    posed_images/<scene>/<frame>.png  depth, uint16 millimetres, 640x480
    posed_images/<scene>/<frame>.jpg  colour, 1296x968 (640x480 in 12 val scenes)
    scans/<scene>/<scene>.txt         ScanNet metadata: fx_depth, fy_depth,
                                      mx_depth, my_depth, colorToDepthExtrinsics

Calibration. The depth camera's intrinsics and the colour->depth extrinsics
come from ScanNet's per-scene <scene>.txt (fetched from the ScanNet release,
kaldir.vc.cit.tum.de/scannet/v2/scans/<scene>/<scene>.txt, after signing the
ScanNet terms of use). A scene without that file is skipped: nothing is
estimated. 345 of the 681 val scenes carry no colorToDepthExtrinsics line; for
those ScanNet itself provides no offset, and identity is used.

Geometry kept as-is from the Occ-ScanNet GT (no re-gridding):
    vox_origin = voxel_origin (world X, Y, Z), cam_pose = pose from the .sens.
    The grid is aligned to the ScanNet world axes, not to the camera's heading,
    so unlike NYU the camera may look along X, Y or a diagonal of the volume.

Colour registration: every depth pixel with a reading is unprojected with
K_depth, moved into the colour camera with the extrinsics and projected with
K_colour; pixels without depth use the rotation-only (infinite-depth) mapping.
Only pixels with depth reach a voxel through `mapping`, so the fill affects 2D
context only.

Usage (from the repo root):
    python -m inference.run_scannet --pretrained_path ./checkpoint/CleanerS_ckpt.pth \
        --scenes scene0011_00 [--max_frames 5]
    python -m inference.run_scannet --pretrained_path ./checkpoint/CleanerS_ckpt.pth --split val
"""

import argparse
import glob
import os
import sys
import time

import cv2
import numpy as np
import torch

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_THIS_DIR)
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

try:
    from .frame_loader import FrameLoader, IMG_H, IMG_W
    from . import model as model_mod
    from .run_inference import load_cfg
    from .utils import get_device, setup_logging, ensure_dir
except ImportError:
    from frame_loader import FrameLoader, IMG_H, IMG_W
    import model as model_mod
    from run_inference import load_cfg
    from utils import get_device, setup_logging, ensure_dir

DEFAULT_ROOT = os.path.join(os.path.dirname(_REPO_ROOT), 'data', 'Scannet')


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--cfg', default='./cfgs/NYU/voxelSSC.yaml')
    p.add_argument('--pretrained_path', required=True)
    p.add_argument('--scannet_root', default=DEFAULT_ROOT)
    p.add_argument('--scans_dir', default=None,
                   help='ScanNet scans/ folder holding <scene>/<scene>.txt '
                        '(default: <scannet_root>/scans)')
    p.add_argument('--scenes', nargs='+', default=None)
    p.add_argument('--split', default=None, choices=['val'],
                   help='every scene that has GT under scannet_root')
    p.add_argument('--max_frames', type=int, default=0, help='per scene, 0 = all')
    p.add_argument('--out_dir', default=None,
                   help='default: ./outputs/scannet')
    p.add_argument('--encoder', default='auto', choices=['auto', 'cuda', 'numpy'],
                   help="TSDF encoder: SSCNet's CUDA kernels or the numpy port (see FrameLoader)")
    p.add_argument('--overwrite', action='store_true')
    args, opts = p.parse_known_args()
    if not args.scenes and not args.split:
        p.error('give --scenes or --split val')
    args.scans_dir = args.scans_dir or os.path.join(args.scannet_root, 'scans')
    if args.out_dir is None:
        args.out_dir = './outputs/scannet'
    return args, opts


# ---------------------------------------------------------------- calibration
def read_scannet_meta(path):
    meta = {}
    with open(path) as f:
        for line in f:
            if '=' in line:
                k, v = line.split('=', 1)
                meta[k.strip()] = v.strip()
    return meta


def load_calibration(scene, scans_dir):
    """-> (K_depth 3x3, T_depth2color 4x4) from ScanNet's <scene>.txt."""
    path = os.path.join(scans_dir, scene, scene + '.txt')
    if not os.path.exists(path):
        raise FileNotFoundError(
            'no %s -- the depth intrinsics come only from ScanNet: fetch '
            'kaldir.vc.cit.tum.de/scannet/v2/scans/%s/%s.txt' % (path, scene, scene))
    m = read_scannet_meta(path)
    K_depth = np.array([[float(m['fx_depth']), 0, float(m['mx_depth'])],
                        [0, float(m['fy_depth']), float(m['my_depth'])],
                        [0, 0, 1]], dtype=np.float64)
    if (int(m.get('depthWidth', IMG_W)), int(m.get('depthHeight', IMG_H))) != (IMG_W, IMG_H):
        raise ValueError('%s: depth is %sx%s, the encoder needs %dx%d'
                         % (path, m.get('depthWidth'), m.get('depthHeight'), IMG_W, IMG_H))
    if 'colorToDepthExtrinsics' in m:
        T_c2d = np.array([float(x) for x in m['colorToDepthExtrinsics'].split()]).reshape(4, 4)
    else:
        T_c2d = np.eye(4)
    return K_depth, np.linalg.inv(T_c2d)


# ------------------------------------------------------------ registration
def register_color_to_depth(color_bgr, depth_m, K_depth, K_color, T_d2c):
    """Resample the colour image onto the depth camera's pixel grid."""
    vs, us = np.meshgrid(np.arange(IMG_H, dtype=np.float64),
                         np.arange(IMG_W, dtype=np.float64), indexing='ij')
    rays = np.stack([(us - K_depth[0, 2]) / K_depth[0, 0],
                     (vs - K_depth[1, 2]) / K_depth[1, 1],
                     np.ones_like(us)], axis=-1)                      # (H,W,3), z = 1

    # infinite depth: rotation only
    far = rays @ T_d2c[:3, :3].T
    # measured depth: full rigid transform
    near = (rays * depth_m[..., None]) @ T_d2c[:3, :3].T + T_d2c[:3, 3]
    valid = depth_m > 0
    pc = np.where(valid[..., None], near, far)

    Kc = K_color[:3, :3]
    with np.errstate(divide='ignore', invalid='ignore'):
        map_x = (Kc[0, 0] * pc[..., 0] / pc[..., 2] + Kc[0, 2]).astype(np.float32)
        map_y = (Kc[1, 1] * pc[..., 1] / pc[..., 2] + Kc[1, 2]).astype(np.float32)
    map_x[~np.isfinite(map_x)] = -1
    map_y[~np.isfinite(map_y)] = -1
    return cv2.remap(color_bgr, map_x, map_y, cv2.INTER_LINEAR,
                     borderMode=cv2.BORDER_CONSTANT, borderValue=0)


# ------------------------------------------------------------------- frames
def list_frames(args):
    if args.scenes:
        scenes = args.scenes
    else:
        scenes = sorted(d for d in os.listdir(args.scannet_root)
                        if d.startswith('scene') and os.path.isdir(os.path.join(args.scannet_root, d)))
    for scene in scenes:
        gts = sorted(glob.glob(os.path.join(args.scannet_root, scene, '*.npz')))
        if args.max_frames:
            gts = gts[:args.max_frames]
        for gt in gts:
            yield scene, os.path.splitext(os.path.basename(gt))[0], gt


def encode_frame(scene, frame, gt_path, args):
    gt = np.load(gt_path)
    img_dir = os.path.join(args.scannet_root, 'posed_images', scene)
    depth_raw = cv2.imread(os.path.join(img_dir, frame + '.png'), cv2.IMREAD_UNCHANGED)
    color = cv2.imread(os.path.join(img_dir, frame + '.jpg'))
    if depth_raw is None or color is None:
        raise FileNotFoundError('missing %s.png/.jpg in %s (run occscannet_remote.py extract)'
                                % (frame, img_dir))
    if depth_raw.dtype != np.uint16 or depth_raw.shape != (IMG_H, IMG_W):
        raise ValueError('%s/%s.png is %s %s, expected uint16 %dx%d'
                         % (scene, frame, depth_raw.dtype, depth_raw.shape, IMG_W, IMG_H))
    depth = depth_raw.astype(np.float32) / 1000.0

    K_color = gt['intrinsic_color']
    K_depth, T_d2c = load_calibration(scene, args.scans_dir)
    rgb = register_color_to_depth(color, depth.astype(np.float64), K_depth, K_color, T_d2c)

    loader = FrameLoader(depth, gt['voxel_origin'], gt['cam_pose'],
                         rgb=rgb.astype(np.float32), cam_K=K_depth.astype(np.float32),
                         drop_invalid_depth=True, encoder=args.encoder)
    return loader.build_tsdf_and_mapping(), rgb


def main():
    setup_logging()
    args, opts = parse_args()
    cfg = load_cfg(args, opts)
    device = get_device()
    pred_root = ensure_dir(os.path.join(args.out_dir, 'prediction'))
    enc_root = ensure_dir(os.path.join(args.out_dir, 'encoded'))

    model = model_mod.load_model(cfg, device=device)
    n = skipped = failed = 0
    t0 = time.time()
    for scene, frame, gt_path in list_frames(args):
        pred_path = os.path.join(pred_root, scene, frame + '.npy')
        if os.path.exists(pred_path) and not args.overwrite:
            skipped += 1
            continue
        try:
            sample, rgb = encode_frame(scene, frame, gt_path, args)
        except FileNotFoundError as e:
            print('[%s/%s] %s' % (scene, frame, e))
            failed += 1
            continue
        pred_label, _ = model_mod.predict(model, sample, device=device)

        os.makedirs(os.path.dirname(pred_path), exist_ok=True)
        np.save(pred_path, pred_label.astype(np.uint8))
        enc_path = os.path.join(enc_root, scene, frame + '.npz')
        os.makedirs(os.path.dirname(enc_path), exist_ok=True)
        np.savez_compressed(enc_path, tsdf=sample['tsdf'].reshape(-1), weight=sample['weight'].reshape(-1),
                            mapping=sample['mapping'])
        n += 1
        if n == 1:
            cv2.imwrite(os.path.join(args.out_dir, 'first_registered_rgb.png'), rgb)
        if n % 50 == 0 or n <= 3:
            print('[%s/%s] observed=%.1f%%  mapped=%d  %.2f s/frame'
                  % (scene, frame, 100 * sample['weight'].mean(),
                     int((sample['mapping'] != IMG_H * IMG_W).sum()),
                     (time.time() - t0) / n), flush=True)
    print('done: %d predicted, %d already present, %d failed -> %s' % (n, skipped, failed, args.out_dir))


if __name__ == '__main__':
    with torch.no_grad():
        main()
