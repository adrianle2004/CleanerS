"""
reconstruction_GT/sweep_eval.py

Score EVERY frame of a sweep, to see which viewpoints the model handles and
which it does not.

    python -m reconstruction_GT.sweep_eval captures/room08
    python -m reconstruction_GT.sweep_eval captures/room08 --every 10
    python -m reconstruction_GT.sweep_eval captures/room08 --keep 1   # add the
                                                # best and worst to the capture

*** WHY ***

One frame gives one number, and that number says as much about the viewpoint as
about the model -- MAKING_GT.md, "What this does not measure". A frame shot
from the corner of a small room has an SC set that is 98% occupied, where
"occupied everywhere" scores 0.98; a frame that looks down at the floor has an
honest one. The only way to tell the two apart is to score a lot of frames and
look at the spread.

The annotation makes that free: the solids describe the ROOM, so every tracked
frame of the sweep can be scored against them without any further work.

*** WHAT IT DOES ***

One pass over the bag. For each sampled tracked frame, in memory:

    where the camera stood        viewpoint_stats: how far, how much floor,
                                  and how much room is hidden behind what it
                                  can see -- the shape of the SC problem
    crop to the NYU geometry      inference/camera.py: match_nyu_fov
    ground truth                  the room's solids through that frame's
                                  world_from_room, voxelized on its grid
    prediction                    CleanerS on that frame's depth + rgb
    score                         test_NYU.py:206-211, over the frustum

It writes one CSV -- nothing else, unless --keep asks for the extremes to be
kept, and then each of them lands on disk as an ordinary frame of the capture,
through the ordinary tools:

    captures/<room>/depth|rgb/<frame>.png        export_frame
    captures/<room>/gt/{Label,TSDF,Mapping}/     voxelize_gt
    captures/<room>/gt/ply/<frame>_gt.ply        show_gt (+ _gt_fov, this view)
    outputs/<room>/prediction/<frame>.npy        run_inference
    outputs/<room>/ply/<frame>.ply

so they can be re-scored, displayed and compared like any other frame.

*** READ THE `occ%` COLUMN ***

`SC` alone ranks a degenerate frame first. A frame whose scored set is 95%
occupied scores ~0.95 for saying "occupied" everywhere, so the table also
carries `occ%` (how much of the SC set is occupied, i.e. what the trivial
baseline would score) and `SC-occ` (how far above that baseline the model
actually got). Sort on `margin` to rank frames by what the model added.
"""

import os
import sys
import csv
import json
import logging
import argparse

import numpy as np

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_THIS_DIR)
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from reconstruction_GT.export_frame import load_poses
from reconstruction_GT.evaluate_gt import score, iou_from_cm
from reconstruction_GT.voxelize_gt import (CLASSES, Grid, IGNORE, frustum_mask,
                                           build_label_volume, compute_label_weight,
                                           downsample_label, load_solids,
                                           transform_solids, VOX_SIZE_HI,
                                           VOX_SIZE_LOW, VOX_UNIT_HI, VOX_UNIT_LOW)

log = logging.getLogger(__name__)
COLUMNS = ['frame', 'cam_z', 'tilt_deg', 'valid_pct', 'floor_pct', 'depth_med',
           'near_pct', 'free_mean', 'scored', 'sc_voxels', 'occ_pct', 'sc_iou',
           'margin', 'ssc_miou', 'classes']


def parse_args():
    p = argparse.ArgumentParser(description='Score every frame of a sweep.')
    p.add_argument('capture_dir')
    p.add_argument('--cfg', default='./cfgs/NYU/voxelSSC.yaml')
    p.add_argument('--pretrained_path', default='./checkpoint/CleanerS_ckpt.pth')
    p.add_argument('--every', type=int, default=20,
                   help='score every Nth tracked frame (default 20)')
    p.add_argument('--limit', type=int, default=0, help='stop after N frames')
    p.add_argument('--min_valid', type=float, default=50.0,
                   help='skip frames with less than this %% valid depth')
    p.add_argument('--out', default=None,
                   help='CSV path (default outputs/<room>/sweep_eval.csv)')
    p.add_argument('--fov', choices=['frustum', 'all'], default='frustum')
    p.add_argument('--rank', choices=['margin', 'sc', 'ssc'], default='margin',
                   help='which column best/worst are taken from (default '
                        'margin: SC minus the trivial baseline)')
    p.add_argument('--restart', action='store_true',
                   help='ignore any existing CSV and score every frame again')
    p.add_argument('--keep', type=int, default=0, metavar='N',
                   help='afterwards, put the N best and N worst frames on disk '
                        'in full: frame, ground truth, prediction and plys')
    return p.parse_args()


def frame_row(idx, depth_c, rgb_c, K_eff, T, spec, model, device, use_frustum):
    """Everything for one frame: ground truth, prediction, score."""
    from inference.frame_loader import FrameLoader, live_cam_pose, load_cuda_encoder
    import inference.model as model_mod

    height = float(T[2, 3])
    up_cam = (T[:3, :3].T @ np.array([0.0, 0.0, 1.0])).tolist()
    A = live_cam_pose(height, 0.0, up_cam).astype(np.float64) @ np.linalg.inv(T)

    # --- ground truth, exactly as voxelize_gt builds it (no rgb, no invalid) ---
    gt_loader = FrameLoader.from_live_camera(
        depth_c, cam_K=np.asarray(K_eff, np.float32), camera_height=height,
        yaw=0.0, drop_invalid_depth=True, up_camera=up_cam)
    gt_loader.encoder = 'cuda' if load_cuda_encoder() is not None else 'numpy'
    gt_sample = gt_loader.build_tsdf_and_mapping()
    tsdf_hi, _ = (gt_loader._build_high_res_tsdf_cuda()
                  if gt_loader.encoder == 'cuda' else gt_loader._build_high_res_tsdf())
    grid = Grid(gt_loader.vox_origin, VOX_SIZE_HI, VOX_UNIT_HI)
    hi = build_label_volume(transform_solids(spec, A), grid)
    label, tsdf_ds = downsample_label(hi, np.asarray(tsdf_hi, np.float32).reshape(-1))
    lw = compute_label_weight(label, tsdf_ds)
    keep = None
    if use_frustum:
        keep = frustum_mask(Grid(gt_loader.vox_origin, VOX_SIZE_LOW, VOX_UNIT_LOW),
                            np.asarray(K_eff, np.float64), gt_loader.cam_pose)

    # --- prediction, exactly as run_inference does it (rgb, invalid kept) ---
    pred_loader = FrameLoader.from_live_camera(
        depth_c, rgb=rgb_c.astype(np.float32), cam_K=np.asarray(K_eff, np.float32),
        camera_height=height, yaw=0.0, up_camera=up_cam)
    pred_label, _ = model_mod.predict(model, pred_loader.build_tsdf_and_mapping(),
                                      device=device)

    cmSSC, cmSC = score(pred_label, label, lw,
                        np.asarray(gt_sample['mapping'], np.int32).reshape(-1), keep)
    _, _, _, iou = iou_from_cm(cmSSC)
    present = [c for c in range(1, 12) if cmSSC[c].sum() > 0]
    _, _, _, iouSC = iou_from_cm(cmSC)
    sc_n = int(cmSC.sum())
    occ = float(cmSC[1].sum()) / max(sc_n, 1)

    return {
        'frame': idx,
        'cam_z': round(height, 3),
        'tilt_deg': round(float(np.degrees(np.arccos(
            min(1.0, -up_cam[1] / np.linalg.norm(up_cam))))), 1),
        'valid_pct': round(100 * float((depth_c > 0).mean()), 1),
        'scored': int(lw.sum()),
        'sc_voxels': sc_n,
        'occ_pct': round(100 * occ, 1),
        'sc_iou': round(100 * float(iouSC[1]), 1),
        'margin': round(100 * (float(iouSC[1]) - occ), 1),
        'ssc_miou': round(100 * float(np.mean(iou[present])) if present else 0.0, 1),
        'classes': len(present),
    }


def viewpoint_stats(depth, cam_K, T, room):
    """Where the camera stood, and what that leaves hidden.

    A frame's SC score is a statement about the volume the camera could NOT
    see, so the shape of that volume is the thing to measure:

      depth_med   median distance to what the frame sees, in metres
      near_pct    how much of the frame is closer than 1.5 m
      floor_pct   how much of it is floor (world z < 12 cm)
      free_mean   the mean distance, along each ray, from the surface the
                  camera hit to where that ray leaves the ROOM. That is the
                  hidden volume the metric will ask the model about: zero when
                  a ray dies on a wall or inside furniture, large when it dies
                  on the near side of something with room behind it.

    `room` is the annotated shell, so this is measured against the same box the
    ground truth is painted in, not against the fused surface.
    """
    from reconstruction_GT.voxelize_gt import solid_rotation
    v, u = np.nonzero(depth > 0)
    step = max(1, len(v) // 20000)                       # ~20k rays is plenty
    v, u = v[::step], u[::step]
    z = depth[v, u]
    P = np.stack([(u - cam_K[0, 2]) * z / cam_K[0, 0],
                  (v - cam_K[1, 2]) * z / cam_K[1, 1], z], axis=1)
    o = T[:3, 3]
    W = P @ T[:3, :3].T + o
    out = {'depth_med': round(float(np.median(z)), 3),
           'near_pct': round(100 * float((z < 1.5).mean()), 1),
           'floor_pct': round(100 * float((W[:, 2] < 0.12).mean()), 1),
           'free_mean': ''}
    if room is None:
        return out
    c = np.asarray(room['center'], np.float64)
    half = np.asarray(room['size'], np.float64) / 2.0
    R = solid_rotation(room)
    d = W - o
    L = np.linalg.norm(d, axis=1)
    d = d / np.maximum(L, 1e-9)[:, None]
    op, dp = R.T @ (o - c), d @ R                        # into the room's frame
    with np.errstate(divide='ignore', invalid='ignore'):
        t1, t2 = (-half - op) / dp, (half - op) / dp
    t_exit = np.minimum.reduce(np.maximum(t1, t2), axis=1)
    free = np.clip(t_exit - L, 0.0, None)                # space behind the hit
    out['free_mean'] = round(float(np.mean(free[np.isfinite(free)])), 3)
    return out


def keep_frame(cap, idx, model, device, args):
    """Put one swept frame on disk the way the file pipeline would: the frame
    itself, its ground truth, its prediction, and both as plys.

    It runs the ordinary tools rather than saving what this script already has
    in memory, so what lands on disk is what `voxelize_gt` and `run_inference`
    produce -- nothing here is a shortcut that could drift from them.
    """
    from reconstruction_GT import export_frame as EF
    from reconstruction_GT import show_gt as SG
    from reconstruction_GT import voxelize_gt as VG
    from inference.run_inference import process_one_live_frame, resolve_frame_camera
    from inference.utils import ensure_dir

    argv = sys.argv
    try:
        sys.argv = ['export_frame', cap, '--frame', str(idx)]
        EF.main()                                   # depth/, rgb/, frame_meta
        stem = 'live_%06d' % idx
        sys.argv = ['voxelize_gt', cap, '--frame', stem]
        VG.main()                                   # Label/, TSDF/, Mapping/
        for extra in ([], ['--fov', 'frustum']):
            sys.argv = ['show_gt', cap, '--frame', stem, '--export'] + extra
            SG.main()                               # gt/ply/<stem>_gt[_fov].ply
    finally:
        sys.argv = argv

    out_root = os.path.join('outputs', os.path.basename(cap))
    pred_dir = ensure_dir(os.path.join(out_root, 'prediction'))
    ply_dir = ensure_dir(os.path.join(out_root, 'ply'))
    meta = json.load(open(os.path.join(cap, 'meta.json')))
    flags = argparse.Namespace(camera_height=None, yaw=None)
    cam_K, height, yaw, up, extra = resolve_frame_camera(meta, stem, flags)
    process_one_live_frame(os.path.join(cap, 'depth', stem + '.png'),
                           os.path.join(cap, 'rgb', stem + '.png'),
                           cam_K, height, yaw, model, device, pred_dir, ply_dir,
                           'surface', up, extra)
    return stem


def main():
    from inference.utils import setup_logging, get_device
    setup_logging()
    args = parse_args()
    import cv2
    import inference.model as model_mod
    from inference.camera import match_nyu_fov
    from inference.run_inference import load_cfg

    cap = args.capture_dir.rstrip('/')
    poses = load_poses(cap)
    spec = load_solids(os.path.join(cap, 'gt', 'solids.json'))
    print('%d tracked frames, %d solids' % (len(poses),
                                           len(spec.get('solids', []))), flush=True)

    cfg_args = argparse.Namespace(cfg=args.cfg, pretrained_path=args.pretrained_path)
    device = get_device()
    model = model_mod.load_model(load_cfg(cfg_args, {}), device=device)

    # bag_reader delivers every frame at any speed, so the network can run
    # inside this loop without changing which frame an index means --
    # MAKING_GT.md, gotcha 10.
    from reconstruction_GT.bag_reader import iter_bag

    # an interrupted run keeps its CSV: frames already in it are not scored
    # again, so guard.sh can restart this without losing the GPU minutes
    out = args.out or os.path.join('outputs', os.path.basename(cap),
                                   'sweep_eval.csv')
    os.makedirs(os.path.dirname(out) or '.', exist_ok=True)
    rows, done = [], set()
    if os.path.exists(out) and not args.restart:
        with open(out) as f:
            for r in csv.DictReader(f):
                if set(COLUMNS) <= set(r):
                    rows.append({k: r[k] for k in COLUMNS})
                    done.add(int(float(r['frame'])))
        if done:
            print('resuming: %d frames already in %s' % (len(done), out),
                  flush=True)
    handle = open(out, 'a' if done else 'w')
    writer = csv.DictWriter(handle, COLUMNS)
    if not done:
        writer.writeheader()
        handle.flush()
    # align='color' -- the same arrangement export_frame writes and the model
    # was trained on: depth projected into the colour camera, colour left
    # whole, unprojected with the colour intrinsics. Scoring the sweep in a
    # different arrangement from the frames it recommends would rank
    # viewpoints on data nobody will ever evaluate.
    for bf in iter_bag(os.path.join(cap, 'scan', 'scan.bag'), align='color'):
        i = bf.index
        if i not in poses or i % args.every or i in done:
            continue
        if 100 * float((bf.depth > 0).mean()) < args.min_valid:
            continue
        bgr = cv2.cvtColor(bf.rgb, cv2.COLOR_RGB2BGR)
        depth_c, rgb_c, K_eff = match_nyu_fov(bf.depth, bgr, bf.K)
        # the uint16 millimetres export_frame would write, so these numbers are
        # the numbers the file pipeline produces
        depth_c = (np.clip(depth_c * 1000.0, 0, 65535).astype(np.uint16)
                   .astype(np.float32) / 1000.0)
        # trajectory.txt holds DEPTH-camera poses; this frame is expressed in
        # the colour camera, so the pose moves with it (export_frame does the
        # same composition)
        T = poses[i] @ np.linalg.inv(np.asarray(bf.color_from_depth, np.float64))
        row = frame_row(i, depth_c, rgb_c, K_eff, T, spec, model, device,
                        args.fov == 'frustum')
        row.update(viewpoint_stats(depth_c, K_eff, T, spec.get('room')))
        rows.append(row)
        writer.writerow(row)                 # on disk before the next frame
        handle.flush()
        os.fsync(handle.fileno())
        # print, not log: loading the model installs mmcv/mmseg's own root
        # logger, and everything logged after that disappears
        print('%3d  frame %4d  floor %4.1f%%  SC %5.1f (trivial %4.1f)  '
              'SSC %5.1f' % (len(rows), i, row['floor_pct'], row['sc_iou'],
                             row['occ_pct'], row['ssc_miou']), flush=True)
        if args.limit and len(rows) >= args.limit:
            break

    if not rows:
        raise SystemExit('no frames scored -- try --every 1 or a lower --min_valid')

    handle.close()

    key = {'margin': 'margin', 'sc': 'sc_iou', 'ssc': 'ssc_miou'}[args.rank]
    for r in rows:                           # resumed rows arrive as strings
        for k in COLUMNS:
            if r[k] == '' or r[k] is None:
                r[k] = 0
            elif k in ('frame', 'scored', 'sc_voxels', 'classes'):
                r[k] = int(float(r[k]))
            elif not isinstance(r[k], (int, float)):
                r[k] = float(r[k])
    rows.sort(key=lambda r: -r[key])
    head = ('%5s %6s %6s %7s %7s %7s %7s %7s %7s'
            % ('frame', 'cam z', 'tilt', 'floor%', 'SC set', 'occ%', 'SC', 'margin',
               'SSC'))
    line = lambda r: ('%5d %6.2f %5.1f° %6.1f%% %7d %6.1f%% %7.1f %7.1f %7.1f'
                      % (r['frame'], r['cam_z'], r['tilt_deg'], r['floor_pct'],
                         r['sc_voxels'], r['occ_pct'], r['sc_iou'], r['margin'],
                         r['ssc_miou']))
    print('\n%d frames scored, best %s first\n' % (len(rows), args.rank))
    print(head)
    if len(rows) <= 12:
        for r in rows:
            print(line(r))
    else:
        for r in rows[:5]:
            print(line(r))
        print('   ...  %d frames ...' % (len(rows) - 10))
        for r in rows[-5:]:
            print(line(r))
    sc = np.array([r['sc_iou'] for r in rows])
    ss = np.array([r['ssc_miou'] for r in rows])
    oc = np.array([r['occ_pct'] for r in rows])
    print('\nSC  %.1f .. %.1f  (median %.1f)   trivial baseline %.1f .. %.1f'
          % (sc.min(), sc.max(), np.median(sc), oc.min(), oc.max()))
    print('SSC %.1f .. %.1f  (median %.1f)' % (ss.min(), ss.max(), np.median(ss)))
    print('\n%s' % out)

    if args.keep:
        picked = rows[:args.keep] + rows[-args.keep:]
        labels = (['best'] * args.keep + ['worst'] * args.keep)
        print('\nkeeping %d frames (%s):' % (len(picked), args.rank))
        kept = []
        for r, what in zip(picked, labels):
            kept.append(keep_frame(cap, r['frame'], model, device, args))
            print('  %-5s frame %4d -> %s' % (what, r['frame'], kept[-1]))
        print('\n  ground truth   %s/gt/{Label,TSDF,Mapping}/<frame>.npz'
              '\n  GT as a ply    %s/gt/ply/<frame>_gt.ply  (+ _gt_fov)'
              '\n  prediction     outputs/%s/prediction/<frame>.npy'
              '\n  prediction ply outputs/%s/ply/<frame>.ply'
              % (cap, cap, os.path.basename(cap), os.path.basename(cap)))
        print('\n  compare one:  python inference/display_overlay.py '
              'outputs/%s/ply/%s.ply --frame-grid'
              % (os.path.basename(cap), kept[0]))


if __name__ == '__main__':
    main()
