"""
inference/export_scannet_ply.py

Writes .ply files for the best and the worst frame of every Occ-ScanNet scene,
built exactly the way test_NYU.py builds NYU's:

    visualize_3d_predict (test_NYU.py):  pred[label_weight == 0] = 0
                                         labeled_voxel2ply(pred)

label_weight is NYU's rule, GT object | tsdf < -0.5, from the GT and the tsdf
run_scannet.py encoded -- the same mask evaluate_scannet.py's `cleaners` rows
score. A GT .ply goes next to each prediction, masked the same way (every GT
object voxel is inside label_weight, so it is the whole annotation minus 255).

"Best" and "worst" rank by the NYU-protocol per-frame SSC mIoU in
evaluate_scannet.py's per_frame.csv, so run that first. Frames with fewer than
--min_scored scored voxels or fewer than --min_classes GT classes are skipped:
a frame scoring a handful of voxels of one class gets 0 or 100 by accident.

For each chosen frame, in <out>/<scene>/:
    <frame>_pred.ply   prediction, label_weight-masked
    <frame>_gt.ply     GT, label_weight-masked
    <frame>_rgb.png    colour registered onto the 640x480 depth camera -- the
                       image the model was given
    <frame>.json       sidecar display_solid.py reads for photo colouring:
                       vox_origin, cam_pose, cam_K (depth), rgb, depth, scores
plus <out>/index.csv listing every exported frame.

Usage (from the repo root, after run_scannet.py and evaluate_scannet.py):
    python -m inference.export_scannet_ply --pred_dir ./outputs/scannet
    python inference/display_overlay.py outputs/scannet/ply/scene0011_00/00012_pred.ply
"""

import argparse
import csv
import json
import os
import sys

import cv2
import numpy as np

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _THIS_DIR)
from visualize import labeled_voxel2ply  # noqa: E402
from run_scannet import DEFAULT_ROOT, load_calibration, register_color_to_depth  # noqa: E402

GRID = (60, 36, 60)


def nyu_label_weight(label, tsdf):
    """data/generate_cleaners_data.py:compute_label_weight -- label in 1..253 | tsdf < -0.5."""
    return (np.abs(127 - label) < 127) | (tsdf < -0.5)


def pick_frames(rows, min_scored, min_classes):
    by_scene = {}
    for r in rows:
        if r['ssc_miou_nyu_protocol'] == '':
            continue
        if int(r['scored_voxels_nyu_protocol']) < min_scored or int(r['gt_classes']) < min_classes:
            continue
        by_scene.setdefault(r['scene'], []).append(r)
    picks = []
    for scene, rs in sorted(by_scene.items()):
        rs.sort(key=lambda r: (float(r['ssc_miou_nyu_protocol']), float(r['sc_iou_nyu_protocol'] or 0)))
        picks.append(('best', rs[-1]))
        if len(rs) > 1:
            picks.append(('worst', rs[0]))
    return picks


def export_frame(role, row, args, full_pred=False):
    scene, frame = row['scene'], row['frame']
    gt = np.load(os.path.join(args.scannet_root, scene, frame + '.npz'))
    enc = np.load(os.path.join(args.pred_dir, 'encoded', scene, frame + '.npz'))
    pred = np.load(os.path.join(args.pred_dir, 'prediction', scene, frame + '.npy')).reshape(GRID).astype(np.int64)
    label = gt['target'].reshape(GRID).astype(np.int64)
    lw = nyu_label_weight(label.reshape(-1), enc['tsdf'].reshape(-1)).reshape(GRID)

    out = os.path.join(args.out_dir, scene)
    os.makedirs(out, exist_ok=True)
    pred_m = pred.copy()
    pred_m[~lw] = 0                                   # visualize_3d_predict
    gt_m = label.copy()
    gt_m[~lw] = 0
    written = {}
    if full_pred:                                     # MASK=none: the prediction as the model gave it
        path = os.path.join(out, '%s_pred_full.ply' % frame)
        if ((pred > 0) & (pred < 255)).any():
            labeled_voxel2ply(pred, path)
            written['pred_full'] = os.path.basename(path)
    for name, vol in (('pred', pred_m), ('gt', gt_m)):
        path = os.path.join(out, '%s_%s.ply' % (frame, name))
        if ((vol > 0) & (vol < 255)).any():
            labeled_voxel2ply(vol, path)
            written[name] = os.path.basename(path)

    # the colour image and camera the model actually used, for display_solid.py
    img_dir = os.path.join(args.scannet_root, 'posed_images', scene)
    depth_path = os.path.join(img_dir, frame + '.png')
    depth = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED).astype(np.float64) / 1000.0
    K_color = gt['intrinsic_color']
    color = cv2.imread(os.path.join(img_dir, frame + '.jpg'))
    K_depth, T_d2c = load_calibration(scene, args.scans_dir)
    rgb = register_color_to_depth(color, depth, K_depth, K_color, T_d2c)
    rgb_path = os.path.join(out, frame + '_rgb.png')
    cv2.imwrite(rgb_path, rgb)

    side = dict(dataset='occ-scannet', scene=scene, frame=frame, role=role,
                vox_origin=[float(v) for v in gt['voxel_origin']],
                cam_pose=np.asarray(gt['cam_pose'], dtype=float).tolist(),
                cam_K=np.asarray(K_depth, dtype=float).tolist(),
                rgb=os.path.basename(rgb_path),
                depth=os.path.relpath(depth_path, out), depth_units='millimetres',
                mask='label_weight = GT object | tsdf < -0.5 (test_NYU.py visualize_3d_predict)',
                sc_iou_nyu_protocol=row['sc_iou_nyu_protocol'],
                ssc_miou_nyu_protocol=row['ssc_miou_nyu_protocol'],
                scored_voxels_nyu_protocol=int(row['scored_voxels_nyu_protocol']),
                ply=written)
    with open(os.path.join(out, frame + '.json'), 'w') as f:
        json.dump(side, f, indent=2)
    return side


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--pred_dir', default='./outputs/scannet')
    p.add_argument('--scannet_root', default=DEFAULT_ROOT)
    p.add_argument('--scans_dir', default=None)
    p.add_argument('--csv', default=None, help='default: <pred_dir>/per_frame.csv')
    p.add_argument('--out_dir', default=None, help='default: <pred_dir>/ply')
    p.add_argument('--min_scored', type=int, default=2000,
                   help='skip frames with fewer NYU-protocol scored voxels')
    p.add_argument('--min_classes', type=int, default=2, help='skip frames with fewer GT classes')
    args = p.parse_args()
    args.scans_dir = args.scans_dir or os.path.join(args.scannet_root, 'scans')
    args.csv = args.csv or os.path.join(args.pred_dir, 'per_frame.csv')
    args.out_dir = args.out_dir or os.path.join(args.pred_dir, 'ply')

    if not os.path.exists(args.csv):
        sys.exit('no %s -- run inference.evaluate_scannet first' % args.csv)
    with open(args.csv) as f:
        rows = list(csv.DictReader(f))
    if 'ssc_miou_nyu_protocol' not in rows[0]:
        sys.exit('%s predates the NYU-protocol columns -- rerun inference.evaluate_scannet' % args.csv)

    picks = pick_frames(rows, args.min_scored, args.min_classes)
    print('%d scenes -> %d frames' % (len({r["scene"] for _, r in picks}), len(picks)))
    index = []
    for i, (role, row) in enumerate(picks, 1):
        side = export_frame(role, row, args)
        if side is None:
            continue
        index.append({k: side[k] for k in ('scene', 'frame', 'role', 'ssc_miou_nyu_protocol',
                                            'sc_iou_nyu_protocol', 'scored_voxels_nyu_protocol')})
        if i % 100 == 0:
            print('  %d/%d' % (i, len(picks)), flush=True)
    os.makedirs(args.out_dir, exist_ok=True)
    with open(os.path.join(args.out_dir, 'index.csv'), 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(index[0].keys()))
        w.writeheader()
        w.writerows(index)
    print('wrote %d frames under %s' % (len(index), args.out_dir))


if __name__ == '__main__':
    main()
