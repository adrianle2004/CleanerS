"""
inference/show_nyu.py

Show a CleanerS prediction on an NYU frame next to NYU's own ground truth
(data/NYU/Label, the CleanerS/SSCNet annotation -- not ISO's), both in
display_overlay.py.

Both .ply files are written the way test_NYU.py:visualize_3d_predict writes the
prediction: voxels outside label_weight (the shipped TSDF/<id>.npz arr_1) are
set to empty, then labeled_voxel2ply, which also drops 255. They are named
NYU<id>_pred.ply / NYU<id>_gt.ply, so display_overlay finds the .bin pose,
depth and RGB on its own.

    python inference/show_nyu.py 0001
    python inference/show_nyu.py NYU0015 --only gt
    python inference/show_nyu.py 0001 --screenshot compare.png
    python inference/show_nyu.py --list 10            # best and worst test frames
    python inference/show_nyu.py 0001 -- --show voxels  # after --: passed to display_overlay

Frame scores are the NYU protocol (label_weight & GT != 255), as evaluate.py.
"""

import argparse
import os
import sys

import numpy as np

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _THIS_DIR)
from visualize import labeled_voxel2ply  # noqa: E402
from evaluate import confusion, metrics, IGNORE, MAPPING_SENTINEL  # noqa: E402
from compare_view import add_view_args, open_windows, screenshot_pair  # noqa: E402

REPO = os.path.dirname(_THIS_DIR)
NYU_ROOT = os.path.join(os.path.dirname(REPO), 'data', 'NYU')
PRED_DIR = os.path.join(REPO, 'custom_visual_pred', 'CleanerS', 'prediction')
OUT_DIR = os.path.join(REPO, 'custom_visual_pred', 'CleanerS', 'compare')
GRID = (60, 36, 60)


def load(fid, args):
    pred = np.load(os.path.join(args.pred_dir, 'NYU%s_0000.npy' % fid)).reshape(GRID).astype(np.int64)
    label = np.load(os.path.join(args.nyu_root, 'Label', '%s.npz' % fid))['arr_0'].reshape(GRID).astype(np.int64)
    lw = np.load(os.path.join(args.nyu_root, 'TSDF', '%s.npz' % fid))['arr_1'].reshape(GRID) > 0
    mapping = np.load(os.path.join(args.nyu_root, 'Mapping', '%s.npz' % fid))['arr_0'].reshape(GRID)
    return pred, label, lw, mapping


def scores(pred, label, lw, mapping):
    keep = lw & (label != IGNORE)
    cm = confusion(pred[keep], label[keep])
    sc = keep & (mapping == MAPPING_SENTINEL)
    cm_sc = confusion((pred[sc] > 0).astype(np.int64), (label[sc] > 0).astype(np.int64), n=2)
    return 100 * metrics(cm_sc)['iou'][1], 100 * np.nanmean(metrics(cm)['iou'][1:]), int(keep.sum())


def write_plys(fid, args):
    pred, label, lw, mapping = load(fid, args)
    sc, miou, n = scores(pred, label, lw, mapping)
    print('NYU%s  SC IoU %.1f  SSC mIoU %.1f  (%d scored voxels, NYU protocol)' % (fid, sc, miou, n))
    full = args.mask == 'none'
    if full:
        print('  mask=none: %d predicted-occupied voxels, %.0f%% of them outside label_weight'
              % ((pred > 0).sum(), 100 * ((pred > 0) & ~lw).sum() / max((pred > 0).sum(), 1)))
    out = {}
    for name, vol in (('pred', pred), ('gt', label)):
        vol = vol.copy()
        if not (full and name == 'pred'):
            vol[~lw] = 0                          # visualize_3d_predict
        suffix = 'pred_full' if (full and name == 'pred') else name
        path = os.path.join(args.out_dir, 'NYU%s_%s.ply' % (fid, suffix))
        if ((vol > 0) & (vol < 255)).any():
            labeled_voxel2ply(vol, path)
            out[name] = path
        else:
            print('  %s: nothing to show inside label_weight' % name)
    return out


def list_frames(args):
    ids = [l.strip() for l in open(os.path.join(args.nyu_root, '%s.txt' % args.split)) if l.strip()]
    rows = []
    for fid in ids:
        if not os.path.exists(os.path.join(args.pred_dir, 'NYU%s_0000.npy' % fid)):
            continue
        sc, miou, n = scores(*load(fid, args))
        if n >= 2000:
            rows.append((miou, sc, n, fid))
    rows.sort()
    print('%s split, %d frames with >= 2000 scored voxels (NYU protocol)\n' % (args.split, len(rows)))
    for title, part in (('best', rows[::-1][:args.list]), ('worst', rows[:args.list])):
        print(title)
        for miou, sc, n, fid in part:
            print('  %s  SSC mIoU %5.1f  SC IoU %5.1f  (%d voxels)' % (fid, miou, sc, n))


def main():
    argv = sys.argv[1:]
    extra = []
    if '--' in argv:
        i = argv.index('--')
        argv, extra = argv[:i], argv[i + 1:]
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('frame', nargs='?', help='NYU frame id: 0001 or NYU0001')
    ap.add_argument('--list', type=int, metavar='N', help='print the N best and worst frames and exit')
    ap.add_argument('--split', default='test', choices=['test', 'train'])
    ap.add_argument('--pred_dir', default=PRED_DIR)
    ap.add_argument('--nyu_root', default=NYU_ROOT)
    ap.add_argument('--out_dir', default=OUT_DIR, help='where the two .ply files are written')
    add_view_args(ap)
    ap.add_argument('--mask', choices=['label_weight', 'none'], default='label_weight',
                    help="label_weight: the prediction as test_NYU.py writes it (default); none: the full "
                         "prediction, written as NYU<id>_pred_full.ply (the GT is the same either way)")
    ap.add_argument('--export_only', action='store_true',
                    help='write the .ply files and print their paths, open nothing (used by the .sh launcher)')
    args = ap.parse_args(argv)

    if args.list:
        return list_frames(args)
    if not args.frame:
        ap.error('give a frame id, or --list N')
    fid = args.frame.upper().replace('NYU', '').split('_')[0].zfill(4)

    plys = write_plys(fid, args)
    if args.export_only:
        for k in ('pred', 'gt'):
            print('%s_ply=%s' % (k, plys.get(k, '')))
        return
    order = [k for k in ('pred', 'gt') if k in plys and (args.only in (None, k))]
    if not order:
        sys.exit('nothing to display')
    if args.screenshot:
        screenshot_pair([plys[k] for k in order], ['CleanerS prediction' if k == 'pred' else 'NYU GT' for k in order],
                        args.screenshot, extra)
    else:
        open_windows([plys[k] for k in order], extra)


if __name__ == '__main__':
    main()
