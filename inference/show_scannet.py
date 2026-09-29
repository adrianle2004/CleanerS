"""
inference/show_scannet.py

Show a CleanerS prediction on an Occ-ScanNet frame next to that frame's GT,
both in display_overlay.py.

The two .ply files come from export_scannet_ply.py and are built exactly like
NYU's (test_NYU.py:visualize_3d_predict): voxels outside label_weight =
GT object | tsdf < -0.5 are set to empty. Frames that were not exported as a
scene's best or worst are exported on demand. display_overlay reads the pose,
depth camera and registered colour from the <frame>.json next to them.

    python inference/show_scannet.py scene0702_00                 # that scene's best frame
    python inference/show_scannet.py scene0702_00 --pick worst
    python inference/show_scannet.py scene0702_00 00094
    python inference/show_scannet.py scene0702_00 --screenshot compare.png
    python inference/show_scannet.py --list 10                     # best and worst frames overall
    python inference/show_scannet.py scene0702_00 -- --show voxels  # after --: passed to display_overlay

Needs run_scannet.py and evaluate_scannet.py to have run on --pred_dir.
"""

import argparse
import csv
import os
import sys
import types

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _THIS_DIR)
from compare_view import add_view_args, open_windows, screenshot_pair  # noqa: E402

REPO = os.path.dirname(_THIS_DIR)
DEFAULT_PRED = os.path.join(REPO, 'outputs', 'scannet')
DEFAULT_ROOT = os.path.join(os.path.dirname(REPO), 'data', 'Scannet')


def read_rows(args):
    path = os.path.join(args.pred_dir, 'per_frame.csv')
    if not os.path.exists(path):
        sys.exit('no %s -- run inference.evaluate_scannet on %s first' % (path, args.pred_dir))
    with open(path) as f:
        return [r for r in csv.DictReader(f) if r.get('ssc_miou_nyu_protocol', '') != '']


def choose(rows, args):
    scene_rows = [r for r in rows if r['scene'] == args.scene]
    if not scene_rows:
        sys.exit('no scored frames for %s in per_frame.csv' % args.scene)
    if args.frame:
        match = [r for r in scene_rows if r['frame'] == args.frame.zfill(5)]
        if not match:
            sys.exit('%s has no frame %s; it has: %s' % (args.scene, args.frame, ' '.join(r['frame'] for r in scene_rows)))
        return 'selected', match[0]
    eligible = [r for r in scene_rows if int(r['scored_voxels_nyu_protocol']) >= 2000 and int(r['gt_classes']) >= 2] \
        or scene_rows
    eligible.sort(key=lambda r: float(r['ssc_miou_nyu_protocol']))
    return args.pick, (eligible[-1] if args.pick == 'best' else eligible[0])


def ensure_plys(role, row, args):
    out = os.path.join(args.pred_dir, 'ply', row['scene'])
    full = args.mask == 'none'
    names = {'pred': 'pred_full' if full else 'pred', 'gt': 'gt'}
    plys = {k: os.path.join(out, '%s_%s.ply' % (row['frame'], v)) for k, v in names.items()}
    if not os.path.exists(os.path.join(out, row['frame'] + '.json')) or not os.path.exists(plys['pred']):
        import export_scannet_ply as E
        ns = types.SimpleNamespace(scannet_root=args.scannet_root, pred_dir=args.pred_dir,
                                   scans_dir=os.path.join(args.scannet_root, 'scans'),
                                   out_dir=os.path.join(args.pred_dir, 'ply'))
        E.export_frame(role, row, ns, full_pred=full)
    return {k: p for k, p in plys.items() if os.path.exists(p)}


def list_frames(rows, n):
    rows = [r for r in rows if int(r['scored_voxels_nyu_protocol']) >= 2000 and int(r['gt_classes']) >= 2]
    rows.sort(key=lambda r: float(r['ssc_miou_nyu_protocol']))
    print('%d frames with >= 2000 scored voxels and >= 2 GT classes (NYU protocol)\n' % len(rows))
    for title, part in (('best', rows[::-1][:n]), ('worst', rows[:n])):
        print(title)
        for r in part:
            print('  %s %s  SSC mIoU %5.1f  SC IoU %5.1f  (%s voxels)' % (
                r['scene'], r['frame'], float(r['ssc_miou_nyu_protocol']), float(r['sc_iou_nyu_protocol'] or 0),
                r['scored_voxels_nyu_protocol']))


def main():
    argv = sys.argv[1:]
    extra = []
    if '--' in argv:
        i = argv.index('--')
        argv, extra = argv[:i], argv[i + 1:]
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('scene', nargs='?', help='e.g. scene0702_00')
    ap.add_argument('frame', nargs='?', help='e.g. 00094 (default: the scene\'s --pick frame)')
    ap.add_argument('--pick', choices=['best', 'worst'], default='best')
    ap.add_argument('--list', type=int, metavar='N', help='print the N best and worst frames overall and exit')
    ap.add_argument('--pred_dir', default=DEFAULT_PRED)
    ap.add_argument('--scannet_root', default=DEFAULT_ROOT)
    add_view_args(ap)
    ap.add_argument('--mask', choices=['label_weight', 'none'], default='label_weight',
                    help="label_weight: the prediction masked like test_NYU.py (default); none: the full "
                         "prediction, written as <frame>_pred_full.ply (the GT is the same either way)")
    ap.add_argument('--export_only', action='store_true',
                    help='write the .ply files and print their paths, open nothing (used by the .sh launcher)')
    args = ap.parse_args(argv)

    rows = read_rows(args)
    if args.list:
        return list_frames(rows, args.list)
    if not args.scene:
        ap.error('give a scene (and optionally a frame), or --list N')

    role, row = choose(rows, args)
    print('%s %s (%s)  SC IoU %s  SSC mIoU %s  (%s scored voxels, NYU protocol)' % (
        row['scene'], row['frame'], role, row['sc_iou_nyu_protocol'], row['ssc_miou_nyu_protocol'],
        row['scored_voxels_nyu_protocol']))
    plys = ensure_plys(role, row, args)
    if args.export_only:
        for k in ('pred', 'gt'):
            print('%s_ply=%s' % (k, plys.get(k, '')))
        return
    order = [k for k in ('pred', 'gt') if k in plys and (args.only in (None, k))]
    if not order:
        sys.exit('nothing to display')
    if args.screenshot:
        screenshot_pair([plys[k] for k in order],
                        ['CleanerS prediction' if k == 'pred' else 'Occ-ScanNet GT' for k in order],
                        args.screenshot, extra)
    else:
        open_windows([plys[k] for k in order], extra)


if __name__ == '__main__':
    main()
