"""
inference/nyu_occlusion_sweep.py

Measure, over every NYU frame, how much worse CleanerS gets at a class once
the camera can no longer see it. Written to answer one question about a live
capture -- "why does the bed go through the wall?" -- by checking whether the
same thing happens on the data the model was trained on. It does. See
document/OCCLUSION.md for the numbers and what they mean.

    python -m inference.nyu_occlusion_sweep --out /tmp/nyu_sweep.npz

~9 minutes for 1449 frames on a 3050 Ti (4 GB).

Method: every voxel with a real GT label is put in one of two buckets by what
OUR TSDF says about it,

    visible  = weight > 0 and |tsdf| < 0.25    the surface shell, directly seen
    occluded = tsdf < -0.5                     behind the surface, must be completed

and each bucket gets its own 12x12 confusion matrix. The gap between the two
recalls is the completion ability, isolated from the segmentation ability.

Also computes the benchmark's OWN metrics, so the finding can be stated in
the terms the paper uses. test_NYU.py:206-210 scores two different regions:

    SSC mask = label_weight & (label3d != 255)                    surface + occluded
    SC  mask = label_weight & (mapping == 307200) & (label3d != 255)   occluded only

`mapping == 307200` is MAPPING_SENTINEL: the voxel projects to no image pixel,
i.e. it is not on the visible surface. The published SC number is therefore an
occlusion metric already -- but a class-agnostic one (occupied vs empty). This
script also breaks that same mask down PER CLASS, which the paper does not,
and scores the surface complement for contrast.

Both use the REFERENCE label_weight (TSDF/arr_1) and mapping, matching
test_NYU.py, while the model input stays our Custom_TSDF -- the isolated-encoder
setup described in cleaner/dataset/NYU/NYU.py.

Also counts what lands in `occluded & GT == 255` -- NYU's out-of-room marker.
That is the analogue of a live capture whose voxel grid is deeper than the
room, where everything past the back wall is out-of-room but still rendered.

This reads the PRECOMPUTED Custom_TSDF/Custom_Mapping npz rather than calling
FrameLoader.build_tsdf_and_mapping() per frame, purely for speed. They are the
same arrays -- verified bit-identical (np.array_equal, max diff 0) -- and
--verify_encoder re-checks that on one frame before the sweep starts.
"""

import os
import sys
import time
import argparse

import cv2
import numpy as np
import torch

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_THIS_DIR)
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from inference import model as model_mod
from inference.run_inference import load_cfg
from inference.frame_loader import FrameLoader, IMG_H, IMG_W, MAPPING_SENTINEL

NUM_CLASSES = 12
CLASS_NAMES = ['empty', 'ceiling', 'floor', 'wall', 'window', 'chair',
               'bed', 'sofa', 'table', 'tvs', 'furn', 'objs']


def parse_args():
    p = argparse.ArgumentParser(description=__doc__.split('\n')[3])
    p.add_argument('--data_root', default=os.path.join(_REPO_ROOT, '..', 'data', 'NYU'))
    p.add_argument('--cfg', default=os.path.join(_REPO_ROOT, 'cfgs', 'NYU', 'voxelSSC.yaml'))
    p.add_argument('--pretrained_path',
                   default=os.path.join(_REPO_ROOT, 'checkpoint', 'CleanerS_ckpt.pth'))
    p.add_argument('--tsdf_dir', default='Custom_TSDF',
                   help="which encoder's output to feed the model; 'TSDF' is "
                        'the reference encoder, Custom_TSDF is ours')
    p.add_argument('--mapping_dir', default='Custom_Mapping',
                   help='mapping that feeds the MODEL')
    p.add_argument('--ref_mapping_dir', default='Mapping',
                   help="mapping used for the benchmark's own SC mask")
    p.add_argument('--ref_labelweight_dir', default='TSDF',
                   help='folder holding arr_1, the official eval mask')
    p.add_argument('--out', default='nyu_sweep.npz')
    p.add_argument('--limit', type=int, default=0, help='stop after N frames (debug)')
    p.add_argument('--verify_encoder', action='store_true',
                   help='re-check that the precomputed npz matches FrameLoader '
                        'on one frame before sweeping')
    return p.parse_args()


def find_frames(data_root):
    """-> ({frame_id: rgb_path}, {frame_id: split}). NYU ships RGB per split
    but depth and the npz sidecars in flat folders shared by both. The split
    matters: the published table is test-only (654 frames)."""
    out, split_of = {}, {}
    for split in ('test', 'train'):
        d = os.path.join(data_root, split, 'RGB')
        if not os.path.isdir(d):
            continue
        for f in sorted(os.listdir(d)):
            if f.endswith('.png'):
                out[f[3:7]] = os.path.join(d, f)
                split_of[f[3:7]] = split
    return out, split_of


def verify_encoder(args, key, rgb_path):
    """The sweep's one load-bearing assumption, checked rather than assumed."""
    depth_dir = os.path.join(args.data_root, 'depth')
    loader = FrameLoader.from_nyu_bin(
        os.path.join(depth_dir, 'NYU%s_0000.bin' % key),
        os.path.join(depth_dir, 'NYU%s_0000.png' % key), rgb_path=rgb_path)
    s = loader.build_tsdf_and_mapping()
    c = np.load(os.path.join(args.data_root, args.tsdf_dir, '%s.npz' % key))
    m = np.load(os.path.join(args.data_root, args.mapping_dir, '%s.npz' % key))
    ok_t = np.array_equal(np.asarray(s['tsdf']).reshape(-1), c['arr_0'])
    ok_m = np.array_equal(np.asarray(s['mapping']).reshape(-1), m['arr_0'])
    print('encoder check on %s: tsdf %s, mapping %s'
          % (key, 'IDENTICAL' if ok_t else 'DIFFERS', 'IDENTICAL' if ok_m else 'DIFFERS'))
    if not (ok_t and ok_m):
        raise SystemExit('precomputed npz does not match FrameLoader; the sweep '
                         'would not be measuring our pipeline')


def build_sample(data_root, tsdf_dir, mapping_dir, key, rgb_path):
    c = np.load(os.path.join(data_root, tsdf_dir, '%s.npz' % key))
    tsdf = c['arr_0']
    # our encoder writes arr_2 = weight; the reference TSDF folder has no such
    # array, so fall back to arr_1 (its label_weight) when reading that one
    weight = c['arr_2'] if 'arr_2' in c.files else c['arr_1']
    mapping = np.load(os.path.join(data_root, mapping_dir, '%s.npz' % key))['arr_0']

    m2d = (np.ones((IMG_H, IMG_W)) * -1).reshape(-1).astype(np.int64)
    valid = mapping != MAPPING_SENTINEL
    m2d[mapping[valid]] = np.nonzero(valid)[0]
    return {'tsdf': tsdf.reshape(1, 60, 36, 60).astype(np.float32),
            'weight': weight,
            'mapping': mapping,
            'mapping2d': m2d.reshape(IMG_H, IMG_W),
            'img': cv2.imread(rgb_path).astype(np.float32)}, tsdf, weight


def main():
    args = parse_args()
    frames, splits = find_frames(args.data_root)
    keys = sorted(frames)
    if args.limit:
        keys = keys[:args.limit]
    if not keys:
        raise SystemExit('no RGB found under %r' % args.data_root)

    if args.verify_encoder:
        verify_encoder(args, keys[0], frames[keys[0]])

    class _A(object):
        cfg, pretrained_path = args.cfg, args.pretrained_path
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model_mod.load_model(load_cfg(_A, []), device)
    print('model on %s; %d frames' % (device, len(keys)), flush=True)

    def _new():
        return {'vis': np.zeros((NUM_CLASSES, NUM_CLASSES), np.int64),
                'occ': np.zeros((NUM_CLASSES, NUM_CLASSES), np.int64),
                'sc': np.zeros((NUM_CLASSES, NUM_CLASSES), np.int64),
                'ssc': np.zeros((NUM_CLASSES, NUM_CLASSES), np.int64),
                'surf': np.zeros((NUM_CLASSES, NUM_CLASSES), np.int64),
                'scbin': np.zeros((2, 2), np.int64),
                'p255': np.zeros(NUM_CLASSES, np.int64),
                'n255': 0}
    acc = {'test': _new(), 'train': _new()}
    per_frame = []

    t0 = time.time()
    for i, key in enumerate(keys):
        sample, tsdf, weight = build_sample(
            args.data_root, args.tsdf_dir, args.mapping_dir, key, frames[key])
        label = np.load(os.path.join(args.data_root, 'Label', '%s.npz' % key))['arr_0']
        pred = model_mod.predict(model, sample, device)[0].reshape(-1)
        a = acc[splits[key]]

        # --- our split: what the TSDF says the camera could see ---
        visible = (weight > 0) & (np.abs(tsdf) < 0.25)
        occluded = tsdf < -0.5
        known = label != 255
        for msk, cm in ((visible & known, a['vis']), (occluded & known, a['occ'])):
            np.add.at(cm, (label[msk].astype(np.int64), pred[msk]), 1)
        out_of_room = occluded & (label == 255)
        np.add.at(a['p255'], pred[out_of_room], 1)
        a['n255'] += int(out_of_room.sum())

        # --- the benchmark's own split (test_NYU.py:206-210) ---
        lw = np.load(os.path.join(args.data_root, args.ref_labelweight_dir,
                                  '%s.npz' % key))['arr_1'].astype(bool)
        ref_map = np.load(os.path.join(args.data_root, args.ref_mapping_dir,
                                       '%s.npz' % key))['arr_0']
        m_ssc = lw & known
        m_sc = m_ssc & (ref_map == MAPPING_SENTINEL)      # occluded half
        m_surf = m_ssc & (ref_map != MAPPING_SENTINEL)    # visible surface half
        for msk, cm in ((m_ssc, a['ssc']), (m_sc, a['sc']), (m_surf, a['surf'])):
            np.add.at(cm, (label[msk].astype(np.int64), pred[msk]), 1)
        # the published SC number: occupied-vs-empty over the SC mask
        np.add.at(a['scbin'], ((label[m_sc] > 0).astype(np.int64),
                               (pred[m_sc] > 0).astype(np.int64)), 1)

        wv, wo = (label == 3) & visible, (label == 3) & occluded
        per_frame.append((int(key), wv.sum(), (pred[wv] == 3).sum(),
                          wo.sum(), (pred[wo] == 3).sum()))
        if i % 50 == 0:
            print('%4d/%d  %.1f frames/s'
                  % (i, len(keys), (i + 1) / (time.time() - t0)), flush=True)

    out = {'per': np.array(per_frame), 'keys': np.array([int(k) for k in keys])}
    for sp in ('test', 'train'):
        for k, v in acc[sp].items():
            out['%s_%s' % (sp, k)] = np.asarray(v)
    np.savez(args.out, **out)
    print('done %d frames in %.1f min -> %s'
          % (len(keys), (time.time() - t0) / 60.0, args.out), flush=True)
    report(args.out, split=('test',))
    report(args.out, split=('test', 'train'))


PUBLISHED = {  # CleanerS* (Segformer-B2), Table 1, NYU test set
    'sc': (88.0, 83.5, 75.0),
    'ssc': [46.3, 93.9, 43.2, 33.7, 38.5, 62.2, 54.8, 33.7, 39.2, 45.7, 33.8],
}


def per_class_iou(cm):
    """IoU per class from a GT x pred confusion matrix. Classes absent from
    both GT and prediction return nan rather than 0, so they do not drag a
    mean down as if they had been got wrong."""
    tp = np.diag(cm).astype(np.float64)
    denom = cm.sum(1) + cm.sum(0) - tp
    with np.errstate(invalid='ignore', divide='ignore'):
        return np.where(denom > 0, tp / denom, np.nan)


def _add(acc, z, split, name):
    return sum(z['%s_%s' % (sp, name)] for sp in split) if len(acc) else None


def report(path, split=('test',)):
    z = np.load(path)
    get = lambda name: sum(z['%s_%s' % (sp, name)] for sp in split)
    tag = '+'.join(split)

    # ---- the benchmark's own numbers, to prove the masks are right ----
    b = get('scbin')
    prec = 100.0 * b[1, 1] / max(b[:, 1].sum(), 1)
    rec = 100.0 * b[1, 1] / max(b[1].sum(), 1)
    iou = 100.0 * b[1, 1] / max(b[1].sum() + b[:, 1].sum() - b[1, 1], 1)
    pp, pr, pi = PUBLISHED['sc']
    print('\nSCENE COMPLETION, official mask (%s split)' % tag)
    print('   %-12s %8s %8s %8s' % ('', 'prec.', 'recall', 'IoU'))
    print('   %-12s %7.1f%% %7.1f%% %7.1f%%' % ('ours', prec, rec, iou))
    print('   %-12s %7.1f%% %7.1f%% %7.1f%%' % ('published', pp, pr, pi))

    ssc = 100 * per_class_iou(get('ssc'))
    sc = 100 * per_class_iou(get('sc'))
    surf = 100 * per_class_iou(get('surf'))
    print('\nPER-CLASS IoU ON THE OFFICIAL MASKS (%s split)' % tag)
    print('   %-8s %8s %8s | %8s %8s %9s' % ('class', 'pub SSC', 'our SSC',
                                             'surface', 'occluded', 'drop'))
    for c in range(1, NUM_CLASSES):
        print('   %-8s %7.1f  %7.1f  | %7.1f  %7.1f  %+8.1f'
              % (CLASS_NAMES[c], PUBLISHED['ssc'][c - 1], ssc[c],
                 surf[c], sc[c], sc[c] - surf[c]))
    print('   %-8s %7.1f  %7.1f  | %7.1f  %7.1f  %+8.1f'
          % ('avg', np.mean(PUBLISHED['ssc']), np.nanmean(ssc[1:]),
             np.nanmean(surf[1:]), np.nanmean(sc[1:]),
             np.nanmean(sc[1:]) - np.nanmean(surf[1:])))

    # ---- our own visibility split, over whatever splits were asked for ----
    v, o = get('vis'), get('occ')
    print('\nRECALL BY VISIBILITY, our TSDF split (%s split)' % tag)
    print('   %-8s %10s %8s | %10s %8s | %s'
          % ('class', 'visible n', 'recall', 'occluded n', 'recall', 'drop'))
    for c in range(1, NUM_CLASSES):
        dv, do = v[c].sum(), o[c].sum()
        if dv < 1000 or do < 1000:
            continue
        rv, ro = 100.0 * v[c, c] / dv, 100.0 * o[c, c] / do
        print('   %-8s %10d %7.1f%% | %10d %7.1f%% | %+.1f pt'
              % (CLASS_NAMES[c], dv, rv, do, ro, ro - rv))

    print('\nWHAT OCCLUDED GT-WALL BECOMES (%d voxels)' % o[3].sum())
    for i in np.argsort(-o[3])[:6]:
        print('   %-8s %7.1f%%' % (CLASS_NAMES[i], 100.0 * o[3, i] / o[3].sum()))

    p, n = get('p255'), int(get('n255'))
    print('\nPREDICTED IN occluded & GT==255, i.e. outside the room (%d voxels)' % n)
    for i in np.argsort(-p)[:5]:
        print('   %-8s %7.1f%%' % (CLASS_NAMES[i], 100.0 * p[i] / p.sum()))

    per = z['per']
    ok = (per[:, 1] > 200) & (per[:, 3] > 200)
    rv, ro = per[ok, 2] / per[ok, 1], per[ok, 4] / per[ok, 3]
    print('\nPER-FRAME wall recall, all %d frames (%d with >200 voxels each side)'
          % (len(per), ok.sum()))
    print('   visible  median %.1f%%   occluded median %.1f%%'
          % (100 * np.median(rv), 100 * np.median(ro)))
    for thr in (0.5, 0.3):
        print('   occluded recall < %d%%: %d frames (%.0f%%)'
              % (100 * thr, (ro < thr).sum(), 100 * (ro < thr).mean()))


if __name__ == '__main__':
    main()
