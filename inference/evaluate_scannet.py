"""
inference/evaluate_scannet.py

Scores run_scannet.py predictions against the Occ-ScanNet GT in
data/Scannet/<scene>/<frame>.npz and writes a markdown report.

The scored set is a choice, and each answers a different question:

  all-voxels / target      every voxel with GT != 255 -- the set Occ-ScanNet
                           papers score (ISO, MonoScene). Compare with them here.
  all-voxels / target_fov  same, with voxels outside the colour camera's view
                           removed (they are "empty" in the released GT only
                           because nothing looked at them).
  cleaners / target        NYU protocol of evaluate.py / test_NYU.py:
                           label_weight = GT object | tsdf < -0.5, GT != 255;
                           SC further restricted to mapping == 307200.
  cleaners / target_fov    same on the out-of-view-removed GT.

label_weight is rebuilt from the GT and our own encoded tsdf, exactly as
data/generate_cleaners_data.py's compute_label_weight does for NYU.

Beyond the tables it measures what differs from NYU:
  - NYU-like views: frames whose camera pitch and height fall inside NYU's own
    5th-95th percentile range (measured from the NYU .bin poses), scored
    separately. Grid placement is the same recipe on both datasets (grid centre
    ~3.3 m ahead, heading spread 0-45 deg on both), so pitch is the viewpoint
    difference; height is close.
  - heading: angle between the camera's heading and the nearest grid axis
    (0 = along an axis, 45 = diagonal), pooled and within-scene.
  - pitch: ScanNet is hand-held and looks down more than NYU.
  - where the errors sit: surface voxels a depth pixel reached, space the
    camera saw through, and hidden space.
  - the SC gap at matched difficulty: SC rises with the occupancy of the scored
    set (which is also what a constant "occupied" prediction scores), and NYU's
    is 21 points higher, so the two are compared band by band and split into
    precision and recall, which do not move with it. That table comes from
    outputs/sc_gap.csv -- build it with
        python -m reconstruction_GT.sc_gap --all
    and the section is skipped with a pointer if the file is absent.

Usage (from the repo root):
    python -m inference.evaluate_scannet --pred_dir ./outputs/scannet \
        --report inference/document/EVALUATION_SCANNET.md
"""

import argparse
import csv
import datetime
import glob
import os
import sys

import numpy as np

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _THIS_DIR)
sys.path.insert(0, os.path.dirname(_THIS_DIR))
from evaluate import (confusion, metrics, load_ids, CLASSES, N_CLASSES, IGNORE,  # noqa: E402
                      MAPPING_SENTINEL, DEFAULT_NYU, DEFAULT_PRED,
                      table, sc_breakdown, sc_class_rows, SC_COLUMNS)

DEFAULT_ROOT = os.path.join(os.path.dirname(os.path.dirname(_THIS_DIR)), 'data', 'Scannet')

ROWS = [('all-voxels', 'target'), ('all-voxels', 'target_fov'),
        ('cleaners', 'target'), ('cleaners', 'target_fov')]
# free-space carving: the prediction with every voxel the depth frame saw through (tsdf > 0) set to
# empty. A post-process from the model's own measured input, reported only as a labelled extra row.
CARVE = ('all-voxels + carve', 'target')
# CleanerS's rule with floor taken out: a voxel that is floor in the GT or in the prediction is not
# scored, and mIoU runs over the other 10 classes. Occ-ScanNet's grid sits 5 cm below world z = 0
# rather than below the annotated floor, so its floor lands 1-2 layers above where the model puts it.
NOFLOOR_ROWS = [('cleaners', 'target'), ('cleaners', 'target_fov')]
# The row the report is built around, as EVALUATION.md is built around NYU test:
# test_NYU.py's rule on the released Occ-ScanNet labels.
PRIMARY = ('cleaners', 'target')
FLOOR = 2
MAIN = ('all-voxels', 'target')           # comparable with published tables
STRICT = ('cleaners', 'target_fov')       # CleanerS protocol, nothing unseen

# ISO (ECCV 2024) Table 1, Occ-ScanNet val, classes in NYU order.
PUBLISHED = {
    'ISO': (42.16, [19.88, 41.88, 22.37, 16.98, 29.09, 42.43, 42.00, 29.60, 10.62, 36.36, 24.61], 28.71),
    'MonoScene': (41.60, [15.17, 44.71, 22.41, 12.55, 26.11, 27.03, 35.91, 28.32, 6.57, 32.16, 19.84], 24.62),
}

# ISO paper Table 2, NYUv2 test -- same all-voxels rule (iso/models/iso.py passes no mask to
# sscMetrics.add_batch, which keeps y_true != 255). Input column as the paper gives it.
PUBLISHED_NYU = {
    'ISO': ('RGB', 47.11, [14.21, 93.47, 15.89, 15.14, 18.35, 50.01, 40.82, 18.25, 25.90, 34.08, 17.67], 31.25),
    'NDC-Scene': ('RGB', 44.17, [12.02, 93.51, 13.11, 13.77, 15.83, 49.57, 39.87, 17.17, 24.57, 31.00, 14.96], 29.03),
    'MonoScene': ('RGB', 42.51, [8.89, 93.50, 12.06, 12.57, 13.72, 48.19, 36.11, 15.13, 15.22, 27.96, 12.94], 26.94),
    '3DSketch': ('RGB + TSDF', 38.64, [8.53, 90.45, 9.94, 5.67, 10.64, 42.29, 29.21, 13.88, 9.38, 23.83, 8.19], 22.91),
}

HEADING_BINS = [0, 7.5, 15, 22.5, 30, 37.5, 45.01]
PITCH_BINS = [-90, -40, -25, -10, 10, 90]
REGIONS = ['surface (a depth pixel landed)', 'seen through (tsdf > 0)', 'hidden (neither)']


# NYU (1449 frames) 5th-95th percentiles, used when the .bin headers are absent
NYU_VIEW_FALLBACK = dict(pitch=(-23.34, -0.08), height=(1.07, 1.51), pitch_median=-13.55,
                         height_median=1.34, heading_median=23.24, n=1449, measured=False)


def nyu_view_ranges(nyu_root):
    """Pitch and camera-height ranges that count as an NYU-like view, measured
    from every NYU .bin header (cam_pose; world +Z up, floor at Z = 0)."""
    bins = sorted(glob.glob(os.path.join(nyu_root, 'depth', '*.bin')))
    if not bins:
        return NYU_VIEW_FALLBACK
    P, Z, H = [], [], []
    for b in bins:
        with open(b, 'rb') as f:
            f.read(12)
            pose = np.frombuffer(f.read(64), dtype=np.float32).reshape(4, 4).astype(np.float64)
        h, p = heading_and_pitch(pose)
        P.append(p); Z.append(pose[2, 3]); H.append(h)
    P, Z, H = np.array(P), np.array(Z), np.array(H)
    return dict(pitch=tuple(np.percentile(P, [5, 95])), height=tuple(np.percentile(Z, [5, 95])),
                pitch_median=float(np.median(P)), height_median=float(np.median(Z)),
                heading_median=float(np.nanmedian(H)), n=len(bins), measured=True)


def is_nyu_like(pitch, height, ranges):
    return (ranges['pitch'][0] <= pitch <= ranges['pitch'][1]
            and ranges['height'][0] <= height <= ranges['height'][1])


def scored_sets(protocol, label, tsdf, mapping):
    if protocol == 'cleaners':
        lw = ((label > 0) & (label < 255)) | (tsdf < -0.5)
        keep = lw & (label != IGNORE)
        return keep, keep & (mapping == MAPPING_SENTINEL)
    keep = label != IGNORE
    return keep, keep


def heading_and_pitch(cam_pose):
    """Degrees. heading = angle between the camera's horizontal forward
    direction and the nearest world X/Y axis, in [0, 45]; pitch > 0 = looking up.
    World +Z is up (Occ-ScanNet grid z)."""
    fwd = cam_pose[:3, 2]
    pitch = np.degrees(np.arcsin(np.clip(fwd[2], -1, 1)))
    horiz = np.hypot(fwd[0], fwd[1])
    if horiz < 0.3:
        return np.nan, pitch
    a = np.degrees(np.arctan2(fwd[1], fwd[0])) % 90.0
    return min(a, 90.0 - a), pitch


def sc_ssc(cm_ssc, cm_sc):
    return 100 * metrics(cm_sc)['iou'][1], 100 * np.nanmean(metrics(cm_ssc)['iou'][1:])


# --------------------------------------------------------------- accumulate
def evaluate(args):
    cm = {r: np.zeros((N_CLASSES, N_CLASSES), np.int64) for r in ROWS}
    cm_sc = {r: np.zeros((2, 2), np.int64) for r in ROWS}
    cm_sc12 = np.zeros((N_CLASSES, N_CLASSES), np.int64)     # PRIMARY, SC region, 12 classes
    geo = dict(lw_total=0, inside=0)
    cm_nf = {r: np.zeros((N_CLASSES, N_CLASSES), np.int64) for r in NOFLOOR_ROWS}
    cm_sc_nf = {r: np.zeros((2, 2), np.int64) for r in NOFLOOR_ROWS}
    cm_carve = np.zeros((N_CLASSES, N_CLASSES), np.int64)
    cm_sc_carve = np.zeros((2, 2), np.int64)
    diag = new_diag()
    cm_nl = {r: np.zeros((N_CLASSES, N_CLASSES), np.int64) for r in ROWS}     # NYU-like views only
    cm_sc_nl = {r: np.zeros((2, 2), np.int64) for r in ROWS}
    n_nl = 0
    region_gt = np.zeros((N_CLASSES, 3), np.int64)      # MAIN row, GT object voxels by region
    region_hit = np.zeros((N_CLASSES, 3), np.int64)     # ... of those, predicted correctly
    region_occ = np.zeros((3, 2), np.int64)             # GT occupied voxels by region: [total, predicted occupied]
    bins_h = {b: [np.zeros((N_CLASSES, N_CLASSES), np.int64), np.zeros((2, 2), np.int64), 0]
              for b in range(len(HEADING_BINS) - 1)}
    bins_p = {b: [np.zeros((N_CLASSES, N_CLASSES), np.int64), np.zeros((2, 2), np.int64), 0]
              for b in range(len(PITCH_BINS) - 1)}
    per_scene = {}                                       # scene -> {'aligned'|'diagonal': [cm, cm_sc, n]}
    rows_csv = []
    n = 0

    preds = sorted(glob.glob(os.path.join(args.pred_dir, 'prediction', '*', '*.npy')))
    for i, pred_path in enumerate(preds):
        scene = os.path.basename(os.path.dirname(pred_path))
        frame = os.path.splitext(os.path.basename(pred_path))[0]
        enc_path = os.path.join(args.pred_dir, 'encoded', scene, frame + '.npz')
        if not os.path.exists(enc_path):
            continue                                     # being written right now
        enc = np.load(enc_path)
        gt = np.load(os.path.join(args.scannet_root, scene, frame + '.npz'))
        pred = np.load(pred_path).reshape(-1).astype(np.int64)
        tsdf, mapping = enc['tsdf'].reshape(-1), enc['mapping'].reshape(-1)

        frame_scores = {}
        add_diag(diag, pred, gt['target'].reshape(-1).astype(np.int64), mapping)
        label_c = gt[CARVE[1]].reshape(-1).astype(np.int64)
        keep_c = label_c != IGNORE
        pred_c = np.where(tsdf > 0, 0, pred)
        c_c = confusion(pred_c[keep_c], label_c[keep_c])
        s_c = confusion((pred_c[keep_c] > 0).astype(np.int64), (label_c[keep_c] > 0).astype(np.int64), n=2)
        cm_carve += c_c
        cm_sc_carve += s_c
        for row in ROWS:
            protocol, which = row
            label = gt[which].reshape(-1).astype(np.int64)
            keep, sc = scored_sets(protocol, label, tsdf, mapping)
            c = confusion(pred[keep], label[keep])
            s = confusion((pred[sc] > 0).astype(np.int64), (label[sc] > 0).astype(np.int64), n=2)
            cm[row] += c
            cm_sc[row] += s
            frame_scores[row] = (c, s)
            if row == PRIMARY:
                cm_sc12 += confusion(pred[sc], label[sc])
                geo['lw_total'] += label.size
                geo['inside'] += int((((label > 0) & (label < 255)) | (tsdf < -0.5)).sum())
            if row in NOFLOOR_ROWS:
                k = keep & (label != FLOOR) & (pred != FLOOR)
                ks = k & (mapping == MAPPING_SENTINEL)
                cm_nf[row] += confusion(pred[k], label[k])
                cm_sc_nf[row] += confusion((pred[ks] > 0).astype(np.int64), (label[ks] > 0).astype(np.int64), n=2)

        # where errors sit, on MAIN
        label = gt[MAIN[1]].reshape(-1).astype(np.int64)
        region = np.where(mapping != MAPPING_SENTINEL, 0, np.where(tsdf > 0, 1, 2))
        obj = (label > 0) & (label != IGNORE)
        region_gt += np.bincount((label * 3 + region)[obj], minlength=N_CLASSES * 3).reshape(N_CLASSES, 3)
        region_hit += np.bincount((label * 3 + region)[obj & (pred == label)],
                                  minlength=N_CLASSES * 3).reshape(N_CLASSES, 3)
        for r in range(3):
            m = obj & (region == r)
            region_occ[r] += (int(m.sum()), int((m & (pred > 0)).sum()))

        # geometry of the view
        heading, pitch = heading_and_pitch(gt['cam_pose'])
        height = float(gt['cam_pose'][2, 3])
        nyu_like = is_nyu_like(pitch, height, args.ranges)
        if nyu_like:
            for row in ROWS:
                cm_nl[row] += frame_scores[row][0]
                cm_sc_nl[row] += frame_scores[row][1]
            n_nl += 1
        c, s = frame_scores[MAIN]
        if not np.isnan(heading):
            b = np.searchsorted(HEADING_BINS, heading, side='right') - 1
            bins_h[b][0] += c; bins_h[b][1] += s; bins_h[b][2] += 1
            if heading <= 10 or heading >= 35:
                key = 'aligned' if heading <= 10 else 'diagonal'
                d = per_scene.setdefault(scene, {}).setdefault(
                    key, [np.zeros((N_CLASSES, N_CLASSES), np.int64), np.zeros((2, 2), np.int64), 0])
                d[0] += c; d[1] += s; d[2] += 1
        b = int(np.clip(np.searchsorted(PITCH_BINS, pitch, side='right') - 1, 0, len(PITCH_BINS) - 2))
        bins_p[b][0] += c; bins_p[b][1] += s; bins_p[b][2] += 1

        sc_main, _ = sc_ssc(*frame_scores[MAIN])
        present = [k for k in range(1, N_CLASSES)
                   if frame_scores[MAIN][0][k].sum() + frame_scores[MAIN][0][:, k].sum() > 0]
        iou_main = metrics(frame_scores[MAIN][0])['iou']
        # NYU protocol per frame (test_NYU.py's rule; target and target_fov score identically
        # under it). export_scannet_ply.py ranks frames by these columns.
        c_nyu, s_nyu = frame_scores[('cleaners', 'target')]
        m_nyu = metrics(c_nyu)
        rows_csv.append(dict(scene=scene, frame=frame,
                             sc_iou_nyu_protocol=round(float(100 * metrics(s_nyu)['iou'][1]), 2)
                             if s_nyu.sum() else '',
                             ssc_miou_nyu_protocol=round(float(100 * np.nanmean(m_nyu['iou'][1:])), 2)
                             if np.isfinite(m_nyu['iou'][1:]).any() else '',
                             scored_voxels_nyu_protocol=int(c_nyu.sum()),
                             gt_classes=int(sum(c_nyu[k].sum() > 0 for k in range(1, N_CLASSES))),
                             heading_deg=round(float(heading), 2), pitch_deg=round(float(pitch), 2),
                             camera_height_m=round(height, 3), nyu_like=int(nyu_like),
                             sc_iou_all_voxels=round(float(sc_main), 2),
                             ssc_miou_present_all_voxels=round(float(100 * np.nanmean(iou_main[present])), 2) if present else '',
                             mapped_voxels=int((mapping != MAPPING_SENTINEL).sum()),
                             gt_occupied=int(obj.sum())))
        n += 1
        if (i + 1) % 2000 == 0:
            print('  %d/%d' % (i + 1, len(preds)), file=sys.stderr, flush=True)

    return dict(diag=diag, cm=cm, cm_sc=cm_sc, cm_sc12=cm_sc12, geo=geo, cm_nf=cm_nf, cm_sc_nf=cm_sc_nf, cm_carve=cm_carve, cm_sc_carve=cm_sc_carve, cm_nl=cm_nl, cm_sc_nl=cm_sc_nl, n_nl=n_nl, region_gt=region_gt, region_hit=region_hit, region_occ=region_occ,
                bins_h=bins_h, bins_p=bins_p, per_scene=per_scene, rows=rows_csv, n=n,
                scenes=len({r['scene'] for r in rows_csv}))


def new_diag():
    return dict(cm_surface=np.zeros((N_CLASSES, N_CLASSES), np.int64),
                floor_gt=np.zeros(36, np.int64), floor_pred=np.zeros(36, np.int64),
                floor_as=np.zeros(N_CLASSES, np.int64))


def add_diag(d, pred, label, mapping):
    """Two model-vs-annotation checks. cm_surface: voxels a depth pixel landed in that the GT
    calls an object -- pure recognition, no completion involved. floor_*: floor voxels per
    height layer (grid axis 1), GT vs prediction, and what GT floor gets predicted as."""
    surf = (mapping != MAPPING_SENTINEL) & (label > 0) & (label != IGNORE)
    d['cm_surface'] += confusion(pred[surf], label[surf])
    lab3, pred3 = label.reshape(60, 36, 60), pred.reshape(60, 36, 60)
    d['floor_gt'] += (lab3 == 2).sum(axis=(0, 2))
    d['floor_pred'] += (pred3 == 2).sum(axis=(0, 2))
    d['floor_as'] += np.bincount(pred[label == 2], minlength=N_CLASSES)


def nyu_reference(nyu_root, pred_dir, split='test'):
    """CleanerS on its own NYU split, scored both ways. The cleaners row must
    reproduce document/EVALUATION.md (SC 75.0 / mIoU 47.7) -- a check on this
    script -- and the all-voxels row shows what the Occ-ScanNet protocol does to
    the same predictions, with no domain change."""
    out = {p: [np.zeros((N_CLASSES, N_CLASSES), np.int64), np.zeros((2, 2), np.int64)]
           for p in ('cleaners', 'cleaners, floor removed', 'all-voxels', 'all-voxels + carve')}
    out['diag'] = new_diag()
    n = 0
    for fid in load_ids(nyu_root, split):
        paths = [os.path.join(pred_dir, 'NYU%s_0000.npy' % fid)] + [
            os.path.join(nyu_root, d, '%s.npz' % fid) for d in ('Label', 'TSDF', 'Mapping')]
        if not all(os.path.exists(q) for q in paths):
            continue
        pred = np.load(paths[0]).reshape(-1).astype(np.int64)
        label = np.load(paths[1])['arr_0'].reshape(-1).astype(np.int64)
        tsdf_z = np.load(paths[2])
        tsdf, lw = tsdf_z['arr_0'].reshape(-1), tsdf_z['arr_1'].reshape(-1) > 0
        mapping = np.load(paths[3])['arr_0'].reshape(-1)
        add_diag(out['diag'], pred, label, mapping)
        for protocol, (c, s) in ((k, v) for k, v in out.items() if k != 'diag'):
            p_ = pred
            if protocol.startswith('cleaners'):      # shipped label_weight, exactly as evaluate.py
                keep = lw & (label != IGNORE)
                if protocol.endswith('floor removed'):
                    keep = keep & (label != FLOOR) & (pred != FLOOR)
                sc = keep & (mapping == MAPPING_SENTINEL)
            else:
                keep, sc = scored_sets('all-voxels', label, tsdf, mapping)
                if protocol.endswith('carve'):
                    p_ = np.where(tsdf > 0, 0, pred)
            c += confusion(p_[keep], label[keep])
            s += confusion((p_[sc] > 0).astype(np.int64), (label[sc] > 0).astype(np.int64), n=2)
        n += 1
    return out, n


# ------------------------------------------------------------------ report
def pct(v):
    return '—' if v is None or (isinstance(v, float) and np.isnan(v)) else '%.1f' % v


def class_table(cmx):
    m = metrics(cmx)
    out = ['| class | GT voxels | precision | recall | IoU |', '| --- | ---: | ---: | ---: | ---: |']
    for k, name in enumerate(CLASSES):
        out.append('| `%s`%s | {:,} | %s | %s | %s |'.format(int(cmx[k].sum()))
                   % (name, ' *(not in mIoU)*' if k == 0 else '', pct(100 * m['precision'][k]),
                      pct(100 * m['recall'][k]), pct(100 * m['iou'][k])))
    return '\n'.join(out)


def top_confusions(cmx, top=10):
    off = cmx.copy()
    np.fill_diagonal(off, 0)
    idx = np.dstack(np.unravel_index(np.argsort(off.ravel())[::-1], off.shape))[0][:top]
    total = cmx.sum(axis=1)
    out = ['| GT | predicted as | voxels | share of that GT class |', '| --- | --- | ---: | ---: |']
    for t, p in idx:
        out.append('| `%s` | `%s` | {:,} | %.1f%% |'.format(int(off[t, p])) % (CLASSES[t], CLASSES[p], 100 * off[t, p] / max(total[t], 1)))
    return '\n'.join(out)


def fp_split(cmx):
    out = ['| predicted class | false positives | on GT `empty` | on another class | largest other class |',
           '| --- | ---: | ---: | ---: | --- |']
    for k in range(1, N_CLASSES):
        col = cmx[:, k].copy()
        fp = int(col.sum() - col[k])
        if fp == 0:
            continue
        other = col.copy(); other[[0, k]] = 0
        o = int(other.argmax())
        out.append('| `%s` | {:,} | %.0f%% | %.0f%% | `%s` (%.0f%%) |'.format(fp)
                   % (CLASSES[k], 100 * col[0] / fp, 100 * other.sum() / fp, CLASSES[o], 100 * other[o] / fp))
    return '\n'.join(out)


def bin_table(bins, edges, label):
    out = ['| %s | frames | SC IoU | SSC mIoU |' % label, '| --- | ---: | ---: | ---: |']
    for b in sorted(bins):
        c, s, k = bins[b]
        if k == 0:
            continue
        sc, ssc = sc_ssc(c, s)
        hi = min(edges[b + 1], 45 if label.startswith('heading') else edges[b + 1])
        out.append('| %g – %g° | %d | %.1f | %.1f |' % (edges[b], hi, k, sc, ssc))
    return '\n'.join(out)


def within_scene(per_scene, min_frames=3):
    d_sc, d_ssc = [], []
    for scene, v in per_scene.items():
        if 'aligned' in v and 'diagonal' in v and v['aligned'][2] >= min_frames and v['diagonal'][2] >= min_frames:
            a, g = sc_ssc(v['aligned'][0], v['aligned'][1]), sc_ssc(v['diagonal'][0], v['diagonal'][1])
            d_sc.append(a[0] - g[0]); d_ssc.append(a[1] - g[1])
    return np.array(d_sc), np.array(d_ssc)


def diagnosis_section(res):
    """Why the like-for-like (cleaners) score is still far below NYU: annotation or model?"""
    nyu, n_nyu = res['nyu']
    if not n_nyu:
        return []
    dn, ds = nyu['diag'], res['diag']
    L = ['## Why ScanNet still scores below NYU: annotation or model?\n']
    L.append('Same scoring rule, so the remaining gap is the ground truth, the model, or the input. Three '
             'checks separate them.\n')

    L.append('### 1. Floor height — the grid assumption breaks, and the model follows a habit\n')
    fg_n, fp_n = dn['floor_gt'] / max(dn['floor_gt'].sum(), 1), dn['floor_pred'] / max(dn['floor_pred'].sum(), 1)
    fg_s, fp_s = ds['floor_gt'] / max(ds['floor_gt'].sum(), 1), ds['floor_pred'] / max(ds['floor_pred'].sum(), 1)
    L.append('Share of floor voxels in each height layer (layer 0 = 5 cm below the grid\'s assumed floor to 3 cm '
             'above it; one layer = 8 cm):\n')
    L.append('| | layer 0 | layer 1 | layer 2 | layer 3 | layers 4+ |')
    L.append('| --- | ---: | ---: | ---: | ---: | ---: |')
    for name, v in (('NYU GT', fg_n), ('NYU prediction', fp_n), ('Occ-ScanNet GT', fg_s), ('Occ-ScanNet prediction', fp_s)):
        L.append('| %s | %.0f%% | %.0f%% | %.0f%% | %.0f%% | %.0f%% |' % (name, *(100 * v[:4]), 100 * v[4:].sum()))
    L.append('')
    fa = ds['floor_as'] / max(ds['floor_as'].sum(), 1)
    fa_n = dn['floor_as'] / max(dn['floor_as'].sum(), 1)
    top = np.argsort(fa)[::-1][:4]
    L.append('GT floor voxels are predicted as floor %.0f%% of the time on NYU, and %.0f%% on Occ-ScanNet '
             '(there: %s).\n' % (100 * fa_n[2], 100 * fa[2], ', '.join('`%s` %.0f%%' % (CLASSES[k], 100 * fa[k]) for k in top)))
    L.append('- **Annotation / dataset construction.** Occ-ScanNet puts the bottom of every grid 5 cm below world '
             '*z = 0* (`height_belowfloor = -0.05` in its `generate_gt.py`), assuming the ScanNet floor sits at 0. '
             'NYU places it 5 cm below the *annotated* floor, so NYU\'s floor is always layer 0. ScanNet scans are '
             'not built that way, and the GT floor lands mostly one or two layers up.')
    L.append('- **Model.** The depth it is given shows the floor at that higher layer — the same grid is used for '
             'the input — yet it still paints floor in layer 0. It learned "floor = bottom layer" from NYU, where '
             'that was always true, rather than reading the height from depth.')
    L.append('- Every other class\'s height relative to the floor shifts with it, so this is not only a floor '
             'problem. It is fixable without new data: place each grid 5 cm below the floor the GT annotates, as '
             'NYU does, re-encode and re-run.\n')

    L.append('### 2. Recognition on measured surfaces — the model and the transfer\n')
    L.append('Only voxels a depth pixel landed in **and** the GT calls an object. Nothing here needs completing, '
             'so hidden-part annotation cannot affect it.\n')
    mn, ms = metrics(dn['cm_surface']), metrics(ds['cm_surface'])
    acc_n = np.diag(dn['cm_surface']).sum() / max(dn['cm_surface'].sum(), 1)
    acc_s = np.diag(ds['cm_surface']).sum() / max(ds['cm_surface'].sum(), 1)
    L.append('| | right class | ' + ' | '.join(CLASSES[1:]) + ' |')
    L.append('| --- | ---: |' + ' ---: |' * (N_CLASSES - 1))
    L.append('| NYU recall | %.1f%% | %s |' % (100 * acc_n, ' | '.join('%.0f' % (100 * v) for v in mn['recall'][1:])))
    L.append('| Occ-ScanNet recall | %.1f%% | %s |' % (100 * acc_s, ' | '.join('%.0f' % (100 * v) for v in ms['recall'][1:])))
    L.append('| change | %+.1f | %s |\n' % (100 * (acc_s - acc_n), ' | '.join('%+.0f' % (100 * (b - a)) for a, b in zip(mn['recall'][1:], ms['recall'][1:]))))
    drops = 100 * (ms['recall'][1:] - mn['recall'][1:])
    big = [CLASSES[k + 1] for k in np.argsort(drops)[:4] if CLASSES[k + 1] != 'floor']
    L.append('Classes that drop a little are ordinary transfer loss: a new sensor and new rooms. The largest drops besides floor (%s) are too '
             'big for that alone and may come from class definitions — CompleteScanNet\'s categories mapped onto '
             'NYU\'s 11 need not match what NYU\'s annotators called those objects. Not verified: that needs '
             'CompleteScanNet\'s raw labels.\n' % ', '.join('`%s`' % c for c in big))

    try:
        from reconstruction_GT.sc_gap import markdown_section
        sec = markdown_section()
    except ImportError:
        sec = []
    if sec:
        L += sec
    else:
        L.append('### 3. The gap at matched difficulty\n')
        L.append('Needs `outputs/sc_gap.csv`: `python -m reconstruction_GT.sc_gap --all`.\n')

    L.append('### 4. What is left\n')
    L.append('Completion of hidden parts is scored against a different recipe — CompleteScanNet CAD voxels copied '
             'onto the 8 cm grid by nearest neighbour, against NYU\'s 2 cm solids with the 4×4×4 rule — and '
             '%.1f%% of Occ-ScanNet\'s GT object voxels sit where the depth sensor measured free space. Separating '
             'that from the model needs GT rebuilt NYU\'s way from CompleteScanNet.\n'
             % (100 * res['region_occ'][1, 0] / max(res['region_occ'][:, 0].sum(), 1)))
    return L


def rgbd_vs_rgb_section(res):
    """Does depth input (CleanerS) beat RGB-only (ISO)? Two tests, each under
    ISO's own scoring rule, and only one of them on equal training data."""
    L = ['## RGB-D CleanerS vs RGB-only ISO\n']
    L.append('ISO scores every voxel whose GT is not 255 (`iso/models/iso.py` calls `metric.add_batch(y_pred, '
             'y_true)` with no mask; `sscMetrics.add_batch` keeps `y_true != 255`; mIoU over classes 1–11). That '
             'is this report\'s **all-voxels / `target`** row, so the scoring below is ISO\'s, not CleanerS\'s.\n')
    names = CLASSES[1:]
    head = '| method | input | trained on | SC IoU | ' + ' | '.join(names) + ' | mIoU |'
    sep = '|' + ' --- |' * 3 + ' ---: |' * (len(names) + 2)
    fmt = lambda v: ' | '.join('%.1f' % x for x in v)

    nyu, n_nyu = res['nyu']
    L.append('### 1. NYUv2 test — same training data, same scoring\n')
    L.append('Both models were trained on NYUv2, so this is the fair test of RGB-D against RGB-only.\n')
    L.append(head); L.append(sep)
    verdict_nyu = None
    if n_nyu:
        c, s_ = nyu['all-voxels']
        iou = 100 * metrics(c)['iou']
        sc_c, miou_c = 100 * metrics(s_)['iou'][1], np.nanmean(iou[1:])
        L.append('| **CleanerS** (%d frames, rescored) | RGB + depth | NYUv2 | **%.1f** | %s | **%.1f** |'
                 % (n_nyu, sc_c, fmt(iou[1:]), miou_c))
        verdict_nyu = (sc_c, miou_c)
        c, s_ = nyu['all-voxels + carve']
        iou = 100 * metrics(c)['iou']
        carve_nyu = (100 * metrics(s_)['iou'][1], np.nanmean(iou[1:]))
        L.append('| **CleanerS + free space from depth** | RGB + depth | NYUv2 | **%.1f** | %s | **%.1f** |'
                 % (carve_nyu[0], fmt(iou[1:]), carve_nyu[1]))
    for name, (inp, sc, cls, miou) in PUBLISHED_NYU.items():
        L.append('| %s | %s | NYUv2 | %.1f | %s | %.1f |' % (name, inp, sc, fmt(cls), miou))
    if n_nyu:
        c, s_ = nyu['cleaners']
        iou = 100 * metrics(c)['iou']
        L.append('| *CleanerS, its own `label_weight` rule* | *RGB + depth* | *NYUv2* | *%.1f* | %s | *%.1f* |'
                 % (100 * metrics(s_)['iou'][1], fmt(iou[1:]), np.nanmean(iou[1:])))
    L.append('')
    if verdict_nyu:
        sc_c, miou_c = verdict_nyu
        iso_sc, iso_miou = PUBLISHED_NYU['ISO'][1], PUBLISHED_NYU['ISO'][3]
        better = [k for k, (a, b) in {'SC IoU': (sc_c, iso_sc), 'mIoU': (miou_c, iso_miou)}.items() if a > b]
        L.append('**Under ISO\'s scoring, CleanerS is %s RGB-only ISO on NYUv2**: SC %.1f vs %.1f (%+.1f), mIoU '
                 '%.1f vs %.1f (%+.1f). %s\n' % (
                     'ahead of' if len(better) == 2 else 'behind' if not better else 'mixed against',
                     sc_c, iso_sc, sc_c - iso_sc, miou_c, iso_miou, miou_c - iso_miou,
                     'Depth input does not, by itself, win: CleanerS\'s published-style numbers (italic row) come '
                     'from a scored set that leaves out the free space the camera saw through, where it predicts '
                     'objects almost everywhere; ISO\'s rule counts that space.'
                     if len(better) < 2 else ''))
        L.append('**Why:** CleanerS\'s training loss uses the same `label_weight` mask as its metric '
                 '(`train_utils.py:83-85`, see `EVALUATION.md`), so nothing ever penalised it for what it predicts in '
                 'free space. On NYU test it predicts an object in ~97% of the free space the camera saw far from '
                 'any surface (GT: 94% empty), against 7% of occluded space (measured on 200 test frames). ISO was trained on every non-255 voxel.\n')
        L.append('**CleanerS + free space from depth** sets every voxel the depth frame saw through (`tsdf > 0`) to '
                 'empty — a post-process using only the model\'s own measured input, not part of CleanerS. With it: '
                 'SC %.1f vs ISO %.1f (%+.1f), mIoU %.1f vs %.1f (%+.1f). %s\n' % (
                     carve_nyu[0], iso_sc, carve_nyu[0] - iso_sc, carve_nyu[1], iso_miou, carve_nyu[1] - iso_miou,
                     'So once the training blind spot is patched with the depth it already has, the RGB-D model is '
                     'ahead on both.' if carve_nyu[0] > iso_sc and carve_nyu[1] > iso_miou else
                     'Even patched, it does not overtake RGB-only ISO on both numbers; the space the camera never '
                     'observed (`tsdf == 0`, ~46% of the NYU grid, still filled ~79%) is left as the model predicted.'))
        L.append('Caveat: the ISO rows are the paper\'s numbers. Its NYU ground truth comes from the same SSCNet '
                 'annotation downsampled to 60×36×60, but the two label files were not compared voxel for voxel '
                 'here.\n')

    L.append('### 2. Occ-ScanNet val — same scoring, different training data\n')
    L.append(head); L.append(sep)
    iou = 100 * metrics(res['cm'][MAIN])['iou']
    sc_o, miou_o = 100 * metrics(res['cm_sc'][MAIN])['iou'][1], np.nanmean(iou[1:])
    L.append('| **CleanerS** (%d frames) | RGB + depth | NYUv2 only (zero-shot) | **%.1f** | %s | **%.1f** |'
             % (res['n'], sc_o, fmt(iou[1:]), miou_o))
    iou_c = 100 * metrics(res['cm_carve'])['iou']
    sc_oc, miou_oc = 100 * metrics(res['cm_sc_carve'])['iou'][1], np.nanmean(iou_c[1:])
    L.append('| **CleanerS + free space from depth** | RGB + depth | NYUv2 only (zero-shot) | **%.1f** | %s | **%.1f** |'
             % (sc_oc, fmt(iou_c[1:]), miou_oc))
    for name, (sc, cls, miou) in PUBLISHED.items():
        L.append('| %s | RGB | Occ-ScanNet train | %.1f | %s | %.1f |' % (name, sc, fmt(cls), miou))
    L.append('')
    iso_sc, iso_miou = PUBLISHED['ISO'][0], PUBLISHED['ISO'][2]
    L.append('With free space from depth: %+.1f SC IoU and %+.1f mIoU against ISO.\n'
             % (sc_oc - iso_sc, miou_oc - iso_miou))
    L.append('CleanerS as-is: %+.1f SC IoU and %+.1f mIoU against ISO. Neither row isolates the input: ISO trained on '
             'Occ-ScanNet\'s 45,755 training frames and CleanerS never saw ScanNet. It measures whether a '
             'depth-using model trained elsewhere transfers better than an RGB model trained in-domain.\n'
             % (sc_o - iso_sc, miou_o - iso_miou))
    L.append('A clean RGB-D vs RGB answer on Occ-ScanNet would need CleanerS fine-tuned on Occ-ScanNet train, or ISO '
             'evaluated zero-shot from NYU.\n')
    return L


def report(res, args):
    cm, cm_sc = res['cm'], res['cm_sc']
    L = []
    L.append('# Occ-ScanNet evaluation\n')
    L.append('Generated %s by `inference/evaluate_scannet.py` from `%s`; rerun it after regenerating the '
             'predictions. CleanerS checkpoint `CleanerS_ckpt.pth`, trained on NYU only — every number here is a '
             'zero-shot transfer to ScanNet. Inputs use ScanNet\'s own per-scene depth calibration '
             '(`scans/<scene>/<scene>.txt`).\n' % (datetime.date.today().isoformat(), args.pred_dir))
    L.append('Protocol is `examples/segmentation/test_NYU.py`\'s, the same as `EVALUATION.md`, applied to the '
             'released Occ-ScanNet labels (`target`): **SSC** over voxels where `label_weight > 0` and '
             '`label != 255`, mIoU averaged over classes 1-11; **SC** the same restricted to `mapping == 307200`, '
             'the voxels no depth pixel reached, scored occupied-vs-empty. `label_weight` is rebuilt the way the '
             'NYU files were made: GT object, or `tsdf < -0.5`.\n')
    L.append('`precision = TP/(TP+FP)` — of what was predicted, how much was right. `recall = TP/(TP+FN)` — of '
             'what was there, how much was found. `IoU = TP/(TP+FP+FN)` — both failures in one number, which is '
             'why it is the headline.\n')
    L.append('Other scoring rules — the all-voxel rule Occ-ScanNet papers use, and a view-restricted GT — follow '
             'after the per-class sections, together with the comparison to ISO and what differs from NYU.\n')

    L.append('## `val` split — %d frames, %d scenes\n' % (res['n'], res['scenes']))
    tbl, _, _ = table(cm[PRIMARY], cm_sc[PRIMARY])
    L.append(tbl)
    L.append('')
    L.append(sc_breakdown(cm[PRIMARY], cm_sc[PRIMARY], res['cm_sc12'], res['geo'], 'val', args.pred_dir)
             .replace('against the NYU labels', 'against the Occ-ScanNet labels'))
    L.append('')
    L.append('### Scene completion (SC) per class\n')
    L.append('SC above is one binary number. This breaks it down per class, over the same region: scored voxels '
             '**no depth pixel reached** — the part of each object the model has to complete rather than read off '
             'the input.\n')
    L.append(SC_COLUMNS)
    L.append('')
    L.append(sc_class_rows(cm[PRIMARY], res['cm_sc12'], cm_sc[PRIMARY])[0])
    L.append('')

    L.append('## Floor removed\n')
    L.append('Occ-ScanNet places the grid 5 cm below world *z = 0* instead of below the annotated floor (NYU\'s '
             'rule), so the ScanNet floor usually lands 1–2 layers above the layer the model learned it in. That is '
             'a dataset-construction difference, not a model error, so the same protocol is also scored with floor '
             'taken out: a voxel that is floor in the GT **or** in the prediction is not scored, and mIoU is over the '
             'other 10 classes. The NYU test split is scored the same way for comparison.\n')
    tbl_nf, _, _ = table(res['cm_nf'][PRIMARY], res['cm_sc_nf'][PRIMARY])
    L.append(tbl_nf.replace('SSC mIoU (classes 1-11)', 'SSC mIoU (classes 1-11 except floor)'))
    L.append('')
    nyu, n_nyu = res.get('nyu', ({}, 0))
    if n_nyu:
        L.append('| | SC IoU | SC precision | SC recall | SSC mIoU |')
        L.append('| --- | ---: | ---: | ---: | ---: |')
        for name, (c_, s_), drop in (
                ('Occ-ScanNet val', (cm[PRIMARY], cm_sc[PRIMARY]), False),
                ('Occ-ScanNet val, floor removed', (res['cm_nf'][PRIMARY], res['cm_sc_nf'][PRIMARY]), True),
                ('NYU test', nyu['cleaners'], False),
                ('NYU test, floor removed', nyu['cleaners, floor removed'], True)):
            ms = metrics(s_)
            cls = [k for k in range(1, N_CLASSES) if not (drop and k == FLOOR)]
            L.append('| %s | %.1f | %.1f | %.1f | %.1f |' % (
                name, 100 * ms['iou'][1], 100 * ms['precision'][1], 100 * ms['recall'][1],
                100 * np.nanmean(metrics(c_)['iou'][cls])))
        L.append('')

    L.append('## All scoring rules at a glance\n')
    L.append('| scored set | GT | SC IoU | SC precision | SC recall | SSC mIoU |')
    L.append('| --- | --- | ---: | ---: | ---: | ---: |')
    for row in ROWS:
        ms = metrics(cm_sc[row])
        L.append('| %s | `%s` | %.1f | %.1f | %.1f | %.1f |' % (
            row[0], row[1], 100 * ms['iou'][1], 100 * ms['precision'][1], 100 * ms['recall'][1],
            100 * np.nanmean(metrics(cm[row])['iou'][1:])))
    for row in NOFLOOR_ROWS:
        ms = metrics(res['cm_sc_nf'][row])
        L.append('| cleaners, **floor removed** | `%s` | %.1f | %.1f | %.1f | %.1f |' % (
            row[1], 100 * ms['iou'][1], 100 * ms['precision'][1], 100 * ms['recall'][1],
            100 * np.nanmean(metrics(res['cm_nf'][row])['iou'][[k for k in range(1, N_CLASSES) if k != FLOOR]])))
    nyu, n_nyu = res.get('nyu', ({}, 0))
    if n_nyu:
        for protocol in ('cleaners', 'cleaners, floor removed'):
            c, s_ = nyu[protocol]
            cls = [k for k in range(1, N_CLASSES) if not (protocol.endswith('removed') and k == FLOOR)]
            ms = metrics(s_)
            L.append('| *NYU test, %s* | *NYU `Label`* | *%.1f* | *%.1f* | *%.1f* | *%.1f* |' % (
                protocol, 100 * ms['iou'][1], 100 * ms['precision'][1], 100 * ms['recall'][1],
                100 * np.nanmean(metrics(c)['iou'][cls])))
    L.append('')
    L.append('- **all-voxels / `target`** is how Occ-ScanNet papers score, so it is the row to put next to them.')
    L.append('- **`target_fov`** marks voxels outside the colour camera\'s view as 255. The released GT calls them '
             '`empty` although no camera looked at them, so predicting anything there is scored wrong.')
    L.append('- **cleaners** is CleanerS\'s own NYU protocol: only GT objects plus occluded space (`tsdf < -0.5`) '
             'are scored, and SC only where no depth pixel landed.')
    L.append('- **floor removed**: the same, but a voxel that is floor in the GT *or* in the prediction is not scored, '
             'and mIoU is over the other 10 classes. Occ-ScanNet places the grid 5 cm below world *z = 0* instead of '
             'below the annotated floor (NYU\'s rule), so its floor usually lands 1–2 layers above the layer the model '
             'learned from NYU; removing floor takes that construction difference out of the comparison. The NYU rows '
             'apply the same removal to CleanerS\'s own test split.\n')

    rg = args.ranges
    L.append('## NYU-like views\n')
    L.append('Only frames whose camera looks like an NYU frame: pitch %.1f° to %.1f° and camera height %.2f to '
             '%.2f m, the 5th–95th percentile of NYU\'s own %d frames%s. ScanNet is hand-held and looks down '
             'more (median pitch %.1f° here vs %.1f° on NYU); grid placement is the same recipe on both '
             '(centre ~3.3 m ahead, heading spread 0–45°), so pitch is the viewpoint difference that matters. '
             '**%d of %d frames** qualify.\n'
             % (rg['pitch'][0], rg['pitch'][1], rg['height'][0], rg['height'][1], rg['n'],
                '' if rg['measured'] else ' (fallback constants; NYU .bin headers not found)',
                np.median([r['pitch_deg'] for r in res['rows']]), rg['pitch_median'], res['n_nl'], res['n']))
    if res['n_nl']:
        L.append('| scored set | GT | all frames: SC IoU | all frames: SSC mIoU | NYU-like: SC IoU | NYU-like: SSC mIoU |')
        L.append('| --- | --- | ---: | ---: | ---: | ---: |')
        for row in ROWS:
            a = sc_ssc(cm[row], cm_sc[row])
            b = sc_ssc(res['cm_nl'][row], res['cm_sc_nl'][row])
            L.append('| %s | `%s` | %.1f | %.1f | **%.1f** | **%.1f** |' % (row[0], row[1], a[0], a[1], b[0], b[1]))
        L.append('')
        nyu, n_nyu = res['nyu']
        if n_nyu:
            c, s_ = nyu['cleaners']
            L.append('For reference, CleanerS on NYU test with the cleaners protocol: SC %.1f / SSC mIoU %.1f. The '
                     '**cleaners / NYU-like** row is the closest like-for-like this data allows: same scoring rule, '
                     'similar viewpoints. What still differs is the GT recipe (CompleteScanNet nearest-neighbour '
                     'voxels vs NYU\'s 2 cm CAD solids with the 4×4×4 rule) and the sensor.\n' % sc_ssc(c, s_))
        iou_nl = 100 * metrics(res['cm_nl'][STRICT])['iou']
        L.append('Per class, cleaners / `target_fov`, NYU-like views:\n')
        L.append('| ' + ' | '.join(CLASSES[1:]) + ' |')
        L.append('|' + ' ---: |' * (N_CLASSES - 1))
        L.append('| ' + ' | '.join('%.1f' % v for v in iou_nl[1:]) + ' |\n')

    L.append('## Against published results\n')
    names = CLASSES[1:]
    L.append('| method | trained on | scored set | IoU | ' + ' | '.join(names) + ' | mIoU |')
    L.append('|' + ' --- |' * 3 + ' ---: |' * (len(names) + 2))
    iou = 100 * metrics(cm[MAIN])['iou']
    L.append('| **CleanerS (ours)** | NYU | all-voxels | **%.1f** | %s | **%.1f** |' % (
        100 * metrics(cm_sc[MAIN])['iou'][1], ' | '.join('%.1f' % v for v in iou[1:]), np.nanmean(iou[1:])))
    for name, (sc, cls, miou) in PUBLISHED.items():
        L.append('| %s | Occ-ScanNet | all-voxels | %.1f | %s | %.1f |' % (name, sc, ' | '.join('%.1f' % v for v in cls), miou))
    iou_s = 100 * metrics(cm[STRICT])['iou']
    L.append('| **CleanerS (ours)** | NYU | cleaners, `target_fov` | **%.1f** | %s | **%.1f** |' % (
        100 * metrics(cm_sc[STRICT])['iou'][1], ' | '.join('%.1f' % v for v in iou_s[1:]), np.nanmean(iou_s[1:])))
    nyu, n_nyu = res['nyu']
    for protocol in ('cleaners', 'all-voxels'):
        if n_nyu:
            c, s = nyu[protocol]
            iou_n = 100 * metrics(c)['iou']
            L.append('| CleanerS, NYU test (%d frames) | NYU | %s | %.1f | %s | %.1f |' % (
                n_nyu, protocol, 100 * metrics(s)['iou'][1], ' | '.join('%.1f' % v for v in iou_n[1:]),
                np.nanmean(iou_n[1:])))
    L.append('')
    L.append('ISO and MonoScene numbers are the ISO paper\'s Table 1 (RGB input, trained on Occ-ScanNet train). '
             'CleanerS uses RGB **and** sensor depth but has never seen ScanNet, so the comparison mixes an '
             'input advantage with a domain gap.\n')
    L.append('The two NYU rows score **the same predictions** on CleanerS\'s own test split. The cleaners row '
             'reproduces `document/EVALUATION.md`; the all-voxels row is what the Occ-ScanNet protocol does to '
             'them with no change of dataset. The model fills most of the space the camera saw through with '
             'objects, and NYU\'s `label_weight` never scores that space, so the protocol alone accounts for '
             'part of the drop.\n')

    L.extend(diagnosis_section(res))
    L.extend(rgbd_vs_rgb_section(res))

    for row, title in ((MAIN, 'all-voxels / `target`'), (STRICT, 'cleaners / `target_fov`')):
        L.append('## Per class — %s\n' % title)
        L.append(class_table(cm[row]))
        L.append('')
        L.append('### Biggest confusions\n')
        L.append(top_confusions(cm[row]))
        L.append('')
        L.append('### Which classes invent volume\n')
        L.append('Each class\'s false positives, split into voxels the GT calls `empty` (volume grown into free '
                 'or unobserved space) and voxels of another class (right volume, wrong name).\n')
        L.append(fp_split(cm[row]))
        L.append('')

    L.append('## Where the misses are — all-voxels / `target`\n')
    L.append('Every GT object voxel, by what the depth frame knew about it: a depth pixel landed in it, the '
             'camera saw through it (tsdf > 0, so the sensor says free space), or neither (occluded or out of '
             'view).\n')
    ro = res['region_occ']
    L.append('| region | GT object voxels | share | predicted occupied (any class) |')
    L.append('| --- | ---: | ---: | ---: |')
    for r, name in enumerate(REGIONS):
        L.append('| %s | {:,} | %.1f%% | %.1f%% |'.format(int(ro[r, 0])) % (
            name, 100 * ro[r, 0] / max(ro[:, 0].sum(), 1), 100 * ro[r, 1] / max(ro[r, 0], 1)))
    L.append('')
    L.append('Per-class recall (right class) by region:\n')
    L.append('| class | surface | seen through | hidden |')
    L.append('| --- | ---: | ---: | ---: |')
    for k in range(1, N_CLASSES):
        g, h = res['region_gt'][k], res['region_hit'][k]
        L.append('| `%s` | %s | %s | %s |' % (CLASSES[k], *[
            pct(100 * h[r] / g[r]) if g[r] else '—' for r in range(3)]))
    L.append('')
    seen_through = ro[1, 0]
    L.append('GT object voxels in the *seen through* row contradict the depth sensor: it measured free space '
             'there. They are %.1f%% of all GT object voxels — a direct measure of GT/sensor disagreement '
             '(CAD completion or pose error).\n' % (100 * seen_through / max(ro[:, 0].sum(), 1)))

    L.append('## Does the grid\'s heading matter?\n')
    L.append('Both grids follow the room / scan axes rather than the camera, so the camera can look along an axis '
             '(0°) or diagonally across the volume (45°). NYU training frames span the same range (median %.0f° '
             'off-axis), so a drop with heading would point at the model, not at a dataset difference. Scored on '
             'all-voxels / `target`, confusion matrices pooled per bin.\n' % args.ranges['heading_median'])
    L.append(bin_table(res['bins_h'], HEADING_BINS, 'heading off-axis'))
    L.append('')
    d_sc, d_ssc = within_scene(res['per_scene'])
    if len(d_sc):
        L.append('Within-scene check (removes scene difficulty): for the **%d scenes** with at least 3 frames '
                 'both within 10° of an axis and within 10° of the diagonal, aligned minus diagonal is a median '
                 '**%+.1f SC IoU** (aligned better in %d%% of scenes) and **%+.1f SSC mIoU** (%d%%).\n'
                 % (len(d_sc), np.median(d_sc), 100 * (d_sc > 0).mean(), np.median(d_ssc), 100 * (d_ssc > 0).mean()))
    L.append('### Camera pitch\n')
    L.append('Negative = looking down. NYU\'s median is %.1f°.\n' % args.ranges['pitch_median'])
    L.append(bin_table(res['bins_p'], PITCH_BINS, 'pitch'))
    L.append('')

    rows = res['rows']
    sc_vals = np.array([r['sc_iou_all_voxels'] for r in rows], dtype=float)
    L.append('## Spread across frames\n')
    L.append('Per-frame SC IoU (all-voxels / `target`): median %.1f, 10th percentile %.1f, 90th percentile %.1f. '
             'Per-frame numbers are in `%s`.\n' % (np.nanmedian(sc_vals), np.nanpercentile(sc_vals, 10),
                                                   np.nanpercentile(sc_vals, 90), os.path.basename(args.csv)))
    return '\n'.join(L)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--pred_dir', default='./outputs/scannet')
    p.add_argument('--scannet_root', default=DEFAULT_ROOT)
    p.add_argument('--nyu_root', default=DEFAULT_NYU)
    p.add_argument('--nyu_pred', default=DEFAULT_PRED)
    p.add_argument('--no_nyu', action='store_true', help='skip the NYU reference rows')
    p.add_argument('--report', default=None, help='write the markdown report here (else stdout)')
    p.add_argument('--csv', default=None, help='per-frame table (default: <pred_dir>/per_frame.csv)')
    args = p.parse_args()
    args.csv = args.csv or os.path.join(args.pred_dir, 'per_frame.csv')

    args.ranges = nyu_view_ranges(args.nyu_root)
    res = evaluate(args)
    res['nyu'] = nyu_reference(args.nyu_root, args.nyu_pred) if not args.no_nyu else ({}, 0)
    if res['n'] == 0:
        sys.exit('no frames scored')

    with open(args.csv, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(res['rows'][0].keys()))
        w.writeheader()
        w.writerows(res['rows'])

    text = report(res, args)
    if args.report:
        with open(args.report, 'w') as f:
            f.write(text + '\n')
        print('wrote %s and %s (%d frames)' % (args.report, args.csv, res['n']))
    else:
        print(text)


if __name__ == '__main__':
    main()
