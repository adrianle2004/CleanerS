"""
reconstruction_GT/evaluate_gt.py

Score a capture's predictions against the ground truth you made, with NYU's
own protocol.

    python -m reconstruction_GT.evaluate_gt captures/room07
    python -m reconstruction_GT.evaluate_gt captures/room07 --fov all
    python -m reconstruction_GT.evaluate_gt --nyu            # self-check

*** THE PROTOCOL IS test_NYU.py's ***

Copied from `examples/segmentation/test_NYU.py:206-211`, so a number here means
the same thing as a number in EVALUATION.md:

    label_weightSSC = label_weight & (label3d != 255)
    weightSC        = label_weight & (mapping == 307200) & (label3d != 255)

  SSC  12-class confusion over label_weightSSC, mIoU averaged over classes 1-11
  SC   the same set, restricted to voxels NO depth pixel reached, scored
       occupied-vs-empty. SC is a test of completion, not of perception.

`--nyu` runs the identical code over NYU's 654-frame TEST split (from
`data/NYU/test.txt`) and must reproduce SC 75.0 / SSC 47.7. Scoring all 1449
frames instead reads 80.6 / 59.9, because the model fits the training frames. That is the check that this file measures what the repo
measures; run it whenever this file changes.

*** THE FRUSTUM ***

By default the score covers only voxels inside the camera's view. The grid is
4.8 m wide whatever the lens does, and a capture's ground truth describes the
whole room -- including, for a level camera at desk height, a ceiling and a
floor that were never in frame. NYU can ignore this because its scored set is
already 95-99.6% in-frame; a small room shot from a corner is not.

`--fov all` scores everything annotated, which is the Occ-ScanNet `target`
rule. Report which one you used: they are not comparable.
"""

import os
import sys
import glob
import json
import argparse
import logging

import numpy as np

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_THIS_DIR)
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from reconstruction_GT.voxelize_gt import (
    CLASSES, Grid, frustum_mask, VOX_SIZE_LOW, VOX_UNIT_LOW, IGNORE)

log = logging.getLogger(__name__)
NYU_ROOT = os.path.join(os.path.dirname(_REPO_ROOT), 'data', 'NYU')
NYU_PRED = os.path.join(_REPO_ROOT, 'custom_visual_pred', 'CleanerS', 'prediction')


def confusion(pred, target, n):
    k = (target >= 0) & (target < n)
    return np.bincount(n * target[k].astype(int) + pred[k].astype(int),
                       minlength=n * n).reshape(n, n)


def iou_from_cm(cm):
    tp = np.diag(cm).astype(np.float64)
    fp = cm.sum(axis=0) - tp
    fn = cm.sum(axis=1) - tp
    return tp, fp, fn, tp / np.maximum(tp + fp + fn, 1)


def score(pred, label, lw, mapping, keep=None):
    """test_NYU.py:206-211, on flat arrays. -> (cmSSC 12x12, cmSC 2x2)"""
    label = label.astype(np.int64).reshape(-1).copy()
    if keep is not None:
        label[~keep.reshape(-1)] = IGNORE
    pred = pred.astype(np.int64).reshape(-1)
    lw = lw.reshape(-1) > 0
    ssc = lw & (label != IGNORE)
    sc = ssc & (mapping.reshape(-1) == 307200)
    return (confusion(pred[ssc], label[ssc], 12),
            confusion((pred[sc] > 0).astype(int), (label[sc] > 0).astype(int), 2))


def report(cmSSC, cmSC, title, n_frames):
    tp, fp, fn, iou = iou_from_cm(cmSSC)
    print('\n=== %s (%d frame%s) ===' % (title, n_frames, '' if n_frames == 1 else 's'))
    print('%-9s %9s %9s %9s %8s %8s %8s' %
          ('class', 'TP', 'FP', 'FN', 'prec', 'recall', 'IoU'))
    for c in range(12):
        if cmSSC[c].sum() == 0 and cmSSC[:, c].sum() == 0:
            continue
        p = tp[c] / max(tp[c] + fp[c], 1)
        r = tp[c] / max(tp[c] + fn[c], 1)
        print('%-9s %9d %9d %9d %7.1f%% %7.1f%% %7.1f%%%s'
              % (CLASSES[c], tp[c], fp[c], fn[c], 100 * p, 100 * r, 100 * iou[c],
                 '   (not in mIoU)' if c == 0 else ''))
    present = [c for c in range(1, 12) if cmSSC[c].sum() > 0]
    miou = float(np.mean(iou[present])) if present else 0.0
    print('\nSSC mIoU over the %d classes present: %.1f' % (len(present), 100 * miou))
    if len(present) < 11:
        missing = [CLASSES[c] for c in range(1, 12) if c not in present]
        print('  (absent from this ground truth, so not averaged: %s)'
              % ', '.join(missing))
    tpS, fpS, fnS, iouS = iou_from_cm(cmSC)
    scored = int(cmSC.sum())
    occ = cmSC[1].sum() / max(scored, 1)
    print('SC  IoU %.1f  |  precision %.1f  recall %.1f  |  %d voxels, %.0f%% occupied'
          % (100 * iouS[1], 100 * tpS[1] / max(tpS[1] + fpS[1], 1),
             100 * tpS[1] / max(tpS[1] + fnS[1], 1), scored, 100 * occ))
    if occ > 0.9 or occ < 0.05:
        trivial = cmSC[1].sum() / max(scored, 1)
        print('  WARNING: %.0f%% of the SC set is occupied, so "predict occupied '
              'everywhere"\n  would score IoU %.3f. This frame cannot tell a '
              'good model from a trivial\n  one -- see MAKING_GT.md, "What this '
              'does not measure".' % (100 * occ, trivial))
    return miou, iouS[1]


def eval_capture(cap, pred_dir, use_frustum, only=None):
    meta = json.load(open(os.path.join(cap, 'meta.json')))
    from inference.frame_loader import cam_pose_from_meta, frame_meta
    cmSSC, cmSC = np.zeros((12, 12), np.int64), np.zeros((2, 2), np.int64)
    n, per_frame = 0, []
    for path in sorted(glob.glob(os.path.join(cap, 'gt', 'Label', '*.npz'))):
        stem = os.path.splitext(os.path.basename(path))[0]
        if only and stem not in only:
            continue
        pred_path = os.path.join(pred_dir, stem + '.npy')
        if not os.path.exists(pred_path):
            log.warning('no prediction for %s at %s', stem, pred_path)
            continue
        label = np.load(path)['arr_0']
        gt_t = np.load(os.path.join(cap, 'gt', 'TSDF', stem + '.npz'))
        mapping = np.load(os.path.join(cap, 'gt', 'Mapping', stem + '.npz'))['arr_0']
        keep = None
        if use_frustum:
            # each frame's own lens and pose: a capture may hold frames lifted
            # out of the sweep, shot from a different height and tilt
            fm = frame_meta(meta, stem)
            g = Grid([-2.4, 0.0, -0.05], VOX_SIZE_LOW, VOX_UNIT_LOW)
            keep = frustum_mask(g, np.asarray(fm['cam_K'], np.float64),
                                cam_pose_from_meta(meta, stem))
        a, b = score(np.load(pred_path), label, gt_t['arr_1'], mapping, keep)
        cmSSC += a
        cmSC += b
        n += 1
        # the same frame on its own, so a report can show where the total
        # comes from: one viewpoint's number says as much about the viewpoint
        _, _, _, iou_a = iou_from_cm(a)
        _, _, _, iou_b = iou_from_cm(b)
        present = [c for c in range(1, 12) if a[c].sum() > 0]
        per_frame.append({
            'frame': stem,
            'scored': int(a.sum()),
            'sc_voxels': int(b.sum()),
            'occ_pct': 100.0 * b[1].sum() / max(int(b.sum()), 1),
            'sc_iou': 100.0 * float(iou_b[1]),
            'ssc_miou': 100.0 * float(np.mean(iou_a[present])) if present else 0.0,
            'classes': len(present),
        })
    if not n:
        raise SystemExit('nothing scored: no predictions found in %s' % pred_dir)
    return cmSSC, cmSC, n, per_frame


def eval_nyu(limit, split='test'):
    # only the held-out split: the predictions folder holds all 1449 frames,
    # and scoring train as well reads 80.6 / 59.9 instead of 75.0 / 47.7
    ids_path = os.path.join(NYU_ROOT, '%s.txt' % split)
    if not os.path.exists(ids_path):
        raise SystemExit('no %s -- expected NYU\'s split list there' % ids_path)
    ids = [l.strip() for l in open(ids_path) if l.strip()]
    cmSSC, cmSC = np.zeros((12, 12), np.int64), np.zeros((2, 2), np.int64)
    n = 0
    for fid in ids[:limit] if limit else ids:
        p = os.path.join(NYU_PRED, 'NYU%s_0000.npy' % fid)
        lab = os.path.join(NYU_ROOT, 'Label', fid + '.npz')
        if not (os.path.exists(p) and os.path.exists(lab)):
            continue
        a, b = score(np.load(p), np.load(lab)['arr_0'],
                     np.load(os.path.join(NYU_ROOT, 'TSDF', fid + '.npz'))['arr_1'],
                     np.load(os.path.join(NYU_ROOT, 'Mapping', fid + '.npz'))['arr_0'])
        cmSSC += a
        cmSC += b
        n += 1
    if n == 0:
        raise SystemExit('scored no frames: no NYU%%s_0000.npy under %s\n'
                         'and/or no Label/*.npz under %s' % (NYU_PRED, NYU_ROOT))
    return cmSSC, cmSC, n


def slope_figure(x, y, cap, key, x_label, y_label, title):
    """Scatter the frames and draw the fitted line, so the slope can be seen.

    Written next to the report as gt/figures/slope_<key>.png and linked from
    it. Returns the relative path, or None if the plot could not be made --
    the report then simply has one less picture.
    """
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except Exception:
        return None
    m, b = np.polyfit(x, y, 1)
    out_dir = os.path.join(cap, 'gt', 'figures')
    os.makedirs(out_dir, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7.2, 4.4), dpi=130)
    ax.scatter(x, y, s=16, alpha=0.55, edgecolor='none', color='#2f6fb0',
               label='one sampled frame')
    xs = np.array([x.min(), x.max()])
    ax.plot(xs, m * xs + b, color='#c2403d', lw=2,
            label='least squares: %+.1f %s per unit' % (m, y_label))
    lo, hi = x < np.median(x), x >= np.median(x)
    ax.plot([x[lo].mean(), x[hi].mean()], [y[lo].mean(), y[hi].mean()],
            'o--', color='#3f7d3f', lw=1.6, ms=7,
            label='means of the two halves')
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_title(title, fontsize=10)
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8, framealpha=0.9)
    fig.tight_layout()
    path = os.path.join(out_dir, 'slope_%s.png' % key)
    fig.savefig(path)
    plt.close(fig)
    return os.path.join('figures', os.path.basename(path))


def worked_example(x, y, cap, key, label, step, step_name, n):
    """The arithmetic behind one row of the table, on this room's numbers."""
    cov = float(((x - x.mean()) * (y - y.mean())).mean())
    r = float(np.corrcoef(x, y)[0, 1])
    m = float(np.polyfit(x, y, 1)[0])
    lo, hi = x < np.median(x), x >= np.median(x)
    dx, dy = x[hi].mean() - x[lo].mean(), y[hi].mean() - y[lo].mean()
    resid = float((y - np.polyval(np.polyfit(x, y, 1), x)).std())
    L = ['', '### Worked example: %s' % label, '']
    fig = slope_figure(x, y, cap, key, label, 'SC',
                       '%s: SC against %s, %d frames' % (os.path.basename(cap),
                                                         label, n))
    if fig:
        L += ['![SC against %s](%s)' % (label, fig), '']
    L += ['Each dot is one sampled frame: where the camera was on that axis, '
          'and what CleanerS scored on it. The red line is the least-squares '
          'fit, and the slope of that line is the `per step` column.', '',
          'Every frame contributes one product to the covariance: how far it '
          'sits from the average on each axis, multiplied. Both above average, '
          'or both below, and the product is positive; one of each and it is '
          'negative. Their average is the covariance, which still carries the '
          'units of both axes, so dividing by the two standard deviations '
          'strips the units out and pins the result between -1 and +1. That is '
          'r.', '',
          '```',
          '%-28s mean %8.3f   std %8.3f' % (label, x.mean(), x.std()),
          '%-28s mean %8.1f   std %8.1f' % ('SC', y.mean(), y.std()),
          '',
          'cov    = mean((property - its mean) x (SC - its mean))',
          '       = %+.4f' % cov,
          'r      = cov / (std(property) x std(SC))',
          '       = %+.4f / (%.4f x %.2f)' % (cov, x.std(), y.std()),
          '       = %+.4f' % r,
          'slope  = r x std(SC) / std(property)',
          '       = %+.4f x %.1f / %.3f' % (r, y.std(), x.std()),
          '       = %+.1f SC per unit' % m,
          'step   = %s  ->  %+.1f x %g = %+.1f SC per step'
          % (step_name, m, step, m * step),
          '```', '',
          'The same answer without fitting anything, as a check: split the '
          '%d frames at the median %s and compare the halves.' % (n, label), '',
          '```',
          'lower half: mean %.3f -> mean SC %.1f' % (x[lo].mean(), y[lo].mean()),
          'upper half: mean %.3f -> mean SC %.1f' % (x[hi].mean(), y[hi].mean()),
          '%+.1f SC over %+.3f  ->  %+.1f SC per step (fit said %+.1f)'
          % (dy, dx, dy / dx * step if abs(dx) > 1e-9 else 0.0, m * step),
          '```', '',
          'That is the green dashed line on the plot. The two agree, so the '
          'number is not an artefact of the fit -- but the frames scatter '
          '%.1f SC either side of the line, so it describes the sweep as a '
          'whole and predicts no single frame.' % resid]
    return L


NYU_VIEWPOINTS = 'outputs/nyu_viewpoints.csv'
SCANNET_VIEWPOINTS = 'outputs/scannet_viewpoints.csv'


def reference_sets():
    """The benchmark distributions, measured the same way, for scale.

    Without them a table of correlations invites the reader to take it as a
    fact about SC and SSC. It is a fact about THIS ROOM, and the comparison
    says how much of it generalises.
    """
    import csv
    out = []
    for name, path in (('NYU', NYU_VIEWPOINTS), ('ScanNet', SCANNET_VIEWPOINTS)):
        if not os.path.exists(path):
            continue
        rows = [x for x in csv.DictReader(open(path))]
        if not rows:
            continue
        d = {}
        for k in rows[0]:
            if k in ('frame', 'scene'):
                continue
            vals = []
            for x in rows:
                try:
                    vals.append(float(x[k]) if x[k] not in ('', None) else np.nan)
                except ValueError:
                    vals.append(np.nan)
            d[k] = np.array(vals)
        out.append((name, d, len(rows)))
    return out


def _corr(a, b):
    m = np.isfinite(a) & np.isfinite(b)
    return float(np.corrcoef(a[m], b[m])[0, 1]) if m.sum() > 10 else np.nan


def nyu_reference(props, col, r):
    refs = reference_sets()
    if not refs:
        return []
    names = ' | '.join('%s' % n for n, _, _ in refs)
    L = ['', '### The same thing measured on the benchmarks', '',
         'NYU (what CleanerS was trained and evaluated on) and Occ-ScanNet, '
         'put through this identical measurement: %s.'
         % ', '.join('%s %d frames' % (n, c) for n, _, c in refs), '', '']
    head = '| property of the viewpoint | here |' + ''.join(' %s |' % n for n, _, _ in refs)
    for metric, lab in (('sc_iou', 'SC'), ('ssc_miou', 'SSC')):
        L += ['**Correlation with %s**' % lab, '', head,
              '| --- | ---: |' + ' ---: |' * len(refs)]
        for k, pl in props:
            row = '| %s | %+.2f |' % (pl, r(col(k), col(metric)))
            for _, d, _ in refs:
                row += ' %+.2f |' % _corr(d[k], d[metric]) if k in d else ' - |'
            L.append(row)
        here = np.mean([abs(r(col(k), col(metric))) for k, _ in props])
        row = '| **mean \\|correlation\\|** | **%.2f** |' % here
        for _, d, _ in refs:
            row += ' **%.2f** |' % np.nanmean(
                [abs(_corr(d[k], d[metric])) for k, _ in props if k in d])
        L += [row, '']
    sc_here = np.mean([abs(r(col(k), col('sc_iou'))) for k, _ in props])
    sc_ref = np.nanmean([np.nanmean([abs(_corr(d[k], d['sc_iou']))
                                     for k, _ in props if k in d])
                         for _, d, _ in refs])
    L += ['Viewpoint %s here than in the benchmarks (mean |correlation| with '
          'SC %.2f against %.2f). %s'
          % ('matters far more' if sc_here > sc_ref + 0.1 else 'matters about '
             'as much', sc_here, sc_ref,
             'The rows above therefore describe this capture, not the metric: '
             'in a large scene the shot barely predicts the score.'
             if sc_here > sc_ref + 0.1 else
             'These rows are not peculiar to this capture.'), '',
          '| | this room |' + ''.join(' %s |' % n for n, _, _ in refs),
          '| --- | ---: |' + ' ---: |' * len(refs)]
    for k, lab, unit, fmt in (('cam_z', 'median camera height', ' m', '%.2f'),
                              ('depth_med', 'median distance to the scene', ' m', '%.2f'),
                              ('free_mean', 'median room hidden behind the surface', ' m', '%.2f'),
                              ('occ_pct', 'median SC set occupied', '%', '%.0f'),
                              ('sc_voxels', 'median SC set', ' voxels', '%.0f')):
        row = ('| %s | ' + fmt + '%s |') % (lab, np.median(col(k)), unit)
        for _, d, _ in refs:
            row += (' ' + fmt + '%s |') % (np.nanmedian(d[k]), unit) if k in d else ' - |'
        L.append(row)
    hid = {n: np.nanmedian(d['free_mean']) for n, d, _ in refs if 'free_mean' in d}
    if hid:
        L += ['', 'The gap that drives the rest is **how much room is hidden '
              'behind what the camera sees**: %s, this room %.2f m. A frame '
              'that hides nothing cannot be asked to complete anything, and a '
              'room too small to hide anything cannot produce such a frame.'
              % (', '.join('%s %.2f m' % (n, v) for n, v in hid.items()),
                 np.median(col('free_mean')))]
    return L


def viewpoint_analysis(rows, num, cap):
    """What separates the viewpoints that score well from the ones that do not.

    Everything here is computed from the sweep CSV, so it is measured on this
    room rather than asserted.
    """
    need = ('cam_z', 'depth_med', 'near_pct', 'free_mean', 'sc_voxels', 'occ_pct')
    if not rows or any(k not in rows[0] or rows[0][k] == '' for k in need):
        return ['', '*(Re-run `sweep_eval.py` to add the viewpoint columns this '
                'section is built from.)*']
    col = lambda k: np.array([num(r, k) for r in rows])
    sc, ssc, mg = col('sc_iou'), col('ssc_miou'), col('margin')
    r = lambda a, b: float(np.corrcoef(a, b)[0, 1])
    def strength(x):
        a = abs(x)
        return ('barely related' if a < 0.3 else 'loosely related' if a < 0.6
                else 'closely related')

    L = ['', '## What separates a good viewpoint from a bad one', '',
         'Over these %d frames SC and SSC correlate **%+.2f**, so the two are '
         '%s: ranking the viewpoints by one does not rank them by the other.'
         % (len(rows), r(sc, ssc), strength(r(sc, ssc))), '',
         'Correlation of each measured property of the shot with the two '
         'scores. These are measurements of THIS room, not general claims:',
         '',
         'The numbers are Pearson correlations, not percentages: **+1** means '
         'the score rises in step with that property, **-1** that it falls in '
         'step, **0** that there is no straight-line relationship. Squaring '
         'one gives the share of the spread it accounts for, so +0.70 is about '
         'half of it and +0.30 about a tenth. They are associations, not '
         'causes, and they only see straight lines -- a property that helps up '
         'to a point and hurts after it (the bands below) reports near zero.',
         '',
         'The **per step** columns are the same relationship in this room\'s own units: the slope of the least-squares line through the frames, read over the step in the second column. It is the correlation with the units left in (slope = r x std(score) / std(property)), so it says what the score did ON AVERAGE across the sweep as that property changed -- not what any one frame would score, and only inside the range the sweep covered.',
         '', '| property of the viewpoint | a step of | vs SC | SC per step | vs SSC | SSC per step |',
         '| --- | --- | ---: | ---: | ---: | ---: |']
    # (column, wording, step to quote the slope over, its name, the 10-90
    # spread below which this room did not vary it enough to fit a line)
    props = [('cam_z', 'camera height above the floor', 0.10, '+10 cm', 0.10),
             ('tilt_deg', 'how far the camera looks down', 5.0, '+5 deg', 3.0),
             ('depth_med', 'median distance to what it sees', 0.10, '+10 cm', 0.15),
             ('near_pct', 'share of the frame closer than 1.5 m', 10.0,
              '+10 points', 10.0),
             ('free_mean', 'room hidden behind the visible surface', 0.10,
              '+10 cm', 0.05),
             ('sc_voxels', 'size of the SC set', 1000.0, '+1000 voxels', 300.0),
             ('occ_pct', 'how much of the SC set is occupied', 10.0,
              '+10 points', 10.0)]
    # `floor_pct` is deliberately absent. It is still measured, and still in
    # the CSV and the viewpoint tables, but it does not belong in a
    # correlation: in a room shot from standing height most frames see no
    # floor at all, so there is nothing to correlate, and in a room where it
    # does vary it varies BECAUSE the camera was lowered -- on room07 floor
    # share and camera height correlate -0.92, so a line fitted to one is a
    # line fitted to the other.
    props = [q for q in props if q[0] in rows[0] and rows[0][q[0]] != '']
    thin = [q for q in props
            if np.percentile(col(q[0]), 90) - np.percentile(col(q[0]), 10) < q[4]]
    props = [q for q in props if q not in thin]
    if not props:
        return ['', '*(This sweep did not vary the camera enough to say what '
                'separates its viewpoints.)*']
    props_full = list(props)
    rs = {k: (r(col(k), sc), r(col(k), ssc)) for k, _, _, _, _ in props}

    def slope(k, y, step):
        # least squares, read over one step: the same line the correlation
        # describes, with the units left in (slope = r * std(y) / std(x))
        x = col(k)
        if x.std() < 1e-9:
            return 0.0
        return float(np.polyfit(x, y, 1)[0]) * step

    for k, label, step, step_name, _ in props:
        L.append('| %s | %s | %+.2f | %+.1f | %+.2f | %+.1f |'
                 % (label, step_name, rs[k][0], slope(k, sc, step),
                    rs[k][1], slope(k, ssc, step)))
    props = [(k, lab) for k, lab, _, _, _ in props]
    strongest = lambda i: max(props, key=lambda kl: abs(rs[kl[0]][i]))
    a, b = strongest(0), strongest(1)
    m_sc = float(np.mean([abs(rs[k][0]) for k, _ in props]))
    m_ss = float(np.mean([abs(rs[k][1]) for k, _ in props]))
    verdict = ('the two about equally (mean |correlation| %.2f and %.2f)' % (m_sc, m_ss)
               if abs(m_sc - m_ss) < 0.05 else
               'SC %s than SSC (mean |correlation| %.2f against %.2f)'
               % ('more' if m_sc > m_ss else 'less', m_sc, m_ss))
    ax = col(a[0])
    resid = (float((sc - np.polyval(np.polyfit(ax, sc, 1), ax)).std())
             if ax.std() > 1e-9 else 0.0)
    if thin:
        L += ['', 'Left out, because this sweep barely varied them and a line '
              'through a column that does not move says nothing: %s. They are '
              'still measured, and still in the CSV.'
              % ', '.join('**%s** (%.10g to %.10g)'
                          % (q[1], col(q[0]).min(), col(q[0]).max())
                          for q in thin)]
    L += ['', 'The shot property that moves SC most here is **%s** (%+.2f, '
          'about %.0f%% of the spread in SC); for SSC it is **%s** (%+.2f, '
          'about %.0f%%). Geometry explains %s -- the mean of the absolute '
          'values down each column.'
          % (a[1], rs[a[0]][0], 100 * rs[a[0]][0] ** 2, b[1], rs[b[0]][1],
             100 * rs[b[0]][1] ** 2, verdict),
          '',
          'Even the strongest of them leaves a lot unexplained: frames scatter %.1f SC either side of its line, against an SC range of %.0f to %.0f across the sweep. These are trends over %d frames, not rules for one.' % (resid, sc.min(), sc.max(), len(rows)), '',
          'Those rows are not separate effects either. The properties move '
          'together, so a strong correlation in one row is often the same '
          'relationship seen from another angle:', '']
    L += ['| |' + ''.join(' %s |' % lab for _, lab in props),
          '| --- |' + ' ---: |' * len(props)]
    for k, label in props:
        L.append('| %s |' % label
                 + ''.join(' %+.2f |' % r(col(k), col(k2)) for k2, _ in props))
    pairs = sorted(((abs(r(col(k1), col(k2))), lab1, lab2, r(col(k1), col(k2)))
                    for i, (k1, lab1) in enumerate(props)
                    for k2, lab2 in props[i + 1:]), reverse=True)
    if pairs:
        L += ['', 'The tightest pair here is **%s** and **%s** at %+.2f: close '
              'enough to be largely one measurement, so the table above cannot '
              'say which of the two a score is really following.'
              % (pairs[0][1], pairs[0][2], pairs[0][3])]
    L += ['',
          'The mechanism is the SC set itself: the voxels no depth pixel '
          'reached. Group the frames by how much room is hidden behind what '
          'they see --', '',
          '| hidden room behind the surface | frames | median | SC set | occupied | SC | SSC |',
          '| --- | ---: | ---: | ---: | ---: | ---: | ---: |']
    fm = col('free_mean')
    qs = np.quantile(fm, [0.25, 0.5, 0.75])
    for lo, hi, lab in ((-np.inf, qs[0], 'least (bottom 25%)'),
                        (qs[0], qs[1], '25-50%'), (qs[1], qs[2], '50-75%'),
                        (qs[2], np.inf, 'most (top 25%)')):
        m = (fm > lo) & (fm <= hi)
        if not m.any():
            continue
        L.append('| %s | %d | %.2f m | %.0f | %.0f%% | %.1f | %.1f |'
                 % (lab, int(m.sum()), np.median(fm[m]),
                    np.median(col('sc_voxels')[m]), np.median(col('occ_pct')[m]),
                    np.median(sc[m]), np.median(ssc[m])))
    order = np.argsort(-mg)
    n_end = max(1, len(rows) // 10)
    top_i, bot_i = order[:n_end], order[-n_end:]
    L += ['', 'And the two ends of the ranking, as medians of the best and '
          'worst tenth by `margin`:', '',
          '| | best 10% | worst 10% |', '| --- | ---: | ---: |']
    for k, label in (('cam_z', 'camera height (m)'), ('tilt_deg', 'tilt (deg)'),
                     ('depth_med', 'median depth (m)'),
                     ('near_pct', 'closer than 1.5 m (%)'),
                     ('free_mean', 'hidden room behind surface (m)'),
                     ('sc_voxels', 'SC set (voxels)'),
                     ('occ_pct', 'SC set occupied (%)'),
                     ('sc_iou', 'SC'), ('ssc_miou', 'SSC')):
        if k not in rows[0] or rows[0][k] == '':
            continue
        c = col(k)
        L.append('| %s | %.2f | %.2f |' % (label, np.median(c[top_i]),
                                           np.median(c[bot_i])))
    diffs = []
    for k, label in props:
        c = col(k)
        hi, lo = np.median(c[top_i]), np.median(c[bot_i])
        spread = np.percentile(c, 90) - np.percentile(c, 10)
        if spread > 1e-9:
            diffs.append((abs(hi - lo) / spread, label, hi, lo))
    diffs.sort(reverse=True)
    L += ['', 'What the two ends differ in most, largest first: '
          + '; '.join('**%s** (%.2f against %.2f)' % (lab, hi, lo)
                      for _, lab, hi, lo in diffs[:3]) + '.', '']
    occ_all = col('occ_pct')
    if occ_all.min() > 90:
        L += ['> **No viewpoint in this room can measure completion.** The '
              'least occupied SC set in the whole sweep is still %.1f%% '
              'occupied, so "predict occupied everywhere" scores at least that '
              'on every frame, and the best margin here is %+.1f. SC is not a '
              'usable number for this room at any viewpoint -- quote SSC, and '
              'capture the next room with more depth between the camera and '
              'what it looks at. See MAKING_GT.md, "What this does not '
              'measure".' % (occ_all.min(), mg.max()), '']
    else:
        good = int(((mg > 0)).sum())
        L += ['> %d of %d viewpoints beat the trivial baseline. The ones that '
              'do are the frames where the hidden volume is neither inside a '
              'solid (nothing to find) nor enormous (too much to guess); on '
              'this room that is a median depth of %.2f m against %.2f m for '
              'the worst tenth.' % (good, len(rows),
                                    np.median(col('depth_med')[top_i]),
                                    np.median(col('depth_med')[bot_i])), '']
    L += nyu_reference(props, col, r)
    a_key, a_lab = a[0], a[1]
    a_step, a_step_name = dict((q[0], (q[2], q[3]))
                               for q in props_full)[a_key]
    L += worked_example(col(a_key), sc, cap, a_key, a_lab, a_step, a_step_name,
                        len(rows))
    L += ['', '### How these were measured', '',
          'Per frame, in `reconstruction_GT/sweep_eval.py` (`viewpoint_stats`), '
          'on the same 640x480 crop the model is given:', '',
          '- **median distance** and **closer than 1.5 m**: the depth image '
          'itself, over valid pixels.',
          '- **floor**: those pixels unprojected with the frame\'s tracked '
          'pose, counted where world z < 12 cm. Recorded in the CSV, but '
          'deliberately kept OUT of the correlations above: a room shot '
          'from standing height has almost no floor in any frame, and a '
          'room where it varies varies it by lowering the camera, which '
          'is already a column of its own.',
          '- **hidden room behind the visible surface**: each depth pixel is a '
          'ray; it is continued past the surface it hit to where it leaves the '
          'ANNOTATED room shell, and the mean of that remaining distance is '
          'taken. Zero when the ray dies on a wall or inside furniture, large '
          'when it dies on the near side of something with space behind it. '
          'Measured against the room in `gt/solids.json`, not against the '
          'fused surface, so it is the same volume the ground truth describes.',
          '- **SC set** and **occupied**: the scored set itself -- '
          '`label_weight > 0`, `label != 255`, `mapping == 307200` -- and the '
          'share of it the ground truth calls occupied, which is what a model '
          'predicting "occupied" everywhere would score.', '',
          'Correlations are Pearson over the %d sampled frames. They describe '
          'this room; a larger room, or one shot from a doorway, would spread '
          'differently.' % len(rows)]
    return L


def sweep_section(cap, rank='margin', top=5):
    """The whole sweep, if sweep_eval.py has scored it: what the best and worst
    viewpoints of this room score, and how much of that is the viewpoint."""
    path = os.path.join('outputs', os.path.basename(cap), 'sweep_eval.csv')
    if not os.path.exists(path):
        return []
    import csv
    with open(path) as f:
        rows = [{k: v for k, v in r.items()} for r in csv.DictReader(f)]
    if not rows:
        return []
    num = lambda r, k: float(r[k])
    rows.sort(key=lambda r: -num(r, rank))
    head = ['| frame | cam z | tilt | hidden | SC set | occupied | SC | margin | SSC |',
            '| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |']
    line = lambda r: ('| `live_%06d` | %.2f m | %.1f° | %.2f m | %s | %.1f%% | %.1f | %+.1f | %.1f |'
                      % (int(r['frame']), num(r, 'cam_z'), num(r, 'tilt_deg'),
                         num(r, 'free_mean') if r.get('free_mean') else 0.0,
                         r['sc_voxels'], num(r, 'occ_pct'),
                         num(r, 'sc_iou'), num(r, 'margin'), num(r, 'ssc_miou')))
    sc = np.array([num(r, 'sc_iou') for r in rows])
    ss = np.array([num(r, 'ssc_miou') for r in rows])
    occ = np.array([num(r, 'occ_pct') for r in rows])
    mg = np.array([num(r, 'margin') for r in rows])
    L = ['', '## Across the sweep', '',
         '`reconstruction_GT/sweep_eval.py` scored **%d frames** of this room '
         'against the same annotation (`%s`). One frame gives one number, and '
         'that number says as much about the viewpoint as about the model.'
         % (len(rows), path), '',
         '| | SC | SSC |', '| --- | ---: | ---: |',
         '| best | %.1f | %.1f |' % (sc.max(), ss.max()),
         '| median | %.1f | %.1f |' % (np.median(sc), np.median(ss)),
         '| worst | %.1f | %.1f |' % (sc.min(), ss.min()), '',
         '`margin` is SC minus what "predict occupied everywhere" would score '
         'on that frame, which is its `occupied` column. **Only %d of %d frames '
         'beat that baseline**, and %d have a scored set over 90%% occupied — '
         'in a small room most viewpoints leave the metric nothing to find.'
         % (int((mg > 0).sum()), len(rows), int((occ > 90).sum())), '',
         '### Best %d viewpoints (by %s)' % (top, rank), ''] + head
    L += [line(r) for r in rows[:top]]
    L += ['', '### Worst %d viewpoints' % top, ''] + head
    L += [line(r) for r in rows[-top:]]
    best_raw = max(rows, key=lambda r: (num(r, 'sc_iou'), -num(r, 'occ_pct')))
    if num(best_raw, 'occ_pct') > 95:
        L += ['', '> The highest **raw** SC in the sweep is %.1f, on frame %s — '
              'whose scored set is %.0f%% occupied, i.e. %s voxels of solid '
              'furniture with nothing empty to find. Rank by `margin`, not by '
              'SC.' % (num(best_raw, 'sc_iou'), 'live_%06d' % int(best_raw['frame']),
                       num(best_raw, 'occ_pct'), best_raw['sc_voxels'])]
    L += viewpoint_analysis(rows, num, cap)
    L += ['', 'These are scored in memory as the sweep runs, against the same '
          'ground truth and the same protocol; where a frame was also written '
          'to disk and scored from its files, the two agree to 0.2 SC. '
          '`--keep N` writes the N best and N worst out in full -- frame, '
          'ground truth, prediction and plys -- and they are the `live_...` '
          'rows in the per-frame table above.']
    return L


def markdown(cmSSC, cmSC, title, n_frames, fov, cap, pred_dir, per_frame=None):
    """The same numbers as the printed table, as a report that can be kept."""
    import time
    tp, fp, fn, iou = iou_from_cm(cmSSC)
    present = [c for c in range(1, 12) if cmSSC[c].sum() > 0]
    miou = float(np.mean(iou[present])) if present else 0.0
    tpS, fpS, fnS, iouS = iou_from_cm(cmSC)
    scored, occ = int(cmSC.sum()), cmSC[1].sum() / max(int(cmSC.sum()), 1)
    L = ['# Evaluation — %s' % title, '',
         'Generated %s by `reconstruction_GT/evaluate_gt.py` from `%s`, against '
         'the ground truth in `%s/gt`. %d frame%s.'
         % (time.strftime('%Y-%m-%d'), pred_dir, cap, n_frames,
            '' if n_frames == 1 else 's'), '',
         'Protocol is `examples/segmentation/test_NYU.py:206-211`, the same as '
         '`inference/document/EVALUATION.md`: **SSC** over voxels where '
         '`label_weight > 0` and `label != 255`, mIoU averaged over the classes '
         'present; **SC** the same set restricted to `mapping == 307200` — the '
         'voxels no depth pixel reached — scored occupied-vs-empty.', '',
         '`--fov %s`: %s.' % (fov, 'only voxels inside the camera view are '
                              'scored' if fov == 'frustum' else
                              'every annotated voxel is scored, including what '
                              'the camera never saw'), '',
         '| class | TP | FP | FN | precision | recall | IoU |',
         '| --- | ---: | ---: | ---: | ---: | ---: | ---: |']
    for c in range(12):
        if cmSSC[c].sum() == 0 and cmSSC[:, c].sum() == 0:
            continue
        L.append('| `%s`%s | %d | %d | %d | %.1f%% | %.1f%% | %.1f%% |'
                 % (CLASSES[c], ' *(not in mIoU)*' if c == 0 else '',
                    tp[c], fp[c], fn[c],
                    100 * tp[c] / max(tp[c] + fp[c], 1),
                    100 * tp[c] / max(tp[c] + fn[c], 1), 100 * iou[c]))
    L += ['', '**SSC mIoU (%d classes present): %.1f**' % (len(present), 100 * miou)]
    if len(present) < 11:
        L.append('  Absent from this ground truth, so not averaged: %s.'
                 % ', '.join('`%s`' % CLASSES[c] for c in range(1, 12)
                             if c not in present))
    L += ['', '**SC IoU: %.1f**  |  precision %.1f  recall %.1f  |  %d voxels, '
          '%.0f%% of them occupied'
          % (100 * iouS[1], 100 * tpS[1] / max(tpS[1] + fpS[1], 1),
             100 * tpS[1] / max(tpS[1] + fnS[1], 1), scored, 100 * occ), '',
          'NYU test for reference: SSC 47.7, SC 75.0 (`--nyu` reproduces both).']
    if per_frame and len(per_frame) > 1:
        L += ['', '## Per frame', '',
              'The total above pools every frame. Each on its own:', '',
              '| frame | scored voxels | SC set | occupied | SC | SSC | classes |',
              '| --- | ---: | ---: | ---: | ---: | ---: | ---: |']
        for r in per_frame:
            L.append('| `%s` | %d | %d | %.0f%% | %.1f | %.1f | %d |'
                     % (r['frame'], r['scored'], r['sc_voxels'], r['occ_pct'],
                        r['sc_iou'], r['ssc_miou'], r['classes']))
    if occ > 0.9 or occ < 0.05:
        L += ['', '> **This SC number is not usable.** %.0f%% of the SC set is '
              'occupied, so a model predicting "occupied" everywhere scores '
              'IoU %.3f. The frame cannot separate a good model from a trivial '
              'one: the camera saw the whole room, so the only hidden volume '
              'left is the inside of the annotated solids. SSC above is still '
              'meaningful. See MAKING_GT.md.' % (100 * occ, occ),
              '']
    L += ['',
          '**What the occupancy figure is, and is not.** It counts only '
          'voxels INSIDE the annotated room: everything beyond the shell is '
          '255 and enters neither side of the fraction. So it does not say '
          'the room is full -- it says how much of the hidden volume within '
          'these walls is furniture and wall interior, which rises as the '
          'room gets smaller, because a camera standing in a small room sees '
          'nearly all of its free space. It is a measurement of the '
          'ANNOTATION, and the shell is the lever: on room07, growing the '
          'shell 0.3 m each way moves it from 97.6% to 68.1%, and 0.6 m to '
          '43.0%, without touching a single piece of furniture (wall '
          'thickness barely matters: 4 cm vs 2 cm gives 97.6% vs 97.5%). '
          'Growing the shell is not a legitimate fix -- those voxels are '
          'outside the room, and calling them empty would assert free space '
          'where there is a wall. The honest shell sits at the walls, and '
          'this number is its consequence.',
          '',
          '**SSC is not affected by it.** SSC averages per-class IoU over the '
          'classes present and leaves `empty` out of that average, so a '
          'mostly-occupied set is what it wants rather than a defect, and '
          'most of its set is genuinely hidden -- it measures completion, not '
          'visible segmentation.']
    L += sweep_section(cap)
    return '\n'.join(L) + '\n'


def main():
    from inference.utils import setup_logging
    setup_logging()
    p = argparse.ArgumentParser(description=__doc__.split('\n')[3])
    p.add_argument('capture_dir', nargs='?')
    p.add_argument('--pred_dir', default=None,
                   help='default outputs/<capture name>/prediction')
    p.add_argument('--fov', choices=['frustum', 'all'], default='frustum',
                   help='score only what the camera saw (default), or '
                        'everything annotated')
    p.add_argument('--nyu', action='store_true',
                   help='self-check: score NYU test and reproduce 75.0 / 47.7')
    p.add_argument('--limit', type=int, default=0, help='--nyu: frames to use')
    p.add_argument('--frame', nargs='+', default=None,
                   help='score only these frame stems (default: every frame '
                        'the capture has ground truth for)')
    p.add_argument('--report', nargs='?', const='', default=None, metavar='PATH',
                   help='also write a markdown report (default '
                        '<capture>/gt/EVALUATION.md)')
    args = p.parse_args()

    if args.nyu:
        cmSSC, cmSC, n = eval_nyu(args.limit or None)
        miou, sc = report(cmSSC, cmSC, 'NYU test (self-check)', n)
        ok = abs(100 * sc - 75.0) < 0.5 and abs(100 * miou - 47.7) < 0.5
        print('\nself-check: SC %.1f (expect 75.0), SSC %.1f (expect 47.7) -> %s'
              % (100 * sc, 100 * miou, 'PASS' if ok else 'FAIL'))
        return 0 if ok else 1

    if not args.capture_dir:
        raise SystemExit('give a capture folder, or --nyu')
    cap = args.capture_dir.rstrip('/')
    pred_dir = args.pred_dir or os.path.join('outputs', os.path.basename(cap),
                                             'prediction')
    cmSSC, cmSC, n, per_frame = eval_capture(cap, pred_dir,
                                             args.fov == 'frustum', args.frame)
    title = '%s%s (%s)' % (os.path.basename(cap),
                           ' ' + ', '.join(args.frame) if args.frame else '',
                         'camera view only' if args.fov == 'frustum'
                         else 'everything annotated')
    report(cmSSC, cmSC, title, n)
    if args.report is not None:
        path = args.report or os.path.join(cap, 'gt', 'EVALUATION.md')
        os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
        with open(path, 'w') as f:
            f.write(markdown(cmSSC, cmSC, title, n, args.fov, cap, pred_dir,
                             per_frame))
        print('\nreport: %s' % path)
    print('\nprotocol: test_NYU.py:206-211. Report the fov setting with the '
          'number.\nRun --nyu to confirm this evaluator still reproduces '
          '75.0 / 47.7 on NYU.')
    return 0


if __name__ == '__main__':
    sys.exit(main())
