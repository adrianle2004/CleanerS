"""
reconstruction_GT/sc_gap.py

Why SC is ~79 on NYU and ~55 on Occ-ScanNet, when the scoring rule is the same.

SC on its own cannot answer that, because SC moves with the difficulty of the
frame: the scored set is the hidden space, and a frame whose hidden space is
97% occupied is scored against a task that "say occupied everywhere" already
wins. The occupancy of the scored set IS the trivial baseline -- predict
occupied everywhere and TP = P, FP = N - P, FN = 0, so IoU = P/N exactly -- and
NYU's median baseline (55.3%) is 21 points above Occ-ScanNet's (34.3%). Half
the headline gap is that, and it says nothing about either model or annotation.

So this compares the two **at matched baseline**, and splits SC into precision
and recall, which do not move with it. That separates the two candidate causes:

    recall drops   -> the model fails to complete structure that is annotated
    precision drops -> the model predicts structure the annotation does not have

What it finds (--report): recall is comparable everywhere, precision is not,
and the precision gap closes as the baseline rises -- 55.8% vs 90.2% in the
emptiest band, 97.3% vs 96.3% in the fullest. That is the signature of sparse
ground truth, not of a worse prediction.

*** STAGES ***

Each writes a CSV and is skipped if that CSV exists (--force redoes it):

  nyu-perframe        outputs/nyu_perframe.csv        SC / margin / SSC per NYU test frame
  nyu-viewpoints      outputs/nyu_viewpoints.csv      + camera height, depth, hidden extent
  scannet-viewpoints  outputs/scannet_viewpoints.csv  the same on a sample of Occ-ScanNet val
  pr                  outputs/sc_gap.csv              precision / recall / baseline per frame,
                                                      all four datasets

`nyu-viewpoints` and `scannet-viewpoints` are what `evaluate_gt.py`'s NYU and
ScanNet reference columns read; `pr` is what the matched-baseline table in
`inference/document/EVALUATION_SCANNET.md` reads.

Needs, besides the repo: ../data/NYU, ../data/Scannet, predictions in
custom_visual_pred/CleanerS/prediction (NYU, `inference/run_inference.py`),
outputs/scannet/{encoded,prediction} and outputs/scannet/per_frame.csv
(`inference/run_scannet.py`, `inference/evaluate_scannet.py`), and
outputs/<room>/prediction for the captured rooms.

    python -m reconstruction_GT.sc_gap --all        # build what is missing, then report
    python -m reconstruction_GT.sc_gap --report     # just the tables
"""

import os
import sys
import csv
import json
import glob
import random
import argparse
import logging

import numpy as np

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_THIS_DIR)
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from reconstruction_GT.evaluate_gt import (                      # noqa: E402
    confusion, iou_from_cm, score, NYU_ROOT, NYU_PRED)
from reconstruction_GT.voxelize_gt import (                      # noqa: E402
    Grid, frustum_mask, VOX_SIZE_LOW, VOX_UNIT_LOW, IGNORE)

log = logging.getLogger(__name__)

SCANNET_ROOT = os.path.join(os.path.dirname(_REPO_ROOT), 'data', 'Scannet')
NYU_PERFRAME = os.path.join(_REPO_ROOT, 'outputs', 'nyu_perframe.csv')
NYU_VIEWPOINTS = os.path.join(_REPO_ROOT, 'outputs', 'nyu_viewpoints.csv')
SCANNET_VIEWPOINTS = os.path.join(_REPO_ROOT, 'outputs', 'scannet_viewpoints.csv')
SC_GAP = os.path.join(_REPO_ROOT, 'outputs', 'sc_gap.csv')
ROOMS = ('room07', 'room08')
SCANNET_SAMPLE, SCANNET_SEED = 1500, 2
MAPPING_SENTINEL = 307200            # no depth pixel reached the voxel

# The bands the matched-baseline table uses. Edges, not centres; the last one
# is open at 100% so a fully occupied scored set lands in it.
BANDS = ((0, 20), (20, 35), (35, 50), (50, 65), (65, 80), (80, 101))


def _write(path, rows):
    if not rows:
        raise SystemExit('nothing to write to %s' % path)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w') as f:
        w = csv.DictWriter(f, list(rows[0]))
        w.writeheader()
        for r in rows:
            w.writerow(r)
    log.info('wrote %s with %d rows', os.path.relpath(path, _REPO_ROOT), len(rows))


def read_csv(path):
    """-> list of dicts, numbers as float, '' as nan."""
    out = []
    for r in csv.DictReader(open(path)):
        d = {}
        for k, v in r.items():
            if k in ('frame', 'scene', 'dataset'):
                d[k] = v
                continue
            try:
                d[k] = float(v) if v not in ('', None) else np.nan
            except (TypeError, ValueError):
                d[k] = np.nan
        out.append(d)
    return out


def hidden_extent(world_pts, cam_t, lo, hi):
    """Metres from each measured surface point to where its ray leaves the box.

    This is the "room behind the visible surface" of evaluate_gt.py's
    viewpoint analysis (`free_mean`). On the captured rooms the box is the
    annotated shell; NYU and Occ-ScanNet have no shell, so it is the grid,
    which is what bounds their annotation. It counts the inside of a wardrobe
    exactly as it counts the air behind one -- see USING_THE_MODEL.md.
    """
    d = world_pts - cam_t
    L = np.linalg.norm(d, axis=1)
    d = d / np.maximum(L, 1e-9)[:, None]
    with np.errstate(divide='ignore', invalid='ignore'):
        t1, t2 = (lo - cam_t) / d, (hi - cam_t) / d
    t_exit = np.minimum.reduce(np.maximum(t1, t2), axis=1)
    return np.clip(t_exit - L, 0.0, None)


def grid_box(vox_origin):
    """-> (lo, hi) world corners of the 60x36x60 @ 8 cm grid."""
    lo = np.asarray(vox_origin, np.float64)
    size = np.array([VOX_SIZE_LOW[0], VOX_SIZE_LOW[2], VOX_SIZE_LOW[1]])
    return lo, lo + size * VOX_UNIT_LOW


def unproject(depth, K):
    """-> (points in the camera frame, the valid depths) for depth in metres."""
    H, W = depth.shape
    vs, us = np.meshgrid(np.arange(H, dtype=np.float32),
                         np.arange(W, dtype=np.float32), indexing='ij')
    z = depth.reshape(-1)
    ok = z > 0
    pt = np.stack([(us.reshape(-1) - K[0, 2]) * z / K[0, 0],
                   (vs.reshape(-1) - K[1, 2]) * z / K[1, 1], z], axis=1)[ok]
    return pt, z[ok]


def tilt_of(R):
    """Degrees the optical axis is below horizontal, from a world-from-camera R."""
    up = R.T @ np.array([0.0, 0.0, 1.0])              # world up, in camera coords
    return float(np.degrees(np.arccos(min(1.0, -up[1] / np.linalg.norm(up)))))


# --------------------------------------------------------------- stage: NYU SC

def stage_nyu_perframe():
    """SC, its trivial baseline, the margin and SSC, per NYU test frame."""
    ids = [l.strip() for l in open(os.path.join(NYU_ROOT, 'test.txt')) if l.strip()]
    rows = []
    for k, fid in enumerate(ids):
        p = os.path.join(NYU_PRED, 'NYU%s_0000.npy' % fid)
        lab = os.path.join(NYU_ROOT, 'Label', fid + '.npz')
        if not (os.path.exists(p) and os.path.exists(lab)):
            continue
        a, b = score(np.load(p), np.load(lab)['arr_0'],
                     np.load(os.path.join(NYU_ROOT, 'TSDF', fid + '.npz'))['arr_1'],
                     np.load(os.path.join(NYU_ROOT, 'Mapping', fid + '.npz'))['arr_0'])
        _, _, _, iou = iou_from_cm(b)
        n = int(b.sum())
        occ = 100.0 * b[1].sum() / max(n, 1)          # == the trivial baseline
        _, _, _, i2 = iou_from_cm(a)
        pres = [c for c in range(1, 12) if a[c].sum() > 0]
        rows.append(dict(frame=fid, sc_voxels=n, occ_pct=round(occ, 2),
                         sc_iou=round(100 * float(iou[1]), 2),
                         margin=round(100 * float(iou[1]) - occ, 2),
                         ssc=round(100 * float(np.mean(i2[pres])) if pres else 0, 2)))
        if k % 100 == 0:
            log.info('  %d/%d', k, len(ids))
    _write(NYU_PERFRAME, rows)


def stage_nyu_viewpoints():
    """The same viewpoint quantities sweep_eval records, on NYU test."""
    import cv2
    from inference.frame_loader import FrameLoader, CAM_K
    if not os.path.exists(NYU_PERFRAME):
        stage_nyu_perframe()
    scored = {r['frame']: r for r in csv.DictReader(open(NYU_PERFRAME))}
    rows = []
    for k, fid in enumerate(sorted(scored)):
        b = os.path.join(NYU_ROOT, 'depth', 'NYU%s_0000.bin' % fid)
        p = os.path.join(NYU_ROOT, 'depth', 'NYU%s_0000.png' % fid)
        if not (os.path.exists(b) and os.path.exists(p)):
            continue
        vox_origin, cam_pose = FrameLoader._load_bin_header(b)
        raw = cv2.imread(p, cv2.IMREAD_UNCHANGED).astype(np.uint16)
        # NYU ships the 16 bits rotated by 3, as SSCNet wrote them
        depth = ((raw << 13) | (raw >> 3)).astype(np.float32) / 1000.0
        pt, z = unproject(depth, CAM_K)
        R, t = np.asarray(cam_pose)[:3, :3], np.asarray(cam_pose)[:3, 3]
        Wpt = pt @ R.T + t
        lo, hi = grid_box(vox_origin)
        free = hidden_extent(Wpt, t.astype(np.float64), lo, hi)
        r = scored[fid]
        rows.append(dict(frame=fid,
                         cam_z=round(float(t[2]), 4),
                         tilt_deg=round(tilt_of(R), 2),
                         depth_med=round(float(np.median(z)), 3),
                         near_pct=round(100 * float((z < 1.5).mean()), 1),
                         floor_pct=round(100 * float((Wpt[:, 2] < 0.12).mean()), 1),
                         free_mean=round(float(np.mean(free[np.isfinite(free)])), 3),
                         sc_voxels=int(r['sc_voxels']), occ_pct=float(r['occ_pct']),
                         sc_iou=float(r['sc_iou']), margin=float(r['margin']),
                         ssc_miou=float(r['ssc'])))
        if k % 100 == 0:
            log.info('  %d/%d', k, len(scored))
    _write(NYU_VIEWPOINTS, rows)


def stage_scannet_viewpoints():
    """The same, on a fixed random sample of Occ-ScanNet val, plus floor quality."""
    import cv2
    per_frame = os.path.join(_REPO_ROOT, 'outputs', 'scannet', 'per_frame.csv')
    if not os.path.exists(per_frame):
        raise SystemExit('need %s -- run inference/evaluate_scannet.py first'
                         % os.path.relpath(per_frame, _REPO_ROOT))
    scored = {}
    for r in csv.DictReader(open(per_frame)):
        # a handful of frames scored nothing at all (zero voxels in the set);
        # they carry empty score fields and cannot contribute to a correlation
        if r['sc_iou_nyu_protocol'] == '' or float(r['scored_voxels_nyu_protocol']) <= 0:
            continue
        scored[(r['scene'], r['frame'])] = r
    random.seed(SCANNET_SEED)
    sample = random.sample(sorted(scored), min(SCANNET_SAMPLE, len(scored)))
    posed = os.path.join(SCANNET_ROOT, 'posed_images')
    rows = []
    for i, (scene, frame) in enumerate(sample):
        gtp = os.path.join(SCANNET_ROOT, scene, frame + '.npz')
        dpp = os.path.join(posed, scene, frame + '.png')
        if not (os.path.exists(gtp) and os.path.exists(dpp)):
            continue
        d = np.load(gtp)
        g = d['target'].reshape(VOX_SIZE_LOW)
        vox_origin = np.asarray(d['voxel_origin'], np.float64)
        cam_pose = np.asarray(d['cam_pose'], np.float64)

        # how level the GT floor sits in the grid: fit gy = a*gz + b*gx + c
        gz, gy, gx = np.nonzero(g == 2)
        floor_n, floor_tilt = len(gz), np.nan
        if floor_n >= 50:
            A = np.stack([gz, gx, np.ones_like(gz)], 1).astype(np.float64)
            c, *_ = np.linalg.lstsq(A, gy.astype(np.float64), rcond=None)
            n = np.array([-c[0], -c[1], 1.0])
            n /= np.linalg.norm(n)
            floor_tilt = float(np.degrees(np.arccos(abs(n[2]))))

        depth = cv2.imread(dpp, cv2.IMREAD_UNCHANGED)
        if depth is None:
            continue
        depth = depth.astype(np.float32) / 1000.0
        H, W = depth.shape
        K = np.asarray(d['intrinsic_color'], np.float64)[:3, :3]
        sy, sx = H / 968.0, W / 1296.0        # colour intrinsics -> this image
        K = K * np.array([[sx, 1, sx], [1, sy, sy], [1, 1, 1]])
        pt, z = unproject(depth, K)
        if len(z) < 5000:
            continue
        Wp = pt @ cam_pose[:3, :3].T + cam_pose[:3, 3]
        o = cam_pose[:3, 3]
        lo, hi = grid_box(vox_origin)
        free = hidden_extent(Wp, o, lo, hi)
        r = scored[(scene, frame)]
        rows.append(dict(
            scene=scene, frame=frame,
            cam_z=round(float(o[2]), 4),
            tilt_deg=round(tilt_of(cam_pose[:3, :3]), 2),
            depth_med=round(float(np.median(z)), 3),
            near_pct=round(100 * float((z < 1.5).mean()), 1),
            floor_pct=round(100 * float((Wp[:, 2] < vox_origin[2] + 0.12).mean()), 1),
            free_mean=round(float(np.mean(free[np.isfinite(free)])), 3),
            floor_voxels=floor_n,
            floor_tilt_deg=round(floor_tilt, 2) if floor_n >= 50 else '',
            sc_voxels=int(float(r['scored_voxels_nyu_protocol'])),
            occ_pct=round(100 * float(r['gt_occupied'])
                          / max(float(r['scored_voxels_nyu_protocol']), 1), 2),
            sc_iou=float(r['sc_iou_nyu_protocol']),
            ssc_miou=float(r['ssc_miou_nyu_protocol']),
            nyu_like=int(r['nyu_like'])))
        if i % 200 == 0:
            log.info('  %d/%d', i, len(sample))
    _write(SCANNET_VIEWPOINTS, rows)


# ------------------------------------------------- stage: precision / recall

def pr_from(pred, label, lw, mapping, keep=None):
    """SC with its parts, on one frame's flat arrays.

    The scored set is `score()`'s: label_weight > 0, label != 255, and no depth
    pixel reached the voxel. `occ` is that set's occupancy, which is both the
    difficulty of the frame and what a constant "occupied" prediction scores.
    """
    label = label.astype(np.int64).reshape(-1).copy()
    if keep is not None:
        label[~keep.reshape(-1)] = IGNORE
    sel = ((lw.reshape(-1) > 0) & (label != IGNORE)
           & (mapping.reshape(-1) == MAPPING_SENTINEL))
    if sel.sum() == 0:
        return None
    cm = confusion((pred.astype(np.int64).reshape(-1)[sel] > 0).astype(int),
                   (label[sel] > 0).astype(int), 2)
    tp, fp, fn, iou = iou_from_cm(cm)
    return dict(n=int(cm.sum()),
                occ=round(100 * float(cm[1].sum()) / max(cm.sum(), 1), 2),
                sc=round(100 * float(iou[1]), 2),
                prec=round(100 * tp[1] / max(tp[1] + fp[1], 1), 2),
                rec=round(100 * tp[1] / max(tp[1] + fn[1], 1), 2))


def stage_pr():
    """One row per frame of all four datasets: SC, precision, recall, baseline."""
    from inference.frame_loader import cam_pose_from_meta, frame_meta
    rows = []

    for r in read_csv(NYU_VIEWPOINTS) if os.path.exists(NYU_VIEWPOINTS) else []:
        fid = r['frame']
        p = os.path.join(NYU_PRED, 'NYU%s_0000.npy' % fid)
        lab = os.path.join(NYU_ROOT, 'Label', fid + '.npz')
        if not (os.path.exists(p) and os.path.exists(lab)):
            continue
        m = pr_from(np.load(p), np.load(lab)['arr_0'],
                    np.load(os.path.join(NYU_ROOT, 'TSDF', fid + '.npz'))['arr_1'],
                    np.load(os.path.join(NYU_ROOT, 'Mapping', fid + '.npz'))['arr_0'])
        if m:
            rows.append(dict(dataset='NYU', scene='', frame=fid,
                             free_mean=r['free_mean'], ssc=r['ssc_miou'], **m))
    log.info('NYU: %d frames', sum(r['dataset'] == 'NYU' for r in rows))

    sc_dir = os.path.join(_REPO_ROOT, 'outputs', 'scannet')
    for r in read_csv(SCANNET_VIEWPOINTS) if os.path.exists(SCANNET_VIEWPOINTS) else []:
        scene, frame = r['scene'], r['frame']
        e = os.path.join(sc_dir, 'encoded', scene, frame + '.npz')
        p = os.path.join(sc_dir, 'prediction', scene, frame + '.npy')
        g = os.path.join(SCANNET_ROOT, scene, frame + '.npz')
        if not all(os.path.exists(x) for x in (e, p, g)):
            continue
        en = np.load(e)
        gt = np.load(g)['target'].reshape(-1).astype(np.int64)
        # label_weight rebuilt as data/generate_cleaners_data.py does for NYU:
        # a GT object, or space our own encoder measured as occupied-side TSDF
        lw = (((gt > 0) & (gt != IGNORE)) | (en['tsdf'].reshape(-1) < -0.5)).astype(np.float32)
        m = pr_from(np.load(p), gt, lw, en['mapping'].reshape(-1))
        if m:
            rows.append(dict(dataset='ScanNet', scene=scene, frame=frame,
                             free_mean=r['free_mean'], ssc=r['ssc_miou'], **m))
    log.info('ScanNet: %d frames', sum(r['dataset'] == 'ScanNet' for r in rows))

    for room in ROOMS:
        cap = os.path.join(_REPO_ROOT, 'captures', room)
        mj = os.path.join(cap, 'meta.json')
        if not os.path.exists(mj):
            continue
        meta = json.load(open(mj))
        sweep = {}
        swp = os.path.join(_REPO_ROOT, 'outputs', room, 'sweep_eval.csv')
        if os.path.exists(swp):
            sweep = {r['frame']: r for r in read_csv(swp)}
        n0 = len(rows)
        for stem in meta.get('frames', []):
            pred = os.path.join(_REPO_ROOT, 'outputs', room, 'prediction', stem + '.npy')
            lab = os.path.join(cap, 'gt', 'Label', stem + '.npz')
            if not (os.path.exists(pred) and os.path.exists(lab)):
                continue
            fm = frame_meta(meta, stem)
            keep = frustum_mask(Grid([-2.4, 0.0, -0.05], VOX_SIZE_LOW, VOX_UNIT_LOW),
                                np.asarray(fm['cam_K'], np.float64),
                                cam_pose_from_meta(meta, stem))
            m = pr_from(np.load(pred), np.load(lab)['arr_0'],
                        np.load(os.path.join(cap, 'gt', 'TSDF', stem + '.npz'))['arr_1'],
                        np.load(os.path.join(cap, 'gt', 'Mapping', stem + '.npz'))['arr_0'],
                        keep)
            if not m:
                continue
            sw = sweep.get(stem.replace('live_', '').lstrip('0') or '0', {})
            rows.append(dict(dataset=room, scene='', frame=stem,
                             free_mean=sw.get('free_mean', np.nan),
                             ssc=sw.get('ssc_miou', np.nan), **m))
        log.info('%s: %d frames', room, len(rows) - n0)

    _write(SC_GAP, rows)


# ------------------------------------------------------------------- the table

def matched_baseline(path=SC_GAP):
    """-> (bands, totals) for the report and for EVALUATION_SCANNET.md.

    bands:  [(lo, hi, {dataset: {sc, prec, rec, n}})]   medians within the band
    totals: {dataset: {sc, prec, rec, occ, scored, occupied, n}}
    """
    if not os.path.exists(path):
        return [], {}
    rows = read_csv(path)
    by = {}
    for r in rows:
        by.setdefault(r['dataset'], []).append(r)

    def med(g, k):
        v = [x[k] for x in g if np.isfinite(x[k])]
        return float(np.median(v)) if v else np.nan

    bands = []
    for lo, hi in BANDS:
        cell = {}
        for ds, g in by.items():
            b = [r for r in g if lo <= r['occ'] < hi]
            if b:
                cell[ds] = dict(sc=med(b, 'sc'), prec=med(b, 'prec'),
                                rec=med(b, 'rec'), n=len(b))
        bands.append((lo, hi, cell))
    totals = {}
    for ds, g in by.items():
        totals[ds] = dict(sc=med(g, 'sc'), prec=med(g, 'prec'), rec=med(g, 'rec'),
                          occ=med(g, 'occ'), scored=med(g, 'n'),
                          occupied=float(np.median([r['n'] * r['occ'] / 100 for r in g])),
                          n=len(g))
    return bands, totals


def markdown_section(path=SC_GAP, a='NYU', b='ScanNet'):
    """The matched-baseline comparison as markdown lines, or [] without the CSV."""
    bands, totals = matched_baseline(path)
    if not bands or a not in totals or b not in totals:
        return []
    nm = {'ScanNet': 'Occ-ScanNet'}
    L = ['### 3. The gap at matched difficulty — precision, not recall\n']
    L.append('SC moves with the difficulty of the frame, so part of the headline gap is not a '
             'result. The scored set is the hidden space, and its occupancy is exactly what a '
             'constant "occupied" prediction scores: predict occupied everywhere and TP = P, '
             'FP = N − P, FN = 0, so IoU = P/N. NYU\'s median scored set is %.0f%% occupied '
             'against Occ-ScanNet\'s %.0f%%, so NYU is being asked an easier question before the '
             'model is involved.\n'
             % (totals[a]['occ'], totals[b]['occ']))
    L.append('Matched on that baseline, and split into the two halves of SC that do **not** move '
             'with it (medians within each band, frame counts in brackets):\n')
    L.append('| scored set occupied | %s SC | prec | recall | %s SC | prec | recall |'
             % (nm.get(a, a), nm.get(b, b)))
    L.append('| --- | ---: | ---: | ---: | ---: | ---: | ---: |')
    for lo, hi, cell in bands:
        if a not in cell and b not in cell:
            continue
        row = ['%d–%d%%' % (lo, min(hi, 100))]
        for ds in (a, b):
            c = cell.get(ds)
            row += ['%.1f (%d)' % (c['sc'], c['n']), '%.1f%%' % c['prec'],
                    '%.1f%%' % c['rec']] if c else ['—', '—', '—']
        L.append('| ' + ' | '.join(row) + ' |')
    L.append('')
    lo0, hi0, c0 = bands[0]
    loN, hiN, cN = bands[-1]
    L.append('**Recall is comparable throughout; precision is not, and the precision gap closes '
             'as the baseline rises.** In the emptiest band (%d–%d%% occupied) the same model '
             'predicts at %.1f%% precision on %s against %.1f%% on %s, while recall differs by '
             '%.1f points. In the fullest (%d–%d%%) the two are indistinguishable: precision '
             '%.1f%% against %.1f%%, SC %.1f against %.1f.\n'
             % (lo0, min(hi0, 100), c0[b]['prec'], nm.get(b, b), c0[a]['prec'], nm.get(a, a),
                abs(c0[a]['rec'] - c0[b]['rec']), loN, min(hiN, 100),
                cN[b]['prec'], cN[a]['prec'], cN[b]['sc'], cN[a]['sc']))
    L.append('A recall gap would mean the model fails to complete structure the annotation does '
             'carry. A precision gap that appears only where the annotation says "mostly empty" '
             'means the opposite: the model predicts structure the annotation does not carry, and '
             'is charged for it. There is less to carry — %s\'s median scored set holds %.0f '
             'occupied voxels out of %.0f (%.0f%%) against %s\'s %.0f out of %.0f (%.0f%%) — so '
             'each false positive also costs proportionally more IoU.\n'
             % (nm.get(b, b), totals[b]['occupied'], totals[b]['scored'],
                100 * totals[b]['occupied'] / max(totals[b]['scored'], 1),
                nm.get(a, a), totals[a]['occupied'], totals[a]['scored'],
                100 * totals[a]['occupied'] / max(totals[a]['scored'], 1)))
    L.append('This does not clear the model — the prediction is the same either way — but it puts '
             'the SC gap on the annotation side, alongside section 1\'s floor, rather than on a '
             'failure to complete. `document/USING_THE_MODEL.md` carries the same comparison '
             'against the two hand-annotated rooms, which are denser than either benchmark.\n')
    L.append('Reproduce: `python -m reconstruction_GT.sc_gap --all` (writes `outputs/sc_gap.csv`, '
             'one row per frame of all four datasets with SC, precision, recall and the '
             'baseline).\n')
    return L


def print_report(path=SC_GAP):
    bands, totals = matched_baseline(path)
    if not bands:
        raise SystemExit('no %s -- run --all first' % os.path.relpath(path, _REPO_ROOT))
    order = [d for d in ('NYU', 'ScanNet') + ROOMS if d in totals]
    print('\nSC matched on the trivial baseline (the occupancy of the scored set)')
    print('%-12s' % 'baseline' + ''.join('  %-30s' % ('%s  SC / prec / rec' % d) for d in order))
    for lo, hi, cell in bands:
        line = '%-12s' % ('%d-%d%%' % (lo, min(hi, 100)))
        for ds in order:
            c = cell.get(ds)
            line += ('  %5.1f / %5.1f%% / %5.1f%% (%4d)' % (c['sc'], c['prec'], c['rec'], c['n'])
                     if c else '  %-30s' % '-')
        print(line)
    print()
    for ds in order:
        t = totals[ds]
        print('  %-8s %5d frames   scored median %6.0f of which %6.0f occupied   '
              'SC %5.1f  prec %5.1f%%  rec %5.1f%%  baseline %5.1f%%'
              % (ds, t['n'], t['scored'], t['occupied'], t['sc'], t['prec'],
                 t['rec'], t['occ']))
    print()


STAGES = (('nyu-perframe', NYU_PERFRAME, stage_nyu_perframe),
          ('nyu-viewpoints', NYU_VIEWPOINTS, stage_nyu_viewpoints),
          ('scannet-viewpoints', SCANNET_VIEWPOINTS, stage_scannet_viewpoints),
          ('pr', SC_GAP, stage_pr))


def main():
    from inference.utils import setup_logging
    ap = argparse.ArgumentParser(description=__doc__.strip().splitlines()[2],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--stage', choices=[s[0] for s in STAGES] + ['all'],
                    help='build one stage (or all); default is --report only')
    ap.add_argument('--all', action='store_true', help='same as --stage all, then report')
    ap.add_argument('--report', action='store_true', help='print the tables')
    ap.add_argument('--force', action='store_true', help='rebuild even if the CSV exists')
    a = ap.parse_args()
    setup_logging()
    want = 'all' if a.all else a.stage
    for name, path, fn in STAGES:
        if want not in ('all', name):
            continue
        if os.path.exists(path) and not a.force:
            log.info('%s: %s exists, skipping (--force to redo)',
                     name, os.path.relpath(path, _REPO_ROOT))
            continue
        log.info('=== %s', name)
        fn()
    if a.report or a.all or not want:
        print_report()
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
