"""
inference/evaluate.py

Per-class TP / FP / FN, precision, recall and IoU for the saved predictions,
on either NYU split, written out as a markdown report.

    python inference/evaluate.py                       # the whole report -> stdout
    python inference/evaluate.py --report inference/document/EVALUATION.md
    python inference/evaluate.py --split test          # one split, fewer sections

With or without --report you get the SAME complete document -- --report only
chooses where it goes.

TEST FIRST. Both splits are evaluated, but the per-class tables are printed for
TEST only -- the held-out split the paper reports. Train appears only where a
section compares the two (marked "needs both splits" below). --train-tables
prints the full train tables as well; --split train prints only train.

    test        metrics table (TP/FP/FN, precision, recall, IoU)
                what the SC IoU is made of (the other 25%)
                scene completion (SC) per class, two ways
                how much of each class's prediction is wrong
                how much of each class the model misses
                the biggest single confusions
                which classes invent volume
                how much of each prediction is scored at all
                how often each class is asserted at all
    once        what label_weight is                    (test frames)
                learned habit, or the data?             (needs both splits)
                three kinds of error (false positives)  (needs both splits)
                three kinds of miss (false negatives)   (needs both splits)
                train vs test headline                  (needs both splits)

Every number in the prose is computed at generation time, not typed in, so a
rerun cannot leave the write-up contradicting its own tables.

It reads what is already on disk -- custom_visual_pred/CleanerS/prediction/
against data/NYU/{Label,TSDF,Mapping} -- so it runs in about a minute on CPU
and needs neither the model nor a GPU. Rerun it whenever the predictions are
regenerated.

*** THE PROTOCOL IS test_NYU.py's, NOT AN INVENTION OF THIS FILE ***

Copied from examples/segmentation/test_NYU.py:test() so the numbers here are
comparable with the ones the repo reports:

    SSC   voxels where label_weight > 0 AND label != 255.
          A 12x12 confusion matrix over classes 0..11; mIoU averages classes
          1..11, i.e. `empty` is in the matrix but not in the mean.

    SC    the same, further restricted to mapping == 307200 -- the voxels NO
          depth pixel unprojected into, i.e. the part that has to be COMPLETED
          rather than merely observed. Binary: occupied vs empty.

label_weight (TSDF/*.npz arr_1) is NYU's own scoring mask, GT-occupied |
(tsdf < -0.5). Everything outside it is invisible to these numbers, which is
worth remembering when a class looks better here than it does on screen: a
prediction that spills into unannotated space costs nothing.

*** THE TRAIN SPLIT IS NOT A GENERALISATION NUMBER ***

The checkpoint was trained on train.txt. Its numbers there are reported so the
two can be compared -- the gap is the interesting part -- but they say nothing
about held-out performance. Only `test` does.
"""

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from visualize import DEFAULT_COLORMAP  # noqa: E402

try:
    from utils import CLASSES  # noqa: E402
except Exception:                      # utils drags in torch; not needed here
    CLASSES = ['empty', 'ceiling', 'floor', 'wall', 'window', 'chair', 'bed',
               'sofa', 'table', 'tvs', 'furn', 'objs']

# every class index here indexes the prediction arrays, the colormap and the
# GT alike, so a mismatch would silently mislabel every row of every table
assert len(CLASSES) == 12 and CLASSES[0] == 'empty', CLASSES
assert len(DEFAULT_COLORMAP) >= len(CLASSES), "colormap shorter than CLASSES"

GRID = (60, 36, 60)
# every voxel falls in exactly one, tested in this order (see evaluate())
REGION5 = ['surface — a depth point landed in it',
           'seen free — `tsdf > 0`, the camera looked through it',
           'just behind a surface — `-0.5 <= tsdf < 0`',
           'occluded — `tsdf < -0.5`',
           'never observed — `tsdf == 0`']
VOX_M = 0.08                      # voxel edge, metres
N_CLASSES = 12
IGNORE = 255
MAPPING_SENTINEL = 307200          # frame_loader.MAPPING_SENTINEL, 480 * 640

DEFAULT_NYU = os.path.join(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))), '..', 'data', 'NYU')
DEFAULT_PRED = os.path.join(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))), 'custom_visual_pred', 'CleanerS', 'prediction')


def confusion(pred, true, n=N_CLASSES):
    """n x n counts, rows = ground truth, columns = prediction.

    Same construction as cleaner/utils/metrics.py: a single bincount over
    true * n + pred, which is why it costs nothing to accumulate per frame.
    """
    k = true.astype(np.int64) * n + pred.astype(np.int64)
    return np.bincount(k, minlength=n * n).reshape(n, n)


def metrics(cm):
    """TP / FP / FN and the three rates, per class, from a confusion matrix.

    tp  the diagonal: predicted this class and it was this class
    fn  row sum minus tp: it was this class and we said something else
    fp  column sum minus tp: we said this class and it was something else

    precision = tp / (tp + fp)   of what we claimed, how much was right
    recall    = tp / (tp + fn)   of what was there, how much we found
    iou       = tp / (tp + fp + fn)   both failures in one number

    NOTE test_NYU.py prints precision under the heading "accuracy"
    (metrics.py: acc_per_cls = tp / predicted). Same quantity, misleading name.
    """
    tp = np.diag(cm).astype(np.float64)
    fn = cm.sum(axis=1) - tp
    fp = cm.sum(axis=0) - tp
    with np.errstate(divide='ignore', invalid='ignore'):
        prec = np.where(tp + fp > 0, tp / (tp + fp), np.nan)
        rec = np.where(tp + fn > 0, tp / (tp + fn), np.nan)
        iou = np.where(tp + fp + fn > 0, tp / (tp + fp + fn), np.nan)
    return dict(tp=tp, fp=fp, fn=fn, precision=prec, recall=rec, iou=iou)


def load_ids(nyu_root, split):
    path = os.path.join(nyu_root, '%s.txt' % split)
    if not os.path.exists(path):
        raise SystemExit("no %r -- expected NYU's split list there" % path)
    return [l.strip() for l in open(path) if l.strip()]


def evaluate(ids, nyu_root, pred_dir, verbose=True):
    """Accumulate the SSC and SC confusion matrices over a list of frame ids."""
    cm_ssc = np.zeros((N_CLASSES, N_CLASSES), dtype=np.int64)
    cm_sc = np.zeros((2, 2), dtype=np.int64)
    # per predicted class: how many of its voxels the protocol even looks at.
    # columns: outside label_weight | inside but GT==255 | scored
    cover = np.zeros((N_CLASSES, 3), dtype=np.int64)
    # per class: in how many FRAMES it appears at all, predicted vs annotated.
    # A class the model asserts far more often than the GT does is applying a
    # prior rather than reading the scene.
    frames = np.zeros((N_CLASSES, 2), dtype=np.int64)
    # per class, OUTSIDE the scored set: voxels the GT calls empty, and the
    # subset of those the depth frame also measured as free space (tsdf > 0,
    # i.e. in front of a surface the camera saw). Contradicted by the
    # annotation and by the sensor, and counted by neither.
    unscored = np.zeros((N_CLASSES, 2), dtype=np.int64)
    # the few quantities the prose quotes that no table carries: the check of
    # what label_weight is, how each frame divides around it, and the floor /
    # ceiling geometry. Layer k of axis 1 (up) sits k * 8 cm above layer 0,
    # the floor's layer.
    geo = dict(lw_match=0, lw_total=0, inside=0, never=0, free=0, other=0,
               gt_where=None, fn_where=None, region5=None, cm_sc12=None,
               floor_pred=[], floor_gt=[], ceil_pred_layer=[],
               ceil_gt_layer=[], ceil_pred_fill=[])
    F, C = CLASSES.index('floor'), CLASSES.index('ceiling')
    # every scored GT object voxel, by where it sits relative to the camera:
    #   0 surface    a depth pixel landed in it (mapping != sentinel)
    #   1 seen free  no pixel, but tsdf > 0: the camera looked through it
    #   2 hidden     neither -- occluded or never observed
    # gt_where[c, r] counts all of them; fn_where[c, k, r] the missed ones,
    # k = 0 called `empty`, k = 1 called another class. Every GT object voxel
    # is inside label_weight, so unlike FP no miss is ever unscored.
    gt_where = np.zeros((N_CLASSES, 3), dtype=np.int64)
    # the whole volume by region -- the five rows of REGION5 -- and, per
    # region: voxels, predicted `empty`, GT `empty`, inside label_weight
    region5 = np.zeros((len(REGION5), 4), dtype=np.int64)
    # the full 12x12 confusion inside the SC region, so SC can be broken down
    # per class -- cm_sc above is only its binary occupied/empty collapse
    cm_sc12 = np.zeros((N_CLASSES, N_CLASSES), dtype=np.int64)
    fn_where = np.zeros((N_CLASSES, 2, 3), dtype=np.int64)
    used, skipped = [], []

    for i, fid in enumerate(ids):
        pred_p = os.path.join(pred_dir, 'NYU%s_0000.npy' % fid)
        lab_p = os.path.join(nyu_root, 'Label', '%s.npz' % fid)
        tsdf_p = os.path.join(nyu_root, 'TSDF', '%s.npz' % fid)
        map_p = os.path.join(nyu_root, 'Mapping', '%s.npz' % fid)
        if not all(os.path.exists(p) for p in (pred_p, lab_p, tsdf_p, map_p)):
            skipped.append(fid)
            continue

        pred = np.load(pred_p).reshape(-1).astype(np.int64)
        label = np.load(lab_p)['arr_0'].reshape(-1).astype(np.int64)
        tsdf_z = np.load(tsdf_p)
        lw = tsdf_z['arr_1'].reshape(-1) > 0
        tsdf = tsdf_z['arr_0'].reshape(-1)
        mapping = np.load(map_p)['arr_0'].reshape(-1)

        keep = lw & (label != IGNORE)
        cm_ssc += confusion(pred[keep], label[keep])

        # where each predicted class's voxels fall relative to the scored set
        cover[:, 0] += np.bincount(pred[~lw], minlength=N_CLASSES)
        cover[:, 1] += np.bincount(pred[lw & (label == IGNORE)],
                                   minlength=N_CLASSES)
        cover[:, 2] += np.bincount(pred[keep], minlength=N_CLASSES)
        frames[:, 0] += np.bincount(np.unique(pred), minlength=N_CLASSES)
        frames[:, 1] += np.bincount(np.unique(label[label != IGNORE]),
                                    minlength=N_CLASSES)
        gt_empty_unscored = (~lw) & (label == 0)
        unscored[:, 0] += np.bincount(pred[gt_empty_unscored],
                                      minlength=N_CLASSES)
        unscored[:, 1] += np.bincount(pred[gt_empty_unscored & (tsdf > 0)],
                                      minlength=N_CLASSES)

        is_obj = (label > 0) & (label != IGNORE)
        region = np.where(mapping != MAPPING_SENTINEL, 0,
                          np.where(tsdf > 0, 1, 2))
        gt_where += np.bincount((label * 3 + region)[keep & is_obj],
                                minlength=N_CLASSES * 3).reshape(N_CLASSES, 3)
        r5 = np.select([mapping != MAPPING_SENTINEL, tsdf > 0, tsdf == 0,
                        tsdf >= -0.5], [0, 1, 4, 2], 3)
        for k in range(len(REGION5)):
            s = r5 == k
            region5[k] += (int(s.sum()), int((s & (pred == 0)).sum()),
                           int((s & (label == 0)).sum()), int((s & lw).sum()))
        missed = keep & is_obj & (pred != label)
        fn_where += np.bincount(
            (label * 6 + (pred != 0) * 3 + region)[missed],
            minlength=N_CLASSES * 6).reshape(N_CLASSES, 2, 3)
        geo['lw_match'] += int(((is_obj | (tsdf < -0.5)) == lw).sum())
        geo['lw_total'] += lw.size
        geo['inside'] += int(lw.sum())
        geo['never'] += int((~lw & (tsdf == 0)).sum())
        geo['free'] += int((~lw & (tsdf > 0)).sum())
        geo['other'] += int((~lw & (tsdf < 0)).sum())
        vg, vp = label.reshape(GRID), pred.reshape(GRID)
        if (vg == F).any():
            geo['floor_gt'].append((vg[:, 0, :] == F).mean())
            geo['floor_pred'].append((vp[:, 0, :] == F).mean())
        for vol, key in ((vp, 'ceil_pred'), (vg, 'ceil_gt')):
            per_layer = (vol == C).sum(axis=(0, 2))
            if per_layer.any():
                k = int(per_layer.argmax())
                geo[key + '_layer'].append(k)
                if key == 'ceil_pred':
                    geo['ceil_pred_fill'].append(
                        per_layer[k] / float(GRID[0] * GRID[2]))

        # SC: the same, but only where no depth pixel landed -- the region the
        # model has to COMPLETE rather than merely label
        sc = keep & (mapping == MAPPING_SENTINEL)
        cm_sc += confusion((pred[sc] > 0).astype(np.int64),
                           (label[sc] > 0).astype(np.int64), n=2)
        cm_sc12 += confusion(pred[sc], label[sc])
        used.append(fid)
        if verbose and (i + 1) % 200 == 0:
            print("   %d/%d frames" % (i + 1, len(ids)), file=sys.stderr)
    geo['gt_where'], geo['fn_where'] = gt_where, fn_where
    geo['region5'] = region5
    geo['cm_sc12'] = cm_sc12
    return cm_ssc, cm_sc, cover, frames, unscored, used, skipped, geo


def table(cm_ssc, cm_sc):
    """A markdown table of the per-class numbers, plus the two headline means."""
    m = metrics(cm_ssc)
    sc = metrics(cm_sc)
    rows = []
    rows.append("| class | TP | FP | FN | precision | recall | IoU |")
    rows.append("| --- | ---: | ---: | ---: | ---: | ---: | ---: |")
    for c in range(N_CLASSES):
        rows.append("| `%s`%s | %s | %s | %s | %s | %s | %s |"
                    % (CLASSES[c], " *(not in mIoU)*" if c == 0 else "",
                       "{:,}".format(int(m['tp'][c])),
                       "{:,}".format(int(m['fp'][c])),
                       "{:,}".format(int(m['fn'][c])),
                       _pct(m['precision'][c]), _pct(m['recall'][c]),
                       _pct(m['iou'][c])))
    miou = np.nanmean(m['iou'][1:]) * 100
    rows.append("")
    rows.append("**SSC mIoU (classes 1-11): %.1f**  |  "
                "mean precision %.1f  |  mean recall %.1f"
                % (miou, np.nanmean(m['precision'][1:]) * 100,
                   np.nanmean(m['recall'][1:]) * 100))
    rows.append("")
    rows.append("**SC (binary, unobserved region only): IoU %.1f  |  "
                "precision %.1f  |  recall %.1f**"
                % (sc['iou'][1] * 100, sc['precision'][1] * 100,
                   sc['recall'][1] * 100))
    return "\n".join(rows), miou, sc['iou'][1] * 100


# The ONLY numbers in this file not computed from the data: what the CleanerS
# paper publishes for NYU test (arXiv 2303.09977, Table 1, row "CleanerS").
# Used solely to say whether this run reproduces them. The paper gives no
# missed / invented split -- that part of the report exists only here.
PAPER_SC_TEST = dict(precision=88.0, recall=83.5, iou=75.0)


def sc_breakdown(cm_ssc, cm_sc, cm_sc12, geo, split, pred_dir):
    """What the one SC IoU number is made of, step by step.

    SC IoU = TP / (TP + FP + FN) over a filtered set of voxels, so "75%" is a
    share of the UNION -- voxels occupied in the GT, the prediction or both --
    not of all voxels. This prints the filter, the 2x2, the arithmetic, what
    the remaining share of the union is, and which classes make it up.
    Every count comes from matrices evaluate() already built.
    """
    total, in_lw = geo['lw_total'], geo['inside']
    n_keep, n_sc = int(cm_ssc.sum()), int(cm_sc.sum())
    tn, fp = int(cm_sc[0, 0]), int(cm_sc[0, 1])     # rows GT: 0 empty, 1 occ
    fn, tp = int(cm_sc[1, 0]), int(cm_sc[1, 1])
    union = tp + fp + fn
    f = lambda v: "{:,}".format(int(v))
    pct = lambda v, d: 100.0 * v / max(d, 1)

    missed = cm_sc12[1:, 0]          # GT class c, predicted empty
    invented = cm_sc12[0, 1:]        # GT empty, predicted class c
    assert (missed.sum(), invented.sum()) == (fn, fp)
    top = lambda arr: ", ".join(
        "`%s` %.0f%%" % (CLASSES[1 + k], pct(arr[k], arr.sum()))
        for k in np.argsort(-arr)[:5] if arr[k] > 0)

    paper = ""
    if split == 'test':
        ours = dict(precision=pct(tp, tp + fp), recall=pct(tp, tp + fn),
                    iou=pct(tp, union))
        same = all(round(ours[k], 1) == PAPER_SC_TEST[k] for k in ours)
        paper = (" For reference, the paper publishes only three SC numbers "
                 "for NYU test — precision %.1f, recall %.1f, IoU %.1f "
                 "(Table 1) — and %s. The missed / invented split below is "
                 "not in the paper at all."
                 % (PAPER_SC_TEST['precision'], PAPER_SC_TEST['recall'],
                    PAPER_SC_TEST['iou'],
                    "this run reproduces all three exactly" if same else
                    "this run gets %.1f / %.1f / %.1f" % (
                        ours['precision'], ours['recall'], ours['iou'])))

    rows = ["| class | missed (GT this class, predicted `empty`) | % of all "
            "missed | invented (predicted this class, GT `empty`) | % of all "
            "invented |",
            "| --- | ---: | ---: | ---: | ---: |"]
    for k in range(N_CLASSES - 1):
        rows.append("| `%s` | %s | %.1f%% | %s | %.1f%% |"
                    % (CLASSES[1 + k], f(missed[k]), pct(missed[k], fn),
                       f(invented[k]), pct(invented[k], fp)))

    return "\n".join([
        "### What the SC IoU is made of", "",
        "**Every number in this section is computed from this run's own "
        "predictions** — `%s`, %s `%s` frames — against the NYU labels. None "
        "of it is copied from the paper.%s" % (pred_dir, f(total // int(np.prod(GRID))), split, paper), "",
        "SC IoU of **%.1f%%** does **not** mean %.1f%% of voxels are right "
        "and the rest wrong. It is a share of one particular set, and empty "
        "voxels correctly left empty are not in that set. Step by step:"
        % (pct(tp, union), pct(tp, union)), "",
        "**1. Which voxels are scored.**", "",
        "| filter | voxels left | % of all |",
        "| --- | ---: | ---: |",
        "| every voxel (%s frames × 60 × 36 × 60) | %s | 100%% |"
        % (f(total // int(np.prod(GRID))), f(total)),
        "| `label_weight` true (GT object, or hidden behind a surface) | %s "
        "| %.1f%% |" % (f(in_lw), pct(in_lw, total)),
        "| and GT ≠ `255` (annotated) — the SSC set | %s | %.1f%% |"
        % (f(n_keep), pct(n_keep, total)),
        "| and **no depth pixel landed in it** — the SC set | **%s** | "
        "**%.1f%%** |" % (f(n_sc), pct(n_sc, total)), "",
        "The last filter drops the surfaces the camera saw directly. SC only "
        "scores what the model had to **infer**.", "",
        "**2. Collapse to occupied vs empty.** Occupied is any class 1–11, in "
        "the prediction and in the GT alike. Class names do not matter here: "
        "a chair predicted as a table counts as correct.", "",
        "| | GT occupied | GT `empty` |",
        "| --- | ---: | ---: |",
        "| **predicted occupied** | TP **%s** | FP **%s** |" % (f(tp), f(fp)),
        "| **predicted `empty`** | FN **%s** | TN **%s** |" % (f(fn), f(tn)),
        "",
        "Counts are pooled over all frames, not averaged per frame.", "",
        "**3. The IoU.**", "",
        "```",
        "IoU = TP / (TP + FP + FN) = %s / %s = %.1f%%"
        % (f(tp), f(union), pct(tp, union)),
        "```", "",
        "The denominator is the **union**: every voxel occupied in the GT, in "
        "the prediction, or both. TN — the %s empty voxels correctly left "
        "empty — does not appear in it at all." % f(tn), "",
        "**4. So the other %.1f%% is** the part of the union where the two "
        "disagree:" % (100 - pct(tp, union)), "",
        "| share of the union | what it is | voxels |",
        "| ---: | --- | ---: |",
        "| **%.1f%%** | both say occupied (TP) | %s |" % (pct(tp, union), f(tp)),
        "| **%.1f%%** | **missed** — GT occupied, predicted `empty` (FN) | %s |"
        % (pct(fn, union), f(fn)),
        "| **%.1f%%** | **invented** — predicted occupied, GT `empty` (FP) | "
        "%s |" % (pct(fp, union), f(fp)), "",
        "- **Missed**, by GT class: %s." % top(missed),
        "- **Invented**, by the class the model gave: %s." % top(invented),
        "- **The same errors over the whole SC set**, TN included: "
        "(%s + %s) / %s = **%.1f%% of SC voxels wrong**. IoU is stricter "
        "than that accuracy-style figure because it leaves out the easy "
        "empty-and-correct voxels."
        % (f(fp), f(fn), f(n_sc), pct(fp + fn, n_sc)),
        "- **Precision %.1f%%** — of what the model filled, how much was "
        "really occupied: `TP / (TP + FP)`. **Recall %.1f%%** — of what was "
        "really occupied, how much it filled: `TP / (TP + FN)`."
        % (pct(tp, tp + fp), pct(tp, tp + fn)), "",
        "Every class's share of the missed and invented voxels:", "",
        "\n".join(rows),
    ])


SC_COLUMNS = """\
**The region.** Every voxel below is *scored* (`label_weight` true, GT not
`255`) **and** has no depth pixel in it (`mapping == 307200`). Call a class
`c`; "predicted empty" means the model output class 0.

**What each column means:**

| column | what it counts |
| --- | --- |
| **unseen GT voxels** | voxels in the region whose GT is `c` |
| **% of the class unseen** | *unseen GT voxels* ÷ all scored GT voxels of `c`, seen or not. How much of the class the model has to complete rather than read off the input |
| **IoU, all scored** | the per-class IoU from the metrics table above, over every scored voxel whether seen or not. Copied here for comparison |
| **SC IoU, label must match** | the ordinary IoU inside the region. TP = GT `c`, predicted `c`. FP = predicted `c`, GT anything else (including `empty`). FN = GT `c`, predicted anything else (including `empty`). `TP / (TP + FP + FN)` |
| **SC IoU, any class counts** | occupancy IoU inside the region — did the model fill it, whatever it called it. TP = GT `c`, predicted **any** non-empty class. FP = predicted `c`, GT `empty`. FN = GT `c`, predicted `empty`. A voxel predicted `c` whose GT is *another* class is not an error here: occupied was right |
| **unseen, filled as another class** | GT `c`, predicted a different non-empty class, ÷ *unseen GT voxels*. Filled, but misnamed |
| **unseen, left empty** | GT `c`, predicted `empty`, ÷ *unseen GT voxels*. Not filled at all |

The last two columns and the correctly labelled share add up to 100% of the
class's unseen voxels, so the correct share is 100% minus both. They describe
misses only — false positives appear in the two IoU columns, not here.

The gap between *label must match* and *any class counts* comes from exactly
two kinds of voxel, both counted against the first and neither against the
second: the class's own unseen voxels filled as another class (the
*filled as another class* column), and other classes' unseen voxels predicted
as `c`. Both are volume the model completed but misnamed.

**The two bottom rows.** *mean, classes 1-11* is the plain average of the
eleven rows, every class weighted equally — the same way SSC mIoU is taken.
*pooled* adds up TP, FP and FN over all eleven classes **before** dividing, so
large classes weigh more. For *any class counts* that pooled sum is exactly the
binary SC matrix — the headline **SC IoU** printed under the metrics table —
which the script asserts on every run. Mean and pooled differ only because of
that weighting."""


def sc_class_rows(cm_ssc, cm_sc12, cm_sc):
    """SC broken down per class, two ways, over the SC region only.

    label must match   the ordinary per-class IoU, computed on cm_sc12 -- the
                       unseen part scored exactly as SSC scores everything
    any class counts   occupancy IoU: a GT voxel of class c is found if it is
                       predicted as ANY non-empty class; FP are voxels
                       predicted c where the GT is empty. Summed over the
                       eleven classes these are exactly the TP / FN / FP of
                       the binary SC matrix, so pooled it IS the headline SC
                       IoU -- asserted below, not assumed.

    The two last columns split each class's unseen voxels that were not given
    their own label: filled with another class, or left empty.
    Returns (markdown, mean label IoU, mean occupancy IoU).
    """
    m_ssc, m_lab = metrics(cm_ssc), metrics(cm_sc12)
    tp_occ = cm_sc12[:, 1:].sum(axis=1)          # GT c, predicted any class
    fn_occ = cm_sc12[:, 0]                       # GT c, predicted empty
    fp_occ = cm_sc12[0, :]                       # GT empty, predicted c
    assert (tp_occ[1:].sum(), fn_occ[1:].sum(), fp_occ[1:].sum()) == \
        (cm_sc[1, 1], cm_sc[1, 0], cm_sc[0, 1]), "per-class SC does not " \
        "add up to the binary SC matrix"
    with np.errstate(divide='ignore', invalid='ignore'):
        occ = np.where(tp_occ + fn_occ + fp_occ > 0,
                       tp_occ / (tp_occ + fn_occ + fp_occ), np.nan)

    rows = ["| class | unseen GT voxels | % of the class unseen | IoU, all "
            "scored | SC IoU, label must match | SC IoU, any class counts | "
            "unseen, filled as another class | unseen, left empty |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for c in range(1, N_CLASSES):
        gt_sc = cm_sc12[c].sum()
        other = gt_sc - cm_sc12[c, c] - cm_sc12[c, 0]
        rows.append("| `%s` | %s | %.0f%% | %s | %s | %s | %.0f%% | %.0f%% |"
                    % (CLASSES[c], "{:,}".format(int(gt_sc)),
                       100.0 * gt_sc / max(cm_ssc[c].sum(), 1),
                       _pct(m_ssc['iou'][c]), _pct(m_lab['iou'][c]),
                       _pct(occ[c]), 100.0 * other / max(gt_sc, 1),
                       100.0 * cm_sc12[c, 0] / max(gt_sc, 1)))
    lab_mean = np.nanmean(m_lab['iou'][1:]) * 100
    occ_mean = np.nanmean(occ[1:]) * 100
    pooled = cm_sc[1, 1] / float(cm_sc[1, 1] + cm_sc[1, 0] + cm_sc[0, 1])
    rows.append("| **mean, classes 1-11** | | | **%.1f%%** | **%.1f%%** | "
                "**%.1f%%** | | |" % (np.nanmean(m_ssc['iou'][1:]) * 100,
                                     lab_mean, occ_mean))
    rows.append("| **pooled** — the headline SC IoU | | | | | **%.1f%%** | "
                "| |" % (100 * pooled))
    return "\n".join(rows), lab_mean, occ_mean


def presence_rows(frames, n_frames):
    """How often each class is asserted at all, predicted against annotated.

    Voxel counts can hide this: a class predicted in every single frame while
    the ground truth carries it in a tenth of them is applying a prior, not
    reading the scene, however few voxels it spends on it.
    """
    rows = ["| class | frames predicted | frames in GT | ratio |",
            "| --- | ---: | ---: | ---: |"]
    order = np.argsort(-(frames[:, 0] / np.maximum(frames[:, 1], 1)))
    for c in order:
        if c == 0 or frames[c].sum() == 0:
            continue
        p, g = int(frames[c, 0]), int(frames[c, 1])
        rows.append("| `%s` | %d / %d | %d / %d | %s |"
                    % (CLASSES[c], p, n_frames, g, n_frames,
                       "%.1fx" % (p / g) if g else "--"))
    return "\n".join(rows)


def coverage_rows(cm, cover, unscored):
    """How much of each class's prediction the benchmark actually scores.

    Three fates for a predicted voxel: outside label_weight (never looked at),
    inside it but GT==255 (unannotated -- test_NYU.py drops these), or scored.
    A class with a small scored share is one whose behaviour the mIoU barely
    constrains, however good its IoU looks.

    The last column is the one that separates "unverified" from "wrong": a
    voxel predicted over GT `empty` is the ground truth actively DISAGREEING,
    which is a different thing from it having no opinion.
    """
    tp = np.diag(cm).astype(np.float64)
    over_empty = cm[0, :].astype(np.float64).copy()
    over_empty[0] = 0.0
    rows = ["| class | predicted | scored | scored share | GT `empty`, scored "
            "(counts as FP) | GT `empty`, unscored (free) | of those, camera "
            "measured free space |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for c in range(1, N_CLASSES):
        tot = cover[c].sum()
        if tot == 0:
            continue
        rows.append("| `%s` | %s | %s | %.0f%% | %s | %s | %s |"
                    % (CLASSES[c], "{:,}".format(int(tot)),
                       "{:,}".format(int(cover[c, 2])),
                       100.0 * cover[c, 2] / tot,
                       "{:,}".format(int(over_empty[c])),
                       "{:,}".format(int(unscored[c, 0])),
                       "{:,}".format(int(unscored[c, 1]))))
    return "\n".join(rows)


def spill_rows(cm):
    """Of each class's false positives, how many are over EMPTY space?

    This separates two very different errors that precision alone lumps
    together: calling a chair a sofa (a labelling mistake, the geometry was
    right) versus putting a sofa where there is nothing at all (inventing
    volume). A class high in this table is one that grows beyond the object --
    the "bed sticking through the wall" failure.
    """
    tp = np.diag(cm).astype(np.float64)
    fp = cm.sum(axis=0) - tp
    over_empty = cm[0, :].astype(np.float64).copy()   # GT empty, predicted as c
    over_empty[0] = 0.0
    rows = ["| class | FP total | FP over `empty` | % of its FP on `empty` | "
            "FP over another class |",
            "| --- | ---: | ---: | ---: | ---: |"]
    order = np.argsort(-np.where(fp > 0, over_empty / np.maximum(fp, 1), 0))
    for c in order:
        if c == 0 or fp[c] == 0:
            continue
        rows.append("| `%s` | %s | %s | **%.0f%%** | %s |"
                    % (CLASSES[c], "{:,}".format(int(fp[c])),
                       "{:,}".format(int(over_empty[c])),
                       100 * over_empty[c] / fp[c],
                       "{:,}".format(int(fp[c] - over_empty[c]))))
    return "\n".join(rows)


def _pct(v):
    return "--" if not np.isfinite(v) else "%.1f%%" % (100 * v)


def _partners(cm, c, axis, total, n=3):
    """Top classes on the other side of a confusion, as 'name pct%' strings."""
    if axis == 'fp':                      # predicted c, was actually t
        pairs = ((cm[t, c], CLASSES[t]) for t in range(N_CLASSES) if t != c)
    else:                                 # was c, predicted as p
        pairs = ((cm[c, p], CLASSES[p]) for p in range(N_CLASSES) if p != c)
    top = sorted((v for v in pairs if v[0] > 0), reverse=True)[:n]
    return ", ".join("`%s` %.0f%%" % (nm, 100.0 * v / max(total, 1))
                     for v, nm in top) or "—"


def fp_breakdown(cm):
    """Of everything the model calls X, how much is wrong -- and what was it?

    Denominator is the SCORED predictions (TP + FP), the only ones the ground
    truth can adjudicate. `FP share` is therefore 1 - precision, restated so it
    reads as "how much of what it claims is wrong".
    """
    tp = np.diag(cm).astype(float)
    fp = cm.sum(axis=0) - tp
    pred = cm.sum(axis=0)
    rows = ["| class | scored predictions | FP | % of its claims wrong | "
            "what those really were |",
            "| --- | ---: | ---: | ---: | --- |"]
    order = sorted(range(1, N_CLASSES),
                   key=lambda c: -(fp[c] / max(pred[c], 1)))
    for c in order:
        rows.append("| `%s` | %s | %s | **%.1f%%** | %s |"
                    % (CLASSES[c], "{:,}".format(int(pred[c])),
                       "{:,}".format(int(fp[c])),
                       100.0 * fp[c] / max(pred[c], 1),
                       _partners(cm, c, 'fp', fp[c])))
    return "\n".join(rows)


def fn_breakdown(cm):
    """The mirror: of everything that IS X, how much did the model miss, and
    what did it call it instead?"""
    tp = np.diag(cm).astype(float)
    fn = cm.sum(axis=1) - tp
    actual = cm.sum(axis=1)
    rows = ["| class | scored ground truth | FN | % of it missed | "
            "what it called them |",
            "| --- | ---: | ---: | ---: | --- |"]
    order = sorted(range(1, N_CLASSES),
                   key=lambda c: -(fn[c] / max(actual[c], 1)))
    for c in order:
        rows.append("| `%s` | %s | %s | **%.1f%%** | %s |"
                    % (CLASSES[c], "{:,}".format(int(actual[c])),
                       "{:,}".format(int(fn[c])),
                       100.0 * fn[c] / max(actual[c], 1),
                       _partners(cm, c, 'fn', fn[c])))
    return "\n".join(rows)


def biggest_confusions(cm, top=10):
    """The largest single (predicted -> actually was) pairs in the matrix.

    `empty` appears here as a predicted class even though it is not in the
    mIoU, because "the model put nothing where something is" is one of the
    largest error modes and hiding it would misrepresent the matrix.
    """
    tp = np.diag(cm).astype(float)
    tot = (cm.sum(axis=0) - tp)[1:].sum()
    pairs = sorted(((cm[t, p], CLASSES[p], CLASSES[t])
                    for p in range(N_CLASSES) for t in range(N_CLASSES)
                    if p != t), reverse=True)[:top]
    rows = ["| predicted | actually was | voxels | share of all class FP |",
            "| --- | --- | ---: | ---: |"]
    for v, pn, tn in pairs:
        rows.append("| `%s` | `%s` | %s | %.1f%% |"
                    % (pn, tn, "{:,}".format(int(v)), 100.0 * v / max(tot, 1)))
    return "\n".join(rows)


def label_weight_note(geos):
    """What label_weight is, with the identity re-verified on every frame.

    `geos` is the per-split geometry from evaluate(); the counts are summed
    over every frame this run evaluated.
    """
    g = {k: sum(x[k] for x in geos) for k in
         ('lw_match', 'lw_total', 'inside', 'never', 'free', 'other')}
    n_frames = sum(len(x['floor_gt']) for x in geos)   # only to name a size
    out = g['never'] + g['free'] + g['other']
    exact = g['lw_match'] == g['lw_total']
    other = ("" if g['other'] == 0 else
             "\n| just behind a surface | `< 0` but `>= -0.5` | negative like "
             "the occluded voxels, but above the -0.5 cut, so not in the mask "
             "(%s voxels) |" % "{:,}".format(g['other']))
    return """\
## What `label_weight` is

Everything above turns on it, so: `label_weight` is NYU's own scoring mask, one
bit per voxel, shipped in `data/NYU/TSDF/{{id}}.npz` as `arr_1`. Re-checked on
every test frame — {match} of {total} voxels agree{exact}:

```
label_weight  =  (GT is a class 1..11)  OR  (tsdf < -0.5)
```

Two ways in. Either the annotation says something is there, or the voxel is
**occluded** — behind a surface the depth camera saw, which is what a TSDF
below -0.5 means. Together that is "the scene, plus the space hidden behind
it": the part a completion model is supposed to reason about.

Everything else is **not scored at all**, and it splits into very different
things:

| `label_weight` false | `tsdf` | what it is |
| --- | --- | --- |
| never observed | `== 0` | outside the frustum, past the depth range, or a dropout — the camera has no information |
| **observed free space** | `> 0` | **in front** of a measured surface. The camera looked straight through it and confirmed nothing is there |{other}

Over those frames that is {never_pc:.0f}% and {free_pc:.0f}% of all voxels
respectively, against {inside_pc:.0f}% inside the mask. So **{out_pc:.0f}% of
the volume is outside the metric**, and {free_of_out:.0f}% of that is space the
sensor positively measured as empty.

This is why a class can look excellent in the table above and still be visibly
wrong on screen. `test_NYU.py` restricts further to `GT != 255`, dropping
unannotated voxels inside the mask as well.
""".format(match="{:,}".format(g['lw_match']),
           total="{:,}".format(g['lw_total']),
           exact=", exactly" if exact else
                 " — **not exactly: the identity below does not hold "
                 "everywhere in this data**",
           other=other,
           never_pc=100.0 * g['never'] / g['lw_total'],
           free_pc=100.0 * g['free'] / g['lw_total'],
           inside_pc=100.0 * g['inside'] / g['lw_total'],
           out_pc=100.0 * out / g['lw_total'],
           free_of_out=100.0 * g['free'] / max(out, 1))


TEMPLATE = """\
## Learned habit, or the data?

Every failure worth chasing comes down to one question: is this something the
model **does**, or something the data **never told it**? Two facts settle it.

**The model has already seen the answers for the train split.** A failure that
survives there cannot be blamed on unfamiliar scenes.

**The training loss uses the same mask as the metric**
(`examples/segmentation/train_utils.py:83-85`):

```python
weightSSC = label_weight & (label3d != cfg.ignore_index)
loss3d = criterion(pred_3d[weightSSC], label3d[weightSSC])
```

So anything outside `label_weight` was never penalised **in training either**.
Output out there is not the model getting something wrong — it is output nobody
ever asked about.

Put together: a mistake that happens **on train**, **inside the mask**, is the
model's. A mistake that only happens **outside the mask** is the data's, in the
sense that nothing ever told the model otherwise. A class fine on train but
poor on test simply **did not transfer**.

The verdicts below are judgements; every number in them is computed from the
tables at generation time.

### What the columns are

**IoU** — `TP / (TP + FP + FN)`, the headline per-class score.

**predicted / GT voxels** — how much of a class the model *outputs*, against
how much the ground truth *has*. `ceiling` on test: {te.pred[ceiling]:,} voxels
predicted against {te.gt[ceiling]:,} in the GT, so **{te.vox[ceiling]:.1f}x**.
Read it as "it paints {te.vox[ceiling]:.0f} times as much ceiling as there is".

One asymmetry to keep in mind, because it inflates this column: the numerator
counts every predicted voxel *anywhere in the volume*, while the denominator
can only count GT inside `label_weight` — there is no ground truth anywhere
else to count. So a class whose output is mostly unscored gets a big ratio
partly by construction. That is still the thing worth knowing — `ceiling` really
does emit {te.vox[ceiling]:.0f}x what anyone annotated — but it is not
{te.vox[ceiling]:.0f}x *too much* in a sense the benchmark could ever charge it
for.

**claimed in frames / GT frames** — how many of the {te.n} frames the model puts
the class in at all, against how many the GT annotates it in. `tvs`: predicted
in **{te.fpred[tvs]}** frames, annotated in **{te.fgt[tvs]}**, so
**{te.frm[tvs]:.1f}x**. This one is symmetric and has no caveat, and it is
about *frequency*, not size — a class can be claimed in every frame while being
tiny in each.

The two say different things. A high voxel ratio means "too much of it where it
does appear"; a high frame ratio means "it appears in rooms that do not have
one". `furn` leans to the first ({te.vox[furn]:.1f}x voxels,
{te.frm[furn]:.1f}x frames): it finds real furniture and then over-grows it.
`tvs` leans to the second ({te.vox[tvs]:.1f}x voxels, {te.frm[tvs]:.1f}x
frames): small, but expected almost everywhere.

**FP on `empty` / % of its FP on `empty`** — how much of a class's output
lands on voxels the ground truth explicitly labels `empty`, and what share of
that class's false positives that is. This is the column that separates
*inventing volume* from *misnaming real geometry*. Note the denominator is FP,
not predictions — it is **not** the *% of its claims wrong* column in the
per-split tables, which divides by predictions.

{rows}

### `floor` — the data, not the model

**What it looks like:** a sheet across the volume's bottom layer — on test
{x.floor_pred:.0f}% of that layer is predicted floor, on average, while the GT
annotates {x.floor_gt:.0f}%: the labelled room's footprint, with the model
continuing it out to the volume's edge. That is the floor running through your
wall.

**Train vs test:** {tr.vox[floor]:.1f}x over-production on train,
{te.vox[floor]:.1f}x on test; {tr.spill[floor]:.0f}% / {te.spill[floor]:.0f}%
of its mistakes put where the GT says nothing; claimed in {tr.frm[floor]:.1f}x
/ {te.frm[floor]:.1f}x the frames that have one; IoU {tr.iou[floor]:.1f} to
{te.iou[floor]:.1f}, a {x.gap[floor]:.1f}-point drop — {x.gap_nth_small[floor]}
of any class.

**Verdict — the data.** The sheet sits outside `label_weight`, so no gradient
ever touched it. The GT does say `empty` for {te.unsc_empty[floor]:,} of those
voxels and not one of them counts. The model is not getting the floor wrong; it
was never asked about that space. Most of it is never-observed too — only
{te.unsc_free[floor]:,} voxels sit where the camera measured free space — so it
is a plausible prior, rooms do have floor throughout, extended past the walls
with nothing to support it.

### `wall` — the model, plus a transfer problem

**What it looks like:** {x.pred_nth[wall]} output of any class —
{te.pred[wall]:,} voxels, {x.wall_vol:.0f}% of every volume,
{te.vox[wall]:.1f}x what the GT annotates, spread through the volume rather
than lining its shell.

**Train vs test:** over-production {tr.vox[wall]:.1f}x on train and
{te.vox[wall]:.1f}x on test, and **{tr.spill[wall]:.0f}% of its train
mistakes** put wall where the GT says nothing is. The loss charged it for both,
on frames it had already seen.

**Verdict — the model, plus transfer.** IoU falls {tr.iou[wall]:.1f} to
{te.iou[wall]:.1f} on top. It fails in both directions at once: its false
positives go to {x.fp_partners[wall]}, while {te.fnto[wall>empty]:.0f}% of the
real wall it misses is called `empty` — {te.cm[wall>empty]:,} voxels. It
over-produces wall {te.vox[wall]:.0f}x and still misses {te.miss[wall]:.0f}% of
the wall that is there. Wrong places, not merely too much of it.

### `ceiling` — the model, and it barely transfers

**What it looks like:** a ceiling in {te.fpred[ceiling]} of {te.n} frames while
the GT annotates one in {te.fgt[ceiling]}. The height it picks is right — its
ceiling layer sits {x.ce_pred_lo:.2f}–{x.ce_pred_hi:.2f} m above the floor
layer in the middle half of frames, against {x.ce_gt_lo:.2f}–{x.ce_gt_hi:.2f} m
for the GT's ceilings — so it is a plausible ceiling in a room that has none.
Unlike the floor it is not a sheet; it fills {x.ce_fill:.0f}% of its layer on
average.

**Train vs test:** claimed in {tr.frm[ceiling]:.1f}x the frames that have one on
train against {te.frm[ceiling]:.1f}x on test. Learned, not provoked by new
scenes.

**Verdict — the model, and it barely transfers.** IoU collapses
{tr.iou[ceiling]:.1f} to {te.iou[ceiling]:.1f}. Only {te.scored[ceiling]:.1f}%
of it is scored — a camera looking horizontally never sees a ceiling, so the
TSDF marks it unobserved and the benchmark registers almost none of this.

### `window` — a boundary problem with `wall`, and it doesn't transfer

**What it looks like:** not really its own failure. **{te.fnto[window>wall]:.0f}%**
of the window it misses is called `wall`, and **{te.fpto[window>wall]:.0f}%** of
what it wrongly paints window over is `wall`. A window is a hole in a wall, and
the model cannot place the edge.

**Train vs test:** claimed in {tr.frm[window]:.1f}x the frames that have one on
train, {te.frm[window]:.1f}x on test. Only {tr.spill[window]:.0f}% of its train
mistakes put window where the GT says nothing — low, so it is not inventing
volume the way `bed` and `sofa` do.

**Verdict — mostly transfer.** IoU {tr.iou[window]:.1f} to {te.iou[window]:.1f},
a {x.gap[window]:.0f}-point drop. It has the window/wall boundary roughly right
on frames it memorised and loses it on new ones.

### `bed` — the model, most clearly of any class

**What it looks like:** the bed poking through a wall. {te.spill[bed]:.0f}% of
its false positives sit on an explicit GT `empty` — volume invented where the
annotation says nothing is, not confusion with a sofa or a table.

**Train vs test:** **{tr.spill[bed]:.0f}% on train** ({x.spill_tr_standing[bed]}),
and the absolute count barely moves between splits ({tr.cm[empty>bed]:,} against
{te.cm[empty>bed]:,}). It claims a bed in {tr.frm[bed]:.1f}x the frames that
have one on train, {te.frm[bed]:.1f}x on test.

**Verdict — the model.** Unlike the floor this happens **inside** the mask, so
it was penalised in training and the model does it anyway, on the very frames
it trained on. The {x.gap[bed]:.1f}-point IoU drop is an extra problem, not the
explanation. Same symptom as the floor on screen, opposite cause: the bed's
spill is inside the scored region so it costs precision, the floor's is outside
so it is free.

### `sofa` — the model, the same story as `bed`

**What it looks like:** **{tr.spill[sofa]:.0f}%** of its train mistakes put sofa
where the GT says nothing is — {x.spill_tr_standing[sofa]}. It also claims a
sofa in {te.frm[sofa]:.1f}x the frames that have one ({te.fpred[sofa]}
predicted against {te.fgt[sofa]} annotated).

**Train vs test:** the habit is on both splits ({tr.spill[sofa]:.0f}% train,
{te.spill[sofa]:.0f}% test) and the frame ratio goes {tr.frm[sofa]:.1f}x to
{te.frm[sofa]:.1f}x.

**Verdict — the model.** Inside the mask, penalised in training, done anyway.
IoU {tr.iou[sofa]:.1f} to {te.iou[sofa]:.1f} on top. Read it alongside `bed`:
the two large furniture classes share a failure, growing past the object into
empty space.

### `chair` — the model, but its confusion is with other furniture

**What it looks like:** {tr.spill[chair]:.0f}% of its train mistakes put chair
in empty space, and it over-produces {te.vox[chair]:.1f}x the GT's voxels. Where
it is not inventing volume it is confusing chair with `table`
({te.fpto[chair>table]:.0f}% of its false positives) and `furn`
({te.fpto[chair>furn]:.0f}%).

**Train vs test:** frame ratio {tr.frm[chair]:.1f}x to {te.frm[chair]:.1f}x. The
empty-space share goes {tr.spill[chair]:.0f}% to {te.spill[chair]:.0f}% on test,
but only because its other errors grow.

**Verdict — the model, plus a large transfer loss.** IoU {tr.iou[chair]:.1f} to
{te.iou[chair]:.1f}. Only {te.scored[chair]:.1f}% of its output is scored at
all, {x.scored_nth_low[chair]} of any class.

### `table` — the model, and the worst `furn` confusion

**What it looks like:** {tr.spill[table]:.0f}% of its train mistakes are in
empty space, and its single largest error is with `furn` —
{te.fpto[table>furn]:.0f}% of its false positives are painted over furniture,
{te.fnto[table>furn]:.0f}% of its misses called furniture.

**Train vs test:** {tr.vox[table]:.1f}x to {te.vox[table]:.1f}x voxels,
{tr.frm[table]:.1f}x to {te.frm[table]:.1f}x frames, spill {tr.spill[table]:.0f}%
to {te.spill[table]:.0f}%.

**Verdict — the model, plus transfer.** IoU {tr.iou[table]:.1f} to
{te.iou[table]:.1f}, a {x.gap[table]:.0f}-point drop, and the table/furniture
boundary is where it goes.

### `tvs` — a prior, and the most over-claimed class

**What it looks like:** {x.frm_nth[tvs]} frame ratio in the set. Predicted in
**{te.fpred[tvs]} of {te.n}** frames while the GT annotates one in
**{te.fgt[tvs]}** — {te.frm[tvs]:.1f}x on test, {tr.frm[tvs]:.1f}x on train. It
is also {x.gt_nth_small[tvs]} class by annotated volume, so this costs little
in the mIoU.

**Train vs test:** the over-claiming is on both splits. Unlike the furniture
classes it barely invents volume ({tr.spill[tvs]:.0f}% of train mistakes in
empty space); its errors go to `furn` ({te.fpto[tvs>furn]:.0f}%) and `objs`
({te.fpto[tvs>objs]:.0f}%).

**Verdict — a prior, plus transfer.** The model expects a television in most
rooms. IoU {tr.iou[tvs]:.1f} to {te.iou[tvs]:.1f}, {x.gap_nth[tvs]} drop.

### `furn` — the model, and a large transfer loss

**What it looks like:** {tr.spill[furn]:.0f}% of its train mistakes put
furniture in empty space — {x.spill_tr_standing[furn]} — and it trades heavily
with `objs` in both directions ({te.fpto[furn>objs]:.0f}% of its false positives
over `objs`, {te.fnto[furn>objs]:.0f}% of its misses called `objs`).

**Train vs test:** frame ratio {tr.frm[furn]:.1f}x to {te.frm[furn]:.1f}x, so it
reads the scene rather than assuming furniture. The empty-space habit is on
both splits.

**Verdict — the model, plus transfer.** IoU {tr.iou[furn]:.1f} to
{te.iou[furn]:.1f}, **{x.gap[furn]:.1f} points**, {x.gap_nth[furn]} drop of any
class. With `objs` at {x.gap[objs]:.1f}, the two catch-all classes are where
much of the headline mIoU drop comes from.

### `objs` — mostly just doesn't transfer

**What it looks like:** the catch-all class, trading with `furn` and `wall` in
both directions. {x.iou_nth_low[objs]} test IoU of the eleven,
precision {te.prec[objs]:.1f}%, recall {te.rec[objs]:.1f}%.

**Train vs test:** claimed in {tr.frm[objs]:.1f}x / {te.frm[objs]:.1f}x the
frames that have it — it reads the scene and asserts nothing it should not. Its
IoU simply halves, {tr.iou[objs]:.1f} to {te.iou[objs]:.1f}, {x.gap_nth[objs]}
drop of any class.

**Verdict — transfer.** It labels the catch-all class well on frames it
memorised and badly on new ones. Not a habit. It {x.free_lead[objs]} the
free-space count — {te.unsc_free[objs]:,} voxels, {x.objs_free_pc:.0f}% of
everything it predicts, in space the depth frame measured as *free* and the GT
calls empty — but that lands outside the mask, so like the floor's sheet it is
unasked-for rather than penalised.

### In one line each

| class | verdict |
| --- | --- |
| `floor` | **the data** — the same on train and test, and it happens where training never looked |
| `wall` | **the model**, plus transfer — invents wall on train, inside the mask |
| `ceiling` | **the model** — a ceiling in most rooms, same on train; and it barely transfers |
| `window` | **transfer** — a window/wall boundary it holds on train and loses on test |
| `bed` | **the model** — the clearest case; penalised in training, does it anyway |
| `sofa` | **the model** — the same habit as `bed`, {tr.spill[sofa]:.0f}% of its train mistakes in empty space |
| `chair` | **the model**, plus transfer — invents volume, and confuses chair with table and furn |
| `table` | **the model**, plus transfer — {x.gap[table]:.0f}-point drop, mostly into `furn` |
| `tvs` | **a prior** — a television in {te.fpred[tvs]} of {te.n} frames, the GT has {te.fgt[tvs]} |
| `furn` | **the model**, plus a {x.gap[furn]:.1f}-point transfer loss |
| `objs` | **transfer** — reads the scene fine, labels it badly on unseen frames |

Two things fall out of the set as a whole.

**{x.n_spill} of the eleven put at least {x.spill_bar:.0f}% of their train
mistakes in empty space** — {x.spill_list}. Only {x.clear_list} stay below it.
Inside the mask, penalised, done anyway. Growing objects past their real extent
is the model's characteristic failure, not a quirk of one class.

**{x.n_gap} of the eleven lose more than {x.gap_bar:.0f} IoU points from train
to test**, {x.gap_exceptions}. Whatever else is wrong per class, failure to
transfer is the common factor, and the headline mIoU falling
{x.miou_tr:.1f} to {x.miou_te:.1f} is that in aggregate.
"""


WATCH = ['floor', 'wall', 'ceiling', 'window', 'bed', 'sofa',
         'chair', 'table', 'tvs', 'furn', 'objs']

SPILL_BAR = 25.0    # % of train FP on `empty` counted as "a meaningful share"
GAP_BAR = 15.0      # IoU points counted as "lost in transfer"

_ORDINAL = ['', 'second', 'third', 'fourth', 'fifth', 'sixth', 'seventh',
            'eighth', 'ninth', 'tenth', 'eleventh']
_WORD = ['no', 'one', 'two', 'three', 'four', 'five', 'six', 'seven',
         'eight', 'nine', 'ten', 'eleven']


class _Ns(object):
    """Attribute bag, so TEMPLATE can say {te.iou[wall]} or {x.gap[bed]}."""

    def __init__(self, **kw):
        self.__dict__.update(kw)


def _order(values, largest=True):
    """Class names 1..11, best first by `values` (a name -> number dict)."""
    names = [CLASSES[c] for c in range(1, N_CLASSES)]
    return sorted(names, key=lambda n: -values[n] if largest else values[n])


def _nth(values, word, largest=True):
    """name -> 'the largest' / 'the second-largest' / ... among the eleven."""
    order = _order(values, largest)
    return {n: 'the ' + (_ORDINAL[k] + '-' if k else '') + word
            for k, n in enumerate(order)}


def _standing(values, word):
    """name -> 'the highest of any class' / 'second only to `bed`' /
    'third, behind `bed` and `sofa`'."""
    order = _order(values)
    out = {}
    for k, n in enumerate(order):
        ahead = ["`%s`" % a for a in order[:k]]
        if k == 0:
            out[n] = 'the %s of any class' % word
        elif k == 1:
            out[n] = 'second only to %s' % ahead[0]
        else:
            out[n] = '%s, behind %s and %s' % (
                _ORDINAL[k], ", ".join(ahead[:-1]), ahead[-1])
    return out


def _split_stats(s):
    """Per-class numbers for one split, as name-keyed dicts."""
    cm, cover, frames, unscored, n, _geo = s
    m = metrics(cm)
    gt = cm.sum(axis=1)
    by = lambda arr: {CLASSES[c]: arr[c] for c in range(N_CLASSES)}
    pair = lambda f: {"%s>%s" % (CLASSES[a], CLASSES[b]): f(a, b)
                      for a in range(N_CLASSES) for b in range(N_CLASSES)}
    pred = cover.sum(axis=1)
    return _Ns(
        n=n,
        iou=by(100 * m['iou']), prec=by(100 * m['precision']),
        rec=by(100 * m['recall']),
        miss=by(100 * m['fn'] / np.maximum(gt, 1)),
        pred=by([int(v) for v in pred]), gt=by([int(v) for v in gt]),
        vox=by(pred / np.maximum(gt, 1)),
        fpred=by([int(v) for v in frames[:, 0]]),
        fgt=by([int(v) for v in frames[:, 1]]),
        frm=by(frames[:, 0] / np.maximum(frames[:, 1], 1)),
        spill=by(100 * cm[0, :] / np.maximum(m['fp'], 1)),
        scored=by(100 * cover[:, 2] / np.maximum(pred, 1)),
        unsc_empty=by([int(v) for v in unscored[:, 0]]),
        unsc_free=by([int(v) for v in unscored[:, 1]]),
        # "a>b" keys. cm: GT a, predicted b.  fpto: of a's FP, % that were b.
        # fnto: of a's FN, % called b.
        cm=pair(lambda a, b: int(cm[a, b])),
        fpto=pair(lambda a, b: 100.0 * cm[b, a] / max(m['fp'][a], 1)),
        fnto=pair(lambda a, b: 100.0 * cm[a, b] / max(m['fn'][a], 1)),
        miou=100 * np.nanmean(m['iou'][1:]))


def learned_vs_generalisation(tr, te):
    """Train against test, per class, in plain terms.

    The question is only ever: is a failure something the model DOES, or
    something the data never told it? Two facts settle it.

    First, the model has already been shown the answers for the train split.
    A failure that survives there is not explained by unfamiliar scenes.

    Second, the TRAINING loss uses the same mask as the metric
    (examples/segmentation/train_utils.py:83-85):

        weightSSC = label_weight & (label3d != ignore_index)
        loss3d    = criterion(pred_3d[weightSSC], label3d[weightSSC])

    so anything outside label_weight was never penalised in training either.
    Output there is not the model failing -- it is output nobody asked about.

    Every number in TEMPLATE is a lookup into the two _split_stats below or
    into `x`; ranks ("the largest", "second only to") are computed too, so a
    rerun on new predictions cannot leave the prose contradicting the tables.
    """
    T, E = _split_stats(tr), _split_stats(te)
    cm_te, cover_te, geo = te[0], te[1], te[5]
    names = [CLASSES[c] for c in range(1, N_CLASSES)]

    rows = ["| class | split | IoU | predicted / GT voxels | claimed in frames "
            "/ GT frames | FP on `empty` | % of its FP on `empty` |",
            "| --- | --- | ---: | ---: | ---: | ---: | ---: |"]
    for name in WATCH:
        for split, s in (('train', T), ('test', E)):
            rows.append("| {} | `{}` | {:.1f} | {:.1f}x | {:.1f}x | {:,} | "
                        "{:.0f}% |".format(
                            "`%s`" % name if split == 'train' else "", split,
                            s.iou[name], s.vox[name], s.frm[name],
                            s.cm['empty>' + name], s.spill[name]))

    gap = {n: T.iou[n] - E.iou[n] for n in names}
    spilly = [n for n in _order(T.spill) if T.spill[n] >= SPILL_BAR]
    clear = [n for n in _order(T.spill, largest=False)
             if T.spill[n] < SPILL_BAR]
    lost = [n for n in names if gap[n] > GAP_BAR]
    kept = [n for n in _order(gap, largest=False) if gap[n] <= GAP_BAR]
    q = lambda a, p: np.percentile(a, p) * VOX_M if len(a) else float('nan')
    free = {n: E.unsc_free[n] for n in names}

    x = _Ns(
        gap=gap,
        gap_nth=_nth(gap, 'largest'),
        gap_nth_small=_nth(gap, 'smallest', largest=False),
        pred_nth=_nth(E.pred, 'largest'),
        frm_nth=_nth(E.frm, 'highest'),
        gt_nth_small=_nth(E.gt, 'smallest', largest=False),
        iou_nth_low={n: v[0].upper() + v[1:] for n, v in
                     _nth(E.iou, 'lowest', largest=False).items()},
        scored_nth_low=_nth(E.scored, 'lowest', largest=False),
        spill_tr_standing=_standing(T.spill, 'highest'),
        free_lead={n: 'leads' if _order(free)[0] == n else
                   'is %s in' % _nth(free, 'largest')[n] for n in names},
        fp_partners={n: _partners(cm_te, CLASSES.index(n), 'fp',
                                  cm_te.sum(axis=0)[CLASSES.index(n)]
                                  - cm_te[CLASSES.index(n), CLASSES.index(n)])
                     for n in names},
        wall_vol=100.0 * E.pred['wall'] / (E.n * np.prod(GRID)),
        objs_free_pc=100.0 * E.unsc_free['objs'] / max(E.pred['objs'], 1),
        floor_pred=100 * np.mean(geo['floor_pred']),
        floor_gt=100 * np.mean(geo['floor_gt']),
        ce_pred_lo=q(geo['ceil_pred_layer'], 25),
        ce_pred_hi=q(geo['ceil_pred_layer'], 75),
        ce_gt_lo=q(geo['ceil_gt_layer'], 25),
        ce_gt_hi=q(geo['ceil_gt_layer'], 75),
        ce_fill=100 * np.mean(geo['ceil_pred_fill']),
        n_spill=_WORD[len(spilly)].capitalize(), spill_bar=SPILL_BAR,
        spill_list=", ".join("`%s` %.0f%%" % (n, T.spill[n]) for n in spilly),
        clear_list=", ".join("`%s` (%.0f%%)" % (n, T.spill[n]) for n in clear),
        n_gap=_WORD[len(lost)].capitalize(), gap_bar=GAP_BAR,
        gap_exceptions=(" and ".join("`%s` at %.1f" % (n, gap[n]) for n in kept)
                        + (" being the sole exception" if len(kept) == 1
                           else " the exceptions")) if kept else
                       "without exception",
        miou_tr=T.miou, miou_te=E.miou)
    return TEMPLATE.format(rows="\n".join(rows), tr=T, te=E, x=x)


GAP_FLAT = 5.0      # IoU points. Below this a class transfers intact.
MISS_FLAT = 10.0    # % of a class missed. Below this there is nothing to fix.
HABIT_BAR = 20.0    # % of hidden voxels missed ON TRAIN that counts as a habit


def _fp_group(gap, dest):
    """The false-positive group rule: 1 transfers intact, 2 its FP mostly
    land on `empty`, 3 on another class. See error_groups."""
    return 1 if gap < GAP_FLAT else (2 if dest == 0 else 3)


def error_groups(tr, te):
    """Group the eleven classes by WHY they fail, not by what they are.

    Everything above is per class. This asks the question one level up: the
    eleven failures are not eleven different problems, and a fix aimed at the
    wrong one is wasted. Three questions, asked in this order, each answered
    from the matrices already computed:

      1. Does the class transfer? train IoU - test IoU < GAP_FLAT means the
         model does on unseen frames what it does on trained ones, so inside
         the scored region there is nothing left to learn. Whatever is still
         wrong with it is wrong OUTSIDE that region -- where the metric never
         looks and, because the training loss shares the mask, where the loss
         never charged it either. Nothing ever checked it.

      2. Otherwise: where do most of its false positives land? On `empty`
         means it is putting volume where the annotation says there is none.
         It is growing objects past their extent, inside the mask, on frames
         it was trained on. That is the model.

      3. On another class means the geometry was roughly right and the name
         was wrong -- the classes are not separable as labelled.

    Order matters. Rule 1 goes first because a class can transfer perfectly
    and still have a largest-FP destination, which would misfile it under 2
    or 3 and point the fix at the wrong place.
    """
    cm_tr = tr[0]
    cm_te, cover_te, frames_te, unsc_te, n_te = te[:5]
    m_tr, m_te = metrics(cm_tr), metrics(cm_te)
    i = {n: k for k, n in enumerate(CLASSES)}
    fp_te = cm_te.sum(axis=0) - np.diag(cm_te)
    fp_tr = cm_tr.sum(axis=0) - np.diag(cm_tr)

    info = {}
    for name in WATCH:
        c = i[name]
        n_dest, dest = max((cm_te[t, c], t) for t in range(N_CLASSES) if t != c)
        gap = 100 * (m_tr['iou'][c] - m_te['iou'][c])
        info[name] = dict(
            c=c, iou=100 * m_te['iou'][c], gap=gap, fp=int(fp_te[c]),
            dest=CLASSES[dest], dest_share=100 * n_dest / max(fp_te[c], 1),
            empty=100 * cm_te[0, c] / max(fp_te[c], 1),
            empty_tr=100 * cm_tr[0, c] / max(fp_tr[c], 1),
            wrong=100 * (1 - m_te['precision'][c]),
            unscored=int(unsc_te[c, 0]),
            group=_fp_group(gap, dest))

    def members(g):
        return [n for n in sorted(WATCH, key=lambda n: -info[n]['iou'])
                if info[n]['group'] == g]

    g1, g2, g3 = members(1), members(2), members(3)
    TITLES = {1: "The data never checked it",
              2: "The model grows things into thin air",
              3: "The model cannot tell the classes apart"}

    rows = ["| class | group | test IoU | train − test | largest FP "
            "destination | share of its FP going there |",
            "| --- | ---: | ---: | ---: | --- | ---: |"]
    for g in (1, 2, 3):
        for n in members(g):
            d = info[n]
            rows.append("| `%s` | %d | %.1f | %.1f | `%s` | %.0f%% |"
                        % (n, g, d['iou'], d['gap'], d['dest'],
                           d['dest_share']))

    def lst(names, fmt):
        return ", ".join("`%s` %s" % (n, fmt % info[n]) for n in names)

    def bare(names, fmt):
        # same, minus the name prefix when there is only one class to name --
        # otherwise a one-member group reads "`floor` is ... -- `floor` 4.5%"
        if len(names) == 1:
            return fmt % info[names[0]]
        return lst(names, fmt)

    others = [n for n in WATCH if info[n]['group'] != 1]
    gaps = sorted(info[n]['gap'] for n in others)
    to_furn = [n for n in g3 if info[n]['dest'] == 'furn']
    worst_unscored = max(others, key=lambda n: info[n]['unscored'])
    over_half = [n for n in g2 if info[n]['empty'] > 50]
    g1_names = " and ".join("`%s`" % n for n in g1)
    is_are = "is" if len(g1) == 1 else "are"
    worse = [n for n in g2 if info[n]['empty_tr'] > info[n]['empty']]
    same = [n for n in g2 if n not in worse]
    if not same:
        worse_note = "larger on train than on test in every case"
    elif worse:
        # do not round this into "worse everywhere": it is not true of every
        # class, and the point stands without the overstatement.
        worse_note = ("larger on train than on test for %s; for %s it is "
                      "lower by %s and still %s"
                      % (", ".join("`%s`" % n for n in worse),
                         ", ".join("`%s`" % n for n in same),
                         " / ".join("%.0f point%s" % (
                             info[n]['empty'] - info[n]['empty_tr'],
                             "" if round(info[n]['empty'] -
                                         info[n]['empty_tr']) == 1 else "s")
                             for n in same),
                         " / ".join("%.0f%%" % info[n]['empty_tr']
                                    for n in same)))
    else:
        worse_note = "at much the level it sits at on test"

    parts = [
        "## Three kinds of error (false positives)",
        "",
        "The tables above say what each class gets wrong. This says **why**, "
        "and the eleven classes fall into three groups that want three "
        "different fixes. Membership is decided by the rule below, "
        "recomputed from the confusion matrices every time this report is "
        "generated -- it is not a hand-made list.",
        "",
        "The rule, in order:",
        "",
        "1. **Does it transfer?** `train IoU − test IoU < %.0f` points means "
        "the model already does on unseen frames what it does on frames it "
        "was trained on. Inside the scored region there is nothing left to "
        "learn, so whatever is still wrong is wrong *outside* it — where the "
        "metric never looks and, since [the training loss uses the same "
        "mask](#what-label_weight-is), the loss never charged it either."
        % GAP_FLAT,
        "2. **Otherwise, where do most of its false positives land?** On "
        "`empty` — it is putting volume where the annotation says there is "
        "nothing.",
        "3. **On another class** — the geometry was roughly right and the "
        "name was wrong.",
        "",
        "\n".join(rows),
        "",
    ]

    # ---- group 1 -----------------------------------------------------------
    parts += [
        "### Group 1 — %s: %s" % (TITLES[1], ", ".join("`%s`" % n for n in g1)),
        "",
        "%s %s the only %s to pass test 1, losing %s points against "
        "%.1f–%.1f for the other %d. Inside the scored region %s finished — "
        "%s — and training harder cannot improve a number already the same "
        "on held-out frames as on trained ones."
        % (g1_names, is_are, "class" if len(g1) == 1 else "classes",
           bare(g1, "%(gap).1f"), gaps[0], gaps[-1], len(others),
           "it is" if len(g1) == 1 else "they are",
           bare(g1, "%(wrong).1f%% of its claims wrong at IoU %(iou).1f")),
        "",
        "Its error is somewhere the metric cannot see. %s predicts %s voxels "
        "on ground truth that explicitly says `empty` but sits outside "
        "`label_weight` — the annotation disagrees, and neither the score "
        "nor the training loss counts it. That is the floor slab running "
        "through walls and out of the room: never scored, never penalised, "
        "never corrected."
        % (g1_names,
           " / ".join("{:,}".format(info[n]['unscored']) for n in g1)),
        "",
        "**Be careful with this one.** Every class spills outside the mask — "
        "`%s` puts %s voxels there, far more than %s does. What makes %s "
        "different is not the size of that spill but that it is *all* of its "
        "error: the scored part is already right and does not degrade. For "
        "the other %s the unscored spill sits on top of failures the metric "
        "does see, which is what groups 2 and 3 are about."
        % (worst_unscored, "{:,}".format(info[worst_unscored]['unscored']),
           g1_names, g1_names, _WORD[len(others)]),
        "",
        "**Fix direction: supervision.** Score the space the model is "
        "currently free to fill, or stop asking it to fill space nobody "
        "annotated. Nothing about the network is at fault here.",
        "",
    ]

    # ---- group 2 -----------------------------------------------------------
    parts += [
        "### Group 2 — %s: %s" % (TITLES[2], ", ".join("`%s`" % n for n in g2)),
        "",
        "For these, `empty` is the single largest destination of their false "
        "positives — %s. They are not mislabelling other objects; they are "
        "manufacturing volume where the ground truth says there is none."
        % lst(g2, "%(empty).0f%%"),
        "",
        "This is the model, not the data, and the train split is what settles "
        "it. On frames the checkpoint was **trained on**, with the answers "
        "already shown to it, the spill does not go away — %s — %s. Those "
        "voxels are inside `label_weight`, so the training loss charged for "
        "every one of them, on scenes it had already seen, and the model "
        "produced them anyway. No amount of extra annotation fixes a failure "
        "that survives the annotation it already has."
        % (lst(g2, "%(empty_tr).0f%%"), worse_note),
        "",
        "**Fix direction: the model.** This is the bed sticking through the "
        "wall — an occupancy prior asserting itself past the evidence.",
        "",
    ]

    # ---- group 3 -----------------------------------------------------------
    parts += [
        "### Group 3 — %s: %s" % (TITLES[3], ", ".join("`%s`" % n for n in g3)),
        "",
        "Here the largest destination is another *class*, not `empty`: %s. "
        "The volume is in roughly the right place; the name on it is wrong."
        % ", ".join("`%s` → `%s` %.0f%%"
                    % (n, info[n]['dest'], info[n]['dest_share']) for n in g3),
        "",
        "%s"
        % ("`furn` absorbs %d of the %d, which is the shape of the problem: "
           "`furn` and `objs` are not separable categories so much as two "
           "names for the same clutter, and the model is being marked wrong "
           "for choosing between them."
           % (len(to_furn), len(g3)) if to_furn else
           "The destinations are spread across several classes."),
        "",
        "**Fix direction: the labels.** Merging the pairs that trade, or "
        "reporting them merged, would move these numbers without touching "
        "the network. The geometry these classes produce is already usable.",
        "",
    ]

    # ---- honesty about the rule -------------------------------------------
    parts += [
        "### What the rule does and does not prove",
        "",
        "Tests 2 and 3 compare against the *largest single* destination, not "
        "a majority. %s Group 2 membership is a ranking, then, not a clean "
        "threshold, and a class near the boundary belongs to both stories."
        % ("Only %s %s more than half of %s false positives to `empty`; for "
           "the rest of group 2 `empty` is the biggest single bucket at "
           "%.0f–%.0f%% while the other classes together still outweigh it."
           % (", ".join("`%s`" % n for n in over_half),
              "sends" if len(over_half) == 1 else "send",
              "its" if len(over_half) == 1 else "their",
              min(info[n]['empty'] for n in g2 if n not in over_half),
              max(info[n]['empty'] for n in g2 if n not in over_half))
           if over_half and len(over_half) < len(g2) else
           "No class in group 2 sends a clear majority of its false "
           "positives to `empty`; the split is a ranking throughout."),
        "",
        "The one cross-cutting fact the grouping does not capture: **%d of "
        "the eleven lose %.0f to %.0f IoU points from train to test**, %s "
        "excepted at %.1f. Whatever else is wrong per class, failure to "
        "transfer is common to all three groups, and the headline mIoU "
        "falling from train to test is that in aggregate."
        % (len(others), gaps[0], gaps[-1],
           " and ".join("`%s`" % n for n in g1), info[g1[0]]['gap']
           if g1 else 0.0),
        "",
    ]
    return "\n".join(parts)


def _miss_stats(cm, geo):
    """Per class: how much it misses, where the missed voxels sit, and what
    they were called. Built from the region counts evaluate() collects."""
    gw, fw = geo['gt_where'], geo['fn_where']
    m = metrics(cm)
    gt = cm.sum(axis=1)
    out = {}
    for c in range(1, N_CLASSES):
        fn = fw[c].sum()
        # the region split must account for every miss the matrix counts
        assert fn == int(m['fn'][c]), (CLASSES[c], fn, m['fn'][c])
        n_dest, dest = max((cm[c, t], t) for t in range(N_CLASSES) if t != c)
        out[CLASSES[c]] = dict(
            miss=100.0 * fn / max(gt[c], 1),
            dest=CLASSES[dest], dest_share=100.0 * n_dest / max(fn, 1),
            empty=100.0 * fw[c, 0].sum() / max(fn, 1),
            empty_hidden=100.0 * fw[c, 0, 2] / max(fw[c, 0].sum(), 1),
            miss_surf=100.0 * fw[c, :, 0].sum() / max(gw[c, 0], 1),
            miss_hide=100.0 * fw[c, :, 2].sum() / max(gw[c, 2], 1),
            hidden_share=100.0 * gw[c, 2] / max(gw[c].sum(), 1))
    return out


def miss_groups(tr, te):
    """The false-negative mirror of error_groups: why each class MISSES.

    Simpler than the FP side in one respect: every real object voxel is
    inside label_weight (it is half of the mask's definition), so every miss
    is scored and was penalised in training. There is no "never checked"
    group. What separates the classes instead is WHERE the missed voxel sits
    and WHAT it was called:

      1. Does it miss much? Under MISS_FLAT % there is nothing to fix.
      2. The largest destination of its misses is `empty`: it leaves part of
         the object unfilled. Every one of those sits in hidden space -- the
         model essentially never says `empty` where the camera measured
         anything (see the region table this section prints first).
      3. Another class: the volume was found, the name was wrong.
    """
    E, T = _miss_stats(te[0], te[5]), _miss_stats(tr[0], tr[5])
    names = [CLASSES[c] for c in range(1, N_CLASSES)]
    r5 = te[5]['region5']

    # the FP group each class landed in, by the same rule as error_groups
    m_tr, m_te = metrics(tr[0]), metrics(te[0])
    fp_group = {}
    for n in names:
        c = CLASSES.index(n)
        _, dest = max((te[0][t, c], t) for t in range(N_CLASSES) if t != c)
        fp_group[n] = _fp_group(100 * (m_tr['iou'][c] - m_te['iou'][c]), dest)

    group = {n: 1 if E[n]['miss'] < MISS_FLAT else
             (2 if E[n]['dest'] == 'empty' else 3) for n in names}
    members = lambda g: sorted((n for n in names if group[n] == g),
                               key=lambda n: -E[n]['miss'])
    g1, g2, g3 = members(1), members(2), members(3)
    q = lambda ns: ", ".join("`%s`" % n for n in ns)
    lst = lambda ns, fmt, S=E: ", ".join("`%s` %s" % (n, fmt % S[n])
                                         for n in ns)

    # ---- where the model says `empty` at all -------------------------------
    reg = ["| region (test) | share of the volume | model says `empty` | "
           "GT says `empty` | inside `label_weight` |",
           "| --- | ---: | ---: | ---: | ---: |"]
    for k, name in enumerate(REGION5):
        n_vox, p_empty, g_empty, in_mask = r5[k]
        reg.append("| %s | %.1f%% | %.1f%% | %.1f%% | %.1f%% |" % (
            name, 100.0 * n_vox / r5[:, 0].sum(),
            100.0 * p_empty / max(n_vox, 1), 100.0 * g_empty / max(n_vox, 1),
            100.0 * in_mask / max(n_vox, 1)))
    pc = lambda k, col: 100.0 * r5[k, col] / max(r5[k, 0], 1)
    empty_hidden_min = min(E[n]['empty_hidden'] for n in names
                           if E[n]['empty'] > 0)

    rows = ["| class | group | missed, test | missed, train | largest "
            "destination of its misses | share | missed on the visible "
            "surface | missed in hidden space | FP group |",
            "| --- | ---: | ---: | ---: | --- | ---: | ---: | ---: | ---: |"]
    for g in (1, 2, 3):
        for n in members(g):
            e = E[n]
            rows.append("| `%s` | %d | %.1f%% | %.1f%% | `%s` | %.0f%% | "
                        "%.0f%% | %.0f%% | %d |"
                        % (n, g, e['miss'], T[n]['miss'], e['dest'],
                           e['dest_share'], e['miss_surf'], e['miss_hide'],
                           fp_group[n]))

    over_half = [n for n in g2 if E[n]['empty'] > 50]
    # group 2 by the train split: already failing on trained frames, or not
    habit = [n for n in g2 if T[n]['miss_hide'] >= HABIT_BAR]
    fitted = [n for n in g2 if n not in habit]
    not_half = [n for n in g2 if n not in over_half]
    both2 = [n for n in g2 if fp_group[n] == 2]
    both3 = [n for n in g3 if fp_group[n] == 3]

    parts = [
        "## Three kinds of miss (false negatives)",
        "",
        "The previous section asked where the model puts things that are not "
        "there. This is the mirror: of what **is** there, what does it fail "
        "to find, and why. One thing makes it simpler. Every real object "
        "voxel is inside `label_weight` — that is half of the mask's "
        "definition — so **every miss is scored, and every miss was "
        "penalised in training**. There is no \"never checked\" side here. "
        "What separates the classes is *where* the missed voxel sits and "
        "*what* it was called.",
        "",
        "### Where the model says `empty` at all",
        "",
        "\n".join(reg),
        "",
        "Where the camera looked straight through the space and the GT "
        "agrees it is empty %.0f%% of the time, the model says `empty` in "
        "only **%.1f%%** of it. Where it does say `empty` is occluded space "
        "— **%.0f%%** — which is the one region the mask covers completely "
        "(%.0f%%). The model learned `empty` where the loss taught it, and "
        "little anywhere else. That is the other face of the false-positive "
        "spill above: space the loss never covered gets filled by default."
        % (pc(1, 2), pc(1, 1), pc(3, 1), pc(3, 3)),
        "",
        "For misses it means the model **never erases something the camera "
        "measured**. It says `empty` on %.1f%% of surface voxels, and of the "
        "misses it calls `empty`, at least %.1f%% sit in hidden space for "
        "every class. A miss is one of two things: a hidden part left "
        "unfilled, or the right volume given the wrong name."
        % (pc(0, 1), empty_hidden_min),
        "",
        "The rule, in order:",
        "",
        "1. **Does it miss much?** Under %.0f%% of the class missed on test "
        "means there is nothing here to fix." % MISS_FLAT,
        "2. **Otherwise, where do most of its misses go?** To `empty` — it "
        "leaves part of the object unfilled, and that part is hidden "
        "(at least %.0f%% of the time, for every class)." % empty_hidden_min,
        "3. **To another class** — it found the volume and named it wrong.",
        "",
        "*Missed on the visible surface* is the share of the class's voxels "
        "that hold a depth point which the model got wrong; *missed in "
        "hidden space* the same for voxels the camera could not see. The "
        "last column is the group the class landed in on the "
        "false-positive side, for comparison.",
        "",
        "\n".join(rows),
        "",
    ]

    if g1:
        parts += [
            "### Group 1 — It finds it: %s" % q(g1),
            "",
            "%s %s %s of %s on test and %s on train. Whatever is wrong with "
            "%s is in the false-positive section, not here."
            % (q(g1), "misses" if len(g1) == 1 else "miss",
               " / ".join("%.1f%%" % E[n]['miss'] for n in g1),
               "itself" if len(g1) == 1 else "themselves",
               " / ".join("%.1f%%" % T[n]['miss'] for n in g1),
               "it" if len(g1) == 1 else "them"),
            "",
        ]

    parts += [
        "### Group 2 — It doesn't complete what's hidden: %s" % q(g2),
        "",
        "`empty` is the largest destination of their misses — %s — and "
        "at least %.0f%% of those empty misses are in space the camera could "
        "not see. These classes are mostly hidden to begin with (%s of their "
        "voxels), and that is where they fail: they miss %s of their hidden "
        "voxels, against %s of their visible surface."
        % (lst(g2, "%(empty).0f%%"), empty_hidden_min,
           lst(g2, "%(hidden_share).0f%%"),
           lst(g2, "%(miss_hide).0f%%"), lst(g2, "%(miss_surf).0f%%")),
        "",
        "It is the model, not the data: those voxels are inside the mask "
        "and every one was penalised in training. The train split says "
        "which kind of model failure. %s"
        % " ".join(s for s in (
            ("For %s it is **already there on frames it trained on** — %s of "
             "the hidden voxels missed on train, %s on test. A habit the "
             "loss could not remove." % (q(habit), lst(habit,
             "%(miss_hide).0f%%", T), lst(habit, "%(miss_hide).0f%%"))
             if habit else ""),
            ("For %s the training frames are fitted — %s missed on train — "
             "and the miss appears on test (%s). That is **transfer**, not "
             "habit." % (q(fitted), lst(fitted, "%(miss_hide).0f%%", T),
                         lst(fitted, "%(miss_hide).0f%%"))
             if fitted else "")) if s),
        "",
    ]
    if both2:
        parts += [
            "%s %s also in false-positive group 2: %s grow%s into empty "
            "space *and* leave%s %s own hidden parts unfilled. The trouble "
            "is the shape of the volume, not only its amount — too much where "
            "nothing is, too little where the object continues out of sight."
            % (q(both2), "is" if len(both2) == 1 else "are",
               "it" if len(both2) == 1 else "they",
               "s" if len(both2) == 1 else "", "s" if len(both2) == 1 else "",
               "its" if len(both2) == 1 else "their"),
            "",
        ]
    parts += [
        "**Fix direction: the model's completion** — shape priors for the "
        "part of an object behind what the camera saw.",
        "",
        "### Group 3 — It finds the volume and names it wrong: %s" % q(g3),
        "",
        "The largest destination is another class: %s. These fail on what "
        "the camera **did** see, too — they miss %s of their visible "
        "surface, with a depth point in the voxel."
        % (", ".join("`%s` → `%s` %.0f%%" % (n, E[n]['dest'],
                                              E[n]['dest_share'])
                     for n in g3),
           lst(g3, "%(miss_surf).0f%%")),
        "",
    ]
    if both3:
        parts += [
            "%s %s in false-positive group 3 as well, so the confusion "
            "runs both ways: what %s wrongly claim%s and what %s miss%s are "
            "the same neighbouring classes."
            % (q(both3), "is" if len(both3) == 1 else "are",
               "it" if len(both3) == 1 else "they",
               "s" if len(both3) == 1 else "",
               "it" if len(both3) == 1 else "they",
               "es" if len(both3) == 1 else ""),
            "",
        ]
    parts += [
        "**Fix direction: the labels**, as on the false-positive side.",
        "",
        "### What the rule does and does not prove",
        "",
        "Test 2 ranks by the *largest single* destination, not a majority. "
        + ("Only %s send%s more than half of %s misses to `empty`; for %s "
           "`empty` is the biggest single bucket (%s) while the other "
           "classes together still outweigh it."
           % (q(over_half), "s" if len(over_half) == 1 else "",
              "its" if len(over_half) == 1 else "their", q(not_half) + ",",
              lst(not_half, "%(empty).0f%%"))
           if over_half and not_half else
           "The same caution as on the false-positive side applies.")
        + " So group 2 is a ranking, and a class near its edge belongs to "
        "both stories.",
        "",
        "Hidden space is also the genuinely hard part of the task: nothing "
        "in the input shows the back of a sofa. How much of a group-2 miss "
        "was knowable is not something these numbers can settle. The train "
        "column bounds it from one side only — the model cannot fit it even "
        "when shown the answer.",
        "",
    ]
    return "\n".join(parts)


def main():
    ap = argparse.ArgumentParser(
        description=__doc__.split('***')[0],
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--split', default='both', choices=['train', 'test', 'both'])
    ap.add_argument('--nyu-root', default=DEFAULT_NYU,
                    help="the NYU data directory holding Label/ TSDF/ Mapping/")
    ap.add_argument('--pred-dir', default=DEFAULT_PRED,
                    help="directory of NYU####_0000.npy prediction files")
    ap.add_argument('--report', help="write the report to this path instead of "
                                     "stdout. The content is identical either "
                                     "way")
    ap.add_argument('--top', type=int, default=12,
                    help="how many confusion pairs to list")
    ap.add_argument('--train-tables', action='store_true',
                    help="also print the full per-class tables for the train "
                         "split. Off by default: the checkpoint was trained on "
                         "those frames, so on their own they say nothing about "
                         "held-out performance. Train is still evaluated for "
                         "the sections that compare it with test")
    args = ap.parse_args()

    splits = ['train', 'test'] if args.split == 'both' else [args.split]
    parts = ["# NYU evaluation",
             "",
             "Generated by `inference/evaluate.py`; rerun it after "
             "regenerating the predictions.",
             "",
             "Protocol is `examples/segmentation/test_NYU.py`'s, so these are "
             "comparable with the repo's own numbers: **SSC** over voxels "
             "where `label_weight > 0` and `label != 255`, mIoU averaged over "
             "classes 1-11; **SC** the same restricted to `mapping == 307200`, "
             "the voxels no depth pixel reached, scored occupied-vs-empty.",
             "",
             "`precision = TP/(TP+FP)` — of what was predicted, how much was "
             "right. `recall = TP/(TP+FN)` — of what was there, how much was "
             "found. `IoU = TP/(TP+FP+FN)` — both failures in one number, "
             "which is why it is the headline. (`test_NYU.py` prints precision "
             "under the heading *accuracy*.)",
             ""]
    headline = []
    extras = {}                 # per split, for the write-up and the comparison
    results = {}
    for split in splits:
        ids = load_ids(args.nyu_root, split)
        print("evaluating %s (%d frames)" % (split, len(ids)), file=sys.stderr)
        results[split] = evaluate(ids, args.nyu_root, args.pred_dir)
        if not results[split][5]:
            raise SystemExit("no frames evaluated for %r -- check --pred-dir "
                             "and --nyu-root" % split)

    # test is the evaluation set -- the one the paper reports. Train gets full
    # tables only when it is all that was asked for, or on --train-tables.
    if args.split != 'both':
        shown = splits
    else:
        shown = ['train', 'test'] if args.train_tables else ['test']
    if args.split == 'both':
        parts += ["**Everything below is the `test` split** — the held-out "
                  "frames, and the set the paper reports — unless a section "
                  "says otherwise. The `train` split (the frames the "
                  "checkpoint was trained on) is evaluated too, but it only "
                  "appears where a section compares the two: *Learned habit, "
                  "or the data?*, the two error-group sections, and the "
                  "closing *train vs test* table.%s"
                  % ("" if args.train_tables else
                     " Pass `--train-tables` for the full train tables."), ""]

    for split in splits:
        (cm_ssc, cm_sc, cover, frames, unscored, used, skipped,
         geo) = results[split]
        tbl, miou, sciou = table(cm_ssc, cm_sc)
        sc_tbl, sc_lab, sc_occ = sc_class_rows(cm_ssc, geo['cm_sc12'], cm_sc)
        headline.append((split, len(used), miou, sciou, sc_lab, sc_occ))
        extras[split] = (cm_ssc, cover, frames, unscored, len(used), geo)
        if split not in shown:
            continue
        parts += ["## `%s` split — %d frames%s" % (
                      split, len(used),
                      "" if not skipped else
                      " (%d skipped: no prediction or no label)" % len(skipped)),
                  ""]
        if split == 'train':
            parts += ["> The checkpoint was **trained** on these frames. The "
                      "numbers are here so the gap to `test` can be seen; on "
                      "their own they say nothing about generalisation.", ""]
        parts += [tbl, "",
                  sc_breakdown(cm_ssc, cm_sc, geo['cm_sc12'], geo, split,
                               args.pred_dir), "",
                  "### Scene completion (SC) per class", "",
                  "SC above is one binary number. This breaks it down per "
                  "class, over the same region: scored voxels **no depth "
                  "pixel reached** — the part of each object the model has "
                  "to complete rather than read off the input.", "",
                  SC_COLUMNS, "",
                  sc_tbl, "",
                  "### How much of each class's prediction is wrong", "",
                  "Of everything the model calls a class — counting only the "
                  "predictions the ground truth can adjudicate — how much is "
                  "wrong, and what those voxels really were. *% of its claims "
                  "wrong* is `FP / (TP + FP)`, i.e. 1 − precision, restated so "
                  "it reads directly. Do not confuse it with *% of its FP on "
                  "`empty`* two tables down, which divides by FP rather than "
                  "by predictions and answers a different question.", "",
                  fp_breakdown(cm_ssc), "",
                  "### How much of each class the model misses", "",
                  "The mirror image: of everything that *is* a class, how much "
                  "the model failed to find, and what it called those voxels "
                  "instead.", "",
                  fn_breakdown(cm_ssc), "",
                  "### The biggest single confusions", "",
                  biggest_confusions(cm_ssc, args.top), "",
                  "### Which classes invent volume", "",
                  "Splitting each class's false positives into *put it where "
                  "there was nothing* versus *called another object by the "
                  "wrong name*. Precision alone conflates the two; only the "
                  "first grows an object past its real extent.", "",
                  spill_rows(cm_ssc), "",
                  "### How much of each prediction is scored at all", "",
                  "`label_weight` and the `GT != 255` rule between them hide "
                  "most of the output from the metric. The last three columns "
                  "are the point: a prediction over GT `empty` is the "
                  "annotation actively disagreeing, and the same voxel counts "
                  "as a false positive or as nothing at all depending only on "
                  "which side of the mask it falls. The final column is the "
                  "strongest form: the depth frame measured free space there "
                  "(`tsdf > 0`, in front of a surface the camera saw), so both "
                  "the annotation and the sensor say nothing is there, and no "
                  "metric counts it.", "",
                  coverage_rows(cm_ssc, cover, unscored), "",
                  "### How often each class is asserted at all", "",
                  "Voxel counts hide this one. A class predicted in every "
                  "frame while the ground truth carries it in a fraction of "
                  "them is applying a prior rather than reading the scene.", "",
                  presence_rows(frames, len(used)), ""]

    if 'test' in extras:
        parts += ["---", "", label_weight_note([extras['test'][5]]),
                  "", "---", ""]
        if 'train' in extras:
            parts += [learned_vs_generalisation(extras['train'], extras['test']),
                      "", "---", "",
                      error_groups(extras['train'], extras['test']),
                      "", "---", "",
                      miss_groups(extras['train'], extras['test']),
                      "", "---", ""]

    if len(headline) == 2:
        parts += ["## train vs test", "",
                  "| split | frames | SSC mIoU | SC IoU | SC mIoU, label "
                  "must match | SC mIoU, any class counts |",
                  "| --- | ---: | ---: | ---: | ---: | ---: |"]
        for s, n, mi, sci, scl, sco in headline:
            parts.append("| `%s` | %d | %.1f | %.1f | %.1f | %.1f |"
                         % (s, n, mi, sci, scl, sco))
        gap = headline[0][2] - headline[1][2]
        parts += ["", "Train − test mIoU gap: **%.1f points**." % gap, ""]

    text = "\n".join(parts)
    if args.report:
        with open(args.report, 'w') as f:
            f.write(text + "\n")
        print("wrote %s" % args.report)
    else:
        print(text)


if __name__ == '__main__':
    main()
