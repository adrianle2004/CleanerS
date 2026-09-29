"""
reconstruction_GT/verify_voxelizer.py

Prove voxelize_gt.py agrees with data somebody else made, before any of your
annotation depends on it. MAKING_GT.md, "Verifying your toolchain".

    python -m reconstruction_GT.verify_voxelizer            # both datasets
    python -m reconstruction_GT.verify_voxelizer --nyu-only

The risk being bought off is silent: a wrong axis order or half-voxel offset
produces a perfectly plausible volume that simply scores badly, and you would
blame the model. So each check is against an independent artifact, never
against this code's own output.

NYU (2 cm, 240x144x240, from the shipped .bin + Label/ + TSDF/):

  A  grid round-trip      index -> world -> index is the identity
  B  solids round-trip    a 2 cm box at a voxel centre fills exactly that voxel
  C  downsample kernel    numpy downsample_label == SSCNet's CUDA DownSampleLabel
  D  label3d              downsample(shipped .bin volume) == shipped Label npz
  E  label_weight         compute_label_weight(...) == shipped TSDF arr_1
  F  axis control         GT surface agrees with the depth-derived surface, and
                          disagrees under a swapped axis order

Occ-ScanNet (8 cm, 60x36x60, from data/Scannet/<scene>/<frame>.npz, already
converted to CleanerS layout by data/occscannet_gt.py):

  G  grid round-trip      same, at a different unit and a per-frame origin
  H  solids round-trip    same
  I  floor placement      floor sits in the lowest height layers (reported
                          against NYU's, which differ -- see the floor-align
                          note in occscannet_floor_align.py)

D and E are the load-bearing ones: they reproduce files this repo did not
write, bit for bit.
"""

import os
import sys
import glob
import json
import struct
import argparse

import numpy as np

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_THIS_DIR)
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from reconstruction_GT.voxelize_gt import (
    Grid, paint_box, downsample_label, compute_label_weight,
    VOX_SIZE_HI, VOX_UNIT_HI, VOX_SIZE_LOW, VOX_UNIT_LOW, IGNORE)

NYU_ROOT = os.path.join(os.path.dirname(_REPO_ROOT), 'data', 'NYU')
SCANNET_ROOT = os.path.join(os.path.dirname(_REPO_ROOT), 'data', 'Scannet')

# 37 SSCNet classes -> the 12 CleanerS ones
SEG_MAP = np.array([0, 1, 2, 3, 4, 11, 5, 6, 7, 8, 8, 10, 10, 10, 11, 11, 9, 8,
                    11, 11, 11, 11, 11, 11, 11, 11, 11, 10, 10, 11, 8, 10, 11,
                    9, 11, 11, 11])

_results = []


def check(name, ok, detail=''):
    _results.append((name, bool(ok)))
    print('  %-22s %s  %s' % (name, 'PASS' if ok else 'FAIL', detail))
    return ok


def load_bin(path):
    """SSCNet .bin -> (vox_origin, cam_pose, 240x144x240 uint8 of 12 classes)."""
    with open(path, 'rb') as f:
        vox_origin = np.array(struct.unpack('3f', f.read(12)), np.float64)
        cam_pose = np.array(struct.unpack('16f', f.read(64)), np.float64).reshape(4, 4)
        rest = f.read()
    vals = np.frombuffer(rest, np.uint32)
    labels, counts = vals[0::2], vals[1::2]
    out = np.repeat(np.where(labels == 255, 255, SEG_MAP[labels % 37]),
                    counts).astype(np.uint8)
    assert out.size == int(np.prod(VOX_SIZE_HI)), out.size
    return vox_origin, cam_pose, out.reshape(VOX_SIZE_HI)


def grid_round_trip(grid, rng, n=400000):
    """A: every index must survive index -> world -> index."""
    idx = np.stack([rng.integers(0, s, n) for s in grid.shape], axis=1)
    back = grid.world_to_index(grid.index_to_world(idx))
    return np.array_equal(idx, back)


def solids_round_trip(vol_truth, grid, rng, n=200):
    """B/H: put a one-voxel box at a voxel's centre; it must fill that voxel
    and only that voxel.

    Not circular: it leaves index space for world metres, goes through the
    same oriented-box test real annotation uses, and comes back.
    """
    occ = np.argwhere((vol_truth > 0) & (vol_truth != IGNORE))
    if len(occ) == 0:
        return False, 'no occupied voxels'
    pick = occ[rng.choice(len(occ), size=min(n, len(occ)), replace=False)]
    centres = grid.index_to_world(pick)
    bad_count = bad_place = 0
    for (i, j, k), c in zip(pick, centres):
        probe = np.zeros(grid.shape, np.uint8)
        # slightly under one voxel, so floating point cannot catch a neighbour
        paint_box(probe, grid, c, [grid.unit * 0.98] * 3, 0.0, 7)
        hit = np.argwhere(probe == 7)
        if len(hit) != 1:
            bad_count += 1
        elif not np.array_equal(hit[0], [i, j, k]):
            bad_place += 1
    return (bad_count == 0 and bad_place == 0,
            '%d probes, %d wrong count, %d wrong voxel' % (len(pick), bad_count, bad_place))


def cuda_downsample(hi_label, tsdf_hi):
    """SSCNet's own kernel, for check C."""
    from inference.frame_loader import load_cuda_encoder
    dp = load_cuda_encoder()
    if dp is None:
        return None
    vox_size = np.array(VOX_SIZE_HI, np.float32)
    n_low = int(np.prod(VOX_SIZE_LOW))
    label = np.zeros(n_low, np.float32)
    tsdf_ds = np.zeros(n_low, np.float32)
    readin = np.ascontiguousarray(hi_label, np.float32).reshape(-1).copy()
    tsdf = np.ascontiguousarray(tsdf_hi, np.float32).reshape(-1).copy()
    cwd = os.getcwd()
    os.chdir(os.path.join(_REPO_ROOT, '..', 'data', 'utils'))
    try:
        dp.DownSampleLabel(readin, vox_size, 4, tsdf, label, tsdf_ds)
    finally:
        os.chdir(cwd)
    return label.astype(np.int32), tsdf_ds


def nyu_frames(n):
    ids = sorted(os.path.splitext(os.path.basename(p))[0]
                 for p in glob.glob(os.path.join(NYU_ROOT, 'Label', '*.npz')))
    return ids[:n]


def verify_nyu(n_frames, rng):
    from inference.frame_loader import FrameLoader, load_cuda_encoder
    print('\n=== NYU (2 cm, %s) ===' % (VOX_SIZE_HI,))
    ids = nyu_frames(n_frames)
    if not ids:
        check('nyu data', False, 'no Label/*.npz under %s' % NYU_ROOT)
        return

    first = True
    lab_ok = lw_ok = cuda_ok = True
    lab_detail = lw_detail = cuda_detail = ''
    for fid in ids:
        bin_path = os.path.join(NYU_ROOT, 'depth', 'NYU%s_0000.bin' % fid)
        png_path = os.path.join(NYU_ROOT, 'depth', 'NYU%s_0000.png' % fid)
        vox_origin, cam_pose, hi = load_bin(bin_path)
        grid = Grid(vox_origin, VOX_SIZE_HI, VOX_UNIT_HI)

        loader = FrameLoader.from_nyu_bin(bin_path, png_path)
        loader.encoder = 'cuda' if load_cuda_encoder() is not None else 'numpy'
        tsdf_hi, _ = (loader._build_high_res_tsdf_cuda() if loader.encoder == 'cuda'
                      else loader._build_high_res_tsdf())
        tsdf_hi = np.asarray(tsdf_hi, np.float32).reshape(-1)

        label, tsdf_ds = downsample_label(hi, tsdf_hi)
        lw = compute_label_weight(label, tsdf_ds)

        shipped_label = np.load(os.path.join(NYU_ROOT, 'Label', '%s.npz' % fid))['arr_0']
        shipped = np.load(os.path.join(NYU_ROOT, 'TSDF', '%s.npz' % fid))
        shipped_lw = shipped['arr_1'].astype(np.float32).reshape(-1)

        same_label = int((label == shipped_label.astype(np.int32).reshape(-1)).sum())
        same_lw = int((lw == shipped_lw).sum())
        if same_label != label.size:
            lab_ok = False
            lab_detail = '%s: %d/%d' % (fid, same_label, label.size)
        if same_lw != lw.size:
            lw_ok = False
            lw_detail = '%s: %d/%d (%d differ)' % (
                fid, same_lw, lw.size, lw.size - same_lw)

        cu = cuda_downsample(hi, tsdf_hi)
        if cu is None:
            cuda_detail = 'DataProcess unavailable'
        else:
            if not np.array_equal(cu[0], label):
                cuda_ok = False
                cuda_detail = '%s: %d differ' % (fid, int((cu[0] != label).sum()))
            elif not np.allclose(cu[1], tsdf_ds, atol=1e-5):
                cuda_ok = False
                cuda_detail = '%s: tsdf mean differs' % fid

        if first:
            first = False
            check('A grid round-trip', grid_round_trip(grid, rng))
            ok, det = solids_round_trip(hi, grid, rng)
            check('B solids round-trip', ok, det)
            # F: the GT surface and the depth-derived surface must overlap, and
            # must stop overlapping if the two horizontal axes are swapped.
            # Restricted to the ANNOTATED region: outside it the GT says
            # nothing, so including it just dilutes both numbers.
            gt_occ = (hi > 0) & (hi != IGNORE)
            annotated = hi != IGNORE
            surf = (np.abs(tsdf_hi.reshape(VOX_SIZE_HI)) < 0.2) & annotated
            def agree(a):
                return float((a & surf).sum()) / max(1, int(surf.sum()))
            right = agree(gt_occ)
            swapped = agree(np.swapaxes(gt_occ, 0, 2))
            check('F axis control', right > 3 * swapped,
                  'observed surface that is GT-occupied: %.2f correct vs '
                  '%.2f axis-swapped' % (right, swapped))

    check('C downsample kernel', cuda_ok, cuda_detail or
          '%d frames match DataProcess.DownSampleLabel' % len(ids))
    check('D label3d vs shipped', lab_ok, lab_detail or
          '%d frames bit-exact' % len(ids))
    check('E label_weight vs shipped', lw_ok, lw_detail or
          '%d frames bit-exact' % len(ids))


def verify_scannet(n_frames, rng):
    print('\n=== Occ-ScanNet (8 cm, %s) ===' % (VOX_SIZE_LOW,))
    files = sorted(glob.glob(os.path.join(SCANNET_ROOT, '*', '*.npz')))[:n_frames]
    if not files:
        check('scannet data', False, 'no npz under %s' % SCANNET_ROOT)
        return
    rt_ok = sr_ok = True
    rt_detail = sr_detail = ''
    floors = []
    for f in files:
        d = np.load(f, allow_pickle=True)
        target = d['target']
        grid = Grid(d['voxel_origin'], VOX_SIZE_LOW, VOX_UNIT_LOW)
        if not grid_round_trip(grid, rng, n=100000):
            rt_ok = False
            rt_detail = os.path.basename(f)
        ok, det = solids_round_trip(target, grid, rng, n=60)
        if not ok:
            sr_ok = False
            sr_detail = '%s: %s' % (os.path.basename(f), det)
        layers = np.nonzero(target == 2)[1]          # axis 1 is height
        if layers.size:
            floors.append((layers.min(), np.median(layers)))
    check('G grid round-trip', rt_ok, rt_detail or '%d frames' % len(files))
    check('H solids round-trip', sr_ok, sr_detail or '%d frames' % len(files))
    if floors:
        lo = np.array([a for a, _ in floors])
        med = np.array([b for _, b in floors])
        check('I floor in low layers', med.max() <= 5,
              'floor layer: min %d, median of medians %.1f (NYU: hi-res 2-3, '
              'i.e. low-res 0)' % (lo.min(), float(np.median(med))))
    else:
        check('I floor in low layers', False, 'no floor class in these frames')


def main():
    p = argparse.ArgumentParser(description=__doc__.split('\n')[3])
    p.add_argument('--frames', type=int, default=4, help='NYU frames to check')
    p.add_argument('--scannet-frames', type=int, default=6)
    p.add_argument('--nyu-only', action='store_true')
    p.add_argument('--scannet-only', action='store_true')
    p.add_argument('--seed', type=int, default=0)
    args = p.parse_args()
    rng = np.random.default_rng(args.seed)

    if not args.scannet_only:
        verify_nyu(args.frames, rng)
    if not args.nyu_only:
        verify_scannet(args.scannet_frames, rng)

    failed = [n for n, ok in _results if not ok]
    print('\n%d checks, %d failed%s' % (len(_results), len(failed),
                                        (': ' + ', '.join(failed)) if failed else ''))
    return 1 if failed else 0


if __name__ == '__main__':
    sys.exit(main())
