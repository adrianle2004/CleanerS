"""
inference/tsdf_torch.py

The TSDF encode in torch, straight to 60x36x60 -- the fast path behind
`FrameLoader(encoder='torch')`. Same geometry as
`FrameLoader._build_high_res_tsdf`, line for line; only the arithmetic moves.

*** WHY THIS IS FASTER THAN BOTH EXISTING ENCODERS ***
The only non-elementwise part of the encode is the distance transform, and it
is TRUNCATED at VOX_MARGIN / VOX_UNIT_HI = 12 voxels: `refined = min(d/12, 1)`.
So nothing outside the 25^3 box around a voxel can change its value, and
squared Euclidean distance over a bounded box is SEPARABLE -- it is
dx^2 + dy^2 + dz^2, a sum of independent terms, so the minimum over the box is
three 1-D minimisations of `f[i-d] + d^2`, |d| <= 12, along z then y then x.

That replaces, per voxel:
  - SSCNet's CUDA kernel (datautil.cu SquaredDistanceTransform__), which
    brute-forces every one of the ~15,600 candidates in that same box, and
  - scipy's exact unbounded O(N) transform, which is CPU-bound,
with 75 whole-array shifted minima -- the shape of work a GPU saturates.
Measured on an RTX 3050 Ti laptop, 240x144x240 = 8.29 M voxels per frame:

    numpy + scipy (encoder='numpy')          ~1.0 s
    DataProcess   (encoder='cuda')           ~1.1 s
    this module, fp32                         ~110 ms
    this module, fp16 EDT                      ~80 ms

Exact, re-measurable figures: `python -m inference.benchmark --what encode`,
which is where every number in document/HARDWARE.md comes from.

Validated on all 654 NYU test frames: SSC mIoU 47.7285 / SC 74.9528 with the
numpy encoder (bit-identical to the on-disk reference predictions, 654/654)
against 47.7285 / 74.9530 with this one. See document/HARDWARE.md.

*** TWO MEMORY DECISIONS ***
`slab`: the projection stage is independent per voxel, so it runs a z-slab at
a time and is block-meaned straight into the 60x36x60 result -- the high-res
TSDF is never held whole. Only the distance field has to exist in full,
because the transform reaches +-12 voxels across slab boundaries. That is the
floor: one 240x144x240 array plus the temporaries of one 1-D pass. Peak 779 ->
67 MB, bit-identical output, same speed.

`edt_dtype`: fp16 is EXACT for this transform, not an approximation. Every
value in it is a whole number no larger than 3 * 12^2 = 432, and fp16 holds
every integer to 2048 exactly. Verified bit-identical over the NYU test set.
Only the "infinity" sentinel has to change: 1e9 overflows half, so the fp16
path uses 60000.0, which is still unreachable by any real squared distance.
"""

import numpy as np
import torch

try:
    from .frame_loader import VOX_UNIT_HI, VOX_SIZE_HI, VOX_MARGIN, SAMPLE_RATIO
except ImportError:                       # run directly, no package context
    from frame_loader import VOX_UNIT_HI, VOX_SIZE_HI, VOX_MARGIN, SAMPLE_RATIO

SR = int(round(VOX_MARGIN / VOX_UNIT_HI))                      # 12
NX, NY, NZ = (int(VOX_SIZE_HI[0]), int(VOX_SIZE_HI[1]), int(VOX_SIZE_HI[2]))
BIG_F32 = 1e9
BIG_F16 = 60000.0                         # 1e9 is +inf in half
DEFAULT_SLAB = 24                         # z-layers per projection pass


def available():
    """True when encoder='torch' can run on the GPU. It also runs on the CPU
    (~1.3x faster than numpy there), but that is not worth defaulting to."""
    return torch.cuda.is_available()


def _edt1d(f, axis, big=None):
    """Exact squared Euclidean distance transform along `axis`, |d| <= SR.

    `out[i] = min over |d| <= SR of (f[i-d] + d^2)`, done as 2*SR shifted
    minima. The shift is a `narrow` VIEW, not a `torch.roll`: roll copies the
    whole array and then wraps, so the SR cells that came in from the far face
    had to be blanked to the sentinel or the grid would be a torus and a
    surface at z=0 would cast distance into z=239. A view simply does not
    write the cells a shift cannot reach, which is the same thing and moves
    half the bytes -- 60 -> 29 ms for the three passes on an RTX 3050 Ti,
    415 -> 184 on its CPU, bit-identical both ways.

    `d^2` is added to the whole array once per d rather than to each of the
    two slices, which is one kernel launch instead of two and measurably
    faster than either the roll version or a per-slice add.

    `big` is accepted and ignored: nothing needs blanking any more.
    """
    out = f.clone()
    n = f.shape[axis]
    shifted = torch.empty_like(f)
    for d in range(1, SR + 1):
        torch.add(f, float(d * d), out=shifted)
        a = out.narrow(axis, d, n - d)                  # f[i-d] + d^2
        torch.minimum(a, shifted.narrow(axis, 0, n - d), out=a)
        a = out.narrow(axis, 0, n - d)                  # f[i+d] + d^2
        torch.minimum(a, shifted.narrow(axis, d, n - d), out=a)
    return out


def encode_low(depth, cam_K, cam_pose, vox_origin, drop_invalid_depth=True,
               device=None, slab=DEFAULT_SLAB, edt_dtype=None):
    """-> (tsdf_low, weight_low), both (60,36,60) float32 numpy arrays.

    `weight_low` is the block MEAN of the high-res observed mask, left
    unthresholded so the caller can apply the same `>= 0.5` the numpy path
    applies. `tsdf_low` is final.

    Mirrors _build_high_res_tsdf exactly, including the order that matters:
    unobserved voxels are zeroed at HIGH resolution, before the 4x4x4 block
    mean, so boundary blocks keep their fractional values.
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    device = torch.device(device)
    if edt_dtype is None:
        # fp16 is exact here but slow on CPU, where there are no half kernels
        edt_dtype = torch.float16 if device.type == 'cuda' else torch.float32
    big = BIG_F32 if edt_dtype == torch.float32 else BIG_F16

    depth_t = torch.as_tensor(np.ascontiguousarray(depth, np.float32), device=device)
    H, W = depth_t.shape
    flat_depth = depth_t.reshape(-1)
    fx, fy = float(cam_K[0, 0]), float(cam_K[1, 1])
    cx, cy = float(cam_K[0, 2]), float(cam_K[1, 2])
    cam_pose = np.asarray(cam_pose, np.float32)
    R = torch.as_tensor(np.ascontiguousarray(cam_pose[:3, :3]), device=device)
    t = torch.as_tensor(np.ascontiguousarray(cam_pose[:3, 3]), device=device)
    org = torch.as_tensor(np.ascontiguousarray(vox_origin, np.float32), device=device)

    # ---- surface stamp: unproject every pixel, seed the voxel it lands in ----
    # The seed array IS the distance field, so no separate occupancy grid is
    # held: surface voxels start at 0, everything else at the sentinel.
    vs, us = torch.meshgrid(torch.arange(H, device=device, dtype=torch.float32),
                            torch.arange(W, device=device, dtype=torch.float32),
                            indexing='ij')
    pc = torch.stack([(us - cx) * depth_t / fx,
                      (vs - cy) * depth_t / fy, depth_t], -1).reshape(-1, 3)
    pw = pc @ R.T + t
    gz = torch.floor((pw[:, 0] - org[0]) / VOX_UNIT_HI).long()
    gx = torch.floor((pw[:, 1] - org[1]) / VOX_UNIT_HI).long()
    gy = torch.floor((pw[:, 2] - org[2]) / VOX_UNIT_HI).long()
    inb = (gx >= 0) & (gx < NX) & (gy >= 0) & (gy < NY) & (gz >= 0) & (gz < NZ)
    if drop_invalid_depth:
        inb &= flat_depth > 0
    f = torch.full((NZ * NY * NX,), big, device=device, dtype=edt_dtype)
    f[(gz[inb] * NY + gy[inb]) * NX + gx[inb]] = 0.0
    del pc, pw, gz, gx, gy, inb, us, vs

    # ---- the truncated distance to the nearest surface voxel ----
    f = _edt1d(f.view(NZ, NY, NX), 0)
    f = _edt1d(f, 1)
    f = _edt1d(f, 2)
    if f.dtype != torch.float32:
        f = f.float()
    f.sqrt_().div_(SR).clamp_(max=1.0)

    # ---- voxel centres -> camera -> pixel, for the sign and the known mask ----
    r = SAMPLE_RATIO
    tsdf_low = torch.empty(NZ // r, NY // r, NX // r, device=device)
    w_low = torch.empty(NZ // r, NY // r, NX // r, device=device)
    yg = torch.arange(NY, device=device, dtype=torch.float32).view(1, NY, 1)
    xg = torch.arange(NX, device=device, dtype=torch.float32).view(1, 1, NX)
    w1 = org[1] + xg * VOX_UNIT_HI
    w2 = org[2] + yg * VOX_UNIT_HI
    if slab % r:
        raise ValueError('slab must be a multiple of %d, not %r' % (r, slab))

    for z0 in range(0, NZ, slab):
        z1 = min(z0 + slab, NZ)
        nz = z1 - z0
        zg = torch.arange(z0, z1, device=device, dtype=torch.float32).view(-1, 1, 1)
        wp = torch.stack([(org[0] + zg * VOX_UNIT_HI).expand(nz, NY, NX).reshape(-1),
                          w1.expand(nz, NY, NX).reshape(-1),
                          w2.expand(nz, NY, NX).reshape(-1)], -1)
        cp = (wp - t) @ R
        del wp
        zc = cp[:, 2].contiguous()
        valid = zc > 1e-6
        u = torch.zeros_like(zc)
        v = torch.zeros_like(zc)
        u[valid] = fx * cp[valid, 0] / zc[valid] + cx
        v[valid] = fy * cp[valid, 1] / zc[valid] + cy
        del cp
        up = torch.round(u).long()
        vp = torch.round(v).long()
        del u, v
        ib = valid & (up >= 0) & (up < W) & (vp >= 0) & (vp < H)
        dm = torch.zeros_like(zc)
        dm[ib] = flat_depth[vp[ib] * W + up[ib]]
        del up, vp, valid
        known = ib & (dm >= 0.5) & (dm <= 8.0)
        del ib
        diff = dm - zc
        del dm, zc
        sign = torch.ones_like(diff)
        sign[known] = torch.where(diff[known].abs() < 1e-4,
                                  torch.ones_like(diff[known]),
                                  torch.sign(diff[known]))
        del diff
        tsdf = torch.ones_like(sign)
        tsdf[known] = sign[known]
        refined = f[z0:z1].reshape(-1)
        better = known & (refined < tsdf.abs())
        tsdf[better] = refined[better] * sign[better]
        tsdf[~known] = 0.0                 # at HIGH res, before the block mean
        kf = known.float()
        del sign, better, refined

        b = tsdf.view(nz, NY, NX).reshape(nz // r, r, NY // r, r, NX // r, r)
        tsdf_low[z0 // r:z1 // r] = b.mean(dim=(1, 3, 5))
        b = kf.view(nz, NY, NX).reshape(nz // r, r, NY // r, r, NX // r, r)
        w_low[z0 // r:z1 // r] = b.mean(dim=(1, 3, 5))
        del tsdf, known, kf, b

    return (tsdf_low.cpu().numpy().astype(np.float32),
            w_low.cpu().numpy().astype(np.float32))
