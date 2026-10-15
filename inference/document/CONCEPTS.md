# The six terms in HARDWARE.md, explained

[HARDWARE.md](HARDWARE.md) says what the fast path costs.
[README.md](README.md#tsdf_torchpy--the-fast-tsdf-encode) says how it is
written. This page answers what its words mean: why 12 voxels, what a 1-D pass
is, what slabbing and projection are, and what fp16 and AMP are.

Nothing here is new work. Every claim is in the code — `inference/tsdf_torch.py`
and `inference/frame_loader.py` — and this is the reading of it.

---

## 1. Why 12 voxels?

It is arithmetic on two constants already in `frame_loader.py`:

```python
VOX_UNIT_HI = 0.02      # 2 cm per high-res voxel
VOX_MARGIN  = 0.24      # 24 cm
search_region = int(round(VOX_MARGIN / VOX_UNIT_HI))   # -> 12
```

0.24 m / 0.02 m = **12 voxels**. That is the **T in TSDF — Truncated**. The
encoder stores, per voxel, how far away the nearest surface is and which side
of it the voxel lies on, but only out to 24 cm:

```python
refined = np.minimum(unsigned_dist / search_region, 1.0)
```

Divide the distance (in voxels) by 12, clamp at 1. A voxel 5 voxels from a
surface stores 5/12 = 0.42; a voxel 12 voxels away and a voxel 200 voxels away
both store 1.0.

**Why truncate at all.** Beyond 24 cm the exact distance carries no information
for completion — "far from anything" is the whole of what the network needs —
and clamping keeps the input bounded. It is SSCNet's value, inherited by
CleanerS, and the pretrained checkpoint was trained on that exact
normalisation, so it is not a tunable: changing it means retraining.

**The consequence that matters.** Nothing more than 12 voxels away can change
a voxel's value. That is what makes the fast method legitimate rather than an
approximation.

---

## 2. What "three 1-D passes" means

The job: for each of 240 x 144 x 240 = 8.29 M voxels, the distance to the
nearest surface voxel.

- Brute force, which is what SSCNet's kernel does: check all ~15,600 voxels in
  the +-12 box around each one.
- The trick: squared distance is **dx^2 + dy^2 + dz^2** — a sum of independent
  terms. So the minimum over the box can be taken one axis at a time.

Start with 0 at surface voxels and the sentinel ("infinity") everywhere else.
One dimension, a surface voxel at the left end:

```
start        0    inf  inf  inf
one pass     0    1    4    9      each cell: min over d of (left[i-d] + d^2)
sqrt         0    1    2    3      the distance, in voxels
```

Two dimensions, surface at the top-left corner:

```
after the x pass          then the y pass
0  1  4  9                0  1  4  9
.  .  .  .                1  2  5  10
.  .  .  .                4  5  8  13
```

Cell (1,1) came out 2: it took row 0's value of 1 and added 1^2. And
sqrt(2) = 1.414 is exactly its diagonal distance to the corner. Each pass
slides along **one** axis only — hence "1-D" — and three of them, z then y
then x, give the exact 3D answer.

**75 shifted comparisons per voxel instead of 15,600 lookups**, each one a
whole-array shift and minimum, which is the shape of work a GPU saturates.
`_edt1d` in `tsdf_torch.py` is one pass: 25 `torch.roll`s and a running
`torch.minimum`.

(The one subtlety: `roll` wraps, so the cells that come round from the far face
are blanked to the sentinel. Without that the grid would be a torus and a
surface at z=0 would cast distance into z=239.)

---

## 3. What slabbing is

The high-res grid is 8.29 M voxels, so **one float32 array over it is 33 MB** —
and the projection step needs about a dozen alive at once. The first working
version peaked at 779 MB.

Slabbing: do **24 z-layers at a time**, ten chunks in all. Each temporary is a
tenth the size, and as soon as a chunk is done it is reduced to its 60x36x60
contribution (the 4x4x4 block mean) and thrown away. The full-resolution
volume never exists.

| | peak |
| --- | ---: |
| full resolution, nothing fused | 779 MB |
| slab 24, fp32 | 192 MB |
| slab 24, fp16 *(the default)* | **97 MB** |

It works because **each voxel's projection does not depend on any other
voxel** — no neighbours, so chunk boundaries cannot matter and the result is
bit-identical. The distance transform is the opposite: it reaches +-12 voxels,
across slab boundaries, so it keeps one full array. That is why 97 MB is the
floor and not less.

---

## 4. What "projection" means here

Two opposite directions, and the encoder uses both:

| | direction | used for |
| --- | --- | --- |
| **un**projection | pixel + depth -> 3D point | which voxels a surface fills |
| **projection** | 3D point -> pixel | each voxel's sign |

The projection stage asks, of all 8.29 M voxel centres: *which pixel would
this voxel appear in?* Convert its world position into camera coordinates,
then

```
u = fx*X/Z + cx       v = fy*Y/Z + cy
```

Read the depth the camera measured at that pixel and compare it with the
voxel's own distance Z:

- voxel **nearer** than the measured surface -> free space, sign **+1**
- voxel **further** -> behind the surface, sign **-1** (this is where
  `tsdf < -0.5` meaning "occluded" comes from)
- lands on no valid pixel — behind the camera, outside the FOV, or a measured
  depth outside 0.5-8 m -> **unknown**, written as 0

It is the same perspective divide a renderer does, run backwards: from voxels
to pixels, to ask what the camera can say about each voxel.

The surface stamp is the other direction, and it is **not** interchangeable:
`_build_low_res_mapping` unprojects each pixel into the voxel it lands in. The
voxel-centre version of that table marks ~70% of the grid where the reference
marks ~2%, and agrees with it on only ~30% of voxels.

---

## 5. What fp32 and fp16 are

Number formats — how many bits per value:

| | bits | precision | largest | where it shows up |
| --- | ---: | --- | ---: | --- |
| fp64 | 64 | ~16 digits | huge | scipy's default |
| **fp32** | 32 | ~7 digits | 3.4e38 | the normal choice |
| **fp16** | 16 | ~3 digits | **65504** | half the bytes, tensor cores |

Both the gain and both the gotchas in this work come straight out of that
table:

- fp16 is **exact** for the distance transform, not a trade-off. Every value
  in it is a whole number no larger than 3 * 12^2 = 432, and fp16 represents
  every integer up to 2048 exactly. Verified: 0 differing voxels out of 8.29 M.
- the sentinel had to change. The numpy code uses **1e9** for "infinity", and
  1e9 > 65504, so in fp16 it overflows to infinity and the arithmetic breaks:
  `value cannot be converted to type at::Half without overflow`. 60000 works,
  and is still unreachable by any real squared distance.
- the handful of voxels where the torch encoder disagrees with the numpy one is
  **fp32 against scipy's fp64**, not the GPU — 2,047 of 84.76 M low-res voxels,
  max 0.068.

(An Ampere card running torch 1.10 was never doing pure fp32 convolutions
either: TF32 is on by default, which is a 19-bit mantissa.)

---

## 6. What AMP is

**A**utomatic **M**ixed **P**recision. One context manager, in
`model.predict`:

```python
with torch.cuda.amp.autocast(enabled=use_amp(device, amp)):
    pred_3d, _, _, _ = model(**data)
```

It intercepts each operation and picks the format per operation: convolutions
and matrix multiplies are cast to **fp16**, where tensor cores do the work
several times faster; sums, normalisations and softmax stay **fp32**, where
fp16 would lose too much. The weights themselves stay fp32 throughout.

- *Mixed* — different formats for different operations.
- *Automatic* — nothing in the model has to be annotated.

**Why it is `autocast` and not `model.half()`.** `half()` converts everything
blindly, and `cleaner/models/complete/unet3d.py:177` does

```python
zero = (segres == 0).float()
```

which forces fp32 back into the middle of a half graph; the next convolution
dies with *Input type (torch.cuda.FloatTensor) and weight type
(torch.cuda.HalfTensor) should be the same*. `autocast` handles that mismatch
by design, so the cast is a no-op rather than a crash.

Measured: the forward pass goes **~350 -> ~205 ms**, and about 1 voxel in 500
changes class — on all 654 NYU test frames, SSC mIoU 47.7288 -> 47.7285 and
SC 74.9528 -> 74.9530, the fourth decimal.

It is CUDA-only. `torch.cpu.amp` exists in 1.10 but is bfloat16-only and runs
into the same cast, which is why `--amp` resolves to off on a machine with no
card.
