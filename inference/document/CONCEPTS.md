# The terms the other pages lean on, explained

[HARDWARE.md](HARDWARE.md) says what the fast path costs.
[README.md](README.md#tsdf_torchpy--the-fast-tsdf-encode) says how it is
written. [WORLD_BUILDING.md](WORLD_BUILDING.md) says where the live room's
pose comes from. This page answers what their words mean: why 12 voxels,
what a 1-D pass is, what slabbing and projection are, what fp16 and AMP
are, and what a raycast is.

Nothing here is new work. Every claim is in the code — `inference/tsdf_torch.py`,
`inference/frame_loader.py` and `inference/tracker.py` — and this is the
reading of it.

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
`_edt1d` in `tsdf_torch.py` is one pass: 24 shifted minima and the array
itself.

Each of those minima is parallel over all 8.29 M voxels at once — that is the
whole point, and it is why the pass is memory-bound rather than
arithmetic-bound. The three passes are **not** parallel with each other and
cannot be: the y pass reads what the z pass wrote. Separability works by
applying them in sequence.

So the only thing that makes a pass faster is moving fewer bytes. The shift
is a `narrow` **view** rather than a `torch.roll` for exactly that reason:
roll copies the whole array and wraps, so the cells that come round from the
far face had to be blanked to the sentinel, or the grid would be a torus and
a surface at z=0 would cast distance into z=239. A view never writes the
cells a shift cannot reach, which is the same answer for half the traffic —
**60 -> 29 ms** for the three passes, bit-identical.

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
| slab 24, fp32 | 96 MB |
| slab 24, fp16 *(the default)* | **67 MB** |

It works because **each voxel's projection does not depend on any other
voxel** — no neighbours, so chunk boundaries cannot matter and the result is
bit-identical. The distance transform is the opposite: it reaches +-12 voxels,
across slab boundaries, so it keeps one full array. That is why 67 MB is the
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

---

## 7. What a raycast is

This one is not from [HARDWARE.md](HARDWARE.md) but from
[WORLD_BUILDING.md](WORLD_BUILDING.md), where the live room gets its pose.
It is the step that turns the accumulated volume back into something that
looks like a camera frame, and it is what makes frame-to-model tracking
possible at all: to compare this frame with the model, you need the model
*as a frame*.

**The word.** To *cast a ray* is to shoot a line out of the camera through
one pixel and ask what it hits first. Doing that for every pixel is
rendering — turning a stored 3-D thing into a 2-D image. So the raycast
runs the pipeline backwards: depth images went in to build the volume, and
a depth image comes back out of it, from any pose you care to ask about.

**Careful — two different truncations.** §1 above is the *encoder's* TSDF,
the network's input: 12 voxels of 2 cm, 24 cm, in `tsdf_torch.py`. The
volume being raycast here is the *tracker's*, a separate thing in open3d
with its own truncation of 8 voxels of 2 cm, 16 cm. Same idea, different
volume, different number.

**Why a ray can find a surface in it at all.** Each voxel holds a
truncated signed distance: how far to the nearest surface, and which side
of it this voxel is on. The surface is never stored as triangles or
points — it is the **zero level set**, the place where the stored number
changes sign. A ray finds it by watching for that change:

```
  camera
    ●──── the ray through one pixel ────────────────────────►

  stored:  +1.0     +1.0     +0.6     +0.2  │ −0.3     −1.0
           ░░░░     ░░░░     ▒▒▒▒     ▓▓▓▓  │ ████     ████
           far from a surface    closing in │  behind it
                                            ▲
                                            │
          the zero crossing, found by interpolating
          between the last + and the first −
```

**The naive version, and why nobody runs it.** March every ray from 0.1 m
to 3 m in 2 cm steps: 145 samples a pixel, 307 200 pixels, **44.5 M
lookups a frame**, nearly all of them in empty space. Three things cut
that down, and all three are visible in the call.

**Cut 1 — only consider the blocks in front of you.** The volume is a hash
map of 16³ blocks, not a dense array, and
`compute_unique_block_coordinates` returns just the blocks whose
projection lands inside this camera's frustum. Measured, 31 frames into
room07: **203 blocks of the 236 active**. Early in a sweep nearly
everything is still in view, which is the worst case for this cut; by the
end of a room it is a small fraction.

**Cut 2 — build a coarse map of how far to march.** Before any ray moves,
open3d projects those blocks and builds a **range map**: for each tile of
8×8 pixels, the nearest and farthest depth at which any block lies. It is
literally in the output of the call —

```
  render_attributes=['depth', 'vertex', 'normal']

    -> {'depth':  (480, 640, 1),
        'vertex': (480, 640, 3),
        'normal': (480, 640, 3),
        'range':  ( 60,  80, 2)}      <- 480/8 x 640/8, (near, far)
```

so each ray starts at its tile's near bound and stops at its far bound,
instead of sweeping the whole 0.1–3 m. What that is worth, measured on the
host with depth and normals rendered:

| `range_map_down_factor` | range map | cost |
| ---: | --- | ---: |
| 1 | 480×640 | 250.0 ms |
| 2 | 240×320 | 87.4 ms |
| **8** — the default, and ours | **60×80** | **24.9 ms** |
| 16 | 30×40 | 20.0 ms |

A tenfold difference out of nothing but knowing where to start and stop.
And once it exists, the nominal clip distance barely matters —
`depth_max` of 1 m costs 22.3 ms, 3 m costs 17.0 ms, 6 m costs 18.6 ms —
because it is the range map, not the clip, that decides how far a ray is
really marched.

**Cut 3 — the stored value tells you how big a step is safe.** This is
§1's truncation earning its keep a second time. If a voxel reads "0.16 m
from the nearest surface", the ray can jump 0.16 m in one stride with no
risk of passing through anything. So the truncation band doubles as the
step size, which is measurable: widen it and the march gets cheaper while
finding the same surfaces.

| truncation | band | cost | pixels filled |
| ---: | --- | ---: | ---: |
| 1 voxel | 0.02 m | 38.5 ms | 86.8% |
| 2 voxels | 0.04 m | 28.4 ms | 86.8% |
| 4 voxels | 0.08 m | 23.3 ms | 86.7% |
| **8 voxels — ours** | **0.16 m** | **19.2 ms** | **86.8%** |
| 16 voxels | 0.32 m | 19.4 ms | 86.7% |

Past 8 voxels it plateaus: the strides are no longer what the cost is made
of. Note the last column — the band changes the *speed*, not the answer.

**What comes out.** Three images, not one:

| | |
| --- | --- |
| `depth` | the distance to the zero crossing, per pixel |
| `vertex` | the same thing as a 3-D point — the unprojection already done |
| `normal` | the gradient of the stored field, normalised: which way the surface faces |

The ICP in [WORLD_BUILDING.md §10.1](WORLD_BUILDING.md#101-icp-six-numbers-from-a-6x6-system)
needs exactly `vertex` and `normal`, which is why it takes them from the
raycast rather than deriving them from the rendered depth.

**What a hole in it means.** Measured on room07: **94.5%** of pixels filled
after one frame, **85.8%** after twenty, 86.5% after thirty-one. It goes
*down*, and that is the interesting part. After one frame the raycast is
taken from the pose that just integrated that frame, so the model is
exactly what the camera saw. Once the camera moves, some rays cross space
nobody has looked at yet, find no sign change anywhere, and return
nothing. **An empty pixel in a raycast is not a failure, it is the model
saying "I have never seen this."**

**One knob where tracking and the mesh disagree.** The call takes a
`weight_threshold` — ignore voxels seen fewer than this many times:

```
  weight_threshold = −1   ->  96.3% filled      what the tracker uses
  weight_threshold =  3   ->  86.8% filled      what the mesh uses
  weight_threshold = 10   ->  83.1% filled
```

The mesh uses 3 (`Tracker.MIN_WEIGHT`) because one-frame depth noise in
something a person is looking at is just wrong. The tracker leaves it off,
so a voxel seen once is already a legitimate thing to track against. That
is the right trade for tracking — the model keeps up with the camera as it
moves into new space, and ICP's own 7 cm gate plus the fitness floor deal
with the noise — and the wrong one for a mesh on screen.

**Where it is called.** `inference/tracker.py`, at the end of every
`step()`, believed frame or not:

```python
self.model.synthesize_model_frame(self.raycast, self.scale, 0.1,
                                  self.depth_max, self.trunc, False)
```

— the frame to render into, the depth scale, a 0.1 m near clip, the 3 m
far clip, the 8-voxel truncation, and `enable_color=False`, because
tracking is depth-only geometry and rendering the colour would be work
thrown away.
