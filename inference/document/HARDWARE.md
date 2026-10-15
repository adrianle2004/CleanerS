# Hardware — what a frame costs, on what

Measured on one machine, the dev box in [SETUP.md](../../SETUP.md): RTX 3050 Ti
Laptop (**4 GB**, sm_86, ~192 GB/s), i7-12700H (14c/20t), driver 595,
torch 1.10.1+cu111. Figures from a second machine are welcome here.

Every number comes from replaying `captures/room07` through the live path at
640x480, one frame at a time, model loaded once, median of 17 frames after 3
warm-up. NYU replay reads and resizes PNGs on top of that and runs ~10% slower.

[CONCEPTS.md](CONCEPTS.md) explains the terms this page leans on — the 12-voxel
truncation, the 1-D passes, slabbing, projection, fp16, AMP — from the code.

## The fast path is two flags

```bash
python -m inference.run_live ... --encoder auto --amp auto   # the default
python -m inference.run_live ... --encoder numpy --amp off   # the old path
```

`--encoder` picks the arithmetic for the TSDF encode (`auto`, `torch`, `numpy`,
`cuda`); `--amp` picks the precision of the forward pass (`auto`, `on`, `off`).
Both resolve to the old behaviour on a machine without a card, so nothing gets
slower anywhere. The batch tools (`run_inference.py`) default the other way —
`numpy` / `off` — because their job is to reproduce the published numbers.

## Where the time went

A frame cost **1327 ms**, and **74% of it was the TSDF encode**: 982 ms of
numpy and scipy on the CPU, never once touching the GPU. That is why no card at
all (0.33 fps) and a 4 GB card (0.75 fps) came out within a factor of two of
each other — the card sat idle for three quarters of every frame.

So the job was never "make the model faster". It was: move the encode onto the
card, then shrink whatever is left over.

## A frame, part by part

| part | before | now | runs on |
| --- | ---: | ---: | --- |
| depth + RGB decode | ~28 ms | ~28 ms | CPU |
| TSDF encode | 982 ms | **116 ms** | CPU -> GPU |
| model forward | 346 ms | **201 ms** | GPU |
| `.ply` write | 83 ms | **0** | CPU |

### Depth + RGB decode — ~28 ms, unchanged

Read a 16-bit depth frame and an 8-bit colour frame, then crop and resize both
to the 640x480 NYU field of view (`match_nyu_fov`). Pure CPU, pure I/O and
memcpy. It is small, and it is the one part to overlap with the GPU work rather
than make faster — see [the ceiling](#the-ceiling).

### The TSDF encode — 982 -> 116 ms, the whole story

This part turns one depth image into the model's input: a 60x36x60 grid where
each cell holds the truncated signed distance to the nearest surface. Getting
there means building a 240x144x240 high-res grid (8.29 M voxels), finding for
every voxel its distance to the nearest occupied voxel, then block-meaning the
result down by 4 in each axis.

Nearly all of the cost is that distance search. Three things fixed it. The
figures in the three headings below are the encode stage timed on its own,
where the numpy reference measures 1026 ms; the 982 ms above is the same work
inside the end-to-end harness.

**1. Do the transform in torch, not scipy — 1026 -> 153 ms.** The distance
transform is *truncated* at 12 voxels (`VOX_MARGIN/VOX_UNIT_HI`), so only the
+-12 box around a voxel can ever matter. And squared Euclidean distance over a
bounded box is **separable**: `dx^2 + dy^2 + dz^2` is a sum of independent
terms, so the minimum over the whole box is three 1-D minimisations of
`min(f[i-d] + d^2)` run along z, then y, then x, 25 shifts each. That is **75
whole-array shifted minima on the GPU** in place of a per-voxel search, and it
is exact inside the truncation radius, not an approximation.

**2. Slab the projection and fuse the block mean — same speed, 779 -> 192 MB
peak.** Projection (camera pixel -> voxel) is independent per voxel, so it can
run one 24-layer z-slab at a time and each slab can be block-meaned straight
into its slice of the 60x36x60 result. The high-res grid is then never held
whole. Only the distance field has to be, because the transform reaches across
slab boundaries. This bought no time — it bought the ability to run on a 4 GB
card at all while the model is also resident.

**3. Run the transform in fp16 — 153 -> 110 ms, peak -> 97 MB.** Every value in
a truncated squared-distance field is a whole number no larger than
3 * 12^2 = 432, and fp16 represents every integer up to 2048 exactly. So half
precision here is not a trade-off, it is free: the result is bit-identical to
fp32. The only thing that had to change is the sentinel for "nothing found
yet" — 1e9 overflows half, 60000 does not.

The 116 ms of the encode stage is 107 ms of transform and projection on the
GPU plus 9 ms of the two mapping tables, `mapping` and `mapping2d`, which
record for every voxel which camera pixel it fell on. Those stay in numpy:
they are index bookkeeping rather than arithmetic, and 9 ms of CPU work is not
worth moving — it is also work that could be overlapped instead.

#### Why the prebuilt CUDA extension is not the fast path

| encoder | needs | ms |
| --- | --- | ---: |
| numpy | scipy | 1026 |
| `cuda` (`datautil.cu`) | a prebuilt `libdatautil.so` | 1101 |
| torch | nothing beyond torch | **116** |

The repo ships SSCNet's own CUDA kernels, and they were the obvious first
candidate. They are **slower than scipy**:
`SquaredDistanceTransform__` brute-forces the same +-12 box, ~15,600 candidates
per voxel, 8.29 M times, while scipy's exact O(N) transform does it on a CPU in
less. End to end the extension gives 1447 ms a frame, worse than numpy's 1327.
(An earlier note in `SETUP.md` claimed 285 ms for this path; it does not
reproduce, and the note is corrected.)

So the torch encoder is also the most portable of the three: rebuilding the
extension needs a system CUDA toolkit and gcc-12 hardcoded in its
`CMakeLists.txt`, whereas anyone with an NVIDIA card gets the fast path from
torch alone.

### The model forward — 346 -> 201 ms, with AMP

TSDFNet plus Unet3d, 14.3 M parameters of 3D convolution over the fixed
60x36x60 grid. `--amp auto` wraps the forward in `torch.cuda.amp.autocast`, so
the convolutions run on fp16 tensor cores while anything numerically touchy
stays fp32.

It has to be `autocast` and not `model.half()`: `unet3d.py:177` does
`(segres == 0).float()`, which forces an fp32 tensor back into a half graph and
breaks the next convolution. `autocast` is what handles that mixed case.

This is the part where precision reaches the output: about 1 voxel in 500
comes out differently, and the benchmark score does not move — see
[does it change the answer](#does-it-change-the-answer).

### Writing the result — 83 ms, now opt-in

A `.ply` per frame cost 83 ms, a fifth of the frame, for a file nobody reads
during a live run. `--save` now defaults to `none` and `ply/` and
`prediction/` are created on first write, so a run that saves nothing leaves no
directories behind. `--save ply|npy|both` and `--out_dir` bring it back.

## End to end

One harness, 20 frames, each change added to the one above it:

| configuration | encode | infer | frame | fps |
| --- | ---: | ---: | ---: | ---: |
| `numpy` / `amp off` | 982 | 346 | 1327 ms | 0.75 |
| `cuda` / `amp off` | 1101 | 346 | 1447 ms | 0.69 |
| `torch` / `amp off` | 117 | 346 | 463 ms | 2.16 |
| `torch` / `amp auto` | **116** | **201** | **317 ms** | **3.16** |
| ... plus a `.ply` | 116 | 202 | 400 ms | 2.50 |

**1327 -> 317 ms, 4.2x.** The encoder is 2.9x of that and AMP the remaining
1.5x. `run_live` itself reports 0.71 -> 2.88 fps over the same 20 frames, the
gap being the frame source the staged figures leave out.

## Speed, by hardware condition

| condition | per frame | fps |
| --- | ---: | ---: |
| *the old path, any machine* | *1.33 s* | *0.75* |
| no CUDA card | 3.06 s | 0.33 |
| CUDA, under 4 GB | 0.32-0.48 s | 2.1-3.2 |
| CUDA, 4 GB *(this box)* | **0.32 s** | **3.16** |
| ... plus the `.ply` write | 0.40 s | 2.50 |
| CUDA, faster than this box | >= 0.21 s | <= 4.8 |

Measured, except two rows. The under-4 GB row *fits* by measurement (allocator
capped on this card) but its speed is this card's compute, so read it as an
upper bound — a smaller card has less bandwidth too. The last row is
arithmetic, not a benchmark: see [the ceiling](#the-ceiling).

**A card is what buys the 4x, not the code.** On CPU the model costs 2.03 s
against 0.20, so the faster encoder barely registers there — 0.33 to 0.35 fps —
and there is no live display without a card either.

## The live window

`--display` draws the prediction as it runs, in the same loop, so its cost is
simply added to the frame. Measured on the same replay, 12 frames:

| `--colour` | what it paints with | +ms |
| --- | --- | ---: |
| *no window* | | 0 |
| `class` | the semantic colour | 4 |
| `zbuffer` | the photo pixel each voxel is front-most on | 15 |
| `all` | the three photo rules stacked | 110 |

The colour/depth panels beside it (`--no_panels` to drop them) cost a further
~15 ms, 2.75 -> 2.64 fps.

`all` cost 424 ms before this work, which would have capped the display at
1.3 fps. The same observation that fixed the encoder fixed it: its nearest-voxel
query is a Chebyshev cube on an integer lattice, so a point can reach at most
2^3 = 8 lattice sites — a bounded scatter, no KD-tree. **424 -> 98 ms,
bit-identical**, and `display_solid` inherited it (`points` 138 -> 34,
`surface` 215 -> 85). The mesh itself is never the problem: 9,029 voxels build
in 5 ms and recolour in 0.8.

## The ceiling

The 0.32 s of compute is four pieces, plus ~28 ms the source costs on top
(`run_live` measures 0.345 s a frame against the 0.317 s of staged work):

| piece | ms | shrinkable |
| --- | ---: | --- |
| depth + RGB decode | ~28 | overlappable |
| TSDF transform + projection | 107 | yes |
| `mapping` + `mapping2d` | 9 | overlappable |
| model forward | **201** | **no** |

The forward pass is 3D convolutions over a fixed grid, cost independent of the
scene, and every frame is an independent single-shot completion, so the whole
thing runs each time with nothing to reuse. That is 58% of the 0.345 s budget
(63% of the 0.317 s of compute) and not addressable here.

So ~0.21 s is an **asymptote, not a target**: what would remain if the encode
cost vanished, which it cannot, since encode and model want the same GPU.
Pipelining improves utilisation, not totals. Reachable from 0.345 s is ~0.31 s,
by threading the 37 ms of CPU work. Past that needs TensorRT/ONNX, running the
model every Nth frame while the viewer draws at camera rate, a faster card, or
a world-fixed volume updating only what changed — and that last one needs the
localisation `run_live.py` already flags as missing.

## Does it change the answer?

Per part: slabbing and fp16 are **bit-identical** to the plain fp32 torch
encode (0 of 8.29 M voxels differ), and the lattice scatter is bit-identical to
the KD-tree it replaced. torch-vs-numpy differs on 2,047 of 84.76 M low-res
voxels (0.0024%, max 0.068), from float32-vs-float64 and `floor` boundaries.
AMP moves about 1 voxel in 500 — end to end, 4,427 of 2.59 M predicted voxels
over 20 frames, 0.171%.

**None of it moves the score.** NYU 654 test frames, `inference/evaluate.py`,
baseline bit-identical to `custom_visual_pred/` on 654/654:

| | SSC mIoU | SC IoU |
| --- | ---: | ---: |
| numpy + fp32 | 47.7285 | 74.9528 |
| torch encode + fp32 | 47.7288 | 74.9528 |
| torch encode + AMP | 47.7285 | 74.9530 |

## Hardware gates

**VRAM.** The model sets the floor, not the encoder. The encoder's own peak,
measured with nothing else resident:

| encoder configuration | peak allocated |
| --- | ---: |
| slab 24, fp16 *(the default)* | **97 MB** |
| slab 24, fp32 | 192 MB |
| unslabbed, fp16 | 374 MB |
| the first prototype, nothing fused | 779 MB |

The whole pipeline, model included: **931 MB** allocated / 1238 MB reserved on
the fast path, against 1115 / 1666 on numpy + fp32.

How little it can be squeezed into, by capping this card's allocator, slabbed
encode, one frame:

| usable VRAM | AMP | fp32 |
| ---: | --- | --- |
| uncapped (4 GB) | OK, 363 ms | OK |
| 1536 MB | OK, 529 ms | OK |
| 1280 MB | OK, 539 ms | **OOM** |
| 1152 MB | OK, 553 ms | **OOM** |
| 1024 MB | **OOM** | **OOM** |

Floors ~**1.1 GB** with AMP, ~**1.4 GB** without. A card that only just clears
them pays ~50% in time from allocator churn — same code, same GPU, only the
memory it may cache changed. So on a small card AMP is not a speed flag, it is
what makes it fit.

**Architecture.** `get_arch_list()` is `sm_37 50 60 61 70 75 80 86 compute_37`.

| arch | status |
| --- | --- |
| sm_70-86 (Turing, Ampere) | measured here; AMP helps |
| sm_50-61 (Maxwell, Pascal) | encoder fine, **AMP expected to cost time** |
| sm_89/90/120 (RTX 40-50, Hopper) | **not supported by the pinned wheel** |
| sm_87 (Jetson Orin) | needs NVIDIA's own torch build |

Consumer Pascal runs fp16 at a fraction of its fp32 rate, so treat AMP as off
below sm_70 until someone times it. For sm_89 and up, only `compute_37` PTX
could JIT and this wheel's cudnn has no kernels for them: it needs a newer
torch, which means re-pinning mmcv-full 1.5.0 and mmseg as well.

Driver >= 450.80.02 for cu111; the conda build bundles its runtime, so no
system CUDA toolkit is needed to run.

**No CUDA.** The torch encoder is one function with a `device` argument, and on
CPU it is still 1.3x faster than numpy (786 vs 1026 ms). AMP is CUDA-only
(`torch.cpu.amp` bf16 hits the same `unet3d.py:177`), and the model is 2032 ms
on CPU against 201, so the frame goes 3.06 s -> 2.87 s and the display stays
out of reach. Apple Silicon and AMD fall here too: torch 1.10 predates MPS.

**Resolution order**, so no machine ends up slower than before:

```
--encoder auto   -> torch  when a CUDA card is visible (fp16 EDT, slab 24)
                 -> numpy  otherwise
--encoder torch  -> torch on whatever device there is; on CPU it uses an
                    fp32 transform, since there are no half kernels there
--encoder numpy  -> the reference encoder: what to check against
--encoder cuda   -> datautil.cu; kept for comparison, 1101 ms
--amp auto       -> on when the model is on a CUDA device, off otherwise
```

`auto` stops at numpy on a CPU-only box rather than taking torch's 1.3x,
because that gain does not justify moving anyone off the validated path by
default, and it buys 6% of a 3-second frame; `--encoder torch` is there for
whoever wants it.

## Status

**Landed.** The encoder is `inference/tsdf_torch.py`, reached as
`FrameLoader(encoder='torch')`; `from_live_camera()` now takes `encoder=` and
`device=`, which it never did. `run_live` defaults to `--encoder auto --amp
auto`, `run_inference` to `--encoder numpy --amp off`, and `run_scannet`'s
existing `--encoder auto` now resolves to torch rather than to the kernels —
pass `--encoder cuda` there to reproduce the published Occ-ScanNet numbers on
the path that made them.

To re-measure on another machine:

```bash
python -m inference.run_live --cfg ./cfgs/NYU/voxelSSC.yaml \
    --pretrained_path ./checkpoint/CleanerS_ckpt.pth \
    --source folder:captures/room07/depth --save none --max_frames 20
```

It logs the resolved `encoder=` and `amp=` on the first line, then prints
encode and infer separately — the split that decides which row above a machine
belongs in.
