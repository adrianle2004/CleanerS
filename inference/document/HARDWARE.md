# Hardware — what a frame costs, on what

Measured on one machine, the dev box in [SETUP.md](../../SETUP.md): RTX 3050 Ti
Laptop (**4 GB**, sm_86, ~192 GB/s), i7-12700H (14c/20t), driver 595,
torch 1.10.1+cu111. Figures from a second machine are welcome here.

Every number comes from replaying `captures/room07` through the live path at
640x480, one frame at a time, model loaded once, median of 20 frames after 3
warm-up -- the script's defaults. NYU replay reads and resizes PNGs on top of
that and runs ~10% slower.

[CONCEPTS.md](CONCEPTS.md) explains the terms this page leans on — the 12-voxel
truncation, the 1-D passes, slabbing, projection, fp16, AMP — from the code.

## Reproducing this page

Every table below is one section of one script, so none of it has to be taken
on trust:

```bash
python -m inference.benchmark --pretrained_path ./checkpoint/CleanerS_ckpt.pth
python -m inference.benchmark --pretrained_path ... --what encode,agree
```

It prints markdown on stdout, a section per table, and needs nothing but the
checkpoint and the five room07 depth frames tracked in git -- cycled with
`--loop`, since cost per frame does not depend on the scene. `--what vram`
wants the card to itself: capped child processes cannot fit if the parent is
still holding the model. Its flags are in
[README.md](README.md#benchmarkpy--regenerate-hardwaremd).

The one thing it cannot do is the NYU 654-frame scores, which need the
dataset. [Status](#status) names the three runs behind those.

Absolute figures move 10-15% between runs on this machine: a sustained load
throttles a laptop GPU, and a hot CPU slows the numpy encoder by more than
that again. The ratios between rows are the stable part, and they are what
this page argues from.

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

A frame cost **1264 ms**, and **73% of it was the TSDF encode**: 926 ms of
numpy and scipy on the CPU, never once touching the GPU. That is why no card at
all (0.35 fps) and a 4 GB card (0.79 fps) came out within a factor of two and a
bit of each other — the card sat idle for three quarters of every frame.

So the job was never "make the model faster". It was: move the encode onto the
card, then shrink whatever is left over.

## A frame, part by part

| part | before | now | runs on |
| --- | ---: | ---: | --- |
| depth + RGB decode | ~28 ms | ~28 ms | CPU |
| TSDF encode | 926 ms | **118 ms** | CPU -> GPU |
| model forward | 348 ms | **205 ms** | GPU |
| `.ply` write | 82 ms | **0** | CPU |

### Depth + RGB decode — ~28 ms, unchanged

Read a 16-bit depth frame and an 8-bit colour frame, then crop and resize both
to the 640x480 NYU field of view (`match_nyu_fov`). Pure CPU, pure I/O and
memcpy. It is small, and it is the one part to overlap with the GPU work rather
than make faster — see [the ceiling](#the-ceiling).

### The TSDF encode — 926 -> 118 ms, the whole story

This part turns one depth image into the model's input: a 60x36x60 grid where
each cell holds the truncated signed distance to the nearest surface. Getting
there means building a 240x144x240 high-res grid (8.29 M voxels), finding for
every voxel its distance to the nearest occupied voxel, then block-meaning the
result down by 4 in each axis.

Nearly all of the cost is that distance search. Three things fixed it. The
figures in the three headings below are the encode stage timed on its own,
where the numpy reference measures 979 ms; the 926 ms above is the same work
inside the end-to-end loop, on a CPU the loop has already warmed up.

**1. Do the transform in torch, not scipy — 979 -> 153 ms.** The distance
transform is *truncated* at 12 voxels (`VOX_MARGIN/VOX_UNIT_HI`), so only the
+-12 box around a voxel can ever matter. And squared Euclidean distance over a
bounded box is **separable**: `dx^2 + dy^2 + dz^2` is a sum of independent
terms, so the minimum over the whole box is three 1-D minimisations of
`min(f[i-d] + d^2)` run along z, then y, then x, 25 shifts each. That is **75
whole-array shifted minima on the GPU** in place of a per-voxel search, and it
is exact inside the truncation radius, not an approximation.

**2. Slab the projection and fuse the block mean — 357 -> 96 MB peak, +3 ms.**
Projection (camera pixel -> voxel) is independent per voxel, so it can run one
24-layer z-slab at a time and each slab can be block-meaned straight into its
slice of the 60x36x60 result. The high-res grid is then never held whole. Only
the distance field has to be, because the transform reaches across slab
boundaries. Three milliseconds for a quarter of the memory — and that is what
lets the encoder run on a 4 GB card while the model is also resident. (An
early version that fused nothing peaked at 779 MB. It is not in the tree, so
the script cannot reproduce that one.)

**3. Run the transform in fp16 — 153 -> 110 ms, and 191 -> 96 MB.** Every value
in a truncated squared-distance field is a whole number no larger than
3 * 12^2 = 432, and fp16 represents every integer up to 2048 exactly. So half
precision here is not a trade-off, it is free: the result is bit-identical to
fp32. The only thing that had to change is the sentinel for "nothing found
yet" — 1e9 overflows half, 60000 does not.

The 118 ms of the encode stage is 110 ms of transform and projection on the
GPU plus ~8 ms of the two mapping tables, `mapping` and `mapping2d`, which
record for every voxel which camera pixel it fell on. Those stay in numpy:
they are index bookkeeping rather than arithmetic, and 8 ms of CPU work is not
worth moving — it is also work that could be overlapped instead.

#### Why the prebuilt CUDA extension is not the fast path

| encoder | needs | ms |
| --- | --- | ---: |
| numpy | scipy | 979 |
| `cuda` (`datautil.cu`) | a prebuilt `libdatautil.so` | 1133 |
| torch | nothing beyond torch | **118** |

The repo ships SSCNet's own CUDA kernels, and they were the obvious first
candidate. They are **slower than scipy**:
`SquaredDistanceTransform__` brute-forces the same +-12 box, ~15,600 candidates
per voxel, 8.29 M times, while scipy's exact O(N) transform does it on a CPU in
less. End to end the extension gives 1438 ms a frame, worse than numpy's 1264.
(An earlier note in `SETUP.md` claimed 285 ms for this path; it does not
reproduce, and the note is corrected.)

So the torch encoder is also the most portable of the three: rebuilding the
extension needs a system CUDA toolkit and gcc-12 hardcoded in its
`CMakeLists.txt`, whereas anyone with an NVIDIA card gets the fast path from
torch alone.

### The model forward — 348 -> 205 ms, with AMP

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

A `.ply` per frame cost 82 ms, a fifth of the frame, for a file nobody reads
during a live run. `--save` now defaults to `none` and `ply/` and
`prediction/` are created on first write, so a run that saves nothing leaves no
directories behind. `--save ply|npy|both` and `--out_dir` bring it back.

## End to end

One harness, 20 frames, each change added to the one above it:

| configuration | encode | infer | frame | fps |
| --- | ---: | ---: | ---: | ---: |
| `numpy` / `amp off` | 926 | 339 | 1264 ms | 0.79 |
| `cuda` / `amp off` | 1094 | 344 | 1438 ms | 0.70 |
| `torch` / `amp off` | 118 | 348 | 466 ms | 2.15 |
| `torch` / `amp auto` | **118** | **205** | **322 ms** | **3.10** |
| ... plus a `.ply` | 116 | 205 | 403 ms | 2.48 |

**1264 -> 322 ms, 3.9x.** The encoder is 2.7x of that and AMP the remaining
1.5x. Three runs of this table on this machine gave 3.7x, 3.8x and 3.9x, and
an earlier one 4.2x; the numpy encode is the variable part, being the row that
depends on how hot the CPU already is.

Peak VRAM while doing it, model included: **930 MB** allocated on the fast
path against **1114 MB** on fp32 — the model dominates either way, and what a
card actually has to clear is in [hardware gates](#hardware-gates).

## Speed, by hardware condition

| condition | per frame | fps |
| --- | ---: | ---: |
| *the old path, any machine* | *1.26 s* | *0.79* |
| no CUDA card | 2.76 s | 0.36 |
| CUDA, under 4 GB | 0.32-0.67 s | 1.5-3.1 |
| CUDA, 4 GB *(this box)* | **0.32 s** | **3.10** |
| ... plus the `.ply` write | 0.40 s | 2.48 |
| CUDA, faster than this box | >= 0.21 s | <= 4.8 |

Measured, except the last row. The under-4 GB row is this card with its
allocator capped, so it *fits* by measurement but runs at this card's speed and
bandwidth; read it as an upper bound, and note the 0.67 s end of it is what
being at the memory floor costs. The last row is arithmetic, not a benchmark:
see [the ceiling](#the-ceiling).

**A card is what buys the ~4x, not the code.** On CPU the model costs 1.95 s
against 0.21, so the faster encoder barely registers there — 0.35 to 0.36 fps —
and there is no live display without a card either.

## The live window

`--display` draws the prediction as it runs, in the same loop, so its cost is
simply added to the frame. Measured on the same replay, 12 frames:

| `--colour` | what it paints with | paint |
| --- | --- | ---: |
| `class` | the semantic colour | ~0 ms |
| `zbuffer` | the photo pixel each voxel is front-most on | 8 ms |
| `all` | the three photo rules stacked | 106 ms |

Painting is what differs between the modes; `class` is a colormap lookup and
costs nothing. On top of it every mode pays the same mesh build — 2 ms for a
median 3,148 voxels on these frames — plus the window redraw, which together
took the whole-window cost of `class` to 4 ms and `zbuffer` to 15 ms when
measured through `--display`. A run with `--display` prints that split itself
on exit (`display: N frames | colour .. ms | mesh+draw .. ms`). The
colour/depth panels beside it (`--no_panels` to drop them) cost a further
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
(`run_live` measures ~0.35 s a frame against the 0.32 s of staged work):

| piece | ms | shrinkable |
| --- | ---: | --- |
| depth + RGB decode | ~28 | overlappable |
| TSDF transform + projection | 110 | yes |
| `mapping` + `mapping2d` | ~8 | overlappable |
| model forward | **205** | **no** |

The forward pass is 3D convolutions over a fixed grid, cost independent of the
scene, and every frame is an independent single-shot completion, so the whole
thing runs each time with nothing to reuse. That is 59% of the ~0.35 s budget
(64% of the 0.32 s of compute) and not addressable here.

So ~0.21 s is an **asymptote, not a target**: what would remain if the encode
cost vanished, which it cannot, since encode and model want the same GPU.
Pipelining improves utilisation, not totals. Reachable from ~0.35 s is ~0.32 s,
by threading the 36 ms of CPU work. Past that needs TensorRT/ONNX, running the
model every Nth frame while the viewer draws at camera rate, a faster card, or
a world-fixed volume updating only what changed — and that last one needs the
localisation `run_live.py` already flags as missing.

## Does it change the answer?

Two of the four changes are **bit-identical** to the plain fp32 torch encode —
slabbing and fp16, 0 of 648,000 voxels over 5 frames — as is the lattice
scatter that replaced the viewer's KD-tree. The two that are not:

| comparison | > 1e-6 | of | worst |
| --- | ---: | ---: | ---: |
| torch vs numpy, NYU frames | 7 | 518,400 | 0.016 |
| torch vs numpy, a live grid | 4,938 | 648,000 | 0.078 |
| AMP vs fp32, predicted labels | 1,083 | 648,000 | 0.17% |

Count differences **above 1e-6**, not differences at all: float32 against
float64 moves about 4% of voxels in the last bit of a float32, which is
arithmetic noise and says nothing about either encoder. The script reports both
columns so the distinction cannot be quietly lost.

**Why a live grid disagrees more than NYU does.** Not the arithmetic — where
the grid sits. A live grid is floor-anchored, so its origin comes out round
(`[-2.4, 0, -0.05]` on these frames) against a 2 cm voxel, and depth arrives
quantised to the millimetre. Measured points therefore land exactly on voxel
faces in bulk, and float32 and float64 round such a tie to opposite sides.
Nudge the same grid 3.1 mm off the lattice and the disagreement falls from 775
voxels to 44; shift it exactly half a voxel and it rises to 1,927. The
magnitude never exceeds one step of the transform (1/12 = 0.083), which is what
a tie between two equidistant neighbours is worth: both answers are right, and
they are two spellings of the same distance.

AMP is the one change that reaches the output — about 1 voxel in 500.

**None of it moves the score.** NYU 654 test frames, `inference/evaluate.py`,
baseline bit-identical to `custom_visual_pred/` on 654/654:

| | SSC mIoU | SC IoU |
| --- | ---: | ---: |
| numpy + fp32 | 47.7285 | 74.9528 |
| torch encode + fp32 | 47.7288 | 74.9528 |
| torch encode + AMP | 47.7285 | 74.9530 |

## Hardware gates

**VRAM.** The model sets the floor, not the encoder. What the encode stage
allocates above whatever is already resident:

| encoder configuration | ms | peak |
| --- | ---: | ---: |
| slab 24, fp16 *(the default)* | 110 | **96 MB** |
| slab 24, fp32 | 153 | 191 MB |
| whole volume, fp16 | 107 | 357 MB |
| whole volume, fp32 | 150 | 357 MB |

Unslabbed, both dtypes land on the same 357 MB: the projection buffers are
fp32 either way and they are then what dominates. The whole pipeline, model
included, peaks at **930 MB** allocated on the fast path against **1114 MB** on
numpy + fp32.

How little it fits in, one frame per cell with this card's allocator capped,
each in a process of its own:

| usable VRAM | AMP | fp32 |
| ---: | --- | --- |
| uncapped (3772 MB) | OK, 321 ms | OK, 464 ms |
| 1536 MB | OK, 320 ms | OK, 462 ms |
| 1280 MB | OK, 320 ms | OK, 463 ms |
| 1152 MB | OK, 319 ms | **OOM** |
| 1024 MB | OK, 670 ms | **OOM** |
| 896 MB | **OOM** | **OOM** |
| 768 MB | **OOM** | **OOM** |

Floors ~**1.0 GB** with AMP, ~**1.3 GB** without. Above the floor the cap costs
nothing at all — 320 ms at 4 GB and at 1280 MB alike. At the floor it costs
**2.1x** (670 ms), all of it allocator churn: same code, same GPU, only how
much it may cache changed. So on a small card AMP is not a speed flag, it is
what makes it fit — and a card that merely clears the floor will not run at
these speeds.

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
CPU it is still 1.2x faster than numpy (776 vs 941 ms for the stage). AMP is
CUDA-only (`torch.cpu.amp` bf16 hits the same `unet3d.py:177`), and the model
is ~1.95 s on CPU against 0.21, so the frame goes 2.90 s -> 2.76 s — 0.35 to
0.36 fps — and the display stays out of reach. `--what frame --device cpu`
measures this row. Apple Silicon and AMD fall here too: torch 1.10 predates
MPS.

**Resolution order**, so no machine ends up slower than before:

```
--encoder auto   -> torch  when a CUDA card is visible (fp16 EDT, slab 24)
                 -> numpy  otherwise
--encoder torch  -> torch on whatever device there is; on CPU it uses an
                    fp32 transform, since there are no half kernels there
--encoder numpy  -> the reference encoder: what to check against
--encoder cuda   -> datautil.cu; kept for comparison, 1133 ms
--amp auto       -> on when the model is on a CUDA device, off otherwise
```

`auto` stops at numpy on a CPU-only box rather than taking torch's 1.2x,
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

To redo this page on another machine, see
[reproducing this page](#reproducing-this-page). For the one figure that
matters — which row of [speed by hardware
condition](#speed-by-hardware-condition) a machine is in — `run_live` answers
on its own:

```bash
python -m inference.run_live --cfg ./cfgs/NYU/voxelSSC.yaml \
    --pretrained_path ./checkpoint/CleanerS_ckpt.pth \
    --source folder:captures/room07/depth --loop \
    --save none --max_frames 20
```

`--loop` because only five depth frames of room07 are tracked in git; without
it the run stops at five, well short of `--max_frames`.

It logs the resolved `encoder=` and `amp=` on the first line, then prints
encode and infer separately. On this box: **0.76 fps** with `--encoder numpy
--amp off` (encode 956, infer 360) against **2.88** on the defaults (encode
136, infer 203). The gap to the staged tables above is the frame source those
leave out.

The NYU scores are the three runs of `run_inference.py` named in
[does it change the answer](#does-it-change-the-answer), scored with
`inference/evaluate.py`; the benchmark script deliberately does not attempt
them, since they need the dataset rather than a frame.
