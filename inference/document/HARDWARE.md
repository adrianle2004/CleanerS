# Hardware — what a frame costs, on what

Measured on one machine, the dev box in [SETUP.md](../../SETUP.md): RTX 3050 Ti
Laptop (**4 GB**, sm_86, ~192 GB/s), i7-12700H (14c/20t), driver 595,
torch 1.10.1+cu111. Figures from a second machine are welcome here.

Most numbers come from replaying `captures/room07` through the live path at
640x480, one frame at a time, model loaded once, median of 20 frames after 3
warm-up -- the script's defaults. NYU replay reads and resizes PNGs on top of
that and runs ~10% slower.
[Speed by hardware condition](#speed-by-hardware-condition) is the exception:
those rows are a D455 on this desk, measured through `run_live` rather than
staged, because that is the question that row answers.

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

A frame cost **1262 ms**, and **73% of it was the TSDF encode**: 921 ms of
numpy and scipy on the CPU, never once touching the GPU. That is why no card at
all (0.35 fps) and a 4 GB card (0.79 fps) came out within a factor of two and a
bit of each other — the card sat idle for three quarters of every frame.

So the job was never "make the model faster". It was: move the encode onto the
card, then shrink whatever is left over.

## A frame, part by part

| part | before | now | runs on |
| --- | ---: | ---: | --- |
| depth + RGB decode | ~28 ms | ~28 ms | CPU |
| TSDF encode | 921 ms | **90 ms** | CPU -> GPU |
| model forward | 341 ms | **205 ms** | GPU |
| `.ply` write | 89 ms | **0** | CPU |

### Depth + RGB decode — ~28 ms, unchanged

Read a 16-bit depth frame and an 8-bit colour frame, then crop and resize both
to the 640x480 NYU field of view (`match_nyu_fov`). Pure CPU, pure I/O and
memcpy. It is small, and it is the one part to overlap with the GPU work rather
than make faster — see [the ceiling](#the-ceiling).

### The TSDF encode — 921 -> 90 ms, the whole story

This part turns one depth image into the model's input: a 60x36x60 grid where
each cell holds the truncated signed distance to the nearest surface. Getting
there means building a 240x144x240 high-res grid (8.29 M voxels), finding for
every voxel its distance to the nearest occupied voxel, then block-meaning the
result down by 4 in each axis.

Nearly all of the cost is that distance search. Four things fixed it. The
figures in the four headings below are the encode stage timed on its own,
where the numpy reference measures 975 ms; the 921 ms above is the same work
inside the end-to-end loop, on a CPU the loop has already warmed up.

**1. Do the transform in torch, not scipy — 975 -> 110 ms.** The distance
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

**3. Run the transform in fp16 — 110 -> 80 ms, and 96 -> 67 MB.** Every value
in a truncated squared-distance field is a whole number no larger than
3 * 12^2 = 432, and fp16 represents every integer up to 2048 exactly. So half
precision here is not a trade-off, it is free: the result is bit-identical to
fp32. The only thing that had to change is the sentinel for "nothing found
yet" — 1e9 overflows half, 60000 does not.

**4. Shift with a view, not `torch.roll` — 60 -> 29 ms for the three passes.**
`roll` copies the whole array and wraps, so the 12 cells that come round from
the far face had to be blanked to a sentinel or the grid would be a torus and
a surface at z=0 would cast distance into z=239. A `narrow` view never writes
the cells a shift cannot reach, which is the same answer for half the traffic;
and `d^2` is added to the whole array once per `d` rather than to each of the
two slices, one kernel launch fewer. **Bit-identical** — 10 of 10 room07
volumes byte-for-byte, through both encoders. It is also what took the
encoder's peak from 96 to 67 MB, since roll's full-array temporaries are gone.
On CPU the same change is worth more: the three passes go 415 -> 184 ms, and
the whole CPU encode stage 776 -> 363.

The 90 ms of the encode stage is 80 ms of transform and projection on the
GPU plus ~10 ms of the two mapping tables, `mapping` and `mapping2d`, which
record for every voxel which camera pixel it fell on. Those stay in numpy:
they are index bookkeeping rather than arithmetic, and 10 ms of CPU work is not
worth moving — it is also work that could be overlapped instead.

#### Why the prebuilt CUDA extension is not the fast path

| encoder | needs | ms |
| --- | --- | ---: |
| numpy | scipy | 975 |
| `cuda` (`datautil.cu`) | a prebuilt `libdatautil.so` | 1153 |
| torch | nothing beyond torch | **90** |

The repo ships SSCNet's own CUDA kernels, and they were the obvious first
candidate. They are **slower than scipy**:
`SquaredDistanceTransform__` brute-forces the same +-12 box, ~15,600 candidates
per voxel, 8.29 M times, while scipy's exact O(N) transform does it on a CPU in
less. End to end the extension gives 1446 ms a frame, worse than numpy's 1262.
(An earlier note in `SETUP.md` claimed 285 ms for this path; it does not
reproduce, and the note is corrected.)

So the torch encoder is also the most portable of the three: rebuilding the
extension needs a system CUDA toolkit and gcc-12 hardcoded in its
`CMakeLists.txt`, whereas anyone with an NVIDIA card gets the fast path from
torch alone.

### The model forward — 341 -> 205 ms, with AMP

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

A `.ply` per frame cost 89 ms, a fifth of the frame, for a file nobody reads
during a live run. `--save` now defaults to `none` and `ply/` and
`prediction/` are created on first write, so a run that saves nothing leaves no
directories behind. `--save ply|npy|both` and `--out_dir` bring it back.

## End to end

One harness, 20 frames, each change added to the one above it:

| configuration | encode | infer | frame | fps |
| --- | ---: | ---: | ---: | ---: |
| `numpy` / `amp off` | 921 | 341 | 1262 ms | 0.79 |
| `cuda` / `amp off` | 1100 | 346 | 1446 ms | 0.69 |
| `torch` / `amp off` | 89 | 390 | 479 ms | 2.09 |
| `torch` / `amp auto` | **90** | **231** | **331 ms** | **3.02** |
| ... plus a `.ply` | 88 | 236 | 413 ms | 2.42 |

**1262 -> 331 ms, 3.8x**, of which the encoder is 2.6x and AMP 1.4x.

Read the frame column with the model's drift in mind: the forward pass
measures **195-231 ms** across runs on this laptop, and the rows of this table
are timed in sequence in one process, so the later ones run on a hotter card.
Timed on its own with warm-up it is ~205 ms, and the cleanest whole-frame
figure is the uncapped row of [hardware gates](#hardware-gates), which gets a
fresh process: **291 ms**, against 321 ms before the encoder's fourth change.

Peak VRAM while doing it, model included: **930 MB** allocated on the fast
path against **1114 MB** on fp32 — the model dominates either way, and what a
card actually has to clear is in [hardware gates](#hardware-gates).

## Speed, by hardware condition

A D455 on this box, `run_live` end to end, 60 frames a row, started from a
cold card:

| live camera | per frame | fps |
| --- | ---: | ---: |
| *the old path, `--encoder numpy --amp off`* | *1.70 s* | *0.59* |
| **the defaults** | **0.35 s** | **2.87** |
| ... the same run, card now hot | 0.38 s | 2.65 |
| ... with the window, `--colour all` | 0.41 s | 2.41 |
| ... plus a `.ply` every frame | 0.48 s | 2.08 |

**4.9x**, and on the camera rather than a replay: the numpy encode costs
1268 ms a frame there against the fast path's 90.

Then what the hardware itself is worth:

| other conditions | per frame | fps |
| --- | ---: | ---: |
| no CUDA card, `--device cpu` | 2.27 s | 0.44 |
| ... the same, `--encoder numpy` | 2.72 s | 0.37 |
| **CUDA, 4 GB — this card** | **0.28 s** | **3.55** |
| ... the same card, `--amp off` | 0.42 s | 2.37 |
| ... the same card, capped to 1.0-1.5 GB | 0.29 s | 3.4 |
| CUDA, faster than this box | >= 0.21 s | <= 4.8 |
| *(for scale: replaying from disk, `run_live`)* | *0.30 s* | *3.28* |

These five are **staged**: encode plus forward pass on one frame in a fresh
process, no camera and no window, which is what makes them comparable to each
other and *not* to the live rows above. Only `faster than this box` is
arithmetic rather than a benchmark — see [the ceiling](#the-ceiling). The
italic row is `run_live` on a replay, there to bridge the two tables.

Three things to read with them:

- **A camera costs ~45 ms a frame over a replay**, which is why the same code
  does 3.28 fps from disk and 2.87 from the sensor: 34 ms of frame source
  against 4, plus align and undistort on a 1280x720 stream.
- **Capping the card to 1 GB costs nothing**: 0.29 s against 0.28 uncapped,
  inside the run-to-run spread. The encoder's 67 MB peak leaves the allocator
  enough room to stop churning, where at 96 MB the same 1 GB cap cost 2.1x.
  What a small card still costs you is fp32 — under ~1.3 GB it OOMs and AMP
  stops being a speed flag and becomes the thing that makes it fit.
- **Read every figure here as +-15%.** Cold, this card runs the defaults at
  2.87 fps; hot, 2.65; after an hour of sustained benchmarking, 2.1-2.3. Its
  clocks fall from 1725 to 1035 MHz of a possible 2100 and `nvidia-smi`
  reports SW power capping. A laptop GPU moves these numbers by more than
  most of the work on this page did.

**A card is still the single biggest factor, but the CPU case is no longer
hopeless.** The two CPU rows are the same machine with and without the torch
encoder — **2.72 -> 2.27 s, 0.37 to 0.44 fps**, a real 19%. The earlier
"barely registers" was written when that encoder was only 1.2x numpy on a
CPU, before the transform stopped copying the array for every shift; it is
2.5x there now.

What caps it is the forward pass, which `--what frame --device cpu` puts at
~1.8 s against 0.20 on the card — **84% of a CPU frame**. No amount of
encoder work reaches that. Comparing within one table rather than across
them: staged, the card is worth **8x** (0.44 -> 3.55); live, the code is
worth **4.9x** (0.59 -> 2.87). At 0.44 fps the live window is not practical
either, since `--colour all` would add its painting to a frame that already
takes two and a half seconds.

## The live window

`--display` draws the prediction as it runs, in the same loop, so its cost is
simply added to the frame. Measured on the same replay, 12 frames:

| `--colour` | what it paints with | paint |
| --- | --- | ---: |
| `class` | the semantic colour | ~0 ms |
| `zbuffer` | the photo pixel each voxel is front-most on | 10 ms |
| `all` | the three photo rules stacked | 129 ms |

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
| TSDF transform + projection | 80 | yes |
| `mapping` + `mapping2d` | ~10 | overlappable |
| model forward | **205** | **no** |

The forward pass is 3D convolutions over a fixed grid, cost independent of the
scene, and every frame is an independent single-shot completion, so the whole
thing runs each time with nothing to reuse. That is 64% of the ~0.32 s budget
(69% of the 0.29 s of compute) and not addressable here.

So ~0.21 s is an **asymptote, not a target**: what would remain if the encode
cost vanished, which it cannot, since encode and model want the same GPU — see
[parallelism](#parallelism) for the measurement that settles that. Past it
needs TensorRT/ONNX, running the model every Nth frame while the viewer draws
at camera rate, a faster card, or a world-fixed volume updating only what
changed — and that last one needs the localisation `run_live.py` already flags
as missing.

## Parallelism

Nothing in the pipeline overlaps anything: no threads, no CUDA streams, no
`multiprocessing` anywhere in `inference/`. The loop is decode, encode, model,
paint, draw, in that order, and the parallelism is all *inside* the GPU kernels
— the 75 shifted minima of the transform, the 3D convolutions on tensor cores
— plus torch's 14 intra-op threads, which only serve CPU tensors.

What overlapping would buy, each stage measured alone, then issued alongside
the model forward:

| the model forward, 203 ms, plus | alone | serial | together |
| --- | ---: | ---: | ---: |
| encode, on a second CUDA stream | 79 | 282 | 280 |
| the `mapping` tables, in a thread | 7 | 209 | 205 |
| `--colour all` painting, in a thread | 135 | 338 | **223** |

- **GPU against GPU is dead.** Two streams saved 2 ms of 79. The card is
  already saturated by one stage at a time, which is also why ~0.21 s is a
  floor rather than a target: there is no idle GPU for the encode to hide in.
- **CPU against GPU works, and the GIL is not in the way** — numpy and cv2
  release it, so a plain `threading.Thread` overlapped 92-100% of the CPU
  work. The `mapping` tables are only worth 6 ms, though. The frame source
  reads as ~28 ms here but on a live camera that is `wait_for_frames`
  *waiting for the sensor*, not CPU work to hide: the sensor is already
  producing the next frame while we work.
- **The painting is the one real win, and it is now taken.** `live_view`
  paints frame N-1 on a worker thread while the main thread encodes and
  infers frame N, then draws the mesh and the camera panels together from
  that same older frame — so the two halves of the window always agree, at
  the cost of the window running one frame behind. `--sync_display` turns it
  off. The window says what it got: `96 of it hidden behind inference`.

The mechanism is worth 28%, measured with no window in the way — the same
live camera, the paint inline against the paint on a worker:

| painter | encode | infer | wait | frame | fps |
| --- | ---: | ---: | ---: | ---: | ---: |
| inline | 87 | 199 | 91 | 377 ms | 2.65 |
| thread | 94 | 201 | **1** | 295 ms | **3.38** |
| process | 86 | 203 | 1 | 289 ms | 3.46 |

`infer` is unchanged across all three and a **process is no better than a
thread**, so the GIL is not a limit here and multiprocessing would buy
nothing.

With the window open, what the overlap buys is better stated as *the paint
stopped costing anything*, because end-to-end fps is no longer governed by
it. Interleaved runs, 60 frames each, live camera:

| round | no display | `class` | `all` |
| ---: | ---: | ---: | ---: |
| 1 | 3.0 | 2.6 | **2.6** |
| 2 | 2.9 | 2.6 | 2.2 |
| 3 | 2.3 | 2.2 | 2.1 |

In round 1, `--colour all` matches `--colour class` despite 96 ms more CPU
work a frame. That is the whole point, and it is the cleanest evidence
there is. What the rows below it show is the card: `infer` drifts 204 ->
300 ms across the three rounds whatever the configuration, `nvidia-smi`
reports SW power capping, and a 3050 Ti Laptop that boosts to 2100 MHz
settles far lower under sustained load. **Thermal state moves these numbers
by more than the optimisation does** — which is worth knowing before
reading any single fps figure on this page as a constant.

The window itself costs ~0.4 fps (3.0 -> 2.6 in a cool round), and the
overlap cannot touch that: open3d's GL rendering wants the same card as the
inference. On a bag replay, whose frame source is 4 ms against a camera's
34, the same change reads 2.32 -> 2.99 fps.

In `--colour class` the paint is a colormap lookup, so the overlap switches
itself off rather than buy nothing for a frame of lag.

That leaves the forward pass, now ~64% of the frame, untouched by any of
this.

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
| slab 24, fp16 *(the default)* | 80 | **67 MB** |
| slab 24, fp32 | 110 | 96 MB |
| whole volume, fp16 | 77 | 357 MB |
| whole volume, fp32 | 104 | 357 MB |

Unslabbed, both dtypes land on the same 357 MB: the projection buffers are
fp32 either way and they are then what dominates. The whole pipeline, model
included, peaks at **930 MB** allocated on the fast path against **1114 MB** on
numpy + fp32.

How little it fits in, one frame per cell with this card's allocator capped,
each in a process of its own:

| usable VRAM | AMP | fp32 |
| ---: | --- | --- |
| uncapped (3772 MB) | OK, 291 ms | OK, 431 ms |
| 1536 MB | OK, 290 ms | OK, 432 ms |
| 1280 MB | OK, 289 ms | OK, 434 ms |
| 1152 MB | OK, 289 ms | **OOM** |
| 1024 MB | OK, 291 ms | **OOM** |
| 896 MB | **OOM** | **OOM** |
| 768 MB | **OOM** | **OOM** |

Floors ~**1.0 GB** with AMP, ~**1.3 GB** without, and the cap now costs
nothing anywhere above them — 291 ms at 4 GB and at 1 GB alike. It used to
cost **2.1x** at 1 GB, all of it allocator churn; dropping the encoder's peak
from 96 to 67 MB left enough room that it stopped churning. So on a small card
AMP is not a speed flag, it is what makes it fit.

**VRAM with `--accumulate`**, which is the configuration that actually
gets close to the card. The model is not alone on it any more: the
tracker keeps a 2 cm TSDF of the room and the display window wants the
same GPU. Measured on this 4 GB card against the attached D455
(1280x720 at 30, cropped to NYU), reading `nvidia-smi` once a second:

| what is running | peak | track | mesh |
| --- | ---: | ---: | ---: |
| model + tracker, no window | 3 125 MB | 14 ms | 3 ms |
| model + tracker + window | **3 413 MB** | 14 ms | 3 ms |
| model + window, `--track_device cpu` | 2 473 MB | 55 ms | 20 ms |

683 MB of headroom with everything up, and it **plateaus**: 3 413 MB at
8 s in and 3 413 MB still at 40 s, because the tracker's volume is a
fixed `--track_blocks` and the accumulating map lives in host memory.
A long sweep does not creep up on the card. `--track_device cpu` is the
escape if something else wants the GPU, and it costs 41 ms a frame,
which against a ~480 ms frame is a tenth of the rate.

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

**No CUDA.** The torch encoder is one function with a `device` argument, and
on CPU it is now **2.5x** faster than numpy for the stage (373 vs 927 ms) —
the view change is worth more there than on the GPU, because a CPU pays full
price for every byte `roll` copied. AMP is CUDA-only (`torch.cpu.amp` bf16
hits the same `unet3d.py:177`), and the model is ~1.8 s on CPU against 0.20 —
84% of the frame — so the frame goes 2.72 s -> 2.27 s, 0.37 to 0.44 fps, and
the display stays out of reach. `--what frame --device cpu` measures this
row. Apple Silicon and AMD fall here too: torch 1.10 predates MPS.

**Resolution order**, so no machine ends up slower than before:

```
--encoder auto   -> torch  when a CUDA card is visible (fp16 EDT, slab 24)
                 -> numpy  otherwise
--encoder torch  -> torch on whatever device there is; on CPU it uses an
                    fp32 transform, since there are no half kernels there
--encoder numpy  -> the reference encoder: what to check against
--encoder cuda   -> datautil.cu; kept for comparison, 1153 ms
--amp auto       -> on when the model is on a CUDA device, off otherwise
```

`auto` stopped at numpy on a CPU-only box because torch's 1.2x there did not
justify moving anyone off the validated path by default. That reasoning is now
weaker — the stage is 2.6x faster on CPU and the frame 26% — but the rule
stands until someone wants it changed, and `--encoder torch` is there
meanwhile.

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
encode and infer separately. On this box: **0.75 fps** with `--encoder numpy
--amp off` (encode 964, infer 362) against **3.13** on the defaults (encode
106, infer 205), and 3.24 replaying the bag. The gap to the staged tables
above is the frame source those leave out.

The NYU scores are the three runs of `run_inference.py` named in
[does it change the answer](#does-it-change-the-answer), scored with
`inference/evaluate.py`; the benchmark script deliberately does not attempt
them, since they need the dataset rather than a frame.
