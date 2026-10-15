# Hardware — what a frame costs, on what

Measured on one machine, the dev box in [SETUP.md](../../SETUP.md): RTX 3050 Ti
Laptop (**4 GB**, sm_86, ~192 GB/s), i7-12700H (14c/20t), driver 595,
torch 1.10.1+cu111. Figures from a second machine are welcome here.

[CONCEPTS.md](CONCEPTS.md) explains the terms this page uses — the 12-voxel
truncation, the 1-D passes, slabbing, projection, fp16, AMP — from the code.

Every number below comes from replaying `captures/room07` through the live path
at 640x480, one frame at a time, model loaded once, median of 17 frames after
3 warm-up. NYU replay reads and resizes PNGs on top of that and runs ~10%
slower.

The fast path is: TSDF encode in torch, slabbed, fp16 transform, plus AMP on
the forward pass. It is what `run_live` does by default on a machine with a
CUDA card, and it is two flags:

```bash
python -m inference.run_live ... --encoder auto --amp auto   # the default
python -m inference.run_live ... --encoder numpy --amp off   # the old path
```

`--encoder` picks the arithmetic (`auto`, `torch`, `numpy`, `cuda`) and `--amp`
the forward pass (`auto`, `on`, `off`). Both resolve to the old behaviour on a
machine without a card, so nothing gets slower anywhere. The batch tools
(`run_inference.py`) default the other way — `numpy` / `off` — their job is to
reproduce the published numbers exactly.

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

**A card is what buys the 4x, not the code.** The old bottleneck was a numpy
encode that never touched the GPU, which is why no card at all (0.33 fps) and a
4 GB card (0.75 fps) were within a factor of 2 of each other. On CPU the model
costs 2.03 s against 0.20, so the faster encoder changes nothing there — 0.33
to 0.35 fps — and there is no live display without a card.

## What changed, what it bought, why

| change | effect |
| --- | --- |
| encode in torch, not numpy/scipy | 1026 -> 153 ms |
| slab the projection, fuse the block mean | peak 779 -> 192 MB, same speed |
| the distance transform in fp16 | 153 -> **110 ms**, peak -> **97 MB** |
| AMP on the forward pass | 346 -> **201 ms** |

Why each one works:

- **torch encode.** The distance transform is truncated at 12 voxels
  (`VOX_MARGIN/VOX_UNIT_HI`), so only the +-12 box can matter — and a bounded
  box is separable: three 1-D passes of `min(f[i-d] + d^2)`, 25 shifts each,
  replace an exact float64 transform on the CPU. Exact inside the truncation
  radius.
- **Slabbing.** Projection is independent per voxel, so it can run a z-slab at
  a time and be block-meaned straight into the 60x36x60 result. The high-res
  grid is never held whole; only the distance field must be, since the
  transform reaches across slab boundaries.
- **fp16.** Squared distances inside the box are integers <= 3*12^2 = 432, and
  fp16 holds every integer to 2048 exactly, so the transform loses nothing.
  Only the sentinel changes: 1e9 overflows half, 60000 does not.
- **AMP.** fp16 tensor cores on the 3D convolutions. `model.half()` does not
  work — `unet3d.py:177` does `(segres == 0).float()`, forcing fp32 back into
  the graph — so it has to be `autocast`, which handles the mixed case.

Agreement with the validated numpy encoder: slabbing and fp16 are
**bit-identical** (0 of 8.29 M voxels); torch-vs-numpy differs on 2,047 of
84.76 M low-res voxels (0.0024%, max 0.068) from float32-vs-float64 and `floor`
boundaries; AMP moves about 1 voxel in 500 (measured end to end: 4,427 of
2.59 M predicted voxels over 20 frames, 0.171%).

**None of it moves the score.** NYU 654 test frames, `inference/evaluate.py`,
baseline bit-identical to `custom_visual_pred/` on 654/654:

| | SSC mIoU | SC IoU |
| --- | ---: | ---: |
| numpy + fp32 | 47.7285 | 74.9528 |
| torch encode + fp32 | 47.7288 | 74.9528 |
| torch encode + AMP | 47.7285 | 74.9530 |

## The live window

`--display` draws the prediction as it runs, in the same loop, so its cost is
simply added to the frame. Measured on the same replay, 12 frames:

| `--colour` | what it paints with | +ms | fps |
| --- | --- | ---: | ---: |
| *no window* | | 0 | 2.88 |
| `class` | the semantic colour | 4 | 2.54 |
| `zbuffer` | the photo pixel each voxel is front-most on | 15 | 2.65 |
| `all` | the three photo rules stacked | 110 | 2.10 |

The colour/depth panels beside it (`--no_panels` to drop them) cost a further
~15 ms, 2.75 -> 2.64 fps.

`all` cost 424 ms before this, which would have made it 1.3 fps. Its distance
query is now a scatter over the voxel lattice rather than a KD-tree search --
bit-identical, 4.3x faster -- which is what moved it into the budget. The mesh
itself is never the problem: 9,029 voxels build in 5 ms and recolour in 0.8.

## The ceiling

The 0.32 s of compute is four pieces, plus ~28 ms the source costs on top
(`run_live` measures 0.345 s a frame against the 0.317 s of staged work):

| piece | ms | on | shrinkable |
| --- | ---: | --- | --- |
| depth + RGB decode | ~28 | CPU | overlappable |
| TSDF transform + projection | 107 | GPU | yes |
| `mapping` + `mapping2d` | 9 | CPU | overlappable |
| model forward | **201** | GPU | **no** |

TSDFNet + Unet3d are 3D convolutions over the fixed 60x36x60 grid, 14.3 M
parameters, cost independent of the scene — and every frame is an independent
single-shot completion, so the whole forward runs each time with nothing to
reuse. That is 58% of the budget and not addressable here.

So ~0.21 s is an **asymptote, not a target**: what remains if the encode cost
vanished, which it cannot, since encode and model want the same GPU.
Pipelining improves utilisation, not totals. Reachable from 0.345 s is ~0.31 s,
by threading the 37 ms of CPU work. Past that needs TensorRT/ONNX, running the
model every Nth frame while the viewer draws at camera rate, a faster card, or
a world-fixed volume updating only what changed — and that last one needs the
localisation `run_live.py` already flags as missing.

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

| usable VRAM | AMP | fp32 | AMP time |
| ---: | --- | --- | ---: |
| uncapped (4 GB) | OK | OK | 363 ms |
| 1536 MB | OK | OK | 529 ms |
| 1280 MB | OK | **OOM** | 539 ms |
| 1152 MB | OK | **OOM** | 553 ms |
| 1024 MB | **OOM** | **OOM** | — |

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
--encoder cuda   -> datautil.cu; kept for comparison, 1101 ms (see below)
--amp auto       -> on when the model is on a CUDA device, off otherwise
```

`auto` stops at numpy on a CPU-only box rather than taking torch's 1.3x,
because that gain does not justify moving anyone off the validated path by
default, and it buys 6% of a 3-second frame; `--encoder torch` is there for
whoever wants it.

## Why the CUDA extension is not the fast path

| encoder | needs | ms |
| --- | --- | ---: |
| numpy | scipy | 1026 |
| `cuda` (`datautil.cu`) | a prebuilt `libdatautil.so` | 1101 |
| torch | nothing beyond torch | **116** |

Rebuilding the extension needs a system CUDA toolkit and gcc-12, hardcoded in
its `CMakeLists.txt`.

`SquaredDistanceTransform__` brute-forces the same +-12 box, ~15,600 candidates
a voxel, 8.29 M times, and scipy's exact O(N) transform beats it on a CPU. (The
285 ms recorded for this path in an earlier note does not reproduce.) The torch
encoder is therefore also the most portable of the three — anyone with an
NVIDIA card and no built extension gets the fast path from torch alone.

## Status

**Landed.** The encoder is `inference/tsdf_torch.py`, reached as
`FrameLoader(encoder='torch')`; `from_live_camera()` now takes `encoder=` and
`device=`, which it never did. `run_live` defaults to `--encoder auto --amp
auto`, `run_inference` to `--encoder numpy --amp off`, and `run_scannet`'s
existing `--encoder auto` now resolves to torch rather than to the kernels —
pass `--encoder cuda` there to reproduce the published Occ-ScanNet numbers on
the path that made them.

What `run_live` itself reports, 20 frames of room07, `--save none`, so including
the frame source the staged figures above leave out:

| | fps | encode | infer |
| --- | ---: | ---: | ---: |
| `--encoder numpy --amp off` | 0.71 | 1043 ms | 366 ms |
| `--encoder auto --amp auto` | **2.88** | **137 ms** | **200 ms** |

To re-measure on another machine:

```bash
python -m inference.run_live --cfg ./cfgs/NYU/voxelSSC.yaml \
    --pretrained_path ./checkpoint/CleanerS_ckpt.pth \
    --source folder:captures/room07/depth --save none --max_frames 20
```

It logs the resolved `encoder=` and `amp=` on the first line, then prints
encode and infer separately — the split that decides which row above a machine
belongs in.
