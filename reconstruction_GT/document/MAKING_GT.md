# Making ground truth for a D455 capture

CleanerS is evaluated against per-voxel ground truth that NYU ships and a
D455 does not. This file is how to hand-make that ground truth for a real
scene you captured yourself, in a format the existing evaluation code
consumes unmodified.

Companion to [README.md](../../inference/document/README.md) (the plumbing) and
[CAMERA_TUNING.md](../../inference/document/CAMERA_TUNING.md) (the capture knobs). Read those first;
this one assumes the grid conventions they establish.

## Contents

- [Status](#status)
- [The one thing you make](#the-one-thing-you-make)
- [The frame contract](#the-frame-contract)
- [What the evaluation actually scores](#what-the-evaluation-actually-scores)
  - [How SC is computed](#how-sc-is-computed)
  - [The scored set, measured](#the-scored-set-measured)
- [The GT format, decoded](#the-gt-format-decoded)
- [Where NYU's ground truth came from](#where-nyus-ground-truth-came-from)
  - [2D or 3D](#2d-or-3d)
- [Annotate solids, not surfaces](#annotate-solids-not-surfaces)
  - [Measured: objects are solid, structure is thin](#measured-objects-are-solid-structure-is-thin)
  - [The GT is not derived from depth](#the-gt-is-not-derived-from-depth)
  - [NYU vs NYUCAD](#nyu-vs-nyucad)
- [How others built CAD ground truth](#how-others-built-cad-ground-truth)
- [Approaches considered](#approaches-considered)
- [Procedure](#procedure)
- [The first room, and what it taught](#the-first-room-and-what-it-taught)
- [Grid contract](#grid-contract)
- [The downsample rule](#the-downsample-rule)
- [What to mark 255](#what-to-mark-255)
- [Verifying your toolchain](#verifying-your-toolchain)
- [Gotchas](#gotchas)
- [Scope and effort](#scope-and-effort)

---

## Status

The loop runs end to end. `room07` was captured, annotated, voxelized and
scored on 2026-09-24: **SSC mIoU 46.1** against NYU's 47.7, zero-shot on a
D455 capture with hand-made ground truth.

| step | tool | state |
| --- | --- | --- |
| 0. labeling guide | — | **not written** (decide `furn` vs `objs` before the next room) |
| 1. still capture | `inference/capture.py --preview` | done |
| 2. pin the grid | `inference/frame_loader.py` `live_cam_pose()` | done |
| 3. record the sweep | `reconstruction_GT/record_scan.py` | done, verified on a D455 |
| 4. fuse to a room mesh | `reconstruction_GT/fuse_scan.py` | done: room07 fused 918/934 frames, fitness 0.87 |
| 4b. correct the height | `reconstruction_GT/refine_height.sh` | done: floor fit set room07 to 1.053 m (tape read 1.15) |
| 5. place solids | `reconstruction_GT/box_editor.py` | done: 23 solids on room07, boxes / cylinders / wedges / CAD meshes |
| 6. export solids | — | not needed: the editor writes `gt/solids.json` |
| 7-9. voxelize, downsample, mask | `reconstruction_GT/voxelize_gt.py` | done, **bit-exact vs NYU on 8 frames** |
| 10. evaluate | `reconstruction_GT/evaluate_gt.py` | done, **reproduces NYU 75.0 / 47.7** |
| display | `reconstruction_GT/show_gt.py`, `inference/display_overlay.py` | done |
| verification | `reconstruction_GT/verify_voxelizer.py` | 9 checks, all passing |

The whole loop, for a new room:

```bash
./reconstruction_GT/record_room.sh room08              # capture + sweep + fuse
./reconstruction_GT/refine_height.sh room08            # floor -> camera_height
python -m reconstruction_GT.box_editor captures/room08 # annotate  (BOX_EDITOR.md)
python -m reconstruction_GT.voxelize_gt captures/room08
python -m inference.run_inference --cfg ./cfgs/NYU/voxelSSC.yaml \
    --pretrained_path ./checkpoint/CleanerS_ckpt.pth \
    --live_dir captures/room08 --out_dir ./outputs/room08
python -m reconstruction_GT.evaluate_gt captures/room08 --report
```

The three live-path gaps listed here before are gone: rather than teach
`run_inference.py` and `evaluate.py` about captures, `evaluate_gt.py` scores
the saved `.npy` predictions directly. `run_inference` still draws its `.ply`
with `--mask surface` for a live frame, which affects the picture only, never a
number.

**The camera must be on USB 3 and the udev rules installed.** On USB 2 a D455
records ~5 fps at 848x480, which is untrackable -- consecutive frames are too
far apart and every one is dropped for low fitness. Without
`/etc/udev/rules.d/99-realsense-libusb.rules` the accelerometer fails with
`Permission denied` on its sysfs nodes and the bag carries no gravity.
`1_check_camera.sh` checks both.

---

## The one thing you make

Everything CleanerS evaluates against derives from a single array:

```
a 240 x 144 x 240 volume of class ids, 2 cm voxels    (8,294,400 entries)
```

That is the whole hand-made artifact. Both arrays the evaluator reads —
`label3d` at 60x36x60 and the `label_weight` mask — are *computed* from it by
code that already exists and is verified to reproduce NYU's shipped files
bit-exactly.

```
  your 240^3 label volume ---+
                             +--> DownSampleLabel --> label3d (60,36,60)
  your capture's high-res ---+                        tsdf_low (60,36,60)
  TSDF (frame_loader)                                      |
                                                           v
                          label_weight = (label in 1..253) | (tsdf_low < -0.5)
```

You do not hand-make `label_weight`. Its second term comes from your own
capture's TSDF, so the occluded region is derived, not annotated.

---

## The frame contract

Ground truth is made **for one 640x480 capture frame**, the same file a normal
inference session consumes. Nothing here is scored against the sweep, the
fused mesh, or the sensor's native resolution.

```
captures/room06/
  depth/live_000000.png    640x480 uint16 PLAIN mm   <- the model input
  rgb/live_000000.png      640x480 BGR
  meta.json                cam_K (post-crop), camera_height, yaw, frames,
                           and frame_meta for frames shot from elsewhere
  scan/room.ply            annotation scaffolding only -- never scored
  gt/...                   what you make (layout below)
```

A capture may hold several frames. The still frame is the one measured with a
tape; frames lifted out of the sweep afterwards
(`reconstruction_GT/export_frame.py`) land in the same `depth/` and `rgb/`
under their own stem -- `live_000445` for bag frame 445, the same
`live_%06d` capture.py gives its own -- and state what differs about them
under `frame_meta` in `meta.json`:

```json
"frames": ["live_000000", "live_000445"],
"frame_meta": {"live_000445": {"camera_height": 1.0789,
                               "up_camera": [...], "cam_K": [[...]],
                               "world_from_room": [[4x4]]}}
```

`world_from_room` is the one thing that is not a measurement of that frame: it
is the tracked pose, and it exists so the annotation stays singular. The solids
are written once, in the room's world; `voxelize_gt` moves them through this
matrix into each frame's grid (`transform_solids`). Nothing is annotated twice,
and nothing about the still frame changes when a new frame is added.

`capture.py --fov nyu` resizes and crops the D455 frame to 640x480 at NYU's
intrinsics, and `meta.json` records the intrinsics that crop actually achieved.
Three of the four arrays at evaluation time come from **that PNG**, via
`FrameLoader.from_live_camera` exactly as `run_inference --live_dir` builds
them:

| array | source | depth-dependent |
| --- | --- | --- |
| `tsdf` (model input) | the 640x480 depth PNG | yes |
| `mapping` (SC's `== 307200` term) | the same PNG, unprojected | yes |
| `label_weight` second term (`tsdf_low < -0.5`) | the same PNG | yes |
| `label3d` | your solids, voxelized | **no** |

So: **the solids are per room; the GT files are per frame.** Re-voxelizing the
same solids against another frame's grid is cheap, but each frame gets its own
`label_weight`, because its occluded region is its own.

And the depth must be the **same file**, not a re-export. Recapture it, change
`--fov`, `--no_undistort`, `--filters` or `--depth_max`, and the TSDF, the
mapping and the mask all move with it; scores across such frames are not
comparable. Note `--hole_fill` in particular invents surface the sensor never
saw, which the encoder then treats as observed.

---

## What the evaluation actually scores

From `examples/segmentation/test_NYU.py:206-211`, in full:

```python
label_weightSSC = label_weight & (label3d != 255)
weightSC        = label_weight & (mapping == 307200) & (label3d != 255)
```

- **SSC** — 12-class confusion over `label_weightSSC`, mIoU averaged over
  classes 1..11.
- **SC** — binary occupied/empty over `weightSC`, which additionally requires
  `mapping == 307200` (unmapped, i.e. *not* directly seen by any pixel). SC is
  therefore a test of completion, not of perception.

`mapping` you already produce in `frame_loader.build_tsdf_and_mapping()`.

### How SC is computed

SC collapses the 12 classes to binary occupancy and scores only voxels the
network had to **infer**:

```python
weightSC  = label_weight & (mapping == 307200) & (label3d != 255)
pred_SC   = (pred_3d[weightSC].argmax(dim=1) > 0).long()
target_SC = (label3d[weightSC] > 0).long()
```

`mapping` is built by unprojecting each **pixel** into a voxel, so a voxel is
"mapped" when a directly measured surface point landed in it. Requiring
`mapping == 307200` therefore discards every directly observed surface voxel —
1.42% of the grid, measured — which is what makes this *completion* rather
than perception.

Reported numbers come from `cleaner/utils/metrics.py:133-140`, index `[1]`:

| CSV column | formula |
| --- | --- |
| `prec.` | `tp / predicted` |
| `recall` | `tp / actual` |
| `IoU` | `tp / (tp + fp + fn)` |

Everything is voxel-based. No mesh or point cloud is involved at evaluation
time — `label3d` and `label_weight` on the grid are the entire interface.

### The scored set, measured

Aggregated over 40 shipped frames (129,600 voxels each):

| | share of grid |
| --- | --- |
| `label_weight` | 46.04% |
| scored (`lw & label != 255`) | 17.07% |
| SC set (`+ mapping == 307200`) | 15.65% |
| of the SC set, occupied | 25.0% |
| occupied **and** mapped — dropped by SC | 1.42% |

SC set provenance: 608,168 voxels from the `tsdf_low < -0.5` term, 112,394
from `label > 0` alone, 90,475 from both. The occluded region supplies most
of it.

**The per-frame spread is enormous.** Over 200 frames:

```
scored %        min 0.4   p25  2.9   median  6.7   p75 20.2   max 56.4
SC set %        min 0.2   p25  2.2   median  5.8   p75 19.0   max 55.5
SC occupied %   min 4.3   p25 22.6   median 47.2   p75 81.3   max 100.0
```

Two consequences:

1. **Aggregate the confusion matrix across frames; never average per-frame
   IoUs.** `test_NYU.py` accumulates a single `cmSC_` over the whole loop. With
   only a handful of annotated scenes this matters more, not less — the
   occupancy rate inside the SC set ranges from 4% to 100%.
2. **Budget for the median, ~7%, but expect frames well outside it.** A frame
   looking at a bare wall scores almost nothing; one looking into a furnished
   corner scores half the grid.

**255 is not a failure state — it is most of the ground truth.**

One structural fact worth knowing: on every frame checked, **no occupied GT
voxel lies outside `label_weight`** (`label > 0 & ~lw` is 0). `label_weight`
is a superset of the annotation, widened by the occluded region.

---

## The GT format, decoded

NYU's ground truth lives inside the same `.bin` files `frame_loader` reads the
header of. `_load_bin_header` stops at byte 76; the rest is the label volume.

```
bytes    0-11   vox_origin    3  x float32
bytes   12-75   cam_pose     16  x float32   (row-major 4x4)
bytes   76-EOF  RLE payload   (value uint32, count uint32) pairs
```

The counts sum to exactly `240*144*240 = 8,294,400`. NYU0001 is 86,494 pairs
in 692,028 bytes.

Values live in a 37-entry class space and are collapsed to CleanerS's 12 by
`segmentation_class_map` (`../data/utils/preget_for_nyu_multiprocess.py:61`):

```python
segmentation_class_map = np.array([
    0,1,2,3,4,11,5,6,7,8,8,10,10,10,11,11,9,8,11,11,11,11,11,
    11,11,11,11,10,10,11,8,10,11,9,11,11,11])
```

255 bypasses the map and passes through as ignore.

### The 12 classes

| id | name | id | name |
| --- | --- | --- | --- |
| 0 | empty | 6 | bed |
| 1 | ceiling | 7 | sofa |
| 2 | floor | 8 | table |
| 3 | wall | 9 | tvs |
| 4 | window | 10 | furn |
| 5 | chair | 11 | objs |
| | | 255 | ignore |

You are annotating directly into this space, so the 37-class map is only
relevant if you want your files to be format-compatible with NYU's `.bin`.
Writing 0..11 and 255 straight into the volume is simpler and is what the
downsampler consumes.

**Write a labeling guide before you start**, specifically for `furn` (10) vs
`objs` (11). That boundary is genuinely ambiguous, and if you decide it case
by case your mIoU is measuring your annotation policy rather than the model.
NYU's working convention: `furn` is large storage and support pieces,
`objs` is everything portable sitting on top of them.

---

## Where NYU's ground truth came from

"NYU" in the SSC literature is not the NYU Depth v2 release. It is a
repackaging by later groups, and the 3D ground truth is theirs:

| stage | who | what |
| --- | --- | --- |
| NYU Depth v2 (2012) | Silberman et al. | RGB + Kinect v1 depth, **2D** annotation only: per-pixel labels (894 classes), instance masks, support relations, per-frame accelerometer. Holes filled by `fill_depth_colorization.m` (colorization inpainting) — an inpaint, not a measurement |
| 3D annotation (2015) | Guo, Zou & Hoiem | people fitted **CAD models and boxes** to each scene, plus the room layout |
| voxelization (2017) | SSCNet, Song et al. | the fits voxelized to 240x144x240 @ 2 cm, RLE'd into `depthbin/*.bin` with `vox_origin` and a gravity-aligned `cam_pose` |
| NYUCAD (2016) | Firman et al. | depth **re-rendered from the CAD fits** — same GT, clean input |
| data prep | 3D-Sketch / TorchSSC | `.bin` -> `Label/`, `TSDF/`, `Mapping/` npz, which this repo reads (`README.md`, "Data preparation") |

`../data/utils/preget_for_nyu.py:35` still points at
`/home/jason/sscnet-master/data/depthbin/`: the repo consumes SSCNet's files
directly.

The `.bin` world frame: gravity-aligned, floor at z = 0, camera at
`(0, 0, height)` (1.18-1.38 m over the first six frames), and
`vox_origin.z = -0.05` on every frame. NYU could gravity-align because it
recorded an accelerometer; the D455's IMU is the analogue.

### 2D or 3D

CleanerS uses **one** annotation: the 3D volume. Both of its losses come out of
it (`examples/segmentation/train_utils.py`):

```python
label2d = label3d[mapping2d[mapping2d != -1]]           # :76  2D target
weightSSC = label_weight & (label3d != 255)             # :84  3D mask
loss = loss2d * 0.25 + loss3d                           # :88
```

The 2D target is **manufactured**: each pixel takes the class of the voxel its
depth unprojects into. NYUv2's own 2D labels are never used, and are not even
downloaded here (`data/NYU/train` and `test` hold only `RGB/`). So for a D455
you make only the 3D volume; the 2D target, `label3d` and `label_weight` all
follow from it.

NYU and NYUCAD share that one volume. `data/NYUCAD/` has no `Label/`, and
`NYU.py:99` always reads labels from the NYU root; only the TSDF and depth are
swapped. NYUCAD's TSDF feeds the **teacher during distillation training**
only: `VoxelSSC.forward(img, mapping2d, tsdf=None, **kwargs)` swallows
`tsdf_CAD` at inference. For inference and evaluation, NYUCAD is not needed.

---

## Annotate solids, not surfaces

This is the whole method, and it is why fusing a point cloud cannot produce
NYU-equivalent GT.

NYU's annotators fitted **3D CAD models** to each object plus planes for the
room layout, then voxelized the *filled* models. That is why the GT contains
the back and the interior of every object — which is precisely what semantic
scene **completion** asks the network to predict.

A fused mesh only contains surfaces that some camera saw. Score against that
and you are testing the model on the one part of the task it did not have to
infer, while marking its correct completions as false positives.

**So your annotation primitive is a solid:** an oriented box, a closed room
shell, or a mesh from a small library. A box around a cabinet fills its unseen
interior. That is the target.

### Measured: objects are solid, structure is thin

Decoding the `.bin` volumes and counting, per class, the voxels whose six
neighbours share their class ("solidity": ~0 for a shell, high for a filled
body):

| NYU0005 | voxels | solidity |
| --- | --- | --- |
| floor | 22,901 | 0.00 |
| wall | 63,832 | 0.01 |
| chair | 6,187 | 0.44 |
| table | 16,622 | 0.32 |
| furn | 340,398 | 0.87 |
| objs | 63,887 | 0.61 |

NYU0001 agrees: floor 0.00, wall 0.01, chair 0.71, furn 0.81, objs 0.77. A
vertical section through NYU0005's largest `furn` object is a 0.96 x 0.94 m
block filled straight through, although the sensor only saw its front face.

- **Objects** are filled solids.
- **Floor, walls, ceiling** are thin slabs. The floor is **2 high-res voxels**
  (4 cm) thick, and always the same two layers: high-res layers 2 and 3 of the
  144-cell height axis, centres at z = 0.00 and 0.02 m, so the slab straddles
  the floor plane (NYU0001/0003/0005/0010/0020; NYU0002 has no floor in view).
  Build your floor slab the same way.

Why this rules out fused-surface GT: voxelize a fused mesh and that `furn`
object is a shell of ~13% of its 340,398 voxels. A model trained on NYU
predicts the solid block, and against a shell GT its IoU on that object is
capped near 0.13 however right it is.

### The GT is not derived from depth

Worth stating plainly, because the reverse is a natural assumption: depth is
the model **input**; the labels come from an independent CAD annotation. The
two are separate measurements of the same room, and they disagree.

Exactly which parts touch depth:

| | source | depth-dependent |
| --- | --- | --- |
| `label3d` | the RLE label volume (CAD) | **no** — `downSample__` computes `tsdf_downsample`, but the label vote reads only `readin_bin` |
| `label_weight` | `(label in 1..253) \| (tsdf_low < -0.5)` | **partly** — the occluded term is depth-derived |
| `mapping` | unprojection | yes |

So the annotation is depth-free; the evaluation mask is not. The same scene
with a different depth input yields the same `label3d` and a different
`label_weight`.

### NYU vs NYUCAD

The two datasets share their ground truth and differ only in the depth fed to
the model. On disk here, `../data/NYUCAD/` contains **only** `depth/` — no
`Label/` — because it reuses NYU's.

| frame 0001 | NYU | NYUCAD |
| --- | --- | --- |
| valid depth | 86.3% | 96.0% |
| range | 1.22-3.70 m | 1.66-3.64 m |
| PNG size | 147 KB | 45 KB |

NYUCAD's depth is the fitted CAD models **re-rendered** as a depth map: no
sensor dropouts, and smooth enough to compress to a third the size. It is the
perfect-sensor condition, where the input is consistent with the GT by
construction. Published NYUCAD scores run well above NYU scores for that
reason alone.

`cfgs/default.yaml:49` points at `data/NYU`, so this repo evaluates the real
sensor condition. That is the right comparison for a D455.

### Two consequences for your annotation

**1. NYU's own GT is not sub-centimetre accurate.** On commonly-valid pixels
of frame 0001, its real depth and its CAD GT disagree by **0.086 m mean,
0.038 m median** — about one 8 cm voxel. Do not chase precision your reference
does not have.

**2. Do not accidentally build NYUCAD.** If you annotate by tracing the fused
point cloud from the same capture you feed the model, your ground truth becomes
derived from the model's own input. That is the NYUCAD condition, and it
inflates every score — you would be testing whether the model can reproduce
what it was given. Fit solids to the **room**: real furniture dimensions, a
tape measure where you can reach, boxes sized to objects rather than hugging
the point cloud. The fused cloud is for placing solids, not for defining them.

---

## How others built CAD ground truth

Neither CleanerS nor Occ-ScanNet made its own GT. Occ-ScanNet (ISO, ECCV 2024)
crops per-frame 60x60x36 grids at 8 cm out of **CompleteScanNet**, then
inspected them by hand. CompleteScanNet comes from SCFusion (3DV 2020), whose
generator, `App/TrainingDataGenerator/exe_GroundtruthGenerator_ScanNetScan2CAD.cpp`
in [ShunChengWu/SCFusion](https://github.com/ShunChengWu/SCFusion), was read
for this section:

1. **Alignment is human.** Scan2CAD annotators picked a matching ShapeNet model
   per object and clicked keypoint pairs; the pose is the least-squares fit
   (translation, rotation, scale). SCFusion only applies it
   (`Scan2CADObjectLoader.cpp:94`).
2. **Label** by ShapeNet category -> NYU40 -> the 11 classes.
3. **Voxelize** each CAD mesh (a shell).
4. **Fill** with connected components (`--fill 1`, `FillObjWithCC`): the
   largest component is outside air, one is the shell, and every other enclosed
   component takes the object's class.
5. **Delete the scanned object**: scan points within 10 cm of any CAD point
   are removed, except ceiling, floor and wall.
6. **Merge** the remaining scan (mostly structure) with the filled objects.

Consequences: objects without a Scan2CAD alignment stay hollow and partial;
structure is scanned surface; a misaligned model erases real geometry within
10 cm of it.

The alignment tool is open too:
[skanti/Scan2CAD-Annotation-Webapp](https://github.com/skanti/Scan2CAD-Annotation-Webapp).
It is point-pair clicking, not dragging (Retriever -> KeypointAligner -> Kabsch,
`client/js/apps/KeypointAligner/controller/ProgressBarController.js:36`). Last
commit 2020-02-28, needs Node.js + MongoDB, ScanNet-style labeled meshes and a
thumbnailed CAD library. Not worth standing up for this; see step 5 for the
tools used instead.

---

## Approaches considered

| approach | verdict |
| --- | --- |
| **Fused surface as GT** — voxelize the multi-view mesh directly | rejected: objects become shells (~13% of NYU's solid `furn` voxels), and GT is derived from the model's own sensor — the NYUCAD condition |
| **Pseudo-GT** — lift a 2D segmenter's labels onto the fused mesh | rejected for scoring: another model's opinion. Usable only as a regression check between code versions |
| **Full CAD alignment** (Scan2CAD-style) | closest to NYU, but CAD retrieval plus keypoints per object is hours per room |
| **Boxes + room shell**, placed against the fused mesh | **chosen.** Solid like NYU. Measured on NYU's own objects, a box is right for `furn` (fills 87% of its bounding box) and `objs` (~100%), but a chair fills only 15% and a table 14% -- box those and their IoU is capped near 0.15. `box_editor.py`'s `C` key places a CAD mesh instead, which is what NYU did for its 6 furniture categories |

For small clutter, where box-fitting is fiddly and the voxel count low, marking
the region 255 is acceptable — see [What to mark 255](#what-to-mark-255).

---

## Procedure

End to end, for one room. Tools and their state are in [Status](#status).

### 0. Write a labeling guide

Before annotating anything: which class each kind of object gets, especially
`furn` (10) vs `objs` (11). NYU's working convention: `furn` is large storage
and support pieces, `objs` is everything portable on top of them. Decide it
once, in writing, or mIoU measures your policy rather than the model.

### 1. Take the still evaluation frame

```bash
python -m inference.capture --preview --out_dir captures/room02
```

Level the tripod (the grid has no pitch/roll term, Gotcha 5) and measure the
lens height with a tape. Type it into the preview; it goes into `meta.json`
as a measured fact and places the whole grid. Watch the depth pane for the
failures the preview warns about (min-Z dropout under 0.6 m, dark or
translucent surfaces).

### 2. Pin the grid

The frame's `vox_origin` and `cam_pose` come from
`FrameLoader.from_live_camera`, via `live_cam_pose()` in
`inference/frame_loader.py`:

```python
vox_origin = [-2.4, 0.0, -0.05]
cam_pose   = [[ cos(yaw), 0, -sin(yaw), 0             ],
              [ sin(yaw), 0,  cos(yaw), 0             ],
              [ 0,       -1,  0,        camera_height ],
              [ 0,        0,  0,        1             ]]
```

Your annotation lives in that world frame or it lines up with nothing. World
Z=0 is the floor; the grid is 4.8 m wide, 4.8 m deep, 2.88 m tall.

### 3. Record a sweep of the room

A single view has large occlusion holes and is unpleasant to annotate against,
so record more views and fuse them **purely as an annotation aid**. Straight
after step 1, into the same folder, **without touching the camera**:

```bash
python -m reconstruction_GT.record_scan --preview --out_dir captures/room02
```

It writes `scan/scan.bag` (raw depth, colour and accelerometer; 848x480 @
15 fps, ~30 MB/s) and `scan/record.json`. It never writes `meta.json`.

- **Hold still through the HOLD STILL countdown** (3 s, `--hold`). The mesh is
  anchored on a frame from the hold, so that frame must be the still frame's
  pose.
- Then lift the camera and sweep **slowly**, within ~3 m of surfaces (D455
  error is ~2% at 4 m). See every object from several sides.
- Point at the **floor** at some point: `fuse_scan`'s floor check needs it.
- Keep furniture or a corner in view. Tracking uses depth geometry only, and a
  view of one flat wall lets the camera slide undetected.
- q in the window, `--seconds`, or Ctrl-C stops it. It refuses to overwrite an
  existing bag.

### 4. Fuse the sweep into a room mesh

```bash
python -m reconstruction_GT.fuse_scan captures/room02
```

Open3D dense SLAM on the GPU: each frame is tracked against a raycast of the
model so far (frame-to-model, so drift does not compound), then integrated
into a TSDF at 1 cm, and meshed. It skips the first 1 s (auto-exposure), which
must be shorter than the hold; it refuses otherwise.

Writes to `scan/`:

| file | contents |
| --- | --- |
| `room.ply` | coloured mesh, **world frame** (`live_cam_pose` from the measured `camera_height`), metres |
| `trajectory.txt` | bag frame index + 4x4 `world_T_cam` per fused frame |
| `fuse_report.json` | frames fused and dropped, tracking fitness, parameters, checks |

Frames whose tracking fails (fitness < 0.1) or implies impossible hand motion
(> 1 m/s or > 120 deg/s) are dropped; 1 s of consecutive drops ends fusion and
the report says where. Without `meta.json` the mesh is written in first-frame
camera coordinates as `room_camera_frame.ply` instead.

**Read `fuse_report.json` before opening the mesh.** Three checks, all
warn-only — nothing is corrected or written back:

| check | question | warns at |
| --- | --- | --- |
| `still_match` | did the camera move between the still frame and the recording? | median depth difference > 3 cm |
| `imu_tilt` | was the camera level during the hold, as `live_cam_pose` assumes? | > 2 deg |
| `floor` | does the lowest large up-facing surface sit at z = 0? | > 4 cm or > 2 deg; meaningless if the floor was never seen (it is then furniture) |

A `still_match` warning means the mesh is offset from the grid: re-record from
the still pose. A floor warning with the floor in view means `camera_height`
or level is off: re-measure and recapture the still frame; do not edit
`meta.json`.

The mesh is scaffolding, not GT. Its colour-camera pinhole model ignores lens
distortion (~3 cm at frame corners), which is fine for placing boxes.

### 5. Place solids against the mesh

Open `scan/room.ply` in CloudCompare (installed) or Blender (not installed).

- **Room shell** — written for you, not annotated. `box_editor.py` fits it the
  way Guo & Hoiem annotated NYU: vote the wall normals into a Manhattan frame,
  then take the strongest plane near each edge of the cloud. Floor, ceiling and
  walls (classes 2, 1, 3) come out as **thin slabs**, 2 high-res voxels, the
  floor straddling z = 0 exactly as NYU's does (see
  [Measured](#measured-objects-are-solid-structure-is-thin)). The shell matters
  most for SC: it is what licenses writing 0 rather than 255 across occluded
  space. `--no_ceiling` leaves the ceiling out, as NYU does on the third of its
  frames that have none.
- **Each object** — an oriented box (or a CAD mesh), filled. Classes 4..11.
  Size boxes to the **object**, with a tape measure where you can reach, not
  hugging the point cloud; err small, since the downsample dilates (see
  [The downsample rule](#the-downsample-rule)).
- **Unsure regions** -> leave unannotated; they become 255.

Tools:

- **`box_editor.py`** — this repo's tool, and the one to use:
  `python -m reconstruction_GT.box_editor captures/room07` places boxes,
  cylinders and wedges in the fused cloud, one keypress per face, with the room
  shell fitted for you. Every key is in [BOX_EDITOR.md](BOX_EDITOR.md).
- **CloudCompare** — useful for *looking* at `scan/room_cloud.ply`, and
  `Tools -> Registration -> Align (point pairs picking)` fits a CAD mesh by
  clicked point pairs, the same method as Scan2CAD.

### 6. Export the solids

One JSON per room, in the world frame, read by `voxelize_gt.py` — its
docstring holds the schema:

```json
{
  "room": {"min": [-2.2, 0.35, 0.0], "max": [2.0, 4.4, 2.5], "thickness": 0.04},
  "solids": [
    {"class": "furn",  "center": [0.4, 2.1, 0.45], "size": [1.2, 0.6, 0.9], "yaw_deg": 12.0},
    {"class": "table", "center": [-0.8, 1.8, 0.37], "size": [1.0, 0.8, 0.74]},
    {"class": "sofa",  "mesh": "sofa_01.ply"}
  ]
}
```

`room` lays down the floor, ceiling and wall slabs (2, 1, 3) at `thickness`
metres, marks its interior 0 and everything outside 255. A `mesh` solid is
filled by ray-cast occupancy and must be closed.

Writing this by hand is fine for a few boxes. **Not written yet**: a ~20-line
Blender script that walks the scene objects and dumps the same file.

### 7. Voxelize to 240 x 144 x 240

For each voxel index, the world coordinate of its centre and its flat offset:

```python
# axis assignment (README: Coordinate conventions)
#   gz <- world X,  gx <- world Y (depth),  gy <- world Z (height)
world_X = vox_origin[0] + (gz + 0.5) * 0.02
world_Y = vox_origin[1] + (gx + 0.5) * 0.02
world_Z = vox_origin[2] + (gy + 0.5) * 0.02

flat = gz * 240 * 144 + gy * 240 + gx        # high-res
flat = gz *  60 *  36 + gy *  60 + gx        # low-res
```

Point-in-solid test per voxel centre gives the class id. Then:

- inside the room shell, inside no object -> **0** (empty)
- outside the room shell, or any region you chose not to annotate -> **255**

Write the result as `float32`, flat, length 8,294,400 — that is what
`DownSampleLabel` expects.

Boxes are solid by the point-in-box test. A CAD mesh is not: voxelize its
surface, then fill it SCFusion's way — connected components over the empty
voxels, where the largest component is outside air and every other enclosed
component takes the object's class. Unlike SCFusion, no scan geometry is
merged in, so its 10 cm scan-deletion step does not apply.

This is `reconstruction_GT/voxelize_gt.py`, which does steps 7 to 9 in one
command:

```bash
python -m reconstruction_GT.voxelize_gt captures/room06
```

It reads `gt/solids.json` and `meta.json`, rebuilds this frame's high-res TSDF
from its depth PNG, voxelizes the solids, downsamples, and writes the files in
step 9. Run [the verification](#verifying-your-toolchain) first.

Voxelize once per **frame**, using that frame's `vox_origin` and `cam_pose`
(step 2) -- not once per room. `label3d` itself never touches depth; the mask
built in step 8 does. See [The frame contract](#the-frame-contract).

A room's solids serve every still frame taken in it: re-voxelize against each
frame's grid.

### 8. Run the existing pipeline

```python
import numpy as np, DataProcess as dp

vox_size = np.array([240, 144, 240], dtype=np.float32)
SR = 4

label    = np.zeros(129600, dtype=np.float32)
tsdf_ds  = np.zeros(129600, dtype=np.float32)
dp.DownSampleLabel(hi_res_label, vox_size, SR, tsdf_hi, label, tsdf_ds)
label = label.astype(np.int32)

lw = np.zeros(label.shape, dtype=np.float32)
lw[np.abs(127 - label) < 127] = 1.0      # label in 1..253
lw[tsdf_ds < -0.5]            = 1.0      # the occluded region
```

`tsdf_hi` is the high-res TSDF for the same frame, flat float32, from
`frame_loader._build_high_res_tsdf()` (or `dp.TSDF`) -- built from
`captures/<room>/depth/<stem>.png`, the same 640x480 PNG the model is fed, via
`FrameLoader.from_live_camera` with `cam_K`, `camera_height` and `yaw` read
from that capture's `meta.json`.

`compute_label_weight` already exists at
`../data/generate_cleaners_data.py:220` if you would rather import it.

### 9. Write the files

| path | key | contents |
| --- | --- | --- |
| `Label/{id}.npz` | `arr_0` | `label3d`, int32, 129600, **255 retained** |
| `TSDF/{id}.npz` | `arr_0` | the model-input TSDF |
| | `arr_1` | `label_weight`, float32, 129600 |

`run_inference.py --mask label_weight` already reads `arr_1` from
`{tsdf_dir}/{id}.npz`, so this layout drops straight into the existing code
path, and an eval script mirrors `test_NYU.py:200-211` line for line.

For a capture rather than an NYU frame, keep that layout under the capture and
key it by the **frame stem**, so one folder holds everything about the frame:

```
captures/room06/gt/Label/live_000000.npz     arr_0  label3d, 255 retained
captures/room06/gt/TSDF/live_000000.npz      arr_0  model-input TSDF
                                             arr_1  label_weight
```

This needs the three live-path fixes in [Status](#status): the id is the whole
stem here, not `frame_name[3:7]`.

### 10. Evaluate

```bash
python -m inference.run_inference --cfg ./cfgs/NYU/voxelSSC.yaml \
    --pretrained_path ./checkpoint/CleanerS_ckpt.pth \
    --live_dir captures/room07 --out_dir ./outputs/room07
python -m reconstruction_GT.evaluate_gt captures/room07 --report
```

`evaluate_gt.py` is `test_NYU.py:206-211` over the saved `.npy` predictions,
accumulating one confusion matrix across frames and never averaging per-frame
IoUs. `--report` writes `gt/EVALUATION.md` beside the ground truth.

**`--nyu` is the check that it measures what the repo measures**: the same code
over NYU's 654-frame test split must print SC 75.0 / SSC 47.7. Run it whenever
the evaluator changes. It earned its keep the first time -- it caught the
evaluator scoring all 1449 frames instead of the test split, which reads
80.6 / 59.9.

**The frustum.** By default only voxels inside the camera's view are scored.
The grid is 4.8 m wide whatever the lens does, while your ground truth
describes the whole room, so a level camera at desk height annotates a ceiling
and a floor it never saw. NYU can ignore this -- 95-99.6% of its scored voxels
are in frame anyway -- but a small room shot from a corner cannot: on room07
only **37.5%** of the scored set was in view. `--fov all` scores everything
annotated (the Occ-ScanNet `target` rule). The two are not comparable, so say
which you used.

### 11. Look at it

```bash
python -m reconstruction_GT.show_gt captures/room07 --cloud   # GT over the sweep
python inference/display_overlay.py outputs/room07/ply/live_000000_gt.ply
python inference/display_overlay.py outputs/room07/ply/live_000000.ply
```

`show_gt` writes `gt/ply/<frame>_gt.ply` with the repo's own
`labeled_voxel2ply`, and a copy under `outputs/<room>/ply/` where
`display_overlay.py` resolves the capture by itself. `--cloud` is the check
worth doing: every coloured voxel should sit on or inside the grey surface it
came from.

---

## Grid contract

Repeated from README because getting it wrong is silent:

- Voxel grid 240x144x240 at 2 cm, downsampled 60x36x60 at 8 cm.
- **Array axis 1 (144 / 36 cells) is room height.**
- Axis assignment `gz <- world X`, `gx <- world Y` (depth),
  `gy <- world Z` (height). `vox_origin[0,1,2]` pairs with world `X, Y, Z`.
- Low-res arrays reshape to `(60, 36, 60)` = `(Z, Y, X)`.

Sanity check after voxelizing, before spending any more time:

```python
lab = label.reshape(60, 36, 60)
for cid, name in ((2, 'floor'), (1, 'ceiling')):
    z = np.where(lab == cid)[1] * 0.08
    print('%-8s %5d  %.2f-%.2f m' % (name, z.size, z.min(), z.max()))
```

Floor should sit in 0.00-0.08 m and ceiling near the room height. If they do
not, the grid is wrong and nothing downstream is worth looking at.

---

## The downsample rule

`downSample__` (`../data/utils/datautil.cu:166`) decides each output voxel from
its 4x4x4 = 64 high-res children:

```
count[0..11] = class tallies,  count[12] = tally of 255
tsdf_downsample = mean of the 64 TSDF values

if count[empty] + count[ignore] > 0.95 * 64:      # i.e. >= 61 of 64
    label = argmax over all 13 bins       (empty or 255 can win)
else:
    label = argmax over classes 1..11     (empty and 255 CANNOT win)
```

Read the `else` branch carefully: **once about 4 of the 64 sub-voxels carry a
class, the output voxel is guaranteed to be occupied.** Empty and ignore are
excluded from the vote entirely.

Two consequences:

1. **Your boxes do not need millimetre precision.** They need to be roughly
   the right size in roughly the right place. The GT is effectively dilated
   relative to the solids you draw.
2. **Thin annotation errors are amplified, not averaged away.** A box drawn
   5 cm too large adds a full 8 cm shell of occupied GT around the object.
   Err small rather than large.

---

## What to mark 255

Use 255 generously. It is 77-78% of NYU's grid.

| region | mark |
| --- | --- |
| inside the room shell, inside no object | 0 (empty) |
| inside an annotated solid | its class 1..11 |
| outside the room shell | 255 |
| inside the room but you did not annotate it | 255 |
| beyond where you can see well enough to judge | 255 |

The last row is the important one. If you cannot tell whether a region is
occupied, marking it 255 costs you a few scored voxels; guessing costs you a
wrong number you will later trust.

**But do not over-use it on free space.** 75% of the SC set is *empty*
voxels — they are genuine negatives, and SC precision is measured against
them. Mark the occluded volume 255 wholesale and you delete most of the
negatives, leaving a metric that is trivially easy to score well on.

This makes the **room shell the load-bearing annotation**. It is what licenses
you to write 0 rather than 255 across the occluded volume. Fit the walls,
floor and ceiling carefully even where you cannot see them well; for SC, the
box matters more than the objects inside it.

---

## Verifying your toolchain

One command, run before annotating anything and after any change to the grid
code:

```bash
python -m reconstruction_GT.verify_voxelizer
```

It checks `voxelize_gt.py` against data this repo did not write — NYU's shipped
`.bin`/`Label/`/`TSDF/`, and Occ-ScanNet's per-frame npz, which use a different
voxel size (8 cm) and a per-frame origin.

```
=== NYU (2 cm, (240, 144, 240)) ===
  A grid round-trip      PASS
  B solids round-trip    PASS  200 probes, 0 wrong count, 0 wrong voxel
  F axis control         PASS  observed surface that is GT-occupied: 0.15 correct
                               vs 0.03 axis-swapped
  C downsample kernel    PASS  8 frames match DataProcess.DownSampleLabel
  D label3d vs shipped   PASS  8 frames bit-exact
  E label_weight vs shipped PASS  8 frames bit-exact

=== Occ-ScanNet (8 cm, (60, 36, 60)) ===
  G grid round-trip      PASS  8 frames
  H solids round-trip    PASS  8 frames
  I floor in low layers  PASS  floor layer: min 0, median of medians 1.0

9 checks, 0 failed
```

**D and E are the ones that matter**: the shipped `Label/*.npz` and the
`label_weight` in `TSDF/*.npz` are reproduced entry for entry from the raw
`.bin`, which can only happen if the axis order, the half-voxel offsets, the
37-to-12 class map, the downsample vote and the mask rule are all right.

B and H are not circular: a one-voxel box is placed at a voxel's world centre
and must come back filling exactly that voxel, so index space is left for
metres and returned through the same oriented-box test real annotation uses.

F is the control for the failure that has no symptom: with the two horizontal
axes swapped, the GT stops agreeing with the depth the same frame produced.

---

## Gotchas

**1. Keep 255 in `Label/{id}.npz`.**
`preget_for_nyu_multiprocess.py:106` runs `label[label == 255] = 0` **after**
computing `label_weight`. The shipped `Label/` files nonetheless retain 255
(verified: `np.unique` on 0001 gives `[0 1 2 3 5 8 10 11 255]`), and the
evaluator depends on it — `label3d != 255` is what excludes unannotated voxels
from both metrics. Zero it and every unannotated voxel silently becomes a
scored "empty".

**2. `label_weight` must be computed before any 255 remapping.**
`np.abs(127 - label) < 127` selects 1..253. Run it on a volume where 255 has
already become 0 and you lose nothing directly, but the ordering is easy to
invert by accident when refactoring.

**3. The extension resolves its shared library relative to CWD.**
See above. It is the most common reason the toolchain check fails on a machine
where it previously worked.

**4. Your grid is not NYU's grid.**
NYU computes `vox_origin` per scene from the room extent — measured across the
first 12 frames, x ranges -0.9 to -4.7 and y ranges 0.06 to 0.98. The live path
uses a fixed `[-2.4, 0, -0.05]`. That is a real difference, and it means your
numbers are not directly comparable to published NYU results. They are
comparable across your own scenes and code changes, which is what you need.

**5. There is no pitch/roll term.**
`from_live_camera` assumes the camera is level. NYU's real frames are not:
measured tilt across the first 12 ranges from 0.31 to 7.65 degrees, and camera
heights from 1.18 to 1.38 m. If your tripod is not level, your GT and your
prediction disagree for a reason that has nothing to do with the model. Level
the tripod, and measure the lens height.

**6. Box GT is coarser than CAD GT.**
A box around a chair fills the space under the seat, which NYU's fitted CAD
model would leave empty. Expect your absolute mIoU to sit below published NYU
numbers for this reason alone.

**7. Do not use `getLabelWeight` from `../data/utils/reprojection.py`.**
It adds a *random* sample of observed-empty voxels capped at 2x the
foreground. NYU's shipped masks are not that: they carry 4.9x-38x more
background than foreground (frames 0001, 0003, 0010, 0100) and match the
deterministic `(label in 1..253) | (tsdf_low < -0.5)` rule bit-exactly. Use the
rule in step 8.

**8. The first fused frame is the evaluation pose.**
Anything that moves the camera between the still capture and the end of the
hold offsets the whole mesh from the grid. `fuse_scan`'s `still_match` check
catches it; the fix is re-recording, not adjusting.

**9. Estimates never go into `meta.json`.**
`camera_height` and `yaw` are measured facts. The fusion checks only warn; no
tool writes a fitted value back. If a check disagrees with a measurement,
re-measure.

**10. "Frame i" of a bag depends on how fast you read it.**
`o3d.t.io.RSBagReader` is a playback device: it drops frames when the consumer
is slower than the recording. Two reads of frame 150 of room08's bag, in one
process, gave 368598 and 338168 valid depth pixels -- not the same image. An
index is then only comparable between loops that happen to run at the same
speed, which is not a property anything can rely on: `sweep_eval.py` put a
network inside the loop and scored frames against other frames' poses, off by
22 SC points on a frame shot during fast motion.

Everything that reads a bag now goes through `reconstruction_GT/bag_reader.py`
(pyrealsense2, `playback.set_real_time(False)`), which delivers every frame
however slow the consumer is: `fuse_scan.py`, which writes the indices into
`trajectory.txt`, `export_frame.py`, which reads them back, and `sweep_eval.py`.
Its self-check is that a fast pass and a pass with 0.2 s of work per frame
agree on every frame. Two conventions differ between the libraries and mixing
them fuses nothing at all ("No block is touched in TSDF volume"): librealsense's
depth scale is metres per raw unit (0.001), Open3D's `integrate()` wants raw
units per metre (1000). `bag_metadata()` returns both, named apart.

A capture must be fused and read by the SAME reader -- the two disagree on how
many frames the bag holds (1065 against 1058 on room08), so indices written by
one do not address the other. A capture fused before this change has to be
re-fused, and any frame exported from it re-exported.

---

## Scope and effort

**Annotate the room, not the frame.** This is the multiplier. One room's solids
serve every capture taken in it — re-voxelize against each frame's grid and
every subsequent frame inherits GT for free. Thirty eval frames per annotation
session instead of one.

Rough budget:

| | cost |
| --- | --- |
| voxelizer + eval script | ~150 lines, once |
| toolchain verification | 20 min, once |
| per room: capture, sweep, fit solids | ~2 h |
| per additional frame in a known room | minutes |

Five to ten annotated rooms is a credible smoke test — enough to catch a broken
class or a systematic geometry error, not enough to publish a benchmark number.
Say so when you report it.

---

## The first room, and what it taught

`room07`, 2026-09-24: a bedroom, 23 solids, scored against the NYU protocol.

```
SSC mIoU 46.1   over the 8 classes present   (NYU test: 47.7)
SC  IoU  84.6                                 (NYU test: 75.0)
   window 85.9 | bed 75.2 | wall 54.2 | chair 49.6 | objs 45.4 | table 32.2
```

SSC landing within 2 points of NYU, zero-shot on a different sensor, a
different room and hand-made ground truth, is the headline. Three lessons came
with it, and all three are about the CAPTURE rather than the annotation:

**1. The SC number was worthless, and the evaluator says so.** 98% of room07's
SC set is occupied, so "predict occupied everywhere" scores IoU 0.981 -- better
than CleanerS's real 0.750 on NYU. SC scores only what the camera could not
see, and in a small room shot from the corner the only hidden volume left is
the inside of the solids you annotated, which is occupied by definition. NYU
frames run 10-76% occupied in that set; the trivial baseline there scores
0.10-0.33. **Capture for occlusion**: shoot from a doorway, or across a larger
room, so there is real empty space the camera cannot see.

**2. A level camera at desk height sees no floor and no ceiling.** With the
camera at 1.05 m and a 63 x 49 degree crop, the floor only enters the frame
beyond ~2.3 m and the ceiling beyond ~3.3 m -- past the far wall. Both were
annotated, neither was ever seen, and `floor` and `ceiling` ended up with 26
and 5 scored voxels. Their IoUs (13%) are noise, not measurements.

**3. The tape lost to the floor fit.** The measured camera height was 9.7 cm
out; three independent sweeps agreed on 1.08 m. See
[refine_height.sh](#status).

## What this does not measure

Worth stating alongside any number you produce:

- **Only the classes present in your rooms.** room07 has 8 of 11 -- no sofa,
  no tvs, no furn -- so its mIoU is not comparable to one over all eleven.
  Report the per-class table, not just the mean.
- **Only in-frustum, annotated volume.** 4.8% of the grid on room07.
- **Not completion, unless the SC set has empty space in it.** Check the
  occupied share the evaluator prints before quoting an SC number.
- **Not the crop decision.** The 63-degree crop is validated separately by the
  ablation in [CAMERA_TUNING.md](../../inference/document/CAMERA_TUNING.md#frame-size-and-fov--do-not-skip-the-crop).
- **Not depth accuracy.** GT here describes the scene, not the sensor — see
  [The GT is not derived from depth](#the-gt-is-not-derived-from-depth). A
  systematic depth bias moves prediction and GT together only if you annotated
  against that same biased cloud, which is the failure mode described under
  [NYU vs NYUCAD](#nyu-vs-nyucad).
