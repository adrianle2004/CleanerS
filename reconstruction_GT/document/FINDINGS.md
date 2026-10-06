# What hand-made ground truth established

Two real rooms were captured with a D455, annotated by hand, voxelised NYU's
way and used to score CleanerS. This is the write-up of what that produced:
the numbers, the one methodological trap it exposed, and what the captures can
and cannot be used for.

Everything here is reproducible from the repo. Each section names the command.

- How the ground truth is made: [MAKING_GT.md](MAKING_GT.md)
- The tool reference: [README.md](README.md), [BOX_EDITOR.md](BOX_EDITOR.md)
- How to use the model that came out of it:
  [../../inference/document/USING_THE_MODEL.md](../../inference/document/USING_THE_MODEL.md)

---

## 1. What was built

| | room07 | room08 |
| --- | ---: | ---: |
| swept frames fused | 793 | 1050 |
| frames scored in the sweep (`--every 5`) | 159 | 210 |
| frames kept in full (still + 2 best + 2 worst) | 5 | 5 |
| annotated solids | 23 | 18 |
| still frame's camera height, from a floor-plane fit | 1.0858 m | 0.6436 m |
| median camera height over the sweep | 1.03 m | 1.06 m |
| floor tilt inside the grid | 0.01° | 0.00° |
| object voxels, share of the whole grid | 10.2% | 8.2% |

(The still was shot from a tripod at a fixed height; the sweep was hand-held,
so its median height differs — room08's still is the low one at 0.64 m.)

Both grids are 60×36×60 at 8 cm, `vox_origin = [-2.4, 0.0, -0.05]`, the same
geometry CleanerS trained on. Depth is aligned **into the colour camera** and
unprojected with the colour intrinsics, which is NYU's arrangement and the
reason the colour frame survives whole.

Every camera parameter is measured, never inferred: heights come from fitting
the fused floor (35,924 and 30,002 points), and the provenance string lives in
each `meta.json`. An SSC output is never fed back as a reference.

```bash
python -m reconstruction_GT.verify_voxelizer      # 9 checks, NYU bit-exact
python -m reconstruction_GT.evaluate_gt --nyu     # must print SC 75.0 / SSC 47.7
```

Both pass as of this write-up, which is what licenses everything below: the
voxeliser reproduces NYU's shipped label files bit-exactly, so an error in the
tables is the model's, not the pipeline's.

## 2. The scores

Pooled over the ten kept frames, NYU protocol, camera view only:

| class | IoU | precision | recall | mostly confused with |
| --- | ---: | ---: | ---: | --- |
| `ceiling` | 61.6 | 67.3% | 87.9% | `wall` (25) |
| `chair` | 58.9 | **96.2%** | 60.3% | `empty` (25) |
| `wall` | 56.1 | 85.5% | 62.0% | `empty` (1748) |
| `window` | 50.8 | 85.3% | 55.7% | `empty` (744) |
| `bed` | 44.6 | 80.5% | 50.0% | `empty` (3684) |
| `table` | 28.2 | 33.8% | 63.1% | `bed` (68) |
| `floor` | 21.8 | **21.8%** | 100.0% | — |
| `objs` | 15.8 | 58.9% | 17.8% | `bed` (837) |
| `furn` | 9.6 | 14.0% | 23.5% | `empty` (354) |
| `tvs` | 0.0 | 0.0% | 0.0% | `furn` (101) |
| `sofa` | 0.0 | 0.0% | — | predicted in rooms with no sofa |

Per room, and each room's own report has the full tables:

| | SSC | SC | SC baseline | margin |
| --- | ---: | ---: | ---: | ---: |
| room07 (5 frames) | 39.4 | 72.4 | 97% | **−24.6** |
| room08 (5 frames) | 35.5 | 59.2 | 71% | **−11.8** |
| both pooled (10 frames) | 34.8 | 66.9 | 85% | **−17.8** |

```bash
python -m reconstruction_GT.evaluate_gt captures/room08 --report
```

### The shape of the error: high precision, low recall

The structural classes sit at 80–96% precision and 50–62% recall. **When the
model says something is there it is usually right; when it says a voxel is
empty it is often wrong.** The single largest error mode is real structure
coming back as `empty` — 3,684 `bed` voxels, 1,748 `wall`, 744 `window`.

Treat a CleanerS occupancy map as a **lower bound**. Anything that depends on
free space being genuinely free — navigation, clearance, grasping — should
dilate the prediction, or read `empty` as *unknown* rather than as *free*.

Two outputs to distrust outright: `floor` at **21.8% precision with 100%
recall** paints floor into open space, and the furniture taxonomy
(`furn` 9.6, `tvs` 0.0, `sofa` 0.0) does not survive outside NYU's furniture —
`sofa` voxels were predicted in two rooms containing no sofa.

## 3. The trap: SC scores the task, not only the model

This is the finding that cost the most to establish and matters most for
anyone reusing these captures.

SC is occupied-vs-empty IoU over the voxels **no depth pixel reached** — the
space hidden behind the first surface. In a small room the camera sees nearly
all the free space, so what stays hidden is the inside of the bed and the
inside of the walls: all solid. When that set is 97% occupied, a model that
answers "occupied" for every voxel scores 0.97.

That is not an analogy, it is an identity. Predict occupied everywhere and
TP = P, FP = N − P, FN = 0, so **IoU = P/N — the occupancy of the scored set
is exactly the trivial baseline.** So the number worth quoting is the

> **margin = SC − occupancy of the SC set**

and `evaluate_gt.py` prints the occupancy with every SC number, warns above
90%, and gives the margin per frame in each room's report.

### It is a property of the annotation, not of the ground truth's quality

Growing room07's shell by 0.6 m — the same frames, the same prediction — moved
occupancy from 97.6% to 43.0% and SC from 87.1 to 44.0. Nothing about the
scene changed; the shell decides how much hidden volume exists to score.

### Where the two rooms land

| | frames with a positive margin | best margin | worst |
| --- | ---: | ---: | ---: |
| room07 | **0 of 159** | +0.0 | −43.2 |
| room08 | **32 of 210 (15%)** | **+25.1** | −54.5 |

**room07 cannot measure completion from any viewpoint.** Its least-occupied SC
set in the entire sweep is still 94.2% occupied. Quote its SSC and nothing else.

**room08 can, at a minority of viewpoints**, and those are real measurements,
not noise:

| room08 frame | SC | baseline | margin | hidden extent |
| --- | ---: | ---: | ---: | ---: |
| `live_000180` | 81.9 | 57% | **+25.1** | 0.29 m |
| `live_000580` | 60.3 | 36% | **+24.7** | 0.42 m |
| `live_000235` | 40.9 | 95% | −54.4 | 0.11 m |

For scale, NYU's own margins by band run **+12.6 to +23.2**, so room08's best
viewpoints are measuring completion as informatively as NYU does. Both are
among the frames kept in full, so they are usable as they stand.

The lesson is not "small rooms are useless". It is that **SC must be read next
to its baseline, frame by frame** — a pooled SC over a mixed set of viewpoints
hides the frames that could tell you something. Band medians hid these six.

## 4. Why the benchmarks disagree, measured the same way

The same quantities on NYU test (654 frames), an Occ-ScanNet val sample (1,500)
and both rooms:

```bash
python -m reconstruction_GT.sc_gap --all
```

All columns are medians over frames, SC included — so this SC is not the
pooled score of section 2, which weights every voxel equally:

| | camera height | median depth | hidden extent | of it solid | **empty hidden depth** | median SC |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| NYU | 1.34 m | 2.49 m | 1.72 m | 55% | 0.666 m | 78.9 |
| Occ-ScanNet | 1.43 m | 1.77 m | 1.64 m | 45% | 0.853 m | 54.6 |
| room08 | 1.06 m | 1.67 m | 0.36 m | 87% | 0.041 m | 58.6 |
| room07 | 1.03 m | 1.85 m | 0.40 m | 100% | 0.002 m | 87.2 |

**Distance is the variable, not height.** The relationship saturates rather
than being linear, with a knee near **1 m of hidden extent**: pooled over all
2,523 frames, occupancy of the SC set against hidden extent correlates −0.60
below 1 m and +0.01 above it. NYU (0.73–3.18 m) and Occ-ScanNet (0.63–3.41 m)
sit entirely above the knee; room07 (0.07–0.59 m) and room08 (0.01–0.89 m)
entirely below. One curve, seen from either side of its elbow — which is why
the same measurement looks decisive here and irrelevant on the benchmarks.

Hidden extent is **not** hidden *space*: it counts the inside of a wardrobe
exactly as it counts the air behind one. Multiply by (1 − occupancy) to get the
empty hidden depth, the part SC can actually test. room07 has **2 mm** of it
per ray against NYU's 666 mm.

### NYU 78.9 against Occ-ScanNet 54.6 is mostly not the model

Matching the two on the baseline and splitting SC into precision and recall —
which, unlike SC, do not move with difficulty:

| scored set occupied | NYU prec | recall | Occ-ScanNet prec | recall |
| --- | ---: | ---: | ---: | ---: |
| 0–20% | 90.2% | 87.2% | **55.8%** | 84.1% |
| 35–50% | 88.6% | 84.5% | 71.4% | 74.8% |
| 65–80% | 89.8% | 90.5% | 83.9% | 88.1% |
| 80–100% | 96.3% | 95.0% | **97.3%** | 96.6% |

Recall is comparable throughout; precision is not, and the precision gap
**closes to nothing** where the annotation is dense. That is the signature of
sparse ground truth: the model predicts furniture that is really in the room,
the GT does not carry it, and the prediction is charged for it. Occ-ScanNet's
median frame offers 1,617 occupied voxels to find against NYU's 4,881. Full
table: [../../inference/document/EVALUATION_SCANNET.md](../../inference/document/EVALUATION_SCANNET.md),
section 3.

## 5. What these captures are good for

| | SC (completion) | SSC (semantics) |
| --- | --- | --- |
| **room07** | no — 97% occupied at every viewpoint | **yes (39.4)** |
| **room08** | at its best viewpoints only (+25 margin) | **yes (35.5)** |
| NYU | yes (75.0) | yes (47.7) |
| Occ-ScanNet | floor-free (57.9), but it reads low | no (16.4 — GT too sparse) |

The per-class table in section 2 is the most useful thing these captures
produce, because the ground truth is dense — 10.2% and 8.2% of grid voxels
are objects, against 4.3% for NYU and 2.1% for Occ-ScanNet on the same
denominator (200-frame samples) — hand-annotated, floor present in every
frame, and level to 0.01°. An error in that table is the model's, not a
hole in the annotation — which cannot be said of Occ-ScanNet, where 14% of
frames have no floor at all and the floors that exist are tilted a median 1.94°
inside the grid.

## 6. How much the trajectory drifts, and why it does not matter

Full treatment, including the two ways this is easy to get wrong:
[DRIFT_AND_ALIGNMENT.md](DRIFT_AND_ALIGNMENT.md).

`fuse_scan.py` tracks frame-to-model with no loop closure, so the trajectory
drifts. Measured by running the identical tracker over the identical frames
forwards and then backwards — same geometry, opposite direction of
accumulation, so the two can only disagree by drift:

| | path | residual at the far end | % of path | at the scored frames |
| --- | ---: | ---: | ---: | ---: |
| room07 | 7.21 m | 174.9 mm / 0.28° | 2.42% | 66–80 mm |
| room08 | 14.40 m | 155.9 mm / 1.41° | 1.08% | 94–102 mm |

Ordinary for dense frame-to-model SLAM without loop closure; Intel's own
`rs-kinfu` is KinectFusion and cannot do better by construction. Nothing
integrates the IMU, so accelerometer error has no path into the trajectory at
all.

**That residual is not a distortion of the reconstruction, and it is the
reconstruction the ground truth is built in.** Tested annotation-free — a bent
reconstruction cannot keep a large plane flat — vertical planes fused from the
whole sweep come out flat to 6.7–17.1 mm, and fusing adds **at most 2.6 mm**
over a single frame's view of the same wall. Frame-to-model tracking anchors
each frame to the accumulated model, so error appears as a slow near-global
change of world frame rather than local warping, and everything the ground
truth does is relative, so a change of world frame cancels.

What is left between a frame and the annotation is **36–56 mm**, and that is
dominated by the annotation's own fidelity rather than by placement: a bed is
not a box.

It is corrected per frame, which is REP-105's split rather than a better
tracker: leave the drifting trajectory alone and give each scored frame a
correction in `world_from_room`, the role `map → odom` plays in ROS 2.

```bash
python -m reconstruction_GT.drift_check       captures/room08
python -m reconstruction_GT.refine_frame_pose captures/room08 --write
```

Effect on the scores: room07 SSC 39.4 → 39.5 and SC 72.4 → 72.4, room08 SSC
35.5 → 35.4 and SC 59.2 → 59.6. So pose error contributes about **±0.1** to the
room figures, and the correction is a polish rather than a fix.

Three traps, all documented in the companion doc because each pointed the
wrong way: perturbing the annotation by the *global* residual suggests ±4 SSC
of uncertainty (wrong premise — the error is not independent of the
annotation); *hit rate* looks like the natural alignment objective but rises
whenever the annotation is pushed toward the camera, burying surfaces inside
solids; and ICP *against the mesh* reads only 5–30 mm because the mesh near a
frame was fused from that frame's own temporal neighbours, so it is blind to
the accumulated bend by construction.

Used for what it is good for, hit rate shows room08's two worst frames are
**under-annotated** rather than misaligned: 40% of the surfaces `live_000235`
and `live_000240` measured fall in voxels annotated as empty. That is unmarked
clutter, and it is the real ceiling on those two frames.

## 7. What is not done

- **A room large enough for SC throughout.** room08 manages it at 15% of
  viewpoints; a room clearing ~1 m of hidden extent would manage it
  everywhere. The target from section 4: median depth around 2.5 m, camera
  near the height of what it looks at, furniture occluding furniture.
- **Occ-ScanNet's grid placement.** It puts every grid 5 cm below world
  *z = 0* rather than below the annotated floor, so its GT floor lands one or
  two layers up while the model paints floor in layer 0. Fixable without new
  data: re-place, re-encode, re-run.
- **Occ-ScanNet GT rebuilt NYU's way** from CompleteScanNet, which is what
  would separate its annotation recipe from the model for good. 14.9% of its GT
  object voxels sit where the depth sensor measured free space.
- **The `empty` class spread**: 9.0% IoU in room07 against 47.2% in room08.
  Worth understanding before leaning on either room's `empty` column.
- **Annotation coverage on the worst frames.** room08 `live_000235` / `240`
  have 40% of their measured surfaces falling in voxels annotated as empty —
  unmarked clutter. Marking it would raise those frames' ceiling; section 6
  shows it is not a pose problem.
- **Loop closure in fusion (option B).** The per-frame correction above fixes
  the scored frames, not the mesh. The proper fix for the reconstruction
  itself is Open3D's offline Reconstruction System, which is already in the
  installed wheel at `open3d/examples/reconstruction_system/` (Choi et al.,
  CVPR 2015): fragments, RGBD odometry inside each, then a pose graph whose
  edges are split into reliable odometry edges and `uncertain=True` loop
  closure edges, optimised with `GlobalOptimizationLevenbergMarquardt`. It
  ships a `config/realsense.json`, and `slac.py` additionally corrects the
  non-rigid "bent room" distortion that a rigid pose graph leaves behind.
  Worth doing when the next room is captured, since it rebuilds the mesh and
  the annotation would want re-checking against it — `refine_tilt.sh` and
  `refine_height.sh` already carry solids through `transform_solids`, so the
  annotation can follow a moved world.

## Reproducing all of it

```bash
python -m reconstruction_GT.verify_voxelizer                 # the pipeline is sound
python -m reconstruction_GT.evaluate_gt --nyu                # the evaluator is sound
python -m reconstruction_GT.evaluate_gt captures/room08 --report
python -m reconstruction_GT.sweep_eval   captures/room08 --every 5 --keep 2
python -m reconstruction_GT.sc_gap       --all                # the cross-dataset tables
```

Per-frame CSVs behind every comparison:

```
outputs/room07/sweep_eval.csv      outputs/room08/sweep_eval.csv
outputs/nyu_viewpoints.csv         outputs/scannet_viewpoints.csv
outputs/nyu_perframe.csv           outputs/sc_gap.csv
outputs/scannet/per_frame.csv
```

Long runs go under the guard, which survives a closed lid:

```bash
./reconstruction_GT/guard.sh sweep8 \
    python -u -m reconstruction_GT.sweep_eval captures/room08 --every 5
```
