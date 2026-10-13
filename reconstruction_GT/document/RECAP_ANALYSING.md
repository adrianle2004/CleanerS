# Recap: everything analysed, everything noted

One page over the whole ground-truth effort — what was measured, what it means,
what was wrong along the way, and what is left. Each section says which document
and which command carries the detail.

- The write-up: [FINDINGS.md](FINDINGS.md)
- How far the poses can be trusted: [DRIFT_AND_ALIGNMENT.md](DRIFT_AND_ALIGNMENT.md)
- How the ground truth is made: [MAKING_GT.md](MAKING_GT.md) · [README.md](README.md)
- When to believe the model: [../../inference/document/USING_THE_MODEL.md](../../inference/document/USING_THE_MODEL.md)
- Occ-ScanNet: [../../inference/document/EVALUATION_SCANNET.md](../../inference/document/EVALUATION_SCANNET.md)

---

## 1. The headline

| | SSC | SC | SC baseline | margin |
| --- | ---: | ---: | ---: | ---: |
| room07 (5 frames) | **39.5** | 72.4 | 97% | −24.6 ⚠ |
| room08 (5 frames) | **35.4** | 59.6 | 72% | −12.4 |
| NYU test (654) | 47.7 | 75.0 | 41% | +34 |
| Occ-ScanNet val (19764) | 16.4 | 54.6 | 34% | +20 |

**Quote SSC from the rooms. For SC, read the per-frame margin, never the pooled
number.** Both room reports carry that column.

```bash
python -m reconstruction_GT.evaluate_gt captures/room08 --report
```

## 2. The central finding: SC scores the task, not only the model

SC is occupied-vs-empty IoU over the voxels no depth pixel reached — the space
hidden behind the first surface. In a small room the camera sees nearly all the
free space, so what stays hidden is the inside of the bed and the inside of the
walls: all solid.

**The occupancy of that scored set is exactly what a constant "occupied"
prediction scores.** Predict occupied everywhere and TP = P, FP = N − P,
FN = 0, so IoU = P/N. An identity, not an analogy. So the number that carries
information is

> **margin = SC − occupancy of the scored set**

and `evaluate_gt.py` prints the occupancy with every SC, warning above 90%.

**It is a property of the annotation, not of GT quality.** Growing room07's
shell by 0.6 m — same frames, same predictions — moved occupancy from 97.6% to
43.0% and SC from 87.1 to 44.0. The shell decides how much hidden volume exists
to be scored. Growing it is not a legitimate fix: those voxels are outside the
room, and calling them empty would assert free space where there is a wall.

### Where the two rooms land

| | frames with positive margin | best | worst |
| --- | ---: | ---: | ---: |
| room07 | **0 of 159** | +0.0 | −43.2 |
| room08 | **32 of 210 (15%)** | **+25.1** | −54.5 |

- **room08 does measure completion**, at a minority of viewpoints, and those are
  real: `live_000180` SC 81.9 against a 57% baseline (**+25.1**),
  `live_000580` 60.3 against 36% (**+24.7**). NYU's own margins by band run
  +12.6…+23.2, so those frames are as informative as NYU is.
- **room07 cannot, from any viewpoint.** Its least-occupied scored set in the
  whole sweep is still 94.2% occupied, and its two *highest*-margin frames score
  a perfect SC 100.0 against a 100%-occupied set — the degeneracy at its purest.

**Noted as a mistake:** the first reading of this was "room08 is negative
everywhere", taken from band medians, which average the good viewpoints away.
Read SC frame by frame against its own baseline.

## 3. Why NYU and Occ-ScanNet disagree (78.9 vs 54.6)

Matched on the baseline and split into precision and recall, which — unlike SC —
do not move with difficulty:

| scored set occupied | NYU prec / rec | Occ-ScanNet prec / rec |
| --- | ---: | ---: |
| 0–20% | 90.2% / 87.2% | **55.8%** / 84.1% |
| 20–35% | 89.1% / 83.3% | 65.0% / 75.9% |
| 35–50% | 88.6% / 84.5% | 71.4% / 74.8% |
| 50–65% | 89.6% / 88.5% | 78.1% / 76.3% |
| 65–80% | 89.8% / 90.5% | 83.9% / 88.1% |
| 80–100% | 96.3% / 95.0% | **97.3%** / 96.6% |

**Recall is comparable throughout; precision is not, and the precision gap
closes to nothing where the annotation is dense.** That is the signature of
sparse ground truth, not of a worse prediction: the model predicts furniture
that is really in the room, Occ-ScanNet's GT does not carry it, and the
prediction is charged for it. Its median frame offers 1,617 occupied voxels to
find against NYU's 4,881.

```bash
python -m reconstruction_GT.sc_gap --all
```

## 4. The saturating curve, and what it means for capture

One curve, four datasets, with a knee at about **1 m of hidden extent**. Pooled
over all 2,523 frames, occupancy of the scored set against hidden extent
correlates **−0.60 below 1 m** and **+0.01 above** it.

| | camera height | median depth | hidden extent | of it solid | **empty hidden depth** |
| --- | ---: | ---: | ---: | ---: | ---: |
| NYU | 1.34 m | 2.49 m | 1.72 m | 55% | 0.666 m |
| Occ-ScanNet | 1.43 m | 1.77 m | 1.64 m | 45% | 0.853 m |
| room08 | 1.06 m | 1.67 m | 0.36 m | 87% | 0.041 m |
| room07 | 1.03 m | 1.85 m | 0.40 m | 100% | **0.002 m** |

NYU (0.73–3.18 m) and Occ-ScanNet (0.63–3.41 m) sit entirely above the knee;
room07 (0.07–0.59 m) and room08 (0.01–0.89 m) entirely below. One relationship
seen from either side of its elbow — which is why the same measurement looks
decisive on our rooms and irrelevant on the benchmarks.

**Distance is the variable, not height.** Hidden extent is **not** hidden
*space*: it counts the inside of a wardrobe exactly as it counts the air behind
one. Multiply by (1 − occupancy) for the part SC can actually test — room07 has
**2 mm** of it per ray against NYU's 666 mm.

So the capture target is whatever clears ~1 m of hidden extent; ~2.5 m of median
depth is simply how NYU gets there.

## 5. What the model does, on real captures

Pooled over the ten annotated frames, NYU protocol, camera view only.

**The shape of it: high precision, low recall.** Structural classes sit at
80–96% precision and 50–62% recall. When the model says something is there it is
usually right; when it says a voxel is empty it is often wrong. The largest
error mode is real structure returning as `empty` — 3,684 `bed` voxels, 1,748
`wall`, 744 `window`.

**Treat a CleanerS occupancy map as a lower bound.** For anything depending on
free space genuinely being free — navigation, clearance, grasping — dilate the
prediction, or read `empty` as *unknown* rather than *free*.

**Two outputs to distrust outright:**

- `floor`, at 21.8% precision with 100% recall — it finds every floor voxel and
  claims about three times too many. Useful as "the floor is somewhere here",
  useless as a boundary.
- the furniture taxonomy: `furn` 9.6, `tvs` 0.0, `sofa` 0.0. It predicted 1,080
  `sofa` voxels in two rooms containing no sofa and missed all 258 `tvs`.

## 6. Pose quality — and three measurements that were wrong

The trajectory drifts **156–175 mm** per sweep (1–2.4% of path length), measured
by running the identical tracker over the identical frames forwards and then
backwards. Nothing integrates the IMU, so accelerometer error has no path into
the trajectory at all.

**Three plausible measurements all pointed at a disaster that is not there:**

| what was tried | it said | why it was wrong |
| --- | --- | --- |
| perturb the annotation by the global residual | ±4 SSC of uncertainty | false premise — the pose error is not independent of the annotation, which co-drifts |
| hit rate as an alignment objective | as-built loses on 7 of 8 frames | biased — pushing the annotation toward the camera buries observed surfaces inside solids and inflates it |
| ICP against the fused mesh | 10–30 mm | blind by construction — the mesh near a frame is fused mostly from that frame's own temporal neighbours |

**What settled it**, annotation-free: a bent reconstruction cannot keep a large
plane flat. Vertical planes — the ones a *horizontal* bend would smear, which
the floor check cannot see — come out flat to **6.7–17.1 mm** in the fused
cloud, and fusing the whole sweep adds **at most 2.6 mm** over a single frame's
view of the same wall. Several fused planes are *tighter* than one frame, which
is averaging beating sensor noise.

So the drift is a slow, near-global change of world frame rather than local
warping — what frame-to-**model** tracking does, since every frame is anchored
to the accumulated model. And **everything scored here is relative**: the
annotation and each frame's grid live in the same reconstruction, so a change of
world frame cancels.

### On an absolute reference

ROS 2 has none either. Its `map` frame is globally *consistent* — via loop
closure, or localising against a prior map — not absolute; REP-105 puts an
`earth` frame above it for exactly that reason. Absolute requires an external
sensor independent of the odometry: GNSS, surveyed fiducials, UWB, motion
capture.

What we already pin to something that is **not** the trajectory:

| DOF | reference | verified |
| --- | --- | --- |
| roll, pitch | gravity, from the fitted floor plane | 0.01° / 0.00° |
| z | floor at z = 0 | frames land within 5 mm |
| x, y, yaw | nothing | — |

Three of six, and the unconstrained three are precisely the ones that cancel. If
absoluteness is ever wanted: tape or laser the room dimensions (checks scale and
squareness), or place AprilTag/ArUco markers at surveyed positions — the
trajectory-independent ICP target we lacked.

## 7. Fixes applied to the captures

- **room07's grid.** Floor tilt 1.88° → 0.01°, floor height → 0.0 m. The whole
  sweep was then re-run (159 frames at `--every 5`) because every swept score
  was stale, and the extremes moved: best by margin is now 530/535, worst
  325/320, with only 320 surviving from the pre-fix set.
- **Frame selection matched.** `export_frame` appends, so three frames that were
  extremes of the old tilted grid were still being pooled into room07's report,
  mixing two selection criteria in one number. Removed, leaving still + 2 best +
  2 worst in both rooms — which is what makes them each other's control.
- **Per-frame pose correction**, `refine_frame_pose.py`: REP-105's split rather
  than a better tracker. The trajectory stays as it is; each scored frame gets a
  correction in `world_from_room`, the role `map → odom` plays in ROS 2.

**Noted as an error:** that correction was built expecting ~100 mm and the real
residual is 5–9 mm, a tenth of a voxel. Effect on the scores: room07 SSC
39.4 → 39.5 and SC unchanged, room08 SSC 35.5 → 35.4 and SC 59.2 → 59.6. It is
a **polish, not a fix** — kept because it is measured and recorded with its
fitness and RMSE, not because the ground truth needed it. It is also not what
makes the ground truth sound; the plane-flatness test is. Standing cost: re-run
it after any `export_frame`, since a freshly exported frame carries the raw
tracked pose.

### The error budget

Generated into both room reports so it travels with the score:

| term | size | effect |
| --- | ---: | --- |
| **annotation fidelity** — a box standing in for furniture | 36–56 mm | **dominant**, and not correctable: it is what the GT *defines* |
| frame pose vs the reconstruction | 5–9 mm, corrected | ±0.1 SSC |
| sensor depth noise, single frame | 8–13 mm plane RMS | inside the row above |
| reconstruction self-consistency | ≤2.6 mm | not warped |
| trajectory drift, whole sweep | 156–175 mm | **cancels** — relative scoring |
| camera height: tape vs floor fit | ~60 mm | the tape's, not the geometry's: the two rooms disagree in *opposite* directions, where a scale error would push both the same way |

**The figure worth improving is the annotation, not the poses.**

## 8. Where each dataset can be trusted

| | SC (completion) | SSC (semantics) |
| --- | --- | --- |
| **room07** | no — 97% occupied at every viewpoint | **yes (39.5)** |
| **room08** | at its best viewpoints only (+25 margin) | **yes (35.4)** |
| NYU | yes (75.0) | yes (47.7) |
| Occ-ScanNet | floor-free (57.9), but it reads low | no (16.4 — GT too sparse) |

The per-class table is the most useful thing these captures produce, because the
ground truth is dense — **10.2% and 8.2%** of grid voxels are objects against
**4.3%** for NYU and **2.1%** for Occ-ScanNet on the same denominator —
hand-annotated, floor present in every frame, level to 0.01°. An error in it is
the model's, not a hole in the annotation. That cannot be said of Occ-ScanNet,
where 14% of frames have no floor at all and the floors that exist are tilted a
median 1.94° inside the grid.

## 9. Bugs and traps found along the way

- **`o3d.t.io.RSBagReader` is a playback device** and drops frames when the
  consumer is slow: two reads of "frame 150" gave 368,598 and 338,168 valid
  pixels. So "frame i" was a property of how fast the loop ran, and every frame
  index was paired with the wrong pose. Fixed by `bag_reader.py` using
  librealsense with `set_real_time(False)`.
- **Two depth-scale conventions.** librealsense gives metres per unit (0.001),
  Open3D wants units per metre (1000). Mixing them fuses nothing, silently:
  "No block is touched in TSDF volume".
- **The room shell used `yaw_deg` only** in `build_label_volume`, ignoring a
  `rotation` matrix — corrupting 9,960 voxels on any transformed room.
- **`create_box` winds triangles outward**, but the camera is *inside* the room
  shell and sees its far walls, so a camera-facing cull kept exactly the faces
  the camera cannot see. ICP diverged to 1.2–1.8 m and 20–46°.
- **`bag_metadata` unpacked 5 values from a 6-value `_start`**, breaking
  `fuse_scan` silently — and hidden twice by a `|| true` and by grep filtering in
  the command used to check it.
- **The refine scripts moved the grid but not the annotation** at first, leaving
  solids 0.055 m from the new mesh instead of 0.043 m.
- **An IMU noise figure of 0.185° was threshold-dependent**; the true value is
  ~0.04°. The accelerometer's real problem is **2.5° of direction error**, which
  is why `up_camera` comes from the floor and not from gravity.
- **Logging after `model_mod.load_model` is swallowed** by mmcv/mmseg — use
  `print(flush=True)`.

## 10. What is not done

1. **Annotation coverage.** 40% of the surfaces room08's `live_000235` and
   `live_000240` measured fall in voxels annotated as empty — unmarked clutter.
   That is the dominant error term and the real ceiling on those two frames.
2. **A third, larger room.** The only route to SC across all viewpoints. Target:
   ≥1 m of hidden extent, ~2.5 m median depth, camera near the height of what it
   looks at, furniture occluding furniture.
3. **Loop closure in fusion (option B).** Open3D's offline Reconstruction System
   is already in the installed wheel at `open3d/examples/reconstruction_system/`
   (Choi et al., CVPR 2015): fragments, a pose graph splitting reliable odometry
   edges from `uncertain=True` loop-closure edges, `global_optimization` with
   Levenberg–Marquardt, plus `slac.py` for the non-rigid bend. It gives
   *consistency*, not absoluteness. Worth doing when the next room is captured,
   since it rebuilds the mesh and the annotation would want re-checking —
   `refine_tilt.sh` and `refine_height.sh` already carry solids through
   `transform_solids`, so the annotation can follow a moved world.
4. **Occ-ScanNet's grid placement.** It puts every grid 5 cm below world *z = 0*
   rather than below the annotated floor, so its GT floor lands one or two layers
   up while the model paints floor in layer 0. Fixable without new data.
5. **The `empty` class spread** — 9.0% IoU in room07 against 47.2% in room08,
   unexplained.

## 11. Standing checks

```bash
python -m reconstruction_GT.verify_voxelizer     # 9 checks, NYU bit-exact
python -m reconstruction_GT.evaluate_gt --nyu    # must print SC 75.0 / SSC 47.7
python -m reconstruction_GT.bag_reader <bag>     # frame i is frame i, at any speed
python3 tools/hf_data.py push --dry-run          # git and HF hold two halves
```

Both of the first two pass as of this recap. The last one matters because a
commit routinely moves the git half of a capture while leaving the HF half
behind — re-fusing a scan rewrites `trajectory.txt` (git) *and* `room.ply` (HF),
and pushing only the commit leaves a fresh `pull` pairing a level annotation
with a tilted mesh. Nothing errors; `refine_tilt.sh --from-floor` simply reads
the stale tilt and "corrects" an already-correct grid.
