# Drift, misalignment, and the difference between them

A frame lifted out of a sweep is placed in the room by its tracked pose. If
that pose is wrong, the frame's ground truth is built in the wrong place and
**nothing in the pipeline complains** — the mesh and the annotation move
together, so they stay consistent with each other while both slide away from
the room.

This document is how that was measured, the three ways it is easy to measure
it wrongly, and what the answer turned out to be.

The short answer, in the order the terms were actually separated:

- the **trajectory** drifts 156–175 mm over a sweep, 1–2.4% of path length;
- the **reconstruction** that drift is blamed for is nevertheless flat to
  within sensor noise — fusing the whole sweep smears a wall by at most
  **2.6 mm** over a single frame's view of it;
- so the drift is a slow, near-global change of world frame, not a local
  distortion, and **everything the ground truth does is relative**, so it
  cancels;
- what is left between a frame and the annotation is **36–56 mm**, dominated
  not by placement but by the annotation's own fidelity: a bed is not a box.

The ground truth is sound. Getting to that took discarding three plausible
measurements, each of which pointed the other way, and they are all recorded
below because the mistakes are the useful part.

```bash
python -m reconstruction_GT.drift_check       captures/room08          # measure
python -m reconstruction_GT.refine_frame_pose captures/room08          # report
python -m reconstruction_GT.refine_frame_pose captures/room08 --write  # apply
```

---

## 1. Where the error can enter

`fuse_scan.py` tracks each frame against a raycast of the model it has built
so far (`o3d.t.pipelines.slam.Model.track_frame_to_model`) and chains the
result. Two things follow:

- **Nothing integrates the IMU.** The accelerometer is read once, during the
  opening hold, purely as a cross-check on which way is down
  (`_check_imu_tilt`). There is no dead-reckoning path, so accelerometer bias
  and noise *cannot* accumulate into the trajectory. See
  [MAKING_GT.md](MAKING_GT.md) and `imu_check.py` for what the IMU is worth on
  its own: 0.04° of noise but **2.5° of direction error**, which is why
  `up_camera` is fitted from the floor and not taken from gravity.
- **Visual tracking has no loop closure.** Frame-to-model chaining accumulates
  without bound, and there is no mechanism to notice when the camera returns
  somewhere it has been. This is a structural property of the KinectFusion
  family, not a tuning problem — Intel's own `rs-kinfu` has it too.

So the only route to a bad pose is visual drift.

## 2. Why the existing checks do not catch it

`fuse_report.json` already carries three checks, and it is worth being precise
about what each one *cannot* see:

| check | what it proves | what it misses |
| --- | --- | --- |
| `still_match` | frame 0 agrees with the separately captured still to 19 mm (room08) / 48 mm (room07) | it validates the **origin** of the pose chain, not its accumulation |
| `floor` | the floor fused from every frame is flat to 0.02° / 0.15° and sits at z = 0 | bounds **rotational** and **vertical** drift; completely blind to a horizontal slide |
| `imu_tilt` | gravity agrees with the grid | nothing to do with the trajectory |

Horizontal drift was therefore unmeasured, which is why `drift_check.py` exists.

## 3. Measuring it: run the same frames backwards

Run the *same* tracker over the *same* frames twice, once forwards and once
backwards. Identical geometry, opposite direction of accumulation, so the two
trajectories can only disagree by drift.

Put them in one frame by re-referencing the forward run to its last frame:
`P_fwd(N-1)⁻¹ · P_fwd(i)` is then directly comparable with `P_rev(i)`, and the
residual at frame 0 is the disagreement accumulated over the whole sweep.

A bag can only be read forwards, so frames are cached to a scratch memmap
first (8 s for 1050 frames) and both passes run off the cache.

| | frames | path | residual at far end | % of path | fitness fwd / rev |
| --- | ---: | ---: | ---: | ---: | ---: |
| room07 | 793 | 7.21 m | **174.9 mm / 0.28°** | 2.42% | 0.876 / 0.877 |
| room08 | 1050 | 14.40 m | **155.9 mm / 1.41°** | 1.08% | 0.867 / 0.855 |

Both forward passes reproduce the committed `fuse_report.json` fitness exactly
(0.8759 → 0.876, 0.8671 → 0.867), which is the proof that this is the real
tracker and not an approximation of it.

The residual grows monotonically with distance from the two runs' shared
origin — 0 mm at the anchor, rising to 175 mm at the start — which is coherent
drift rather than jitter. At the frames that actually carry ground truth it is
66–80 mm (room07) and 94–102 mm (room08), i.e. 0.8–1.3 voxels at 8 cm.

**This is a lower bound.** A bias the tracker commits in both directions
cancels and shows as zero. A small residual does not prove the trajectory is
right; a large one proves it is not.

## 4. The first wrong turn: treating that residual as the GT error

The obvious next step is to ask what ~100 mm of misplacement does to the
scores. Shift each frame's annotation by its own residual, rebuild the ground
truth, re-score the *same* prediction:

| | SSC moves | mean abs | SC moves | mean abs |
| --- | --- | ---: | --- | ---: |
| room07 | −9.1 … +8.2 | 2.2 | −3.2 … +4.3 | 1.0 |
| room08 | −9.7 … +7.9 | 4.2 | −7.5 … +15.6 | 6.5 |

Read naively this says the drift is fatal: SSC 35.5 ± 4 is not a number worth
quoting. **The reasoning is wrong, and the flaw is in the premise.**

That perturbation assumes the pose error is *independent* of the annotation.
It is not. The annotation was drawn on the mesh fused from the same
trajectory, so a frame's pose and the local mesh carry the same drift. Where
the scoring happens — a frame against the annotation near it — the drift
largely cancels. Drift is slow and coherent; locally, around any one frame,
the reconstruction is self-consistent, and the annotation sits on it.

So the perturbation measures the right **sensitivity** (the scores really are
sensitive to a voxel of slop) against the wrong **magnitude**. What is needed
is a direct measurement of frame-to-room misalignment.

## 5. The second wrong turn: hit rate is a biased objective

The tempting measurement is *hit rate*: the share of observed-surface voxels
(a depth pixel landed there) that the annotation calls solid. The sensor is
looking **at** a surface, so a well-aligned frame should score high. Shift the
annotation and see whether the as-built position is best.

It is biased, and badly. **Shifting the annotation toward the camera buries
observed surfaces deeper inside solids and raises the hit rate without
improving alignment at all.** The objective is maximised by sinking the room
into the camera, not by aligning it.

Measured: over ±100 mm shifts, the as-built annotation is the best alignment on
only **1 of 8** frames, most shifts that win share a direction, and room08's
`live_000235` appears to improve from 60.1% to 71.2%. All of it is the
artefact. ICP, which is unbiased, puts the true offset at 10–30 mm.

**Trust the inlier RMSE of a registration, not the hit rate.**

What hit rate *is* good for, once alignment has been ruled out separately, is
annotation **coverage** — see section 8.

## 6. Measuring it properly — and what the mesh can and cannot tell you

### The trap in the obvious target

ICP each frame against the fused mesh and the corrections come out at
10–30 mm. That number is real but it answers the wrong question, **because the
mesh is built from the same drifting trajectory.** Worse, the mesh *near* a
frame was fused mostly from frames near it in time, which share its drift
almost exactly. Frame-to-mesh ICP is therefore structurally blind to the
accumulated bend; it measures local tracking noise and nothing else.

Measured without an optimiser (point-to-plane distance at the as-built pose, so
nothing can diverge):

| | frame → mesh | frame → annotation | within 1 voxel |
| --- | ---: | ---: | ---: |
| room07 `live_000320` | 5.3 mm | 43.0 mm | 88% |
| room07 `live_000325` | 5.0 mm | 40.2 mm | 89% |
| room07 `live_000530` | 5.4 mm | 55.5 mm | 91% |
| room07 `live_000535` | 5.5 mm | 55.9 mm | 90% |
| room08 `live_000180` | 9.2 mm | 39.7 mm | 93% |
| room08 `live_000235` | 5.4 mm | 36.5 mm | 73% |
| room08 `live_000240` | 5.9 mm | 39.4 mm | 72% |
| room08 `live_000580` | 9.0 mm | 36.7 mm | 92% |

Frame-to-mesh is 5–9 mm, confirming the blindness: a frame always sits on its
own neighbourhood. Frame-to-**annotation** is 36–56 mm, and that is the
quantity the ground truth actually depends on, because the annotation is one
global human-fitted model rather than a per-frame fusion.

### Is the reconstruction itself bent?

This is the question the forward/reverse residual seems to answer and does not.
Test it annotation-free: **a bent reconstruction cannot keep a large plane
flat**, because surfaces contributed by early frames sit off the ones
contributed late. So fit planes in the fused cloud and compare their thickness
with the same plane seen in a *single* frame, which is pure sensor noise with
no fusion involved.

| | fused plane RMS | single frame, same plane | fusion adds |
| --- | ---: | ---: | ---: |
| room07 floor (45,415 pts) | 13.0 mm | 8.0–8.8 mm | ~10 mm |
| room07 wall 1 (69,907 pts) | 8.6 mm | 15.3 mm | **0.0 mm** |
| room07 walls 2–3 | 17.1 / 11.3 mm | — | — |
| room08 floor (36,394 pts) | 12.1 mm | 13.1 mm | **0.0 mm** |
| room08 wall 1 (39,866 pts) | 14.6 mm | 14.4 mm | **2.6 mm** |
| room08 wall 2 (37,752 pts) | 6.7 mm | 21.2 mm | **0.0 mm** |
| room08 wall 3 (26,573 pts) | 15.4 mm | 26.5 mm | **0.0 mm** |

Walls are the ones that matter, because a *horizontal* bend would not smear a
horizontal floor — the same blind spot the `floor` check has. Vertical planes
observed across the sweep come out flat to **6.7–17.1 mm**, and fusing the
whole sweep adds **at most 2.6 mm** over a single frame. Several fused planes
are *tighter* than one frame's view of them, which is averaging beating sensor
noise.

**So the reconstruction is not meaningfully distorted, and the 156–175 mm
forward/reverse residual does not say that it is.** The two numbers measure
different things: the residual is the disagreement between two independent
integration paths, each internally self-consistent, ending in slightly
different global frames.

That is the nature of frame-to-**model** tracking, as against frame-to-frame.
Every frame is aligned to the accumulated model, so the model acts as an
anchor and errors do not compound locally. Drift appears as a slow,
approximately global transformation rather than a local distortion — which is
exactly why KinectFusion-family systems produce locally excellent, globally
drifting maps.

### Why a global transformation is harmless here

Everything we score is relative. The annotation lives in the reconstruction;
each frame's grid is placed by that frame's pose in the same reconstruction;
the model never sees world coordinates. A rigid change of world frame cancels
out of all of it. Only **local** distortion could corrupt the ground truth, and
that is bounded above at 2.6 mm.

What is left of the 36–56 mm frame-to-annotation residual is therefore mostly
not placement at all — it is **the annotation's own fidelity**: a bed is not a
box. A box approximation of real furniture is wrong by tens of millimetres by
construction, and that is not an error, it is the ground truth's definition.
The one drift-correlated signal visible in it is room07's rise from 40–43 mm
mid-sweep to 55–56 mm late, about 15 mm.

### Not aligning to the annotation, again

With the shell's normals fixed (see below), ICP against the annotation does
converge, and it still cannot be trusted: `live_000240` asks for 141–157 mm and
**8.5–9.0° of rotation**, which is the optimiser exploiting the fact that a box
is a poor model of a bed. `--target solids` stays diagnosis-only, and
`refine_frame_pose.py` keeps the mesh as its target.

> A bug worth recording: `o3d.geometry.TriangleMesh.create_box` winds its
> triangles **outward**, but the camera is *inside* the room shell and sees its
> far walls. The first version of the camera-facing cull therefore kept exactly
> the faces the camera cannot see, and ICP diverged to 1.2–1.8 m and 20–46°.
> The shell's winding is now flipped so its normals point into the room.

## 7. The fix: ROS 2's split, not a better tracker

ROS 2 does not try to make odometry drift-free. REP-105 declares that
impossible and splits the frames instead:

- `odom → base_link` — continuous, smooth, drifts without bound. Good for
  control, useless as a global reference.
- `map → base_link` — drift-free but discontinuous. A localisation node
  computes it and publishes the **correction**, `map → odom`, rather than the
  pose itself.

We have the same split available, and a reason to prefer it: **only a handful
of frames per capture are ever scored.** A globally consistent trajectory is
not needed to get those right. So the trajectory stays exactly as it is — the
mesh and the annotation are built on it — and each scored frame gets its own
correction.

`frame_meta[stem]['world_from_room']` is the slot. It is a per-frame 4×4,
structurally the same thing as `map → odom`.

### The arithmetic

`export_frame.py` stores, for a frame whose tracked pose is `T`
(world-from-camera, colour frame, in the room's world):

```
P = live_cam_pose(T[2,3], 0, up_from(T))     the canonical evaluation pose
A = P @ inv(T)                               = world_from_room
```

A room point reaches the frame's world at `A·p`, with the camera at `P`, and
`inv(P)·A·p == inv(T)·p` — the point in the camera frame, as tracking says. So
`T` comes back out of the record as `T = inv(A) @ P`, no trajectory file
needed, and the frame's observed points land in room coordinates at `T·p_cam`.

ICP returns the correction `C` that takes those observed points onto the room:

```
T_new = C @ T
```

and `camera_height`, `up_camera` and `world_from_room` are rebuilt from `T_new`
by exactly the lines `export_frame.py` uses, so the record stays internally
consistent.

Nothing here is inferred. The correction is measured from this frame's own
depth against the fused room, and its fitness, RMSE, shift and rotation are
recorded beside it in a `pose_refined` block together with the values it
replaced — the same standing as `camera_height` from a floor-plane fit. A
correction larger than `--max_shift` (default 300 mm) or below 0.3 fitness is
refused rather than applied: ICP has found something other than the same room.
`meta.json` is backed up to `meta.json.preicp` before the first write.

## 8. What changed, and what did not

Applied to both rooms, ground truth rebuilt, same predictions re-scored:

| | SSC before | after | SC before | after |
| --- | ---: | ---: | ---: | ---: |
| room07 | 39.4 | **39.5** | 72.4 | **72.4** |
| room08 | 35.5 | **35.4** | 59.2 | **59.6** |

Nothing, in other words — which is the right outcome given section 6.

**Be plain about what this means: the correction was built for a problem that
does not exist at the size it was built for.** It was aimed at ~100 mm of
suspected pose error; the real residual is 5–9 mm, a tenth of a voxel. It
removes that, so it is a polish rather than a fix.

It is kept rather than reverted because it is measured, recorded with its
fitness and RMSE, and not wrong — but it is not what makes the ground truth
sound. The plane-flatness test in section 6 is. The correction carries a small
standing cost too: it has to be re-run after any `export_frame`, because a
freshly exported frame carries the raw tracked pose, and forgetting leaves a
capture half-corrected.

Each room's `gt/EVALUATION.md` now carries an **error budget** listing this term
beside every other known error in the numbers, so the caveat travels with the
score instead of living only here.

The uncertainty on the room SSC figures from pose error is about **±0.1**, not
the ±4 that section 4's sensitivity table suggests in isolation.

### The finding that did fall out of it

Hit rate, used for what it is actually good for, shows room08's two worst
frames are not misaligned — they are **under-annotated**:

| frame | hit rate | reading |
| --- | ---: | --- |
| room07 `live_000320`–`535` | 91.8–94.9% | well annotated |
| room08 `live_000180` | 86.8% | good |
| room08 `live_000580` | 78.6% | acceptable |
| room08 `live_000235` | 59.7% | **40% of measured surfaces annotated as empty** |
| room08 `live_000240` | 59.0% | **41% annotated as empty** |

That is unmarked clutter, not a pose problem, and it is the real ceiling on
those two frames. Marking it would raise them.

## 9. What this does not fix

Two things are genuinely left, and neither is the one this document set out
to find.

**Annotation fidelity, which is now the dominant term.** 36–56 mm of
frame-to-annotation residual is mostly boxes standing in for furniture, and no
amount of pose work touches it. Finer solids, or `shape: mesh`, would.

**Global consistency, which we do not currently need.** The reconstruction is
locally sound (≤2.6 mm of fusion smear) but its world frame still drifts, so a
*single* box cannot fit a wall that was observed at both ends of a long sweep
as well as it fits one observed briefly. Our rooms are small enough that this
stays inside the box-approximation error; a larger room, or a longer sweep,
would not be.

For that, the fix is Open3D's offline **Reconstruction System**, already present
in the installed wheel at `open3d/examples/reconstruction_system/` (Choi et
al., *Robust Reconstruction of Indoor Scenes*, CVPR 2015):

1. split the sequence into fragments, RGBD odometry inside each;
2. build a pose graph whose edges are deliberately two classes — **odometry
   edges** (temporally adjacent, reliable) and **loop-closure edges**
   (non-adjacent, found by global registration, marked `uncertain=True`);
3. `global_optimization` with `GlobalOptimizationLevenbergMarquardt`
   redistributes the accumulated error around the loops, the `uncertain` flag
   driving a line process that lets it *reject* false loop closures.

`slac.py` additionally corrects the non-rigid distortion a rigid pose graph
leaves behind — the "bent room" specifically. A `config/realsense.json` ships
with it.

It is worth doing when the next room is captured, because it rebuilds the mesh
and the annotation would then want re-checking against it. `refine_tilt.sh` and
`refine_height.sh` already carry solids through `transform_solids`, so the
annotation can follow a moved world.

## 10. Re-running it

```bash
# measure: forward vs reverse, writes outputs/<room>/drift_check.json
python -m reconstruction_GT.drift_check captures/room08

# correct the scored frames
python -m reconstruction_GT.refine_frame_pose captures/room08            # report
python -m reconstruction_GT.refine_frame_pose captures/room08 --write    # apply

# then rebuild and re-score what it touched (one --frame at a time)
python -m reconstruction_GT.voxelize_gt captures/room08 --frame live_000180
python -m reconstruction_GT.evaluate_gt captures/room08 --report
```

Re-run `refine_frame_pose` after any `export_frame`, since a newly exported
frame carries the raw tracked pose. The results quoted here are in
`outputs/room07/drift_check.json` and `outputs/room08/drift_check.json`, and
each refined frame's own correction is in its `pose_refined` block in
`captures/<room>/meta.json`.

Summary of all of it, alongside the other ground-truth findings:
[FINDINGS.md](FINDINGS.md), section 6.
