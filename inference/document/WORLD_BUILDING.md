# How the camera knows where it is

The room grows as you walk, so something must be deciding where each new
frame goes. This is that something, in full. It is worth reading before
[VIEWERS.md](VIEWERS.md), because almost every control in the live window —
the hold, the LOST banner, the way a lost pose is found again — is a
consequence of the one mechanism below.

**It is not the IMU.** The D455 has an accelerometer and it is read, but it
contributes exactly one thing, once, on frame 0: which way is down. The
pose comes from the depth image and nothing else. There is no gyroscope
reading, no dead reckoning, no integration of acceleration — all three
drift, and none of them is needed.

## Contents

- [1. A depth frame is a cloud of 300 000 points](#1-a-depth-frame-is-a-cloud-of-300-000-points)
- [2. The room does not move, so if the points moved, the camera did](#2-the-room-does-not-move-so-if-the-points-moved-the-camera-did)
- [3. Finding the motion: ICP](#3-finding-the-motion-icp)
- [4. What it is matched against: the model, not the last frame](#4-what-it-is-matched-against-the-model-not-the-last-frame)
- [5. From steps to a world, and the accelerometer's one job](#5-from-steps-to-a-world-and-the-accelerometers-one-job)
- [6. How far it can jump — measured](#6-how-far-it-can-jump--measured)
- [7. Fitness, and why everything is gated on it](#7-fitness-and-why-everything-is-gated-on-it)
- [8. What this explains about the window](#8-what-this-explains-about-the-window)
- [9. Where the code is](#9-where-the-code-is)

## 1. A depth frame is a cloud of 300 000 points

Every pixel carries a distance. With the intrinsics — `fx`, `fy`, `cx`,
`cy`, the same ones [GEOMETRY.md](GEOMETRY.md) spends its length getting
right — a pixel `(u, v)` holding depth `d` is a point in **camera
coordinates**:

```
      camera                                            the wall
        ●─────────────────────────────────────────────────► +Z
         ╲  the ray through pixel (u, v)
          ╲
           ╲
            ●  the point that pixel saw, d metres away


      x = (u − cx) · d / fx      y = (v − cy) · d / fy      z = d
```

Measured on room07, frame 601: **298 662 of 307 200 pixels** carry a
distance — 97.2% of the image. So a frame is not an image with some depth
attached to it. It is a quarter of a million measured 3-D points per
frame, in the camera's own frame of reference.

## 2. The room does not move, so if the points moved, the camera did

That is the whole principle, and everything else is machinery for
extracting it.

The room is rigid. Between two frames nothing in it has changed shape or
position. So if this frame's points do not land where the previous ones
did, there is exactly one explanation, and it has exactly six numbers in
it — three of rotation, three of translation. Find the single rigid motion
that puts the new points back on the old surface, and you have found how
the camera moved, inverted: shift the camera 2 cm to the right and every
measured point appears 2 cm to the left of where the map says it belongs.

This is why it needs no inertial sensing. The geometry is already an
absolute measurement of position relative to the room; it just has to be
read out.

## 3. Finding the motion: ICP

The catch is that the correspondence is unknown. You have 300 000 new
points and a surface, and nothing tells you which point belongs to which
piece of surface. **Iterative Closest Point** cuts the knot by guessing,
and then using the guess to improve itself:

1. **Guess the motion.** The guess is "the camera has not moved since the
   last frame". On room07 the camera's median step between consecutive
   frames is **8.9 mm** (p90 20 mm, worst 40 mm over the whole 808-frame
   sweep), so the guess is wrong by about a centimetre. That is the entire
   reason this works.
2. **Assume the nearest thing is the matching thing.** Transform the new
   points by the current guess, and pair each one with the closest point
   on the known surface.
3. **Solve.** Given those pairs, the rigid motion that minimises the
   leftover distances has a closed form. Apply it.
4. **Repeat.** With a better alignment the pairings are better, so the next
   solve is better.

Open3D runs this at **three resolutions, coarse first — 6 iterations, then
3, then 1** (`track_frame_to_model`'s default criteria). Large motion is
caught cheaply on a small image where a pixel covers more room; the fine
detail is polished once, at full resolution, where it is nearly aligned
already. The whole thing costs **13 ms** of a 331 ms frame on room07 —
see [DEMO.md](DEMO.md).

### Point-to-plane, and why it matters in a room

The default method is `Method.PointToPlane`, and the choice is not an
optimisation detail. Instead of minimising the straight-line distance
between paired points, it minimises the distance **measured along the
surface normal**:

```
      point-to-point                       point-to-plane

       measured ●                           measured ●
                 ╲                                   │
                  ╲ this distance                    │ this distance
                   ╲                                 │
     ──────────────●─────── surface   ───────────────●─────── surface

                                              sliding the point along
                                              the surface now costs
                                              nothing at all
```

That is physically honest. A flat patch of wall genuinely does not tell
you how far you slid along it; only motion *through* the wall is
observable. Penalising the sliding would invent information, and it
converges several times slower besides.

The same honesty is the method's one failure mode, and it is written at the
top of `tracker.py`: **a view of nothing but one flat wall is degenerate.**
If every normal in the frame points the same way, two of the six numbers
are unconstrained and the solve is free to slide. Keep a corner, a
doorway or some furniture in shot and the normals disagree, which is all
the constraint it needs.

## 4. What it is matched against: the model, not the last frame

Here is the part that makes it hold over eight hundred frames.

Every believed frame is integrated into a **TSDF volume**: 2 cm voxels in
16³ blocks, each voxel storing the signed distance to the nearest surface,
truncated at 8 voxels (16 cm), **averaged over every frame that has seen
it**. One frame's depth is noisy. Fifty frames' average of the same patch
of wall is not.

Before each new frame arrives, that volume is **raycast from the last
believed pose** — a ray through every pixel, stopped where the stored
distance crosses zero. The result is a synthetic depth image of what the
camera *would* see from where it last was, and after 60 frames of room07
**93.9% of its pixels hit a surface**. That clean synthetic frame, not the
previous raw frame, is what ICP aligns against.

```
     ┌───────────────────────────────────────────────────────────┐
     │                                                           │
     ▼                                                           │
  frame N ───┐                                                   │
  (raw depth)│                                                   │
             ├─► ICP ─► step ─► T ← T · step ─► world_T_cam = W·T │
             │                     │                             │
  synthetic ─┘                     ▼                             │
  depth  ▲                 integrate frame N into the TSDF       │
         │                         │                             │
         │                         ▼                             │
         └──────── raycast the volume from T ◄────────────────────┘
```

**Frame-to-frame** tracking would chain each small error into the next, and
over 800 frames a half-millimetre bias per frame is half a metre of
nonsense. **Frame-to-model** compares today's frame with the average of
everything seen so far, so there is no chain for error to accumulate
along. It is the same choice `reconstruction_GT/fuse_scan.py` makes for
producing committed ground truth, and the same volume the `m` key's mesh
is carved out of — see [VIEWERS.md](VIEWERS.md).

## 5. From steps to a world, and the accelerometer's one job

Each ICP result is a *relative* step. The poses are just the running
product:

```
   frame 0        frame 1        frame 2                   frame N
      ●──────────────●──────────────●──────── … ──────────────●
          step₁          step₂          step₃ … stepN

      T = I          T = step₁      T = step₁·step₂    T = step₁ ⋯ stepN

      world_T_cam = W · T
      └── W = live_cam_pose(camera_height, yaw, up_camera) of FRAME 0
```

`W` is built once, from the first frame, and never again. It is where the
accelerometer comes in and the only place it does:

- `rs.stream.accel` is the only motion stream enabled; there is no gyro
  in the pipeline at all.
- At rest the accelerometer reads gravity, so the vector it returns is
  *up*, expressed in camera coordinates. `camera.py:_gravity` smooths it
  (`EMA_UP = 0.25`) and rejects samples taken under acceleration by
  watching the magnitude wander (`MAG_TOL = 0.08` of a slow reference,
  `EMA_MAG = 0.02`). On room07's bag, **798 of 808 samples** passed.
- `live_cam_pose` turns that into world axes: **Z is up**, Y is the
  camera's heading flattened into the horizontal, X is Y × Z, and
  `camera_height` puts Z = 0 on the floor.

So gravity decides the *orientation of the world's axes at frame 0*. It is
never integrated, never accumulated, and a reading that goes wrong halfway
through the sweep cannot move the accumulated map — the single-frame grid
the network is handed still uses that frame's own gravity, but where the
room sits in the world was settled on frame 0. Without it the camera is assumed
level, which a handheld sweep is not: 4 degrees of unnoticed pitch puts a
surface 3 m away 21 cm from where a level grid would draw it.

Because `T` starts at the identity, frame 0's `world_T_cam` *is* its own
`cam_pose`, the whole placement collapses to the identity there, and
`inference/check_world_map.py algebra` can check the arithmetic against
`run_inference` — 7 889 of 7 889 voxels, 0 missing, 0 extra.

## 6. How far it can jump — measured

ICP only converges from a guess that is already close. "Close" is not a
figure of speech, so here it is measured, on room07's busiest stretch
(frames 641–760, 1.33 m of travel). One tracker is warmed on frames
601–660 and then handed a frame *k* ahead, **having seen nothing in
between** — which is exactly what a hold, a run of dropped frames, or a
camera swung away and back hands it:

| frames skipped | the camera really moved | ICP answered | fitness | off by |
| ---: | --- | --- | ---: | ---: |
| 1 | 0.017 m, 0.9° | 0.017 m, 0.9° | 0.868 | **0.000 m** |
| 2 | 0.040 m, 1.7° | 0.042 m, 1.7° | 0.847 | **0.002 m** |
| 5 | 0.105 m, 3.6° | 0.111 m, 3.6° | 0.793 | **0.008 m** |
| 10 | 0.206 m, 5.7° | 0.216 m, 5.9° | 0.691 | **0.019 m** |
| 20 | 0.434 m, 11.4° | 0.121 m, 3.1° | 0.463 | 0.328 m |
| 40 | 0.817 m, 20.2° | 0.128 m, 5.1° | 0.196 | 0.804 m |
| 60 | 1.008 m, 20.9° | 0.165 m, 3.2° | 0.140 | 0.939 m |
| 80 | 1.034 m, 23.0° | 0.235 m, 4.5° | 0.090 | 1.152 m |
| 100 | 1.029 m, 23.0° | 0.232 m, 4.2° | 0.088 | 1.140 m |

Read it in three bands.

**Up to about 0.21 m and 6°, it simply works** — the answer is within
19 mm of what the frame-by-frame run produced, and ICP does not care that
it never saw the intervening frames.

**At 0.43 m and 11° it converges on the wrong answer and does not say so.**
It reports a confident 0.12 m motion for a camera that moved 0.43 m, and
puts the frame a third of a metre from where it belongs. The optimiser has
fallen into a different local minimum: a wrong alignment where the points
also happen to sit flat against *some* surface. Nothing about the
transform it returns looks wrong.

**Past that it collapses** — fitness 0.09 to 0.20, answers unrelated to
anything.

The useful thing is where the boundary falls relative to the gate. The
fitness floor in `Belief` is 60% of this run's own median, capped at
**0.55** — a cap chosen as a tenth below the worst good frame ever
measured, which is a different argument entirely. The last row here that
is *right* reads 0.691. The first row that is *quietly wrong* reads 0.463.
The floor lands between them, so this table is an independent check that
it sits where it has to: row 5 is exactly the frame that must never reach
the map, and it does not.

## 7. Fitness, and why everything is gated on it

Fitness is the fraction of the frame's pixels that found a correspondence
within `depth_diff`, which is **7 cm**. It is not a confidence score the
algorithm invents; it is a count of how much of what the camera is looking
at actually landed where the model predicted.

Measured over both committed captures end to end — 1 871 frames — it never
fell below **0.647**, median **0.90**. Good tracking is not marginal, it is
emphatic: nine pixels in ten landing on the model where predicted. A frame
reading 0.18 is telling you, plainly, that it is looking at something the
model does not contain.

That gap between 0.65 and 0.20 is what the whole gate rests on, and why
the floor is relative rather than fixed — a different sensor or a darker
room matches worse everywhere, and a fixed floor would either kill the run
or wave the jump through. The rule, the measurements behind each constant,
and how a lost pose is earned back are in `Belief`'s docstring and in
[README.md](README.md#trackerpy--where-the-pose-comes-from).

## 8. What this explains about the window

Everything in [VIEWERS.md](VIEWERS.md) that looks like a separate feature
is this mechanism showing through:

| what you see | why |
| --- | --- |
| **holding the window still tracks the camera** | on a live camera your hand keeps moving while you look. Dropping those frames hands the tracker a jump across the whole hold, and §6 is what 0.43 m does. Tracked without being integrated, a 4-second hold on room07 goes from *15 frames and lost* to 45 of 45 |
| **a hold cannot go on forever** | nothing is integrated while held, so the model stops growing while the camera keeps leaving it. 142 frames of continued sweeping exhausted it |
| **"point the camera back at what it was looking at"** | there is no inertial estimate to carry across the gap. The only way back is a frame that matches the model, and the model only contains what has already been seen |
| **three agreeing frames to recover** | one lucky match at the wrong local minimum welds the room into the wrong place permanently. Row 5 of §6 is what a single lucky-looking match can be |
| **keep a corner in view** | §3: flat wall, free slide |
| **a lost pose freezes the map rather than smearing it** | a bad pose does not degrade the map, it corrupts it. Geometry welded into the wrong place never washes out, however many good frames follow |

## 9. Where the code is

| | |
| --- | --- |
| `inference/tracker.py` | `Tracker.step` — the loop in §4, one frame in, `world_T_cam` out |
| | `Belief` — the rule in §7, pure, no open3d |
| | `reject` — the fitness floor and the jump gate |
| `inference/frame_loader.py` | `live_cam_pose` — `W`, the world axes of §5 |
| `inference/camera.py` | `_gravity` — the accelerometer smoothing of §5 |
| `reconstruction_GT/fuse_scan.py` | the same tracking, at 1 cm, for committed ground truth |
| `inference/check_world_map.py` | `algebra` — checks the composition of §5 |
| `tests/test_tracker.py` | the gate and the recovery rule, 26 cases |

Open3D's own documentation for the call this all comes down to:
[`Model.track_frame_to_model`](https://www.open3d.org/docs/release/python_api/open3d.t.pipelines.slam.Model.html).
