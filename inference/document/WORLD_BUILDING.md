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
- [10. The math](#10-the-math) — ICP, the TSDF and the accelerometer, written out
  - [10.1 ICP: six numbers from a 6x6 system](#101-icp-six-numbers-from-a-6x6-system)
  - [10.2 The TSDF: a running mean of signed distances](#102-the-tsdf-a-running-mean-of-signed-distances)
  - [10.3 The accelerometer: one vector, once](#103-the-accelerometer-one-vector-once)

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
2. **Pair the points up using the guess.** Move each new point by the
   current guess and see which pixel of the model it lands on; whatever
   surface sits at that pixel is taken to be what it was looking at. This
   is a *projection*, not a search — classical ICP hunts the nearest
   neighbour in a tree, but against a depth image the answer comes out of
   a division by z, which is what makes 300 000 points affordable. See
   [§10.1](#101-icp-six-numbers-from-a-6x6-system).
3. **Solve.** Given those pairs, the rigid motion that minimises the
   leftover distances has a closed form — six unknowns, one 6x6 system.
   Apply it.
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

Written out, with the degeneracy made exact, that is
[§10.1](#101-icp-six-numbers-from-a-6x6-system).

That is physically honest. A flat patch of wall genuinely does not tell
you how far you slid along it; only motion *through* the wall is
observable. Penalising the sliding would invent information, and it
converges several times slower besides.

The same honesty is the method's one failure mode, and it is written at the
top of `tracker.py`: **a view of nothing but one flat wall is degenerate.**
If every normal in the frame points the same way, three of the six
numbers drop out of the equations entirely and the solve is free to slide
— [§10.1](#101-icp-six-numbers-from-a-6x6-system) puts a number on it. Keep a corner, a
doorway or some furniture in shot and the normals disagree, which is all
the constraint it needs.

## 4. What it is matched against: the model, not the last frame

Here is the part that makes it hold over eight hundred frames.

Every believed frame is integrated into a **TSDF volume**: 2 cm voxels in
16³ blocks, each voxel storing the signed distance to the nearest surface,
truncated at 8 voxels (16 cm), **averaged over every frame that has seen
it**. One frame's depth is noisy. Fifty frames' average of the same patch
of wall is not — the update rule, and the arithmetic that says by how much,
are [§10.2](#102-the-tsdf-a-running-mean-of-signed-distances).

Before each new frame arrives, that volume is **raycast from the last
believed pose** — a ray through every pixel, stopped where the stored
distance crosses zero. The result is a synthetic depth image of what the
camera *would* see from where it last was, and after 60 frames of room07
**93.9% of its pixels hit a surface**. That clean synthetic frame, not the
previous raw frame, is what ICP aligns against. (What a raycast is, and the
three cuts that make one affordable thirty times a second, are
[CONCEPTS.md §7](CONCEPTS.md#7-what-a-raycast-is).)

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
  `camera_height` puts Z = 0 on the floor. The Gram–Schmidt that does it,
  and what the magnitude test is really worth, are
  [§10.3](#103-the-accelerometer-one-vector-once).

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

## 10. The math

Sections 1–9 are the mechanism. This is the same thing with the algebra
left in, for anyone who has to change it rather than explain it. Every
constant named here is the one the code passes.

### 10.1 ICP: six numbers from a 6x6 system

**What is being matched.** The new frame is unprojected into a *vertex
map* `V_s` — §1's equation applied to every pixel, so `V_s[p]` is a 3-D
point in camera coordinates. The raycast of the model gives the same thing
for the target, `V_t`, plus a *normal map* `N_t`, the surface orientation
at each of its pixels. Both are images, not point clouds, and that is the
whole trick.

**Correspondence is a projection, not a search.** For source pixel `p`,
under the current estimate `T_k`:

```
  v' = T_k · V_s[p]                          the point, moved by the guess

  u  = fx · v'_x / v'_z + cx                 where it lands in the model's
  w  = fy · v'_y / v'_z + cy                 image

  q  = (round u, round w)                    its partner pixel
```

and the pair is accepted only if the two agree in depth:

```
  | v'_z − V_t[q]_z |  ≤  depth_outlier_trunc  =  0.07 m
```

That threshold is `Tracker`'s `depth_diff`, and **the fraction of pixels
that survive it is the fitness of §7**. No tree, no search, O(1) per
pixel, every pixel independent — which is what makes a 300 000-point
alignment a 13 ms operation on the card.

**The objective.** With `n = N_t[q]` the model's normal at the partner
pixel and `q_v = V_t[q]` its vertex:

```
  E(T)  =  Σ  ρ( ( T · V_s[p] − q_v ) · n )
           p
```

`ρ` is the Huber norm, not a plain square — quadratic inside
`depth_huber_delta = 0.05 m` and linear outside it, so a handful of badly
paired pixels bend the answer instead of dictating it. The dot with `n` is
the point-to-plane choice of §3: only the component along the surface
normal is counted.

**Linearise.** Between consecutive frames the rotation is under a degree
(§3), so write it as a small-angle rotation vector `ω = (α, β, γ)`:

```
            ⎡  0  −γ   β ⎤
  [ω]ₓ   =  ⎢  γ   0  −α ⎥          R ≈ I + [ω]ₓ       R·v ≈ v + ω × v
            ⎣ −β   α   0 ⎦
```

One pixel's residual then collapses to something linear in the six
unknowns:

```
  r  =  ( v + ω × v + t − q_v ) · n

     =  (ω × v)·n  +  t·n  −  (q_v − v)·n

     =  ω·(v × n)  +  t·n  −  b            b = (q_v − v)·n
```

— the middle step is the scalar triple product identity
`(ω × v)·n = ω·(v × n)`, and it is the reason this is tractable at all:
the rotation unknown comes out of the cross product as a plain dot.

Stack the six unknowns, and each pixel's six coefficients beside them:

```
  x  =  (ω, t)       ∈ ℝ⁶           the answer
  a  =  (v × n, n)   ∈ ℝ⁶           this pixel's row

  r  =  aᵀx − b                     one linear equation per pixel
```

Minimising `Σ r²` gives the normal equations — a 6x6 solve, no matter how
many pixels went into it:

```
  A x = c           A = Σ a aᵀ   (6x6)        c = Σ b a   (6x1)
                        p                        p
```

(The Huber norm makes this *iteratively reweighted* least squares rather
than plain least squares: each pixel enters the two sums with a weight
`w = 1` when `|r| ≤ δ` and `w = δ/|r|` when it is larger, recomputed each
iteration. The shape of the system is unchanged — still 6x6.)

Built exactly as written above from 1 200 synthetic points on three
planes, with a known motion applied, that solve returns the motion to
**0.16 mm and 1.5 microradians** — the residual being the small-angle
approximation and nothing else.

**That sum is the only thing in the whole algorithm that touches all
300 000 pixels.** It is a parallel reduction of 300 000 rank-1 outer
products into one small matrix; everything after it is a 6x6 Cholesky.
That is the shape of the 13 ms, and why the work belongs on a GPU.

Solve, turn `ω` back into a proper rotation matrix, compose it onto the
running estimate, and go round again — 6 iterations at quarter
resolution, 3 at half, 1 at full (§3). The pyramid exists because step 1
depends on `T_k`: a coarse image forgives a worse guess, because one pixel
spans more room.

**Degeneracy, exactly.** Suppose every normal in view is the same
`n = (0, 0, 1)` — one flat wall, square on. Then for a point
`v = (x, y, z)`,

```
  v × n  =  (y, −x, 0)          so    a  =  (y, −x, 0, 0, 0, 1)
```

The third, fourth and fifth entries are zero for *every* pixel in the
frame, however many there are. `A = Σ a aᵀ` therefore has **rank 3, not
6** — built from 1 200 such points its eigenvalues come out

```
  [ 0, 0, 0, 383.6, 392.4, 1201.6 ]        rank 3 of 6, cond = ∞
```

against rank 6 and a condition number of **1.56** for the same count of
points spread over three mutually perpendicular planes, which is what a
room corner is. Three of the six numbers are not merely ill-conditioned,
they are absent from the equations: the two translations parallel to the
wall, *and the rotation about its normal* — spin the camera about its own
optical axis in front of a featureless plane and the depth image is
unchanged.

So "keep a corner in view" is the instruction and the eigenvalues of `A`
are the measurement. Their ratio is the honest statement of how many of
the six numbers the current view actually pins down.

**And why convergence is local.** Step 1 pairs pixels using `T_k`. If the
guess is far enough out that most pixels project onto the *wrong* surface,
the residuals are perfectly consistent with a different pose, `A` is
well-conditioned, and the iteration converges confidently to it. That is
§6's 0.43 m row: fitness 0.463, 0.33 m out, nothing in the returned
transform to say so.

### 10.2 The TSDF: a running mean of signed distances

**What is stored**, read out of the running volume rather than quoted from
the paper: per voxel one `Float32` in `[−1, +1]` and one `UInt16` count,
in 16³ blocks of 2 cm voxels, `--track_blocks` of them (8 000 by default).

**The update.** For a voxel centre `x_w` in world coordinates, with this
frame's pose:

```
  xc   =  (world_T_cam)⁻¹ · x_w        the voxel, in camera coordinates

  u    =  fx · xc_x / xc_z + cx        the pixel it falls on
  w    =  fy · xc_y / xc_z + cy

  D    =  depth(u, w)                 what the sensor measured there

  sdf  =  D − xc_z                    signed distance along the view ray
```

`sdf > 0` means the voxel is **in front of** the measured surface — free
space the ray passed through. `sdf < 0` means behind it.

```
  skip if   D ≤ 0   or   xc_z ≤ 0   or   sdf < −τ

  s  =  min(sdf, τ) / τ                      normalised, so s ∈ [−1, +1]

  τ  =  trunc_voxel_multiplier × voxel  =  8 × 0.02  =  0.16 m
```

**That `sdf < −τ` skip is the honesty of the whole method.** The sensor
measured one surface along that ray and knows nothing whatever about what
lies more than 16 cm behind it, so nothing is written there — it stays
unobserved rather than being declared solid or empty. (Compare
[OCCLUSION.md](OCCLUSION.md): what the *network* does with that same
unobserved space is a completely different question, and it is the one the
room is drawing.)

**The average**, with unit weights:

```
  TSDF  ←  ( W · TSDF  +  s ) / ( W + 1 )
  W     ←  W + 1
```

Measured by integrating one frame of room07 repeatedly: the weight reads
exactly 1, 10, 100, 400 and 1 000 after that many integrations. It is a
plain running mean, and a voxel's stored value is the mean of every
normalised signed distance ever measured for it.

So if a single frame's depth error at that voxel is independent with
standard deviation σ, the mean of `N` of them has standard deviation

```
  σ / √N
```

Fifty looks at a wall is a sevenfold reduction. §4's claim that the model
is a quieter thing to match against than any raw frame is this, and
nothing more. It matters most where the frame is worst, because stereo
depth error grows with the square of range —
`σ_Z = Z² · σ_D / (fx · B)`, [GEOMETRY.md](GEOMETRY.md).

**The surface, and where the normals of §10.1 come from.** The surface is
the level set `TSDF = 0`. The raycast marches a ray per pixel until the
stored value changes sign and interpolates the crossing:

```
                                  TSDF(t₁)
  t*  =  t₁  +  (t₂ − t₁) · ─────────────────────
                             TSDF(t₁) − TSDF(t₂)
```

and the normal is the gradient of the same field, by central differences:

```
  n  =  ∇TSDF / ‖∇TSDF‖
```

That is `N_t`, handed straight to §10.1. Marching cubes — the `m` key —
extracts the same zero level set as triangles, skipping voxels with
`W < MIN_WEIGHT = 3`, which is what drops one-frame depth noise out of the
mesh.

### 10.3 The accelerometer: one vector, once

**What it measures.** Not acceleration — *specific force*:

```
  f  =  a_lin  −  g
```

At rest `a_lin = 0`, so `f = −g`: a vector of magnitude ‖g‖ ≈ 9.81 m/s²
pointing **up**. This is why no integration is involved anywhere. An
accelerometer at rest reads the up direction directly; it is a gyroscope,
integrated into an orientation, that drifts, and there is no gyroscope in
this pipeline.

**Why the magnitude test, and what it is actually worth.** Under motion
the reading is contaminated. For linear acceleration perpendicular to
gravity,

```
  ‖f‖  =  √( ‖g‖² + ‖a_lin‖² )          tilted by  θ = atan( ‖a_lin‖ / ‖g‖ )
```

`camera.py:_gravity` accepts a sample only when its magnitude has not
wandered from a slow reference:

```
  | ‖f‖ − m |  ≤  MAG_TOL · m  =  0.08 · m
```

Work out what that admits: `‖a_lin‖/‖g‖ ≤ √(1.08² − 1) = 0.41`, so a
single accepted sample may still be tilted by `atan(0.41) = 22°`. **The
magnitude test is a coarse rejector, not a guarantee** — what does the
real work is the smoothing underneath it:

```
  u  ←  (1 − 0.25) · u  +  0.25 · f         EMA_UP,  ~4 samples ≈ 0.3 s
  m  ←  (1 − 0.02) · m  +  0.02 · ‖f‖       EMA_MAG, ~50 samples ≈ 3 s
```

The two rates do different jobs. `m` is deliberately slow so the reference
stays near rest and a *sustained* acceleration cannot talk the band across
to meet it. `u` is fast enough to follow a real re-orientation but long
enough to average away the residual tilt of accepted samples, which over a
handheld sweep is zero-mean. On room07's bag, **798 of 808 samples** were
accepted.

**From one vector to a world frame** — Gram–Schmidt, with `ẑ = (0, 0, 1)`
the camera's own optical axis:

```
  e_z  =  f / ‖f‖                                    world Z: up

  e_y  =  ( ẑ − (ẑ·e_z) e_z ) / ‖ ẑ − (ẑ·e_z) e_z ‖  world Y: where the
                                                     camera looks, with the
                                                     up-component removed

  e_x  =  e_y × e_z                                  world X: right-handed


         ⎡ — e_xᵀ — ⎤
  R  =   ⎢ — e_yᵀ — ⎥        rows are the world axes in camera coordinates,
         ⎣ — e_zᵀ — ⎦        so R maps a camera vector into world axes
```

The middle line is the only interesting one: subtracting `(ẑ·e_z) e_z`
removes whatever part of the camera's forward direction points at the sky,
which is exactly "flatten the heading into the horizontal". If the camera
is aimed within a thousandth of straight up or down the projection
vanishes and there is no heading to flatten; `live_cam_pose` falls back to
the camera's −Y axis there.

Then yaw and the floor:

```
         ⎡ cos ψ  −sin ψ  0 ⎤             ⎡               │  0 ⎤
  R_z =  ⎢ sin ψ   cos ψ  0 ⎥      W  =   ⎢   R_z · R     │  0 ⎥
         ⎣   0       0    1 ⎦             ⎢               │  h ⎥
                                          ⎣  0    0    0  │  1 ⎦
```

with `h = camera_height`, which is what puts world Z = 0 on the floor
rather than at the lens. This `W` is the one of §5, frozen at frame 0.

**What the whole exercise buys.** A tilt error `θ` displaces a surface at
range `d` by `d · sin θ`. Assuming the camera is level when it is pitched
4° — ordinary for a handheld sweep, and NYU's own frames run 0.31° to
7.65° — puts a wall 3 m away

```
  3 · sin 4°  =  0.21 m
```

from where the grid draws it: two and a half voxels of the 8 cm world map,
on every surface in the room, for free and in the wrong direction.
