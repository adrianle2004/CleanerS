# Camera tuning

How the D455's capture settings affect CleanerS's output, with measured
numbers rather than guesses. Companion to [README.md](README.md), which covers
the plumbing; this file covers the knobs.

All measurements are on `captures/room01` — a real room, camera at 1.42 m,
D455 on USB 3.2, 1280×720@30.

---

## Quick diagnostic

Two numbers tell you whether a capture is usable, before you look at anything
else.

**1. `observed %`**, printed by `run_inference`:

```
[live_000000] mask=surface (39.4% of grid), observed=45.1%, occupied=3033
```

| observed | verdict |
| --- | --- |
| 40–50% | healthy — an NYU reference frame is 46.3% |
| 25–40% | usable, but the network is completing more than it's seeing |
| < 25% | something is wrong; check the blind spot and the mounting |

**2. Floor/ceiling vertical separation.** The single best evidence that
`camera_height` and the grid orientation are both right:

```python
import numpy as np
p = np.load('outputs/room01/prediction/live_000000.npy').reshape(60, 36, 60)
for cid, name in ((2, 'floor'), (1, 'ceiling')):
    z = np.where(p == cid)[1] * 0.08          # axis 1 is height, 8 cm cells
    print(f'{name:8s} {z.size:5d}  {z.min():.2f}-{z.max():.2f} m')
```

A good capture looks like this — two tight, well-separated bands:

```
floor     3614  0.00-0.08 m
ceiling   1147  2.08-2.80 m
```

Floor voxels above ~0.2 m, or ceiling voxels below ~1.5 m, mean the geometry
is wrong. Fix that before tuning anything else.

---

## How the frame gets to 640×480

`match_nyu_fov` shrinks the frame and then throws half of it away. That looks
wasteful, so it is worth understanding before you try to remove it.

*(This section is the intuition. The algebra — the scale, the crop offsets and
how `K_eff` is derived — is in [GEOMETRY.md §4](GEOMETRY.md#4-matching-nyus-field-of-view).)*

### The one idea: pixels per degree

Every camera has a *zoom-ness*: how many pixels it spends on one degree of the
world.

```
NYU / Kinect  ->   9.1 pixels per degree
D455          ->  11.1 pixels per degree
```

The D455 is more zoomed in per degree — it spends more pixels describing the
same slice of the world. (It also *sees* more degrees in total. That is a
separate problem; hold the thought.)

**Focal length is just this number in different units.** `fx = 518.86` and
"9.1 px/degree" are the same fact. That is all `fx` has ever meant.

### Why it matters

The network learned its idea of the world from Kinect pictures. In those, a
chair 2.5 m away is about **104 pixels wide**, and that is baked into every
filter in the 2D backbone.

| | focal length | a 0.5 m chair at 2.5 m |
| --- | --- | --- |
| NYU (what the network learned) | 518.86 | **103.8 px** |
| D455 native 1280×720 | 637.79 | 127.6 px (1.23×) |
| D455 squashed into 640×480 | 318.89 | 63.8 px (0.61×) |
| D455 after matching | 518.86 | **103.8 px** (1.00×) |

So the job is: make our picture have the Kinect's pixels-per-degree.

### The tempting wrong fix

*"The model wants 640×480 and my camera gives 1280×720, so just resize it."*

```
                 pixels/degree    degrees covered
  D455 native        11.1              90 deg
  after resize        5.6              90 deg    <- half the density
  NYU wants           9.1              63 deg
```

You halved the density and kept all 90 degrees. The chair is now 64 px where
the network wants 104 — the D455 started 1.23× too big and you overshot to
0.61× too small.

**Resizing changes how zoomed-in the picture is. It never changes how much of
the room is in frame.** Those are two different problems, so they get two
different fixes.

### Step 1 — fix the density (this is the focal length match)

We have 11.1 px/degree and want 9.1, so shrink a little:

```
9.1 / 11.1 = 0.81
1280x720  --x 0.81-->  1041x586
```

```
                 pixels/degree    degrees covered
  after step 1        9.1              90 deg    <- density is RIGHT
  NYU wants           9.1              63 deg    <- still too many degrees
```

The chair is now 104 px. Correct — but the picture is 1041 wide and still
shows all 90 degrees.

#### How the shrink is actually done

Three lines of code, but each one has a decision in it.

**1. Pick the scale.** It comes straight from the two focal lengths — no search,
no fitting:

```python
s = fx_target / fx_source = 518.8579 / 637.7892 = 0.813526
```

**2. Pick the new size.** Multiply and round:

```python
W_r = round(1280 * 0.813526) = 1041
H_r = round( 720 * 0.813526) =  586
```

**Both axes use the same `s`.** That is what "isotropic" means, and it is
deliberate: scaling them differently would stretch the image, giving the
picture a different shape from anything the network saw in training. (It would
also let us match `fy` exactly — see [Optional: an exact
match](#optional-an-exact-match) — but the shape change is the reason we don't.)

**3. Resample.** This is the part with real consequences.

Shrinking is not "throw away every 5th pixel." Output pixel 100 corresponds to
input position **122.9** — a fraction, not a real pixel. Something has to
decide what value goes there, and the two sensible answers behave very
differently:

| | what it does | good for |
| --- | --- | --- |
| `INTER_NEAREST` | take pixel 123, copy its value verbatim | **depth** |
| `INTER_AREA` | average pixels 122 and 123 | **colour** |

For colour, averaging is correct — that is what downscaling *should* do, and it
suppresses aliasing.

For depth, averaging is a disaster. Say pixel 122 is the edge of a chair at
1.0 m and pixel 123 is the wall behind it at 3.0 m. Blend them and you get
**2.0 m** — a surface that exists nowhere, floating in the empty air between
the chair and the wall. Worse, the TSDF encoder cannot tell an interpolated
value from a measured one; it stamps that phantom as real observed geometry,
and the network then completes a room around it.

So depth is resampled with `INTER_NEAREST` throughout the pipeline — here, and
in the undistortion remap for the same reason. Every value in the output depth
image is a genuine sensor reading, just relocated. (That is also why the
capture PNGs come back bit-identical: nearest-neighbour moves samples without
altering them.)

#### One caveat: the achieved scale is not exactly `s`

`cv2.resize` takes an integer output size, so it can only scale by exact
ratios. Asking for 0.813526 gets you:

```
x:  1041/1280 = 0.813281      (0.00024 short)
y:   586/720  = 0.813889      (0.00036 over)
```

Recovering the mapping empirically — feed in a linear ramp and read what comes
out — confirms cv2 uses the half-pixel convention,
`u_in = (u_out + 0.5)/scale − 0.5`. Putting both together, the frame after
step 1 really has:

```
                      actual       K_eff reports    difference
  fx                 518.7020        518.8579        +0.16 px
  fy                 518.4642        518.2329        -0.23 px
  cx (pre-crop)      518.9586        519.2080        +0.25 px
  cy (pre-crop)      296.4623        296.4231        -0.04 px
```

`K_eff` is computed from the *requested* scale `s` and the naive `c * s` rather
than from the achieved ratio and the half-pixel convention, so it is off by up
to **0.25 px**. At 3 m that is **1.4 mm — about 1.8% of one 80 mm voxel**, so it
changes nothing measurable, and it is a fixed bias rather than per-frame noise.

Worth knowing because the docstring describes `K_eff` as measured from what was
actually done; strictly it is computed from what was *asked for*. Making it
genuinely measured is a two-line change (use `W_r/W_s`, `H_r/H_s`, and
`(c + 0.5) * scale − 0.5`), but it would shift every stored `meta.json` and
invalidate the bit-identical replay checks, so it has not been applied.

### Step 2 — fix the extent (this is the crop)

We only want 63 degrees. At 9.1 px/degree, 63 degrees is 640 pixels. Keep 640
columns, discard the other 401.

```
                 pixels/degree    degrees covered
  after step 2        9.1              63 deg    <- both RIGHT
```

Cropping throws away *degrees* without touching *density* — the exact opposite
of what resizing does. That is why you need both, and why neither alone works.

```
  1280 x 720, 90 deg, too dense (11.1 px/deg)
  +--------------------------------------+
  |                                      |
  |            whole room                |
  |                                      |
  +--------------------------------------+
                    |
                    |  STEP 1: shrink x0.81  -> density now 9.1 px/deg
                    v
  1041 x 586, still 90 deg, density correct
  +--------------------------------+
  |      +------------------+      |
  |      |  640 x 480       |      |   STEP 2: keep this box,
  |      |  = 63 deg        |      |           throw away the rest
  |      +------------------+      |
  +--------------------------------+
        ^194                  ^207
                    |
                    v
  640 x 480, 63 deg, 9.1 px/deg  ->  looks like a Kinect photo
```

### Why the crop sits at 194, not in the middle

The middle of a 1041-wide image is 520, so a centred crop would start at 200.
We start at **194**.

The crop aligns to the **lens axis** — the point the lens actually looks
straight through — not the middle of the sensor. On the D455 those differ: the
axis sits at pixel 638.22 of 1280, not 640. NYU's images have their axis at
325.58. So we cut exactly enough from the left that ours lands at 325 too.

Crop centrally instead and the whole reconstruction shifts sideways, because
the geometry would believe the camera points somewhere it does not.

### What is left over

| | wanted | achieved | off by |
| --- | --- | --- | --- |
| horizontal density (`fx`) | 518.8579 | **518.8579** | **0** — what step 1 solved for |
| lens axis x (`cx`) | 325.58 | 325.21 | 0.37 px |
| lens axis y (`cy`) | 253.74 | 253.42 | 0.32 px |
| vertical density (`fy`) | 519.4696 | 518.2329 | 1.24 px |

The two **0.3 px** residuals exist because you cannot crop half a pixel —
193.628 has to become 194. Harmless: `match_nyu_fov` reports the shifted
principal point rather than NYU's, so the encoder unprojects with (very nearly)
the truth and the error cancels instead of becoming a systematic voxel shift.
"Very nearly" because of the sub-pixel bias in [One caveat: the achieved scale
is not exactly `s`](#one-caveat-the-achieved-scale-is-not-exactly-s) — 1.4 mm at
3 m, against 80 mm voxels.

The **1.24 px** one is structural. Step 1 shrinks both axes by the same factor
and chose that factor to fix the horizontal density; the D455's vertical
density is not in quite the same proportion to its horizontal as the Kinect's
was, so it lands 0.24% off and no crop can move it. Measured to be harmless —
see [It is `fx`, not the pixel aspect ratio](#it-is-fx-not-the-pixel-aspect-ratio).

### In one line

**Resize fixes how big things look; crop fixes how much you see.** You need
both, in that order — and 1280×720 is the right starting profile because it
makes step 1 a shrink rather than a stretch.

### Optional: an exact match

Four constraints (`fx`, `fy`, `cx`, `cy`) against three knobs (scale, `x0`,
`y0`) leaves one residual, and it lands on `fy`. Adding a fourth knob — scaling
the axes independently, which is still a valid pinhole camera — makes the
system exactly determined:

```
sx = 0.813526   sy = 0.815467      (0.239% apart)
1280x720 -> 1041x587, crop at (194, 43)
achieved  fx=518.8579  fy=519.4696  cx=325.2080  cy=254.1305
residual  fx=+0.0000   fy=+0.0000   cx=-0.3720   cy=+0.3905   <- rounding only
```

Both focal lengths exact, at the cost of a 0.239% vertical stretch. Not
implemented, and not recommended: the ablation shows pixel aspect ratio barely
affects the network, and `K_eff` keeps the geometry exact either way. Listed so
the option is on record.

---

## Matching depth to colour

A second kind of matching, easy to confuse with the first. They fix unrelated
problems:

*(The deproject → transform → project math, and why alignment leaves holes, is
in [GEOMETRY.md §2](GEOMETRY.md#2-depth--colour-align).)*

| | what is wrong | who fixes it |
| --- | --- | --- |
| **FOV match** | one lens has the wrong *zoom* | our code, `match_nyu_fov` |
| **Alignment** | two lenses are in different *places* | the SDK, `rs.align` |

### Your camera has three eyes

```
  [left IR]      [right IR]           [RGB]
      |                                 |
      +--------- 59 mm apart ------------+
```

Depth is computed from the **IR pair**, so it lives in the left IR lens's
viewpoint. Colour comes from the **RGB lens**, 59 mm to the side. So depth
pixel (100, 200) and colour pixel (100, 200) are **not looking at the same
thing** — they are two photographs taken 6 cm apart.

The measured extrinsic is `(-58.967, -0.068, 0.471) mm`: essentially a pure
sideways offset, the other two axes being mechanical tolerance.

### Why 6 cm is a lot

Hold up a finger and blink one eye, then the other. It jumps — and near things
jump further than far things. Same effect here:

```
object at 0.5 m  ->  the two views disagree by  61 px
object at 1.0 m  ->                             31 px
object at 2.0 m  ->                             15 px
object at 4.0 m  ->                              8 px
```

**The disagreement depends on distance.** That is the part that matters: no
fixed shift can correct it, because every pixel needs its own correction based
on how far away that pixel is. Skip it and the network reads the colour of the
wall while looking at the geometry of the chair in front of it.

### The fix

`rs.align(rs.stream.color)`, per pixel:

1. turn the depth pixel into a 3D point, in IR-lens coordinates
2. shift that point by the 59 mm offset, into RGB-lens coordinates
3. project it back onto the RGB image

The output is depth **as if the depth sensor sat inside the colour lens**.

### NYU did the same thing

The Kinect has the same layout — IR pair and a separate RGB camera, a few
centimetres apart — and the NYU toolbox solves it the same way, with
`project_depth_map` reprojecting raw depth onto the RGB image plane. **The
depth maps the dataset ships are already aligned** (the unaligned ones are kept
separately as `rawDepths`).

So this is not us being fancy. Training data was aligned depth, so inference
must be aligned depth — the same reasoning as the FOV crop.

### Three consequences

**1. `cam_K` must be the COLOUR intrinsics.** After alignment the depth image
lives in the colour lens's frame, so it is described by `fx 637.79, cx 638.22`,
**not** the depth stream's own `fx 648.09, cx 635.97`. Using the depth
intrinsics would put a ~10 px error into every unprojection. `camera.py` reads
from `rs.stream.color` for exactly this reason.

**2. The distortion comes along for the ride.** The raw depth stream has *zero*
distortion coefficients — already rectified, so it looks like nothing needs
correcting. After alignment it inherits the colour lens's. That is the trap
behind [`--no_undistort`](#--no_undistort): the distortion arrives with the
alignment, not with the depth sensor.

**3. You get shadow stripes.** Some surfaces the colour lens can see are hidden
from the IR lens behind a nearer object; those pixels return 0. Because the
offset is sideways, the shadows are **vertical bands beside object edges** —
roughly 20 px wide for an object at 1 m against a wall at 3 m. NYU has the same
artifact about a third as wide, the Kinect's lenses being closer together. Not
fatal, since 0 is already handled as invalid, but it is why depth goes missing
beside edges rather than at random.

---

## Frame size and FOV — do not skip the crop

The most tempting change, and the one that costs the most. What follows is what
happens if you remove the crop described above.

### The ablation

Same physical frame, three projections, run through the full pipeline:

| | fx | fy | FOV | observed | agreement with A |
| --- | --- | --- | --- | --- | --- |
| **A** crop (current) | 518.86 | 518.23 | 63.3 × 49.7° | 45.1% | 1.0000 |
| **B** squash to 640×480 | 318.89 | 424.68 | 90.2 × 58.9° | **52.2%** | 0.7742 |
| **C** scale + pad, square px | 318.89 | 318.51 | 90.2 × 74.0° | **52.2%** | 0.7744 |

### Geometry is projection-agnostic

Unprojection is analytically correct for *any* honest `K`, and the results say
so — the floor lands in the same place to within one voxel:

```
A  floor 3614  z = 0.00-0.08 m
B  floor 3620  z = 0.00-0.08 m
C  floor 3619  z = 0.00-0.08 m
```

You also gain genuine coverage: **observed rises 45.1% → 52.2%**. A 63°
frustum only reaches the volume's ±2.4 m walls at 3.9 m depth; a 90° frustum
gets there at 2.4 m, so the near corners stop being pure guesswork.

### Semantics do not survive

Class share of occupied voxels (%):

| class | A crop | B squash | C pad | |
| --- | --- | --- | --- | --- |
| ceiling | 1.7 | 2.7 | 1.9 | |
| floor | 5.4 | 5.4 | 5.4 | |
| wall | 33.6 | 26.1 | 26.2 | |
| window | 6.0 | **0.5** | **0.5** | ← near-total collapse |
| chair | 5.4 | 8.9 | 8.5 | |
| bed | 9.9 | 6.1 | 6.8 | |
| sofa | 4.1 | 2.4 | 2.5 | |
| table | 4.6 | 5.1 | 4.8 | |
| tvs | 0.0 | 0.0 | 0.0 | |
| furniture | 8.6 | **15.8** | **16.8** | ← mass flows in |
| objects | 20.6 | **27.1** | **26.6** | ← and here |

This is the signature of scale mismatch. Every object subtends fewer pixels
than the backbone expects, so the model loses confident large-structure
identity — window, wall, bed, sofa — and dumps the mass into the generic
catch-alls. Windows go from 6% to essentially zero.

The ceiling check makes it obvious:

```
A  ceiling 1147   z = 2.08-2.80 m     plausible
B  ceiling 1806   z = 0.48-2.80 m     ceiling at half a metre
C  ceiling 1295   z = 0.56-2.80 m
```

### It is `fx`, not the pixel aspect ratio

B and C share an `fx` of 318.89 but differ sharply in `fy` (424.68 vs 318.51)
— non-square versus square pixels. They agree with A at **0.7742 and 0.7744**:
indistinguishable.

So non-square pixels cost essentially nothing, and **apparent object scale is
the entire effect**. This also rules out the tempting "pad instead of crop to
keep pixels square" fix — C is no better than B.

### Why it can't be normalised away

The 2D branch is `mit_b2` → `SegFormerHead`, trained on NYU-projection
imagery. Transformers and CNNs are translation-equivariant but **not
scale-equivariant**; no normalisation layer undoes a 1.63× change in apparent
object size. The 3D branch is geometric and doesn't care. CleanerS fuses them,
so the learned half drags the result down.

### If you want the wider FOV anyway

**Tile, don't stretch.** After the isotropic resize the frame is 1041 px wide
and the crop uses 640 of them, leaving 207 px of margin. Take three 640-wide
crops at `x0 = 0, 194, 401`, run the network on each, and merge in world
coordinates. Each tile is a genuine 63.3° in-distribution frame, and together
they span the full 90.2°.

Costs: 3× the ~1.5 s per-frame runtime, plus a merge rule for the overlaps
(max-confidence or majority vote). Not implemented.

The unglamorous alternatives — fine-tuning the 2D branch on D455-projection
data, or stepping the camera back so more of the room fits in 63° — both work.

---

## `--camera_height`

The most important number you type. The voxel grid is floor-anchored (world
Z=0 is the floor, `vox_origin[2] = -0.05`), so this shifts the entire
reconstruction vertically. **Measure it; don't guess.**

NYU's own frames sit at 1.18–1.32 m, which is where the 1.25 default comes
from. `room01` used 1.42 m.

Symptoms of a wrong value: floor voxels predicted above the 0.00–0.08 m band,
ceiling drifting toward the grid's top or bottom edge, and `floor`/`ceiling`
being the first two classes to collapse.

Height is stored in `meta.json` at capture time and read back automatically.
Precedence at inference is **CLI flag → capture's recorded value → 1.25**.

## Mounting — pitch and roll are not compensated

`from_live_camera` assumes the camera is level. There is no pitch/roll term.

A capture pitched upward produced **ceiling 10.3% vs floor 4.2%** — backwards
for a room whose floor is much larger than its visible ceiling. If that ratio
inverts, check the tripod before checking anything else.

Practical setup: tripod, level, scene 1–4 m away, and a measured lens height.

The D455's IMU is present and working (accel 63/250 Hz, gyro 200/400 Hz) and
could supply true gravity alignment. Not wired in.

## Range and the blind spot

The D455 has a **~0.5 m minimum range** — `min-Z = fx·B/D_max`, derived and
measured in [GEOMETRY.md §1](GEOMETRY.md#1-min-z-what-stereo-can-and-cannot-measure).
Anything closer returns invalid, and
a subject held at arm's length is inside it. Diagnosed exactly this once —
valid depth swung 62% → 25.5% → 41.6% across captures until a preview render
showed a phone at arm's length filling the frame.

`--depth_max` (default 10.0 m) marks anything beyond as invalid. This is not
cosmetic: the disparity→depth conversion maps near-zero disparity to tens of
metres, so a filtered frame carries a few hundred nonsense pixels. At uint16
those wrap — a 70 m reading becomes 4 mm, a phantom surface pressed against
the lens. `_to_uint16_mm` clamps to 0 as a second line of defence.

A D455 is only trustworthy to about 6 m regardless.

## `--filters` and `--hole_fill`

`--filters` runs the SDK's spatial + temporal edge-preserving filters, in
disparity space (librealsense's documented order), which is where the sensor's
noise is actually uniform.

The temporal filter needs history, so warmup frames are pushed through it
before the first real frame. **Never set `--warmup 0` with filters on.**

`--hole_fill` is deliberately separate and off by default. It **invents depth
for pixels the sensor never measured**, and the TSDF encoder cannot tell an
invented surface from a real one — it stamps both as observed geometry.
Completing unseen space is the network's job, not a filter's. Turning this on
does not add information; it launders a guess into an observation.

## `--warmup`

Default 30 frames (1 s at 30 fps). A D4xx opens with auto-exposure and
white-balance unconverged: the first frames are dark and the depth
correspondingly sparse.

**Never set this to 0 for a single-shot capture** — that one frame *is* the
unconverged one.

## `--no_undistort`

Leave undistortion on. The D455's colour stream is `inverse_brown_conrady`
with real coefficients, and after align-to-color the depth inherits them,
while `frame_loader` unprojects with a plain pinhole model.

Measured on this unit: **3.34 cm of geometric error at the crop corners**
(1.7 high-res voxels), zero at centre. Correction brings it to **0.0025 cm**.

Note the trap: the *depth* stream's own coefficients are all zero, so it looks
like nothing needs correcting. The distortion arrives with the alignment.

## Stream profile and USB

**USB 3 is mandatory.** On USB 2.1 the SDK *silently removes* high-rate
profiles rather than erroring — measured: 1280×720@5 got 0/20 frames,
640×480@30 got 2/20, 424×240@30 failed to start. The same camera on USB 3.2
does 30/30 in 1.4 s (20.9 fps).

If frames stop arriving, check `usb_type_descriptor` first. Device permissions
are *not* the usual cause.

**Capture at 1280×720, not 640×480.** The FOV match scales by
`fx_NYU/fx_source`. From 1280×720 that is 0.8135 — a downscale, which loses
nothing. From a lower stream resolution the factor exceeds 1 and the code must
*upscale*, inventing pixels. `RealSenseSource` logs
`UPSCALE (prefer a higher stream resolution)` when this happens.

## Thermal drift

Intrinsics move as the ASIC warms — firmware rewrites the calibration:

```
fx 637.7892 @ 32 °C   ->   638.3094 @ 35 °C      cx, cy unchanged
```

About 1.6 px, 0.24%, roughly 4 mm at 3 m. Small, but it is why `cam_K` is
written into `meta.json` per-capture rather than hardcoded, and why
`match_nyu_fov` returns *measured* intrinsics instead of assumed ones.

If you need repeatability across a session, let the camera warm up for a few
minutes before the captures that matter.

---

## Reproducing the ablation

Variant A above was recomputed from `captures/room01/native/` and differs
marginally from the stored `room01` run (floor 3614 vs 3613, ceiling 1147 vs
1093). **Not** because of the PNG round-trip — depth survives that
bit-identically (verified: 0 differing pixels; `depth_scale` is 0.001 so
`metres × 1000` recovers the original z16 integer, and both depth resampling
steps use `INTER_NEAREST`, which relocates samples without averaging).

The cause is RGB rounding in the ablation script: it loads the native colour
frame as float32, runs `INTER_AREA`, then casts with `.astype(np.uint8)`,
which *truncates*, whereas `capture.py` runs `INTER_AREA` on uint8 directly,
which *rounds*. A ±1 per-channel difference into the semantic branch is enough
to move borderline voxels.

Bands and conclusions are unaffected, but compare variants against each other,
not against a stored run.

The `native/` folder exists precisely so this kind of question can be answered
without re-shooting the scene. Don't pass `--no_native` on a capture you might
want to re-examine.
