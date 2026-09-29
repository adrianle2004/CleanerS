# Geometry: from a D455 to a CleanerS input frame

How a raw stereo pair becomes the `(480, 640)` depth image the encoder expects,
and why each step is shaped the way it is. Four stages, in pipeline order:

1. [Min-Z: what stereo can and cannot measure](#1-min-z-what-stereo-can-and-cannot-measure)
2. [Depth → colour (align)](#2-depth--colour-align)
3. [Undistortion to an ideal pinhole](#3-undistortion-to-an-ideal-pinhole)
4. [Matching NYU's field of view](#4-matching-nyus-field-of-view)

Every number here was either read out of `captures/room01/meta.json` or
recomputed from it; the reproductions are marked ✓.

> [CAMERA_TUNING.md](CAMERA_TUNING.md) covers the same pipeline in plain
> language — pixels-per-degree, "your camera has three eyes", which knob to
> turn. Read that one first if you want the intuition. This one is the algebra
> behind it, for when you need to change the code rather than the settings.

### Notation

| symbol | meaning |
| --- | --- |
| `(u, v)` | pixel coordinates, `u` right, `v` down |
| `Z` | depth along the optical axis, metres |
| `K` | `[[fx, 0, cx], [0, fy, cy], [0, 0, 1]]` |
| `d` / `c` / `n` | subscripts: depth stream, colour stream, NYU-matched |
| `B` | baseline, metres |
| `D` | disparity, pixels |

---

## 1. Min-Z: what stereo can and cannot measure

### The one equation

A stereo pair recovers depth by finding the same patch in both images and
measuring how far it moved. From `librealsense/doc/depth-from-stereo.md:44`:

```python
depth[disparity > 0] = (fx * baseline) / (units * disparity[disparity > 0])
```

that is

```
        fx · B
  Z  =  ──────
          D
```

Depth is **inversely** proportional to disparity. Near things shift a lot, far
things barely shift — which is also why stereo precision degrades quadratically
with range (`∂Z/∂D = −fx·B/D² = −Z²/(fx·B)`).

### Where the floor comes from

The matcher searches a **finite** window of disparities, `D ∈ [0, D_max]`. The
largest disparity it can report therefore sets the smallest depth it can
report:

```
             fx · B
  min-Z  =  ────────
             D_max
```

For our capture at 1280×720:

| term | value | source |
| --- | --- | --- |
| `fx_d` | 674.4 px | `(1280/2) / tan(87°/2)`, D455 depth HFOV |
| `B` | 95 mm | D455 stereo baseline |
| `D_max` | 128 | D4xx ASIC disparity search range |
| **min-Z** | **0.501 m** | `674.4 × 0.095 / 128` |
| **measured floor**, `room01` | **0.493 m** | ✓ 1.5% agreement |

**This is a wall, not a taper.** `D_max` is a hard integer bound on the search,
so the matcher does not degrade gracefully as you approach 0.5 m — it simply
runs out of places to look. The histogram of `room01` shows exactly that shape:

```
  0.40-0.45 m :        0     <- nothing at all below the floor
  0.45-0.50 m :    2,580
  0.50-0.55 m :   10,598
  0.55-0.60 m :   35,898
```

Note the D455 is *worse* here than a D435 despite being the better camera: its
95 mm baseline (vs ~50 mm) is what buys the long-range accuracy, and the same
term in the numerator pushes min-Z out. A design trade, not a defect.

### Consequences you will actually hit

Anything inside 0.5 m returns `0` — the desk the camera stands on, a hand, a
subject at arm's length. Diagnosed twice in this repo now
([CAMERA_TUNING.md](CAMERA_TUNING.md#range-and-the-blind-spot)): once a phone at
arm's length, once the desk under the tripod, which cost 79.8% of the bottom
eighth of the frame.

No filter recovers it. Post-processing can only redistribute measurements that
exist — running `room01` through the SDK's own decimation filter moved holes
33.1% → 30.7% and left the 0.493 m floor untouched. This is the same reason
`--hole_fill` is off by default: it would paint over the gap with invented
geometry, and the TSDF encoder cannot tell an invented surface from a measured
one.

### The two levers, and why we pull neither

**Capture resolution.** `min-Z ∝ fx ∝ width`:

| mode | `fx_d` | min-Z | px/deg | into `match_nyu_fov` |
| --- | --- | --- | --- | --- |
| 1280×720 *(ours)* | 674.4 | 0.50 m | 11.8 | ×0.77 downsample |
| 848×480 | 446.8 | **0.33 m** | 7.8 | ×1.16 upsample |
| 640×360 | 337.2 | 0.25 m | 5.9 | ×1.54 upsample |

NYU is 9.1 px/deg. At 1280×720 we sit *above* it and downsample, which is the
right side to be on (§4). 848×480 drops below and interpolates.

**`disparityShift`** (advanced mode, `STDepthTableControl.disparityShift`,
`rs_advanced_mode_command.h:111-117`) moves the search window to
`[S, S + D_max]`, buying near range by surrendering far range:

| shift | min-Z | max-Z |
| --- | --- | --- |
| 0 *(default)* | 0.50 m | ∞ |
| 25 | 0.42 m | 2.56 m |
| 50 | 0.36 m | 1.28 m |
| 100 | 0.28 m | 0.64 m |

**A non-starter for SSC.** The voxel grid is 4.8 m deep, so even `S=25`
truncates it at 2.56 m — trading a hole in front for a missing back half of the
room. Move the camera instead.

---

## 2. Depth → colour (align)

### Why it is mandatory

The depth origin (left IR imager) and the colour lens sit apart by a measured
`t = (−58.967, −0.068, 0.471) mm` — essentially a pure sideways offset, the
other two axes being mechanical tolerance. Two cameras 59 mm apart do not see
the same thing at pixel `(u, v)`, and the disagreement is **depth-dependent**:

```
                 fx · B_dc
  Δu(Z)  =  ─────────────────
                    Z
```

In the final 640×480 matched frame (`fx = 518.86`, `B_dc = 0.059`):

| `Z` | `Δu` |
| --- | --- |
| 0.5 m | 61.2 px |
| 1.0 m | 30.6 px |
| 2.0 m | 15.3 px |
| 4.0 m | 7.7 px |

*(These are the figures quoted in `README.md` and `CAMERA_TUNING.md`; neither
says which frame they are in. It is the matched frame, not the native one.)*

Because `Δu` depends on `Z`, **no fixed shift corrects it** — every pixel needs
its own correction based on how far away that pixel is. Skip alignment and the
network reads the colour of the wall while reading the geometry of the chair in
front of it.

### The math the SDK runs

`rs.align(rs.stream.color)` — `librealsense/src/proc/align.cpp:50-92`. For each
depth pixel with `Z ≠ 0`, three steps:

**① Deproject** — pixel + depth → a 3D point in the depth camera's frame
(`rs.cpp:4342`):

```
  x = (u_d − cx_d) / fx_d
  y = (v_d − cy_d) / fy_d
  P_d = (x·Z,  y·Z,  Z)
```

If the intrinsics carry distortion, `x, y` are first *un*distorted by a
10-iteration fixed-point solve — see §3, this is the direction that needs
iterating.

**② Transform** — into the colour camera's frame by the factory extrinsics
(`rs.cpp:4430`, column-major `R`):

```
  P_c  =  R · P_d  +  t          t = (−58.967, −0.068, 0.471) mm
```

`R` is near-identity (the lenses are coplanar to within calibration tolerance),
so this is dominated by the sideways `t` — which is exactly why the disagreement
is a horizontal shift that scales as `1/Z`.

**③ Project** — into the colour image (`rs.cpp:4270`):

```
  x = P_c,x / P_c,z ,   y = P_c,y / P_c,z
  (apply the distortion polynomial forward — see §3)
  u_c = x·fx_c + cx_c
  v_c = y·fy_c + cy_c
```

### Why alignment creates holes

Two structural details of the implementation:

**It is a forward scatter, not a gather.** The loop iterates over *depth*
pixels and writes into the colour grid, which is `memset` to zero first
(`align.cpp:103`). Any colour pixel that no depth pixel happens to land on
stays `0`. To limit that, the SDK maps both the **top-left and bottom-right
corners** of each depth pixel and fills the whole rectangle between them
(`align.cpp:63-89`) — which is why upsampling to a larger colour image does not
leave a grid of gaps.

**Occlusion is resolved by a z-buffer.** When two depth pixels land on the same
colour pixel, the nearer one wins (`align.cpp:118-120`):

```cpp
out_z[other] = out_z[other] ? std::min(out_z[other], z_pixels[i]) : z_pixels[i];
```

Correct, but it means surfaces the depth camera saw *behind* a foreground edge
are discarded. The residue is an occlusion shadow: a band on one side of every
depth discontinuity, `Δu(Z)` wide, that the colour camera can see but the depth
camera never could. At 0.5 m that band is 61 px.

These shadows are edge-shaped and unavoidable. They are *not* the same thing as
the min-Z dropout in §1, which is region-shaped — in `room01` the largest hole
is a single 292,478-pixel blob whose surviving rim sits at 0.52 m, right on the
min-Z floor.

---

## 3. Undistortion to an ideal pinhole

### Why

`frame_loader.unproject()` uses a plain pinhole model. The D455's **colour**
lens has real distortion, and after align-to-colour the depth inherits it.
Uncorrected, the reprojection is off by **3.34 cm**; corrected, **0.0025 cm**
(`camera.py:_build_undistort_maps`, verified against
`rs2_deproject_pixel_to_point`).

### The direction that matters

`cv2.remap` needs, for each *output* pixel, the *source* pixel to sample. The
output is an ideal pinhole grid, so:

```
  x = (u_out − cx) / fx                      ideal normalised ray
  y = (v_out − cy) / fy

  r² = x² + y²
  f  = 1 + k₁r² + k₂r⁴ + k₃r⁶                radial
  x' = x·f + 2p₁xy + p₂(r² + 2x²)            + tangential
  y' = y·f + p₁(r² + 2y²) + 2p₂xy

  u_src = x'·fx + cx
  v_src = y'·fy + cy
```

**The polynomial is applied forward**, even though the D455 colour model is
named `RS2_DISTORTION_INVERSE_BROWN_CONRADY`. The name describes what the
coefficients were *fitted for* — deprojection — so they already encode the
inverse map. Inverting them again doubles the error instead of removing it.

The SDK is the proof: `rs2_project_point_to_pixel` applies this polynomial
directly (`rs.cpp:4274-4286`), while `rs2_deproject_pixel_to_point` needs a
10-iteration fixed-point solve to go the other way (`rs.cpp:4352-4369`). Going
*to* a pixel is the closed-form direction.

Intrinsics are unchanged by construction — the output is the same camera with
zero distortion — so `cam_K` stays valid.

> **Precision footnote.** The SDK's `INVERSE_BROWN_CONRADY` branch scales `x, y`
> by `f` *before* forming the tangential terms; the `BROWN_CONRADY` branch (and
> our code) uses the unscaled `x, y` there. The difference is second-order in
> `p₁, p₂`, and it is already inside the 0.0025 cm residual measured against the
> SDK's own deprojection — but worth re-checking against this unit's real
> coefficients, which `meta.json` does not currently record.

Depth is remapped `INTER_NEAREST`, RGB `INTER_LINEAR`. Interpolating depth
across a discontinuity invents a surface halfway between foreground and
background, and the TSDF encoder cannot tell that from an observation.

---

## 4. Matching NYU's field of view

### Why

The pretrained checkpoint was trained on NYU Depth v2 — a Kinect v1, `fx =
518.86` at 640×480, a **63.3° × 49.6°** field of view. A D455 is far wider:
**90.2° × 58.9°** on colour. ✓ (both recomputed from `meta.json`)

Feeding the native frame in is *geometrically* fine — the encoder uses whatever
`cam_K` it is handed — but it puts the RGB branch far outside its training
distribution. The same wall occupies a completely different pixel footprint,
and SegFormer features do not transfer for free.

So `match_nyu_fov` (`camera.py:102`) resizes and centre-crops to 640×480 with
NYU's intrinsics.

### The math

Isotropic scale so the source focal length matches NYU's:

```
  s = fx_NYU / fx_src
```

Resample the whole frame by `s`, then crop a 640×480 window whose top-left is
placed so the principal point lands on NYU's:

```
  W_r = round(W_src · s)          H_r = round(H_src · s)
  x₀  = round(cx_src · s − cx_NYU)
  y₀  = round(cy_src · s − cy_NYU)
```

The effective intrinsics are then **measured from what was actually done**, not
assumed:

```
  K_eff = [[fx_src·s,        0,  cx_src·s − x₀],
           [       0, fy_src·s,  cy_src·s − y₀],
           [       0,        0,              1]]
```

That last point is the whole design. If the crop has to be clamped or padded at
a frame edge, `x₀`/`y₀` absorb it and `K_eff` still describes the output
exactly, so the unprojection stays correct. Nothing downstream assumes NYU's
`K`; it reads `K_eff`.

### Worked example — `room01`

```
  source   1280×720,  fx 637.7892  fy 637.0209  cx 638.2195  cy 364.3684
  s      = 518.8579 / 637.7892  =  0.813526
  resize   1280×720  ->  1041×586
  x₀     = round(638.2195 × 0.813526 − 325.58)  =  194
  y₀     = round(364.3684 × 0.813526 − 253.74)  =   43
  crop     640×480 at (194, 43)

  K_eff    fx 518.8579  fy 518.2329  cx 325.2080  cy 253.4230
  meta     fx 518.8579  fy 518.2329  cx 325.2080  cy 253.4230   ✓ exact
```

Resulting FOV **63.3° × 49.7°** against NYU's **63.3° × 49.6°**. ✓

### Three things this pins down

**Stream at 1280×720, not 640×480.** `s = 0.8135 < 1` makes this a *downscale*,
which discards detail rather than inventing it. From 640×480 the same match
would need `s = 1.63` — upscaling, i.e. interpolating pixels that were never
measured. This is the reason the default profile is 1280×720, and it is in
direct tension with §1, where a *lower* capture resolution would improve min-Z.

**`fy` lands 0.24% off and that is fine.** D455 `fy/fx = 0.99880` vs NYU
`1.00118` — genuinely different pixel geometry, which no crop can fix. Harmless
precisely because `K_eff` reports the true value instead of asserting NYU's.

**Do not skip the crop.** Keeping the full 90° FOV leaves the geometry intact
but costs 23% of voxel labels, collapsing the `window` class from 6.0% to 0.5%
([CAMERA_TUNING.md](CAMERA_TUNING.md#frame-size-and-fov--do-not-skip-the-crop)).
And 640×480 is not negotiable regardless: `MAPPING_SENTINEL = 480 × 640 =
307200` is baked into both the encoder and `test_NYU.py`'s SC evaluation mask.

### Applying it twice

`match_nyu_fov` is not idempotent. Run it on an already-matched frame and `s`
is computed from intrinsics that describe the *output*, cropping a second time.
`FolderSource` guards this by setting `self.fov = 'none'` when
`meta['fov'] == 'nyu'` (`camera.py:518`).

---

## The whole chain

```
  D455 stereo pair
       │  matcher, D ∈ [0, 128]        §1  floor at 0.493 m
       ▼
  depth 1280×720  in the DEPTH frame,  K_d
       │  deproject → R,t → project    §2  +occlusion shadows, +scatter holes
       ▼
  depth 1280×720  in the COLOUR frame, K_c = [637.79, 637.02, 638.22, 364.37]
       │  remap, forward polynomial    §3  3.34 cm → 0.0025 cm
       ▼
  depth 1280×720  ideal pinhole,       K_c unchanged
       │  ×0.813526, crop at (194,43)  §4
       ▼
  depth 640×480   NYU-matched,         K_eff = [518.86, 518.23, 325.21, 253.42]
       │  ×1000 → uint16 mm                capture.py
       ▼
  depth/live_000000.png                    the encoder's input
```

`captures/*/native/` holds the frame as it stood after §3 and before §4, so a
capture can be re-cropped without returning to the hardware. Verified:
`match_nyu_fov(native)` reproduces the shipped 640×480 depth **bit-for-bit**
(100.0000% of pixels, `room01`).

---

## References

**In this repo**

| path | what |
| --- | --- |
| `inference/camera.py:102` | `match_nyu_fov` |
| `inference/camera.py:289` | `_build_undistort_maps` |
| `inference/camera.py:363` | `RealSenseSource.frames` — align, filter, remap, match |
| `inference/frame_loader.py:34-46` | `CAM_K`, `IMG_H/W`, `MAPPING_SENTINEL` |
| [CAMERA_TUNING.md](CAMERA_TUNING.md) | capture knobs, measured |
| [README.md](README.md) | module reference, verified device numbers |

**librealsense** — checked out at `/home/adrian-le/CleanerS/librealsense`

| path | what |
| --- | --- |
| `doc/depth-from-stereo.md` | `Z = fx·B/D`, with a worked NumPy stereo matcher |
| `src/proc/align.cpp:50-121` | the align scatter loop and its z-buffer |
| `src/rs.cpp:4270` | `rs2_project_point_to_pixel` — polynomial forward |
| `src/rs.cpp:4342` | `rs2_deproject_pixel_to_point` — 10-iteration solve |
| `src/rs.cpp:4430` | `rs2_transform_point_to_point` |
| `include/librealsense2/h/rs_advanced_mode_command.h:111` | `STDepthTableControl` |
| `doc/rs400/rs400_advanced_mode.md` | advanced mode |
| `doc/post-processing-filters.md` | what the Viewer's filters do |

**Intel** — *not verified from this machine, no network access.* The *D400
Series Datasheet* is the authority on per-SKU Min-Z and baseline; *Tuning depth
cameras for best performance* covers `disparityShift`. Both normally live under
`dev.intelrealsense.com/docs/`.
