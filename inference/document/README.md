# `inference/` — running CleanerS on a live RealSense D455

CleanerS was trained on NYU Depth v2 and hardcodes assumptions a D455 does not
satisfy. This package is the adapter layer that closes the gap, plus a
capture-to-disk tool and two runners.

| CleanerS expects | D455 gives |
| --- | --- |
| 640×480, `MAPPING_SENTINEL = 307200` | 1280×720 |
| Kinect FOV, `fx ≈ 518.9` | `fx ≈ 637.8` (90° horizontal) |
| Depth as bit-rotated PNG, `((d << 13) \| (d >> 3)) / 1000` | plain uint16 millimetres |
| Rectified pinhole | Brown-Conrady distortion on the colour lens |
| GT-derived `label_weight` mask for display | no ground truth at all |

For what the capture settings do to the output — with measured numbers — see
[CAMERA_TUNING.md](CAMERA_TUNING.md). For hand-making ground truth so a live
capture can be evaluated the way NYU is, see [MAKING_GT.md](../../reconstruction_GT/document/MAKING_GT.md),
and for what those captures established once scored —
including why SC has to be read against its own baseline — see
[FINDINGS.md](../../reconstruction_GT/document/FINDINGS.md).
For why furniture gets extruded through the wall behind it — measured over all
1449 NYU frames — see [OCCLUSION.md](OCCLUSION.md). For what the model is
actually reliable at, which classes to distrust, how to hold the camera and
which metric to quote — measured on hand-annotated real rooms against NYU and
Occ-ScanNet — see [USING_THE_MODEL.md](USING_THE_MODEL.md).

## Contents

- [Data flow](#data-flow)
- [Quick start](#quick-start)
- [Module reference](#module-reference) — every file, every function
  - [`camera.py`](#camerapy--frame-sources)
  - [`capture.py`](#capturepy--capture-to-disk)
  - [`frame_loader.py`](#frame_loaderpy--geometry-encoder)
  - [`tsdf_torch.py`](#tsdf_torchpy--the-fast-tsdf-encode)
  - [`model.py`](#modelpy--checkpoint-loading-and-forward-pass)
  - [`visualize.py`](#visualizepy--voxels--ply)
  - [`run_inference.py`](#run_inferencepy--batch--single-frame-runner)
  - [`run_live.py`](#run_livepy--streaming-runner)
  - [`live_view.py`](#live_viewpy--the-live-window)
  - [`utils.py`](#utilspy--shared-helpers)
  - [`display.py`](#displaypy--standalone-viewer)
  - [`display_solid.py`](#display_solidpy--solid-voxel-viewer)
  - [`reconstruct.py`](#reconstructpy--the-scene-the-camera-measured)
  - [`display_overlay.py`](#display_overlaypy--the-two-in-one-window)
  - [`colorize_depth.py`](#colorize_depthpy--depth-pngs--realsense-colour)
  - [`evaluate.py`](#evaluatepy--per-class-metrics)
  - [`nyu_occlusion_sweep.py`](#nyu_occlusion_sweeppy--visible-vs-occluded-recall)
  - [`run_scannet.py`](#run_scannetpy--occ-scannet-runner)
  - [`evaluate_scannet.py`](#evaluate_scannetpy--occ-scannet-metrics)
  - [`export_scannet_ply.py`](#export_scannet_plypy--best-and-worst-frames-as-ply)
  - [`show_nyu.sh` / `show_scannet.sh`](#show_nyush--show_scannetsh--prediction-next-to-gt)
- [The `sample` dict](#the-sample-dict)
- [Coordinate conventions](#coordinate-conventions)
- [VIEWERS.md](VIEWERS.md) — every viewer button and mode, in one place
- [EVALUATION.md](EVALUATION.md) — per-class metrics on both NYU splits, what `label_weight` is, whether each of the eleven classes' failures is the model's doing or the data's, and the groups its false positives and its misses fall into
- [GEOMETRY.md](GEOMETRY.md) — the math: min-Z, align, undistort, FOV match
- [OCCLUSION.md](OCCLUSION.md) — what the model does behind what it cannot see
- [EVALUATION_SCANNET.md](EVALUATION_SCANNET.md) — zero-shot transfer to Occ-ScanNet val, against ISO/MonoScene, and what the scoring protocol does to the numbers
- [HARDWARE.md](HARDWARE.md) — what a frame costs on what: the per-stage timings, the VRAM floor, the minimum card, and the CPU-only case
- [CONCEPTS.md](CONCEPTS.md) — the terms HARDWARE.md uses, explained from the code: why the truncation is 12 voxels, what a 1-D pass of the distance transform actually does, what slabbing and projection are, and what fp16 and AMP are
- [What to notice](#what-to-notice)
- [Verified device numbers](#verified-device-numbers)
- [Documentation drift](#documentation-drift)
- [Open items](#open-items)

---

## Data flow

```
                    ┌─────────────┐
   D455 ──USB3──►   │  camera.py  │  ◄── adapter layer
                    │             │      RealSenseSource / FolderSource / NYUSource
                    │  yields     │      all yield a uniform `Frame`
                    │  Frame      │
                    └──────┬──────┘
                           │
            ┌──────────────┴──────────────┐
            ▼                             ▼
     ┌─────────────┐               ┌─────────────┐
     │ capture.py  │               │ run_live.py │  streaming loop
     │ writes disk │               │  (realtime) │
     └──────┬──────┘               └──────┬──────┘
            │  captures/room01/           │
            │    depth/live_000000.png    │
            │    rgb/live_000000.png      │
            │    native/…                 │
            │    meta.json  ◄─ contract   │
            ▼                             │
   ┌──────────────────────┐               │
   │ run_inference.py     │               │
   │   --live_dir         │               │
   └──────────┬───────────┘               │
              │                           │
              └─────────┬─────────────────┘
                        ▼
              ┌──────────────────┐
              │ frame_loader.py  │  depth → TSDF + 2D↔3D mapping
              └────────┬─────────┘
                       ▼
              ┌──────────────────┐
              │    model.py      │  VoxelSSC → (60,36,60) class logits
              └────────┬─────────┘
                       ▼
              ┌──────────────────┐
              │  visualize.py    │  → prediction/*.npy + ply/*.ply
              └──────────────────┘
```

`run_live.py folder:captures/room01` replays a capture through the streaming
path and produces **bit-identical** output to `run_inference --live_dir`
(verified: 0 of 129,600 voxels differ). That equivalence is the safety net —
anything debugged offline is what runs live.

---

## Quick start

### Capture then infer (D455)

```bash
# --preview opens a window first: type the camera height, then space (or the
# CAPTURE button) saves the frame. Without --preview, pass --camera_height 1.42.
python -m inference.capture --out_dir ./captures/room01 --filters --preview

python -m inference.run_inference \
    --cfg ./cfgs/NYU/voxelSSC.yaml \
    --pretrained_path ./checkpoint/CleanerS_ckpt.pth \
    --live_dir ./captures/room01 --out_dir ./outputs/room01
```

### Stream live

```bash
python -m inference.run_live \
    --cfg ./cfgs/NYU/voxelSSC.yaml \
    --pretrained_path ./checkpoint/CleanerS_ckpt.pth \
    --source realsense --camera_height 1.42 --display
```

### Stream without a camera

A `.bag` is the real sensor, recorded. Same pipeline, same alignment, same
intrinsics — and `captures/*/scan/scan.bag` already holds a 54–71 s sweep of
each room.

```bash
python -m inference.run_live \
    --cfg ./cfgs/NYU/voxelSSC.yaml \
    --pretrained_path ./checkpoint/CleanerS_ckpt.pth \
    --source bag:captures/room07/scan/scan.bag --display --colour all
```

### Verify without hardware

`nyu:` replay is the only mode whose output is directly comparable to the
reference results, because it reproduces the dataset's room-canonical grid.
Use it to prove the install works before trusting live output.

```bash
python -m inference.run_live ... --source nyu:./data/NYU/testset/depth \
    --rgb_dir ./data/NYU/testset/RGB --max_frames 20
```

### View

```bash
python inference/display.py ./outputs/room01/ply/live_000000.ply          # points
python inference/display_solid.py ./outputs/room01/ply/live_000000.ply    # cubes + photo
python inference/reconstruct.py ./outputs/room01/ply/live_000000.ply      # what the camera measured
python inference/display_overlay.py ./outputs/room01/ply/live_000000.ply  # both, overlaid
```

Every button and every mode of all four is written up in
**[VIEWERS.md](VIEWERS.md)**.

PLY coordinates are **voxel indices**, not metres — 60×36×60 at 8 cm each,
on left-handed axes that both viewers reflect so the room reads the same way
round as the RGB frame ([why](#displaypy--standalone-viewer)).
`display_solid.py` draws each voxel as a cube instead of a point, so the
surfaces have no gaps, and paints it with the colour the camera actually saw
([how](#display_solidpy--solid-voxel-viewer)).

---

## Module reference

### `camera.py` — frame sources

Source specs, parsed by `open_source`:

```
realsense                 D455 (or any D4xx) over USB
realsense:1280x720@30     ... with an explicit stream profile
bag:/path/to/scan.bag     a RealSense recording — the same pipeline as the
                          camera, hardware not required
nyu:/path/to/depth        NYU .bin/.png pairs (real vox_origin/cam_pose)
folder:/path/to/pngs      plain 16-bit depth PNGs in millimetres
```

#### `Frame` (dataclass)

The uniform yield type. Every source produces these; consumers never branch on
source kind, only on which optional fields are populated.

| field | type | meaning |
| --- | --- | --- |
| `depth` | (480,640) float32 | **metres**, already decoded |
| `name` | str | frame stem, e.g. `live_000000` |
| `rgb` | (480,640,3) uint8 | BGR |
| `cam_K` | 3×3 | effective intrinsics; `None` → NYU default |
| `vox_origin` | (3,) | NYU replay only — dataset grid placement |
| `cam_pose` | 4×4 | NYU replay only |
| `camera_height` | float | from `meta.json`; live sources only |
| `up_camera` | (3,) | measured gravity in camera coords; `bag:` only |
| `yaw` | float | from `meta.json` |
| `depth_native` | (H,W) float32 | pre-crop frame, when `keep_native=True` |
| `rgb_native` | (H,W,3) uint8 | pre-crop frame |
| `cam_K_native` | 3×3 | intrinsics of the pre-crop frame |

`vox_origin is None` is the switch consumers use: populated means "this frame
carries its own registration"; `None` means "build a floor-anchored grid".

#### `CameraSource`

Base class. `frames()` is a generator, `close()` releases hardware, and it is a
context manager so `with open_source(...) as src:` always shuts the sensor
down — a RealSense left running can need a USB replug before it reopens.

#### `_require_shape(depth, where)`

Raises unless depth is exactly (480, 640). The error message names
`MAPPING_SENTINEL` so the reason is visible at the failure site rather than
buried in `frame_loader`.

#### `match_nyu_fov(depth, rgb, K) -> (depth, rgb, K_eff)`

Isotropically scales by `fx_NYU / fx_source`, then crops 640×480 positioned so
the principal point lands at NYU's. From 1280×720 that is 1041×586 cropped at
`(194, 43)`, discarding 50% of the frame and narrowing 90.2° → 63.3°.

Returns intrinsics **describing the result**, not the ideal target — `x0`/`y0`
are rounded to whole pixels, so `cx`/`cy` land ~0.35 px off NYU's, and the
encoder must see those values or the error becomes a systematic voxel shift.

They are computed from the *requested* scale and a naive `c * s`, not from the
achieved integer ratio and cv2's half-pixel convention, so they carry a fixed
sub-pixel bias of up to 0.25 px — 1.4 mm at 3 m, 1.8% of a voxel. See
[CAMERA_TUNING.md](CAMERA_TUNING.md#one-caveat-the-achieved-scale-is-not-exactly-s).

Depth resamples with `INTER_NEAREST` deliberately: bilinear across a depth
discontinuity averages foreground with background and invents a surface that
was never observed. RGB uses `INTER_AREA`. Crop padding fills depth with 0
(== invalid); anything else would fabricate geometry.

#### `RealSenseSource`

```python
RealSenseSource(width=1280, height=720, fps=30, fov='nyu', emitter=True,
                align_to_color=True, warmup=30, filters=False,
                hole_fill=False, keep_native=False, undistort=True)
```

Constructor order matters and is: enumerate device → `_check_profile` →
start pipeline → build `rs.align(color)` → read depth scale and set emitter →
read **colour** intrinsics into `self.cam_K` → build undistort maps → build the
filter chain → run warmup.

Reading intrinsics from the *colour* stream is not a typo — aligned depth lives
in the colour image plane, so the colour intrinsics are the ones that describe
it. The depth stream's own `fx=fy 648.09, cx 635.97, cy 361.63` would put a
~10 px principal-point error into every unprojection.

**`_build_undistort_maps(intr, eps=1e-6) -> (mapx, mapy) | None`**
`cv2.remap` tables sending an ideal pinhole grid to source pixels. Returns
`None` when every coefficient is below `eps`. The polynomial is applied
**forward** — see gotcha 2; do not "fix" this.

**`_check_profile(rs, dev, width, height, fps)`**
Validates the requested profile before `pipeline.start`, reports the USB link
type from `usb_type_descriptor`, and lists resolutions supported for **both**
depth (`z16`) and colour (`bgr8`). Special-cases USB 2.x with cable advice.
Exists because librealsense otherwise fails with `RuntimeError: Couldn't
resolve requests`, which names neither the cause nor the fix.

**`frames()`**
Per frame: `wait_for_frames` → align → filter chain → scale to metres →
undistort remap (`INTER_NEAREST` depth, `INTER_LINEAR` rgb, `BORDER_CONSTANT`
0) → snapshot native if requested → `match_nyu_fov` → yield.

Undistortion sits **after** align (the aligned depth inherits the colour lens's
distortion) and **before** the crop (the coefficients are defined over the full
sensor frame in native pixel coordinates).

**`close()`** — stops the pipeline, swallowing errors.

#### `BagSource(path, fov='nyu', loop=False, fps=None, imu=True, realtime=False, ...)`

Replays a RealSense `.bag`. **Subclasses `RealSenseSource`** and replaces only
how the pipeline is opened: `_setup_from_profile` and `_process` are the
camera's own, so a recording is aligned, scaled, undistorted and FOV-matched by
exactly the code the live stream uses. That is also why the refactor is safe to
make without hardware — the shared half is the half a bag exercises.

Three things differ from the device:

- **`set_real_time(False)`.** librealsense waits for us instead of dropping
  frames we were too slow to collect, so every recorded frame is seen however
  long inference takes. `--fps` throttles if you want to watch it at a sane
  speed instead. `realtime=True` (`--realtime`) hands that back and makes the
  file behave like a camera — see [pacing](#pacing-a-live-source) below.
- **Colour is `rgb8`.** A recording replays the format it was made with, and
  capture.py's scans are rgb8 where the live stream is requested as bgr8. The
  profile is read and the channels swapped (`_color_swap`), so `Frame.rgb` is
  BGR either way.
- **It carries an accelerometer.** See below.

`camera_height` and `yaw` come from the capture's `meta.json` two directories
up (`captures/room07/` for `captures/room07/scan/scan.bag`). The `cam_K` in
that file is deliberately **not** used: it describes capture.py's own
1280×720 stream, while room07's scan was recorded at 848×480. The recording's
intrinsics are read from its own profile.

#### `BagSource._gravity(fs)` — why the IMU is read

`from_live_camera` builds the voxel grid around the up direction, and a
handheld sweep is never level. Over room07's scan the median tilt is **5.2°**
and the maximum 16.1°, which puts a surface 3 m away 29 cm from where a level
grid draws it.

So the accel vector is read per frame and passed through `Frame.up_camera`.
The magnitude is not trusted and does not need to be — `live_cam_pose`
normalises it, and this unit reads 9.02 m/s² at rest, 8% low. The *direction*
is the good part: 0.108° of noise at rest
(`captures/room07/scan/imu_check.json`). Under motion linear acceleration adds
to gravity, so a sample whose magnitude has wandered from the resting one is
dropped, and the rest are smoothed — `EMA_UP = 0.25` on the direction (~0.2 s,
follows a turn), `EMA_MAG = 0.02` on the magnitude reference, because a
reference that chased the motion it is meant to reject would be no gate at all.

What it buys, over 40 frames of that sweep:

| | measured up | assumed level |
| --- | ---: | ---: |
| floor voxels / frame | **1706** | 1482 |
| ceiling voxels / frame | 12 | 20 |
| …over the 15 frames tilted past 8° | | |
| floor | **1882** | 1472 |
| ceiling | **0** | 14 |

The phantom ceiling is the tilt, hanging in mid-air. `--no_imu` turns the
whole thing off, which is the only way to see it.

#### Pacing a live source

A camera does not wait. `pipeline.wait_for_frames()` returns the **newest**
frameset and librealsense discards what arrived while we were busy, which is
the right behaviour here: each frame is an independent single-shot completion,
so a stale one is worth nothing. The cost is latency, not correctness — the
frame on screen is one sensor period plus one inference old.

A recording is the opposite by default, because a replay that silently skipped
most of a scan would be useless for comparing runs. `--realtime` switches a
`.bag` to the camera's behaviour, which is how to see the live pacing with no
camera attached. Measured on room07 (recorded at 15 Hz, ~0.39 s of work a
frame):

```
pacing=wait for us   808 of 808 frames processed, frame number steps by 1
pacing=real time      30 of 159 processed, 129 skipped, steps by 2 to 6
```

`CameraSource.report()` is how that line reaches the log — one sentence about
what the source did, printed at the end of a run however the run ended, since
a count that only prints on reaching EOF is the count you never see.

#### `NYUSource(path, rgb_dir=None, loop=False, fps=None)`

Replays `.bin`/`.png` pairs, reading each frame's real `vox_origin`/`cam_pose`
from the `.bin` header via `FrameLoader._load_bin_header`, and decoding depth
with the NYU bit rotation via `FrameLoader._load_depth`. RGB lookup tries both
`{id}_rgb.png` and `{id}_colors.png`, since CleanerS-derived dumps differ.

#### `FolderSource(path, rgb_dir=None, cam_K=None, loop=False, fps=None, fov='native')`

Replays plain uint16-millimetre depth PNGs.

**`_find_meta(path) -> (meta, root)`** (static)
Looks for `meta.json` at `path` and then at `dirname(path)`, so both
`folder:captures/room01` and `folder:captures/room01/depth` work. Unreadable
JSON warns and is ignored rather than crashing a replay.

**`__init__`** — when meta is found and `root/depth` exists, it switches the
depth directory to `root/depth` and auto-resolves `root/rgb`. Then it adopts
`cam_K`, `camera_height` and `yaw` from meta, and — critically — sets
`self.fov = 'none'` if `meta['fov'] == 'nyu'`, so an already-matched capture is
not FOV-matched a second time. With no meta it **warns** and falls back to
NYU's `CAM_K`, which is only correct if the frames really came from NYU.

**`_rgb_for(depth_path)`** — tries `{stem}.png`, `{stem}_colors.png`, and
`{stem without _0000}_colors.png`.

**`frames()`** — decodes `raw / 1000.0` (plain mm), optionally FOV-matches, and
attaches `camera_height`/`yaw` to every yielded frame.

#### `open_source(spec, rgb_dir=None, loop=False, fps=None, fov='nyu')`

Parses the spec string and constructs the right source. Note `fov` defaults to
`'nyu'` here, which is what makes `FolderSource`'s double-match guard necessary.

---

### `capture.py` — capture to disk

Writes a folder `run_inference --live_dir` consumes directly.

```
captures/scene01/
  depth/live_000000.png          640×480 uint16 MILLIMETRES   ← inference input
  rgb/live_000000.png            640×480 BGR
  native/live_000000_depth.png   sensor resolution, uint16 mm
  native/live_000000_rgb.png     sensor resolution, BGR
  meta.json
```

`native/` holds the **undistorted but pre-crop** frames — the snapshot is taken
after the remap and before `match_nyu_fov` — so a different crop can be tried
later without re-shooting the scene. That is what made the FOV ablation in
[CAMERA_TUNING.md](CAMERA_TUNING.md) possible.

#### `parse_args()`

| flag | default | notes |
| --- | --- | --- |
| `--out_dir` | *required* | folder to create |
| `--frames N` | 1 | burst of N frames |
| `--interval S` | 0 | seconds between frames |
| `--width/--height/--fps` | 1280/720/30 | native stream profile |
| `--fov {nyu,native}` | `nyu` | `nyu` resizes + centre-crops to 640×480 |
| `--warmup N` | 30 | discarded frames, for auto-exposure; also primes the temporal filter |
| `--filters` | off | SDK spatial + temporal depth filters |
| `--hole_fill` | off | **invents geometry the sensor never saw**; the encoder treats it as observed |
| `--depth_max M` | 10.0 | beyond this → marked invalid (0 keeps everything) |
| `--no_undistort` | off | skip the pinhole resampling; costs up to 3.3 cm at frame corners |
| `--no_native` | off | don't keep the pre-crop sensor frames |
| `--preview` | off | live RGB + depth window; frame the shot, type the height, press space |
| `--camera_height M` | — | **lens height above the floor, in metres — measure it.** Under `--preview` it only pre-fills the on-screen field; without `--preview`, omitting it records 1.25 m and warns |
| `--yaw RAD` | 0.0 | rotation about the vertical axis |

#### `--preview` — frame the shot before saving

Opens one window: the 640×480 RGB that will be saved, beside the same frame's
depth colourised through
[`colorize_depth.colorize`](#colorize_depthpy--depth-pngs--realsense-colour)
(`rs-jet`, equalised — what the RealSense Viewer would show).

| input | does |
| --- | --- |
| `0`–`9`, `.`, `backspace` | edit the **camera height** field |
| click the height field | clear it |
| `space` / `enter` / **CAPTURE** button | save the frame on screen |
| `q` / `esc` / **QUIT** button / window ✕ | stop |

##### The camera height is typed here, not passed as a flag

You measure the lens height standing at the camera, which is exactly when this
window is in front of you — so that is where it is entered. The value goes into
`meta.json` verbatim.

CAPTURE is **disabled** until the field holds a number in 0.2–3.0 m (loose
sanity bounds: NYU's own camera heights span 0.749–1.553 m). Pressing space
with an empty or out-of-range field prints why and saves nothing. This is
deliberate — the grid is floor-anchored, so a wrong height shifts the entire
scene vertically and produces a frame that is silently wrong rather than
obviously broken, and there is no honest default for it.

Digits, `.` and backspace are not bound to anything else in the window, so
there is no focus mode: typing a height can never be mistaken for a capture.
`--camera_height` still works and pre-fills the field, e.g. for a second
capture from the same tripod.

It renders the **matched 640×480 frame, not the sensor's native one** — framing
is about what the network sees, and `match_nyu_fov` discards most of the D455's
width before that point.

Frame numbering follows what was *saved*, so a preview that idled for 400
frames before the first keypress still writes `live_000000`. Exits once
`--frames` have been saved (default 1 — pass `--frames 5` to keep the window
open for five). Falls back to capturing immediately, with a warning, when there
is no `DISPLAY` or OpenCV has no GUI backend.

**The depth pane is the point.** Min-Z dropout, a dark sofa, a translucent
bottle and a textureless wall are all invisible in colour and obvious in depth.
The footer carries a calibrated warning:

| condition | meaning |
| --- | --- |
| `valid < 20%` | aim at a wall 1–4 m away, check the light |
| `> 2%` of depth under 0.6 m | something is at the ~0.5 m min-Z edge and already dropping out |
| `> 20%` of the lower third empty | usually the surface the camera is standing on |
| median `< 1 m` | scene closer than the 4.8 m grid expects |

Thresholds are measured, not guessed. `<0.6 m` separates outright: 8.9% on
`room01` and 10.8% on `test01` — both captures ruined by a desk inside min-Z —
against **0.00%** on every NYU frame the model was trained on. Hole fraction
alone is far weaker (30.5% bad vs 7.1% good).

#### `_to_uint16_mm(depth_m, where, depth_max=0.0, quiet=False)`

Metres float → millimetres uint16, **saturating rather than wrapping**.
`nan/±inf` → 0, values beyond `depth_max` → 0 with an info line, values still
above the 65535 ceiling → 0 with a warning. Without this, uint16 turns a 70 m
reading into a 4 mm one — a phantom surface pressed against the lens.

#### `main()`

Opens the source, saves `--frames` frames, and takes `cam_K_eff = frame.cam_K`
**off the yielded frame** rather than recomputing it, because `match_nyu_fov`
measured it from the crop actually achieved.

Per frame it logs valid-depth percentage and range, and warns below 20% with
concrete advice. Finally it writes `meta.json` and prints the exact
`run_inference` command to run next.

`meta.json` fields: `source`, `stream`, `depth_scale`, `depth_units` (an
explicit string reading *"uint16 millimetres (plain; NOT the NYU bit-rotated
encoding)"*), `fov`, `cam_K` (post-match), `cam_K_native` (pre-match),
`camera_height`, `yaw`, `depth_max`, `undistorted`, `filters`, `frames`.

---

### `frame_loader.py` — geometry encoder

Depth → the tensors the model consumes. This is the validated core: 0.96
correlation and 12,313 / 12,668 unique TSDF values matching the CleanerS
reference for NYU0001.

#### Constants

```python
CAM_K            = [[518.8579, 0, 325.58], [0, 519.4696, 253.74], [0, 0, 1]]
VOX_UNIT_HI      = 0.02                  # 2 cm
VOX_SIZE_HI      = [240, 144, 240]
VOX_MARGIN       = 0.24                  # TSDF truncation, 12 high-res voxels
SAMPLE_RATIO     = 4
VOX_SIZE_LOW     = [60, 36, 60]
VOX_UNIT_LOW     = 0.08                  # 8 cm
IMG_H, IMG_W     = 480, 640
MAPPING_SENTINEL = 307200                # == IMG_H * IMG_W
```

#### `FrameLoader(depth, vox_origin, cam_pose, rgb=None, cam_K=None, drop_invalid_depth=False, encoder='numpy', device=None)`

Four encoders, agreeing to within a handful of boundary voxels and differing
by an order of magnitude in speed. Per frame on the dev box:

| `encoder` | ms | what it is |
| --- | ---: | --- |
| `'numpy'` | 1026 | scipy's exact transform on the CPU — **the default** |
| `'cuda'` | 1101 | SSCNet's `datautil.cu` kernels; raises if unimportable |
| `'torch'` | **116** | [`tsdf_torch.py`](#tsdf_torchpy--the-fast-tsdf-encode) |
| `'auto'` | | `'torch'` with a CUDA card, else `'numpy'` |

`device` is for `'torch'` only. Numpy stays the constructor default so the
NYU-validated path is what anything unaware of the flag keeps getting;
`run_live` asks for `'auto'`, `run_inference` for `'numpy'`.
[HARDWARE.md](HARDWARE.md) has the per-stage breakdown and the VRAM floors.

`'cuda'` runs SSCNet's own `depth2Grid__` + `SquaredDistanceTransform__`
instead of `_build_high_res_tsdf`. The library is preloaded
by absolute path and the call briefly `chdir`s into `data/utils/`, because the
extension `dlopen`s `./build/libdatautil.so`. Checked on ScanNet frames: the
mapping is identical, 0–247 of 8.29 M high-res voxels differ (surface voxels on
a voxel boundary, float32 vs float64), and the low-res TSDF differs by at most
0.019. The kernel returns no observed mask, so `weight` is rebuilt as
`tsdf != 0 | surface` (agrees with numpy on ≥ 99.99% of low-res voxels). It is
**not** the fast path — its transform brute-forces ~15,600 candidates per
voxel, which scipy beats on a CPU.

`drop_invalid_depth` excludes `depth <= 0` pixels from the surface stamp. Leave
`False` to reproduce the NYU reference exactly; set `True` for live sensors.

The effect is **exactly one voxel**: every zero-depth pixel unprojects to
`(0,0,0)` in camera coordinates, so they all land in the voxel containing the
camera. NYU tolerates it because its grid usually excludes the camera; a
floor-anchored live grid *contains* it (verified at high-res voxel
`x=0, y=65, z=120`, in bounds on all three axes), and a D455 reports 0 for
every out-of-range or low-confidence pixel, so the blob would be permanent.

#### `from_nyu_bin(bin_path, png_path, rgb_path=None, encoder='numpy', device=None)` (classmethod)

Reads the header for `vox_origin`/`cam_pose`, decodes depth with the NYU
rotation. `cv2.imread` fails silently by returning `None`, so the RGB load is
explicitly checked.

#### `from_live_camera(depth, rgb=None, cam_pose=None, vox_origin=None, cam_K=None, camera_height=1.25, yaw=0.0, drop_invalid_depth=True, up_camera=None, cam_K_color=None, color_from_depth=None, encoder='auto', device=None)` (classmethod)

Note the default: `encoder='auto'`, unlike the constructor's `'numpy'`. A live
frame has no reference to reproduce, so this path takes the fast encoder
whenever there is a card to run it on.

Builds a **gravity-aligned, floor-anchored** grid in NYU's convention when
`cam_pose` is not supplied:

```python
cam_pose = [[ cos,  0, -sin, 0        ],     # cam +X (right)   -> world ( c, s, 0)
            [ sin,  0,  cos, 0        ],     # cam +Y (down)    -> world ( 0, 0,-1)
            [ 0,   -1,  0,   height   ],     # cam +Z (forward) -> world (-s, c, 0)
            [ 0,    0,  0,   1        ]]
vox_origin = [-2.4, 0.0, -0.05]              # 4.8 m wide, 4.8 m deep, from 5 cm below the floor
```

An identity `cam_pose` sends world Z along the viewing direction instead,
producing a 4.8 m-tall × 2.88 m-deep box and handing the network a scene
rotated 90°. `floor` and `ceiling` are the first predictions to collapse.

#### `_load_bin_header(bin_path)` (static)

Unpacks `3f` `vox_origin` then `16f` `cam_pose` reshaped to 4×4.

#### `_load_depth(png_path)` (static)

NYU decoding only: `((raw << 13) | (raw >> 3)) / 1000.0`. Never use this on a
capture PNG — see gotcha 1.

The `.astype(np.uint16)` on the preceding line is load-bearing, not redundant.
numpy preserves array dtype through `<<`, so on an int32 array there is no
16-bit wrap and the same three values decode as 165,808 m instead of 2.53 m.

#### `build_tsdf_and_mapping() -> sample`

The public entry point. Builds the high-res TSDF, block-mean downsamples it,
thresholds the weight at 0.5, builds the low-res mapping, and inverts that
mapping into `mapping2d`.

With `encoder='torch'` the first two steps are fused — `_build_low_res_tsdf_torch`
returns 60×36×60 directly, because the slabbing that keeps its peak at 97 MB
reduces each z-slab on the spot. The `>= 0.5` weight threshold and everything
downstream are identical either way.

#### `_build_low_res_tsdf_torch() -> (tsdf_low, weight_mean)`

Delegates to `tsdf_torch.encode_low`, passing `self.device`. Returns the
low-res pair, `weight_mean` still unthresholded so the caller applies the same
`>= 0.5` the numpy path applies.

#### `_build_high_res_tsdf() -> (tsdf, weight)`

**The reference encoder, and the slow one: ~1.0 s of a 1.33 s frame.** It is
kept as the default and as the thing every other encoder is checked against;
[`tsdf_torch.py`](#tsdf_torchpy--the-fast-tsdf-encode) is the same steps in
torch at 110 ms. Steps:

1. Unproject every pixel to camera coordinates, transform to world by
   `point_cam @ R.T + t`, and stamp an occupancy grid. Axis assignment is
   `gz ← world X`, `gx ← world Y`, `gy ← world Z`.
2. Build world coordinates for all 8.29 M voxel centres, project them back into
   the image, and read the measured depth at each.
3. A voxel is `known` when it projects in-bounds **and** the measured depth is
   within 0.5–8.0 m. Sign comes from `depth_measured - z_camera`.
4. `distance_transform_edt(~occ)` gives unsigned distance; divided by the
   12-voxel search region and clipped to 1, it refines any `known` voxel whose
   value it improves. **This EDT is ~93% of the encode time (~387 ms).**
5. `tsdf[~known] = 0.0` — and this **must** happen at high resolution, before
   the block mean. SSCNet's `SquaredDistanceTransform__` bare-returns for
   behind-camera / outside-FOV / out-of-range voxels, leaving them zero, and
   `downSample__` then averages the raw 240³ volume. Zeroing after the
   downsample gives a mask that cannot represent partial frustum coverage, and
   boundary blocks lose their fractional values (corr 0.996 vs 1.000).

There is deliberately no `depth > 0` mask in the occupancy stamp, matching
`depth2Grid__` (`datautil.cu:39-62`), which has none — an SSCNet quirk the
reference data and pretrained checkpoint both carry.

#### `_downsample_block_mean(vol, r=4)` (static)

Reshape to `(Z/r, r, Y/r, r, X/r, r)` and mean over the three block axes.

#### `_build_low_res_mapping() -> (129600,) int64`

Per-low-res-voxel → pixel index, or `MAPPING_SENTINEL` for unmapped.

This is the **inverse** of the obvious implementation: it unprojects each
**pixel** and records it in the voxel it lands in, rather than projecting voxel
centres into the image. The difference is not cosmetic — the voxel-centre
version marks ~70% of the grid where the reference marks ~2%, and agreed on
only ~30% of voxels. This array drives the model's 2D→3D feature reprojection.

`float32` throughout, matching the CUDA kernel. Many pixels can fall in one
voxel; numpy keeps the last write, which matches the reference.

---

### `tsdf_torch.py` — the fast TSDF encode

`FrameLoader(encoder='torch')`. The same geometry as `_build_high_res_tsdf`,
line for line; only the arithmetic moves. 116 ms against numpy's 1026 and the
CUDA extension's 1101 — 110 ms of it the encode proper, 9 ms the two mapping
tables, which stay in numpy on both paths.

#### Why it is faster than both

The only non-elementwise part of the encode is the distance transform, and it
is **truncated** at `VOX_MARGIN / VOX_UNIT_HI` = 12 voxels
(`refined = min(d/12, 1)`). Nothing outside the 25³ box around a voxel can
change its value — and squared Euclidean distance over a bounded box is
**separable**, being `dx² + dy² + dz²`, a sum of independent terms. So the
minimum over the box is three 1-D minimisations of `f[i-d] + d²`, `|d| ≤ 12`,
along z then y then x:

```
start        0    ∞    ∞    ∞          surface voxel at the left
one pass     0    1    4    9          each cell: min over d of (f[i-d] + d²)
√            0    1    2    3          distance, in voxels
```

75 whole-array shifted minima replace, per voxel, either SSCNet's brute force
over all ~15,600 candidates in that same box, or scipy's exact unbounded
transform on the CPU. The result is exact, not an approximation.

#### `encode_low(depth, cam_K, cam_pose, vox_origin, drop_invalid_depth=True, device=None, slab=24, edt_dtype=None)`

Returns `(tsdf_low, weight_low)`, both (60,36,60) float32 numpy. `weight_low`
is the block **mean** of the high-res observed mask, left unthresholded so the
caller applies the same `>= 0.5` the numpy path does. Order is preserved where
it matters: unobserved voxels are zeroed at **high** resolution, before the
4×4×4 block mean, so boundary blocks keep their fractional values.

| argument | default | |
| --- | --- | --- |
| `device` | cuda if available | the encoder runs on CPU too, ~1.2× numpy |
| `slab` | 24 | z-layers per projection pass; must divide by 4 |
| `edt_dtype` | fp16 on cuda, fp32 on cpu | the transform's precision |

**`slab`** — the projection stage is independent per voxel, so it runs a z-slab
at a time and each slab is block-meaned straight into the result. The high-res
TSDF is never held whole. Only the distance field must be, since the transform
reaches ±12 voxels across slab boundaries: that is the floor, one
240×144×240 array plus one 1-D pass's temporaries. Peak 779 → 97 MB
(measured with nothing else resident), bit-identical, same speed.

**`edt_dtype`** — fp16 is **exact** here, not a trade. Every value in the
transform is a whole number ≤ 3·12² = 432, and fp16 holds every integer to 2048
exactly; verified bit-identical over the NYU test set. Only the sentinel
changes, since 1e9 overflows half — `BIG_F16 = 60000.0`, still unreachable by
any real squared distance. fp32 on CPU because there are no half kernels there.

#### `_edt1d(f, axis, big)`

One 1-D pass: 25 `torch.roll` shifts and a running `minimum`. `roll` wraps, so
the `d` cells that came round from the far face are blanked to the sentinel —
without that the grid would be a torus and a surface at `z=0` would cast
distance into `z=239`.

#### `available()`

`torch.cuda.is_available()` — what `encoder='auto'` turns on.

#### Agreement

Slabbing and fp16 are bit-identical to the unslabbed fp32 torch encode.
Torch-vs-numpy carries one difference, float32 against scipy's float64, worth
2,047 of 84.76 M low-res voxels (0.0024%, max 0.068). On all 654 NYU test
frames: SSC mIoU 47.7285 / SC 74.9530, against numpy's 47.7285 / 74.9528.

---

### `model.py` — checkpoint loading and forward pass

Mirrors `test_NYU.py`'s verified calls. `cfg.model.NAME == 'VoxelSSC'`:
encoder (`mit_b2`) → `SegFormerHead` → `feature2d`, `cls2d`; `tsdf` →
`TSDFNet` → `tsdf_feat`; then `(feature2d, mapping2d, tsdf_feat)` → `Unet3d` →
`pred_semantic`.

#### `IMG_MEAN` / `IMG_STD`

```python
IMG_MEAN = [123.675, 116.28, 103.53]
IMG_STD  = [58.395, 57.12, 57.375]
```

mmsegmentation's standard `img_norm_cfg`, matching the config's
`datatransforms.val = [ImgNormalize, ToTensor]`. **Omitting this normalisation
collapses predictions onto one or two dominant classes** — raw BGR values into
an ImageNet-pretrained encoder. If results look wrong, check
`cleaner/transform.py` for the exact values `ImgNormalize` uses.

#### `_normalize_img(img_bgr_hwc)`

BGR → RGB, HWC → CHW, then `(img - mean) / std`.

#### `load_model(cfg, device)`

`build_model_from_cfg(cfg.model)` → `load_checkpoint(..., prefix='student.')`
→ `.eval()`. The `student.` prefix matters: the checkpoint holds a
teacher/student pair and the student is the one that takes noisy sensor depth.

#### `_build_forward_kwargs(sample, device)`

Adds a batch dimension, moves to device, and fills placeholders for the
training-only fields: `label3d = zeros` (class 0), `label_weight = ones`,
`tsdf_CAD = tsdf.clone()`.

These are confirmed safe, not assumed: `TSDFNet.forward(self, tsdf)`,
`SegFormerHead.forward(self, x, ...)` and
`Unet3d.forward(self, feature2d, mapping2d, tsdf_feat=None)` reference none of
them. Only `img`, `tsdf` and `mapping2d` drive the computation graph.

#### `use_amp(device, amp=False) -> bool`

Whether the forward pass runs in mixed precision. `amp` is `False`/`None`/
`'off'` (**the default — off**), `True`/`'on'`, or `'auto'`, which means on
whenever the model is on a CUDA device. CUDA only whatever is asked:
`torch.cpu.amp` exists in 1.10 but is bfloat16-only and hits the same cast
described below.

Off by default for the same reason `numpy` is `FrameLoader`'s default encoder —
`sweep_eval.py`, `nyu_occlusion_sweep.py` and `run_scannet.py` all call
`predict` and must keep producing the numbers already published. `run_live`
asks for `'auto'`.

#### `predict(model, sample, device, amp=False) -> (pred_label, weight)`

Decorated `@torch.no_grad()`. Calls `pred_3d, _, _, _ = model(**data)` inside
`torch.cuda.amp.autocast(enabled=use_amp(device, amp))`, then
`flatten(2).permute(0,2,1)` → `(B, 129600, 12)` → argmax → `(60,36,60)` int.

`weight` is returned from `sample`, **not** from the model — same as
`test_NYU.py` pulling `label_weight` from the data dict rather than the output.

AMP halves the forward pass, 346 → 201 ms, and moves the NYU score in the
fourth decimal (47.7288 → 47.7285 SSC mIoU). It has to be `autocast` and not
`model.half()`: `unet3d.py:177` does `(segres == 0).float()`, which forces fp32
back into the middle of a half graph, and the next convolution dies with
*Input type (torch.cuda.FloatTensor) and weight type (torch.cuda.HalfTensor)
should be the same*. `autocast` chooses per operation — convolutions and
matmuls in fp16, reductions and normalisations left in fp32 — so that cast is a
no-op instead of a crash, and the weights stay fp32 throughout. The output is
cast back to float before the argmax.

---

### `visualize.py` — voxels → PLY

A direct port of `labeled_voxel2ply` / `get_xyz` / `colorMap` from
`test_NYU.py`, with per-axis Python loops replaced by `np.meshgrid` (identical
result) and numpy arrays accepted instead of GPU tensors.

#### `DEFAULT_COLORMAP`

13 rows, verbatim from `test_NYU.py`:

| id | class | RGB | | id | class | RGB |
| --- | --- | --- | --- | --- | --- | --- |
| 0 | empty | 22,191,206 | | 7 | sofa | 147,102,188 |
| 1 | ceiling | 214,38,40 | | 8 | table | 30,119,181 |
| 2 | floor | 43,160,4 | | 9 | tvs | 188,188,33 |
| 3 | wall | 158,216,229 | | 10 | furn | 255,127,12 |
| 4 | window | 114,158,206 | | 11 | objects | 196,175,214 |
| 5 | chair | 204,204,91 | | 12 | ignore (255) | 153,153,153 |
| 6 | bed | 255,186,119 | | | | |

#### `get_xyz(size)`

Three `(60,36,60)` int32 index grids, `indexing='ij'`.

#### `labeled_voxel2ply(vox_labeled, ply_filename, colorMap=None)`

Maps labels through the colormap, keeps only `label > 0`, and writes ASCII PLY
with `x y z red green blue`. Coordinates are **voxel indices with no world
scaling** — matching the reference exactly. Label 255 is remapped to 0 first.

#### `save_prediction_ply(pred_label, weight, ply_path, colorMap=None)`

Forces unobserved voxels to `empty` before export, matching `test_NYU.py`'s
`visualize_3d_predict`. Occupied-only filtering happens inside
`labeled_voxel2ply`.

---

### `run_inference.py` — batch / single-frame runner

Three mutually exclusive modes, validated in `parse_args`.

#### `parse_args()`

| mode | flags |
| --- | --- |
| 1 — single NYU frame | `--bin_path` + `--png_path` (+ `--rgb_path`) |
| 2 — batch NYU folder | `--data_dir` (+ `--rgb_dir`) |
| 3 — live capture | `--live_dir` (+ `--camera_height`, `--yaw` overrides) |

Plus `--cfg`, `--pretrained_path` (required), `--out_dir`, `--mask`,
`--tsdf_dir`. Unknown args pass through as `opts` for `cfg.update`.

| flag | default | notes |
| --- | --- | --- |
| `--encoder {numpy,torch,cuda,auto}` | `numpy` | the TSDF arithmetic |
| `--amp {auto,on,off}` | `off` | mixed precision on the forward pass |

Both default to the slow, reproducible path **on purpose**: this is the tool
that produced the published NYU and room numbers, and its output is compared
byte for byte against `custom_visual_pred/`. `--encoder auto --amp on` is ~4×
faster and costs the fourth decimal.

#### `load_cfg(args, opts)`

`EasyConfig().load(args.cfg, recursive=True)` then `update(opts)` — the same
sequence as `test_NYU.py`, minus distributed setup, experiment-directory
generation, and the entire dataloader config. `cfg.rank` is resolved to a real
device index instead of a process rank.

#### `build_vis_mask(sample, frame_name, mode, tsdf_dir) -> (mask, used)`

Display choice only — it never touches the prediction, which is always saved in
full.

| mode | definition | fidelity |
| --- | --- | --- |
| `label_weight` | `arr_1` from `{tsdf_dir}/{id}.npz` (GT-derived) | 1.0000 agreement, 33301/33302 occupied over 5 frames |
| `occluded` | `tsdf < -0.5` | strict subset — no false positives, drops the visible surface |
| `surface` | `occluded \| (observed & \|tsdf\| < 0.25)` | 0.9748 agreement, 32246/33302 |
| `frustum` | every observed voxel | ~2.7× too much; debugging only |
| `auto` | `label_weight` if the npz exists, else `surface` | |

`--mask label_weight` raises `FileNotFoundError` when the npz is missing rather
than silently falling back.

#### `process_one_frame(bin_path, png_path, rgb_path, model, device, pred_dir, ply_dir, mask_mode='auto', tsdf_dir=None)`

NYU path: `from_nyu_bin` → encode → predict → mask → save `.npy` + `.ply`.

#### `load_live_meta(live_dir, args) -> (cam_K, height, yaw)`

Reads `meta.json`, raising an explanatory `FileNotFoundError` if absent —
without the sidecar there is no record of the intrinsics, and NYU's defaults
would misplace every unprojected point. CLI flags override meta.

#### `process_one_live_frame(depth_path, rgb_path, cam_K, camera_height, yaw, model, device, pred_dir, ply_dir, mask_mode)`

Reads depth with `IMREAD_UNCHANGED`, **rejects non-uint16** with a message
explaining that an 8-bit file has already been quantised to 256 depth levels
and is unusable for TSDF, then decodes **plain millimetres**. Forces
`mask_mode` to `surface` when `auto`/`label_weight` is requested, since there is
no ground truth.

Prints `observed=` alongside the mask stats — the health check.

#### `main()`

Loads the model once, then dispatches on mode. Batch mode catches per-frame
exceptions and reports a succeeded/skipped tally; live mode skips frames with
no matching RGB (the model consumes `img` — it is not optional).

---

### `run_live.py` — streaming runner

Single process, model loaded once, one frame at a time.

#### `_restore_log_levels(level=logging.INFO)`

mmcv's `get_logger` (`mmcv/utils/logging.py`, 1.5.0) walks
`logger.root.handlers` and forces every `StreamHandler` to `ERROR`, to mute
duplicate output on non-zero ranks during distributed training. Importing the
model triggers it, so **without this call every log line after `load_model()`
silently vanishes** — including throughput numbers and the "no frames
processed" warning.

#### `_handle_signal(signum, _frame)`

Sets a module-level `_STOP` on SIGTERM/SIGINT; this finishes the current frame
and closes the sensor cleanly.

#### `parse_args()`

| flag | default | notes |
| --- | --- | --- |
| `--source` | `realsense` | `realsense[:WxH@FPS]` \| `nyu:/path` \| `folder:/path` |
| `--fov {nyu,native}` | `nyu` | |
| `--camera_height` | `None` | → capture's meta.json → 1.25 |
| `--yaw` | `None` | → meta.json → 0.0 |
| `--rgb_dir`, `--loop`, `--fps` | | replay sources only |
| `--save {ply,npy,both,none}` | **`none`** | a live run is usually watched, not kept, and a `.ply` costs 83 ms of a 320 ms frame |
| `--out_dir` | `./outputs/live_session` | `prediction/` and `ply/` under it, created only when written to |
| `--zoom` | 2.0 | how much of the window the volume fills |
| `--mask {surface,occluded,frustum}` | `surface` | no `label_weight`: there is no GT live |
| `--encoder {auto,torch,numpy,cuda}` | `auto` | → `torch` with a card, else `numpy` |
| `--amp {auto,on,off}` | `auto` | → on when the model is on a card |
| `--no_imu` | off | `bag:` only — ignore the recorded gravity and assume level |
| `--realtime` | off | `bag:` only — play at the recorded rate and drop what we miss, as a camera does |
| `--display` | off | open the live window, [`live_view.py`](#live_viewpy--the-live-window) |
| `--colour {class,zbuffer,all}` | `zbuffer` | what that window paints the voxels with |
| `--no_panels` | off | `--display` only — drop the colour/depth window beside it |
| `--window WxH` | `1280x800` | size of the 3D window |
| `--panel_width` | 420 | where the colour/depth window starts; it is resizable |
| `--max_frames` | 0 (unlimited) | |
| `--log_every` | 30 | |

The two speed flags default to the fast path here, the opposite of
`run_inference`: 0.71 → 2.88 fps replaying room07, 0.67 → 2.80 on NYU. Both
resolve to the old behaviour on a machine with no CUDA card, so nothing gets
slower anywhere. `--encoder numpy --amp off` is the old path exactly.

#### `build_mask(sample, mode)`

GT-free masks only. Same definitions as `run_inference.build_vis_mask`, minus
the `label_weight`/`auto` branches.

#### `main()`

Installs the signal handlers, loads the model, restores logging, then iterates
`src.frames()`. Per frame it branches on `frame.vox_origin is not None`:

- **populated** → replay of a dataset frame carrying its own registration; use
  `FrameLoader(...)` directly.
- **`None`** → live; resolve height and yaw by **CLI → frame → default**, then
  `from_live_camera`.

`--encoder auto` and `--amp auto` are resolved once before the loop and logged
on the `source=` line, so a run states which path it actually took.

Encode and inference time are accumulated separately and reported as
`encode N ms | infer N ms`, which is how the encode bottleneck was identified.
Both stages end in a `.cpu()` copy, which synchronises, so the split is honest
and not an artefact of CUDA's asynchrony.

---

### `live_view.py` — the live window

`run_live --display`. One open3d window whose voxels are replaced every frame,
driven from the inference loop — no second thread, no marshalling. The
geometry and the colouring are `display_solid`'s functions, imported:
`build_cube_mesh`, `paint_mesh`, `photo_colours`, `fill_hidden`. Same culled
solid cubes, same baked face shading, same colour rules.

[VIEWERS.md](VIEWERS.md#live_viewpy) is the user-facing page — keys, colour
modes, and the two things this window cannot do (pick a voxel, aim its own
camera). What follows is the code.

#### `LiveView(colour='zbuffer', reach=0.75, width=1280, height=800, out_dir=None, title=...)`

Imports open3d and `display_solid` lazily, inside `__init__`, so `run_live`
without `--display` neither needs open3d nor pays to import it.

Pins `display_solid.FIXED_PIVOT = GRID[2] - 1` on construction. `to_view_index`
reflects the depth axis about a pivot taken from the frame's own voxel bounds,
which **moves as the scene changes** — the room would jump between frames.
`display_overlay.py` pins it for the same reason, to put two files in one
coordinate system.

The renderer is unlit (`light_on = False`): the face shading is already baked
into the vertex colours by `build_cube_mesh`'s `FACES` table, and real lighting
on top would tone-map the class colours away from the legend.

#### `update(pred_label, mask, loader, rgb=None, name='', fps=None) -> bool`

The whole API. Takes what `run_live` already hands `save_prediction_ply`, plus
the loader (for `cam_K` / `cam_pose` / `vox_origin` / `depth`) and the colour
frame; `name` and `fps` are for the panel footer only. Returns **False once
the user closes the window**, which stops the run.

| stage | ms |
| --- | ---: |
| colour, `class` | ~0 |
| colour, `zbuffer` | 11-17 |
| colour, `all` | ~100 |
| `build_cube_mesh` + draw | 4-9 |
| the camera panels | ~15 |

#### `_build(idx, colours)` and `_swap(mesh)` — the framing

The camera cannot be aimed from Python in open3d 0.17 (`get_view_control()`
returns a copy), so the framing is done to the geometry, which this file
rebuilds every frame anyway:

- `_swap` fits the camera **once**, to a proxy box covering the whole
  60×36×60 grid, then removes the box. Fitting the first frame's voxels
  instead would make the opening view depend on what the model happened to
  predict then, and put the orbit pivot somewhere arbitrary. The pivot is the
  grid's centre, always.
- `_build` scales each frame's mesh about that same centre by `zoom`. With the
  camera fixed, scaling the content is exactly equivalent to moving the camera
  in or out.

open3d's fit frames the bounding *sphere*, hence the factor: at `zoom=1.0` a
live frame fills 31% of the window height, at 2.0 it fills 65%. An NYU frame
is tighter in the volume and reaches the edges at 2.0, so `--zoom 1.7` suits
replay. `+`/`-`/`0` step it in the window, and `_repaint` redraws the frame
already on screen rather than waiting for the next one.

#### `_draw_panels(depth, rgb, name, fps)`

The second window: the colour frame over the depth frame, colourised with
`colorize_depth.colorize` — the RealSense Viewer's own rs-jet ramp with
per-frame histogram equalisation, the same call `capture.py`'s `--preview`
makes, so the three agree.

It draws the frame **after** `match_nyu_fov`, not the sensor's native one:
640×480 with NYU's intrinsics is what the network sees and therefore what is
worth judging. Equalising per frame means the colours are a shape readout and
not a measurement — they are not comparable between frames — so the footer
carries the numbers instead: the frame name, the fraction of depth pixels the
sensor actually measured, the median range, and the instantaneous rate.

A separate OpenCV window because the open3d one cannot hold widgets at all
(see [VIEWERS.md](VIEWERS.md#why-buttons-and-not-keys)); the two are
positioned side by side at startup, the 3D one at the origin and this one
just past its right edge.

It is a `WINDOW_NORMAL`, **not** `WINDOW_AUTOSIZE`. AUTOSIZE pins the window
to the image and refuses to be resized, so a panel taller than the screen
cannot be shrunk or even fully seen. NORMAL lets it be dragged to any size and
scales the image into it.

`--panel_width` sets where it starts, 420 by default: 420 plus the 3D window's
1280 is 1720 wide and 800 tall, which fits the 1728×1080 desktop a 3456×2160
panel gives at 2×. At 640 the pair wanted 1940×1028 and hung off the edge.

#### `occupied(pred_label, mask) -> (idx, cls)`

The voxels the `.ply` would hold, and their classes — the same selection
`visualize.save_prediction_ply` makes: zero what the mask excludes, keep what
is left occupied. The `(x, y, z)` order of `np.nonzero` matches `get_xyz`'s
meshgrid, so these are the indices `display_solid`'s ply reader would produce.

#### `_swap(mesh)`

One geometry object, its arrays reassigned:

```python
self.geom.vertices = mesh.vertices
self.geom.triangles = mesh.triangles
self.geom.vertex_colors = mesh.vertex_colors
self.vis.update_geometry(self.geom)
```

`update_geometry` handles the changing vertex count (verified: 8,387 → 4,634 →
9,736 voxels across three frames, each drawn correctly). `remove_geometry` +
`add_geometry` would work too and would **refit the camera**, throwing away the
viewpoint the user set — which is the whole reason the swap is done this way.

---

### `utils.py` — shared helpers

| symbol | purpose |
| --- | --- |
| `CLASSES`, `NUM_CLASSES` | 12 NYU classes; the colour table lives in `visualize.py` |
| `get_device(prefer_cuda=True)` | CUDA if available, else CPU |
| `setup_logging(level=INFO)` | `[HH:MM:SS] LEVEL: message` |
| `ensure_dir(path)` | `makedirs(exist_ok=True)`, returns the path |
| `find_frame_pairs(data_dir)` | pairs `.bin`/`.png` by their shared 7-char frame id |
| `load_config(cfg_path)` | **dead code** — see [Documentation drift](#documentation-drift) |

`find_frame_pairs` pairs **by frame id, not sorted-list-position**, which
silently misaligns whenever the two lists differ in count or have a gap. It
warns about unmatched files on both sides rather than dropping them quietly.

### `display.py` — standalone viewer

An open3d viewer for the PLY files. Not imported by anything; run it directly:

```bash
python inference/display.py ./outputs/room01/ply/live_000000.ply
```

Hover the cursor over the cloud and the readout bar **below** the 3D view — not
an overlay on it — reports the voxel under the cursor: its grid index, its RGB
triple, and the class that colour belongs to, alongside a colour legend.
Ctrl+click prints the same line to stdout. The reported index is the
**original** grid index, so it still indexes the prediction arrays directly. A
per-class point census is printed on load.

| flag | meaning |
| --- | --- |
| `--raw-axes` | don't correct the handedness — see below |
| `--point-size` | screen px per point; default sizes them so voxels just meet at the volume's mid-depth |
| `--legacy` | `Visualizer` instead of the `gui` app: same view, no hover readout |

**The file's axes are left-handed, and the viewer reflects them.** Probing
`frame_loader` with single-patch depth images pins the grid order down as

```
  ply x = grid axis 0 -> world X, increasing to the CAMERA'S RIGHT
  ply y = grid axis 1 -> world Z, increasing UP
  ply z = grid axis 2 -> world Y, increasing AWAY from the camera
```

— i.e. `(right, up, forward)`, which is left-handed, because `frame_loader`
swapped world `Y`/`Z` when it wrote `gz`/`gy`/`gx` and never flipped a sign.
Every viewer expects OpenGL's `(right, up, TOWARD the viewer)`, so the scene
renders inside-out: the eye ends up behind the far wall looking back at the
camera, the wall is drawn over the furniture in front of it, and orbiting turns
the room the wrong way. `display.py` reflects the depth axis
(`z -> zmax + zmin - z`) to make the basis right-handed, then puts the eye
where the capture camera stood.

Two things worth knowing before you touch this:

**Left/right was already correct, and the reflection does not change it.** No
camera placement can fix a handedness error — with the raw axes, `front` of
`+z` and `-z` render *pixel-identical* images. The obvious guess, reflecting
`x`, silently breaks a left/right that was right all along.

**Verified against the camera, not by eye.** The PLY was reprojected through
the real camera model (`cam_pose` + the capture's `cam_K`) into a 640×480 label
image and compared with `captures/room01/rgb`: desk objects left, bed right,
floor below, wall behind, near surfaces occluding far ones. The corrected view
reproduces that ordering; the raw one puts the far wall in front. Picking is
verified the same way — 501 sampled pixels, and the class the panel reports
matches the colour actually rendered at that pixel in all of them.

**Picking fails soft.** A mouse handler fires on every motion event, so an
exception inside one is not one traceback but one per frame and an unusable
window. `guard_pick` (in `display_solid.py`, used by both viewers) prints the
first failure in full — with the open3d and python versions — then disables
picking for the session and leaves the camera controls working. If you ever see
that message, the traceback under it is the thing worth reporting.

The `.ply` files themselves are untouched: their layout is what `test_NYU.py`'s
`labeled_voxel2ply` has always written, and other tooling reads them.

Exits non-zero with a warning if the file loads but contains 0 points.

---

### `display_solid.py` — solid-voxel viewer

Same PLY files, same orientation fix, two differences that make a crowded
prediction readable. `display.py` is untouched and still works.

```bash
python inference/display_solid.py ./outputs/room01/ply/live_000000.ply
python inference/display_solid.py ./visual_pred/CleanerS/NYU0015.ply
python inference/display_solid.py <ply> --colour class          # old colours
python inference/display_solid.py <ply> --screenshot out.png    # offscreen
```

**Cubes, not points.** `display.py` draws each voxel as a screen-space point,
so the voxels never quite tile: at any zoom the background shows through and a
wall reads as a grid of dots. This one emits an actual 1×1×1 cube per voxel, so
neighbours share faces and surfaces are solid at every zoom. Faces between two
occupied voxels are dropped — invisible from outside, and most of the geometry
(room01: 12,468 triangles kept of 36,396; NYU0015: 28,844 of 103,356). Each
face carries a fixed shade baked into its vertex colours (+Y 1.00 down to −Y
0.44) and is rendered unlit: real lighting would tone-map the colours and make
the readout disagree with the screen, but with no shading at all a cube grid is
a flat silhouette.

**The colour the camera saw.** Voxel index → world metres → camera frame →
pixel, using the *same* `vox_origin` / `cam_pose` / `cam_K` the frame went
through on the way in — read from the `.bin` header for NYU frames and from
`captures/<scene>/meta.json` for live ones, never guessed. Each voxel is
painted into the image as the perspective-correct square it actually covers,
far to near, so occlusion is resolved and a voxel's colour is the mean of the
pixels it *owns*, not a single sample a one-pixel error could take off the
wrong object. Voxels the camera could not see get `--hidden`: their class
colour dimmed to 0.55 (default, and it reads as "unseen"), or the colour of the
nearest visible voxel of the same class (`--hidden nearest`, for a continuous
surface). The readout says which of the two a voxel is.

**How the query is answered — `points_in_cubes`.** Every rule below that
starts from the measured points asks the same thing: which points lie inside
this voxel's cube, grown to `reach`? That is
`cKDTree(pts).query_ball_point(idx, r, p=inf)` — Chebyshev, so the region is a
cube and not a sphere, which would clip the corners at reach 0.5 and stop being
"inside the voxel".

It is now a **scatter rather than a search**. A point reaches a voxel iff
`|i_a - p_a| <= r` on every axis, which on each axis means the integers
`ceil(p-r) .. floor(p+r)` — at most two of them while `r < 1`. So each point
touches at most 2³ = 8 lattice sites, and the whole query is a bounded scatter
over the grid: no tree to build, and no per-voxel Python list to unpack
afterwards. `photo_from_points_multi` runs several reaches through one
unprojection and one lattice lookup, which is most of the second call's cost.

Measured on NYU0001 (265 k depth points, 9,029 voxels), and **bit-identical**
to the tree across 5 frames × 4 modes × 4 reaches:

| `photo_colours(mode=…)` | before | after |
| --- | ---: | ---: |
| `points` | 138 ms | **34 ms** |
| `surface` | 215 ms | **85 ms** |
| `all` | 424 ms | **98 ms** |

That is what makes `--colour all` usable in the live window. Reaches past 1.5
voxels fall back to the tree (`_points_in_cubes_tree`) rather than loop over a
neighbourhood that is growing cubically.

**Where a voxel's colour comes from — `--visibility`.** The question is which
measured pixels belong to a given cube, and there are three answers here
because they are not the same question — plus `all`, which stacks them.

`points` **(the default)** is the exact one: unproject every valid depth pixel,
round it to the voxel it lands in, average the colour of the pixels that landed
in each. No projection, no tolerance, no parameter — a point is inside a voxel
or it is not, and the voxel size is known. It is also the correct answer to
"did the camera see this voxel", and it is what `TRUE` means in the table below.
**No voxel is ever coloured from another voxel's measurements**; one that has
none of its own gets its class colour, and the render speckles where the
camera's coverage has a ragged edge. That is the honest picture.

`zbuffer` is what this file did before it could read a depth frame at all
(`resolve_frame` only started returning `depth_path` when `reconstruct.py` was
written). It paints the voxels into the image far-to-near and colours whichever
is front-most over each pixel. That is right for a wall facing the camera and
badly wrong for one seen edge-on: along a bed top viewed from standing height
the ray travels a long way horizontally per 8 cm of drop, so consecutive bed
voxels sit far apart in depth, the nearest wins every pixel, and the rest fall
back to the flat class colour.

`surface` is **the same rule with the region grown to `--reach` voxels** (0.75,
i.e. 6 cm). It fills the speckles — but by taking a *neighbouring* voxel's
measurements, so it is an inference, and it is **not** the default. Opt in with
`--visibility surface`.

What it recovers are voxels at the **fringe of the measured region** — the
model says the surface continues, the camera's samples stop just short. It is
*not* a depth-direction rounding, which is what I first assumed: over 80 NYU
frames (30,861 such voxels) the nearest measured point sits at the **same
distance from the camera** (offset along the view averages +0.07 voxels, 55 %
further vs 45 % nearer — no bias) and about **0.8 voxels sideways**.

They are at a fringe because the camera has no data there: 26.5 % of their
image footprints straddle a depth step over 0.3 m (a silhouette), their dropout
rate is 12.5 % against 5.5 % for voxels that did get points, and 4.4 % touch the
image border.

And the model is not misplacing them — NYU's ground truth calls the model's own
cube occupied in **99.99 %** of cases (57.5 % occupies both that cube and the
neighbour the points landed in, 42.5 % only the model's). Just **3 of 30,861**
are the "model picked the wrong cell" case. So reaching 6 cm colours voxels
that really are on the surface, from real measured pixels.

| reach | room01 | room03 | NYU0015 |
| --- | --- | --- | --- |
| 0.50 (= `points`) | 1615 | 2244 | 2230 |
| 0.60 | 1855 | 2567 | 2383 |
| **0.75 (default)** | **2005** | **2711** | **2869** |
| 1.00 | 2051 | 2761 | 3993 |

Voxels with no measurement anywhere near them still get nothing, which is the
honest thing to show: that is completion the model placed in front of a wall,
a median 1.5 voxels from any measured surface and 26 at the 90th percentile.
**Above reach 0.5 this is an inference**, and worth stating plainly: every pixel
used is a real measurement, but it was measured in a *neighbouring* cell, not
this one. Two things bound it. The query box is 6 cm on all three axes —
depth included — so it can only reach the same local surface patch; a
foreground object at a silhouette is far too distant to contribute (measured:
borrowed points never sit more than 9 cm from the voxel in depth). And the
readout says which voxels are which, so a borrowed colour is never passed off
as a measured one.

How good an inference, by hold-out — take the voxels that *do* have their own
points, colour them from outside-only points, and compare with the truth:

| | median error | mean | vs. two arbitrary voxels |
| --- | --- | --- | --- |
| room01 | 0.6 / 255 | 3.0 | 18.5 |
| room03 | 1.4 | 6.8 | 52.9 |
| NYU0015 | 1.4 | 4.8 | 40.6 |
| NYU0284 | 7.8 | 12.0 | 51.0 |
| NYU0515 | 1.4 | 7.1 | 30.1 |

4–20× better than chance, but not exact — NYU0284 is a classroom whose fine
structure changes at roughly the 6 cm scale the reach spans. `--visibility
points` refuses the inference entirely.

`all` is the three of them stacked, each voxel taking its colour from the first
rule that reaches it: its own points, else a neighbour's within `--reach`, else
the photo. The order is by how directly the rule measures *that* voxel, so
nothing is ever downgraded — every voxel `points` could colour keeps exactly
the colour `points` gave it, from its own measurements alone. (That is not true
of `surface`, where even a voxel full of its own points takes the mean over the
grown region.) The later stages only fill in around it, and `photo_colours`
returns a per-voxel `source` code so the readout can say which stage a voxel
came from rather than presenting all three as one claim.

The photo stage is **not** plain `zbuffer`, and it is two stages rather than
one. A pixel is **stage 3** if the depth there agrees with the voxel's own
range to within `--agree` (one voxel, 8 cm), or if there is no depth there at
all — dropout, where the photo is the only evidence and neither the measurement
nor the prediction puts anything in front of the voxel. Every other pixel the
voxel is front-most on is **stage 4**, which only ever colours a voxel nothing
else reached.

That split is the whole difference between a usable last stage and a wrong one.
`zbuffer` makes neither test — it resolves occlusion among the *voxels* and
never consults the depth frame, so a voxel standing behind a measured surface
is still front-most if no other predicted voxel lies on that ray, and it takes
the colour of whatever the camera really saw. Of the voxels the ungated
z-buffer adds beyond the two 3D rules on room01's frame, **88 % own pixels
whose depth is over 0.2 m away** — a median of 1.41 m *in front* of them, i.e.
the foreground object's colour. Demoted rather than discarded, those pixels
still beat a flat class colour for looking at the scene, and `--strict-photo`
drops them for anyone who disagrees.

Hold both stages to the same test as `surface`: take the voxels that do have
their own points, colour them by the rule instead, and compare with the truth.

| | gated (median / 90th) | ungated (median / 90th) | voxels `all` adds | coverage |
| --- | --- | --- | --- | --- |
| room01 | 1.0 / 6.9 | 1.1 / 6.1 | 81 | 66.1 % → 68.8 % |
| room03 | 1.4 / 22.9 | 1.6 / 27.4 | 8 | 34.8 % → 34.9 % |
| room04 | 6.4 / 39.4 | 7.8 / 47.1 | 58 | 65.7 % → 66.3 % |
| NYU0001 | 1.7 / 18.1 | 2.0 / 25.3 | 157 | 31.0 % → 32.7 % |
| NYU0002 | 2.8 / 30.0 | 3.1 / 33.2 | 103 | 20.4 % → 22.3 % |
| NYU0015 | 1.9 / 14.8 | 2.2 / 18.6 | 173 | 33.3 % → 35.3 % |
| NYU0284 | 7.8 / 47.5 | 10.8 / 52.2 | 647 | 40.4 % → 49.2 % |
| NYU0515 | 0.8 / 12.9 | 0.9 / 25.5 | 216 | 14.4 % → 17.8 % |

Errors are of 255, compared on the same voxels within each frame. The gate is
better in all 8 and at the 90th percentile in 7 of 8, and still adds 8–647
voxels a frame beyond `points` + `surface`. Most of what it adds is dropout:
75 of room01's 81, 501 of NYU0284's 647 have *only* no-depth pixels. That is
the stage's reason to exist — depth dropout and colour loss are not
co-located, so the camera's colour survives in plenty of places its ranging
does not.

Stage 4, the pixels that test threw out, costs **3.0–27.3 of 255** (median,
90th 30.5–93.5) by the same hold-out over room01/03/04/07 and
NYU0015/0284/0515 — three to four times stage 3's error, and it adds another
9–1372 voxels a frame. Together the two take coverage from 14.4–66.1 % to
30.1–69.1 %.

**`--hidden` defaults to `nearest` under `--visibility all`**, and only there.
A mode whose point is to colour everything it can should not then paint half
the scene flat, and what is left after all four stages is voxels no pixel ever
reached — in room07, 4380 of 7888, mostly completion standing behind what the
camera could see. `nearest` gives each the colour of the nearest coloured voxel
*of its own class*, which over those 7 frames sits 2.0–7.3 voxels away (median)
and held out costs 3.0–14.3 of 255. It takes room07 from 44.5 % carrying a
photo colour to 79.8 %; the 1597 that stay flat belong to classes with no
coloured member anywhere in the frame, so there is nothing to borrow. The
readout still calls every one of them *hidden — colour inferred*, and
`--hidden class` puts the flat colours back.

Against `TRUE` = the voxels holding at least one measured point:

| rule | says visible | misses | over-claims | colour error where both agree |
| --- | --- | --- | --- | --- |
| room01 — TRUE = 1615 | | | | |
| `zbuffer` | 1027 | 863 | 275 | 4.81 / 255 |
| `surface` | 1984 | 1 | 369 | 2.24 / 255 |
| `points` | 1615 | 0 | 0 | 0 (by definition) |
| `all` | 2095 | 0 | 480 | 0 (by construction) |
| room03 — TRUE = 2243 | | | | |
| `zbuffer` | 1330 | 1147 | 234 | 9.21 / 255 |
| `surface` | 2679 | 5 | 436 | 5.05 / 255 |
| `all` | 2733 | 0 | 489 | 0 (by construction) |
| NYU0015 — TRUE = 2230 | | | | |
| `zbuffer` | 3974 | 242 | 1986 | 10.88 / 255 |
| `surface` | 2779 | 8 | 549 | 2.81 / 255 |
| `all` | 4414 | 0 | 2184 | 0 (by construction) |

`surface`'s "over-claims" are not errors — they are voxels with real measured
points just outside them, and the readout labels them *measured just outside it
(within --reach)* rather than *measured inside this voxel*. The same goes for
`all`, which labels its three stages separately; its zero in the last column is
not a measurement but the stacking order — a voxel with its own points keeps
`points`' answer, so `all` carries `points`' exact core and `surface`'s fringe
at once. (The `zbuffer` and `surface` counts were taken with the earlier
two-stage fill; the single-reach rule that replaced it lands within a few dozen
voxels of them — re-run against the code as it stands, room01 reads
`zbuffer` 1009 / 881 / 275 and `surface` 2005 / 0 / 390. The `all` rows are
from the current code.)

One test that looked appealing and is worthless: rendering the coloured voxels
back through the capture camera and scoring against the RGB frame. The
`zbuffer` rule's visible set is *by construction* the set of voxels that own
pixels in that painter, each coloured from the very pixels it is then scored
on. It grades its own homework and wins.

| flag | meaning |
| --- | --- |
| `--colour photo\|class` | the camera colour (default), or `display.py`'s flat class colours |
| `--hidden class\|nearest` | how to paint the voxels the camera never saw (default `class`, but `nearest` under `--visibility all`) |
| `--visibility points\|surface\|zbuffer\|all` | where a voxel's colour comes from — see above (default `points`, which never borrows) |
| `--reach VOXELS` | `--visibility surface` and `all`: half-width of the region a voxel collects points from (default 0.75) |
| `--agree METRES` | `--visibility all` only: how far the depth frame may disagree with a voxel's range before its photo pixel drops to stage 4 (default 0.08, one voxel) |
| `--strict-photo` | `--visibility all` only: drop stage 4 — colour nothing from a pixel the depth frame contradicts |
| `--flat` | no face shading |
| `--no-cull` | keep the faces between adjacent voxels |
| `--raw-axes` | don't correct the handedness — see `display.py` above |
| `--rgb` / `--bin` / `--meta` | point at the colour image / NYU pose header / capture sidecar by hand |
| `--camera-height` / `--yaw` | override `meta.json`'s values |
| `--screenshot PATH` | render offscreen to a PNG and exit (works headless, EGL) |
| `--no-highlight` | don't outline the voxel under the cursor |

In the window: hover to outline the voxel under the cursor and read it out;
the `colour` and `reframe` buttons sit under the view; Ctrl+click prints the
readout to stdout.

**Picking is a ray march, not a nearest-centre lookup.** Voxel `c` owns the
unit box `[c - 0.5, c + 0.5]`, and those boxes are exactly the cubes on screen,
so the cursor's ray is walked cell by cell (3D-DDA) and the first *occupied*
cell is the answer — by construction it is the voxel you can see. Snapping to
the nearest voxel *centre* instead is only right when you look straight at a
face: on a surface seen edge-on the hit point sits far from its own voxel's
centre and nearer the neighbour's, so hovering a wall could name the floor
voxel in front of it. Measured on a render where every voxel carries a unique
id colour, so the renderer itself says which voxel each pixel shows:

| pixels where the drawn voxel is unambiguous | nearest-centre | ray march |
| --- | --- | --- |
| room01 | 81.4 % | **100.0 %** |
| room03 | 87.4 % | **100.0 %** |
| NYU0015 | 83.9 % | **100.0 %** |

(4,000–4,100 sampled pixels each; silhouette pixels are excluded because an
antialiased edge blends two id colours, so the ground truth there is not a
voxel. The nearest-centre errors were not all same-class either — `wall`
reported as `objs`, `ceiling`, `bed` or `window`.)

**Verified.** The projection is cross-checked against
`frame_loader._build_low_res_mapping`, which builds the voxel↔pixel
correspondence the *other* way round (it unprojects every depth pixel and
records which voxel it lands in), so agreement is not circular — it tests the
axis order and the pose independently. Median error 4.99 px on NYU0015 and
10.74 px on room01, with 93.4 % / 79.6 % of voxels landing inside their own
projected footprint; the residual is the mapping's "last pixel in the voxel
wins" pick, which is arbitrary within the voxel. Separately: the index
reflection is a bijection that leaves ply `x`/`y` untouched, every emitted face
is a unit square, the mesh extends exactly half a voxel past the occupied
indices, and the pick nudge (half a voxel along the view ray, so a click on a
shared face resolves to the cube in *front* of it) resolved 300/300 sampled
face hits to the right voxel.

If no colour image or camera pose can be found for a PLY, it says so and falls
back to `--colour class` rather than inventing a pose.

---

### `reconstruct.py` — the scene the camera measured

The depth frame triangulated into a surface and textured with the RGB frame,
the way a RealSense viewer shows it. **Nothing here is inferred**, which is the
whole point: this is the only geometry in the repo that is pure measurement, so
it is the reference the completion gets checked against — never the reverse.

It opens a window on that surface **alone**, and deliberately knows nothing
about the prediction beyond where to find the frame — the readout gives the
cell under the cursor and the range the camera measured there, nothing else.
Buttons cycle the surface colouring and reframe. To see the surface and the
prediction together use
[`display_overlay.py`](#display_overlaypy--the-two-in-one-window), which
imports what is built here.

```bash
python inference/reconstruct.py ./outputs/room01/ply/live_000000.ply
python inference/reconstruct.py <ply> --surface label     # model's labels on real geometry
python inference/reconstruct.py <ply> --surface match     # filled vs left empty
python inference/reconstruct.py <ply> --screenshot out.png
```

It takes the **prediction** as its argument and finds that frame's depth, RGB
and pose exactly the way `display_solid` does — `.bin` header for NYU, the
capture's `meta.json` for a live frame.

### `display_overlay.py` — the two in one window

The prediction and the measurement together, so they can be compared. Same
family as `display.py` (points) and `display_solid.py` (cubes); this one adds
the measured surface underneath, built by `reconstruct.py`.

```bash
python inference/display_overlay.py ./outputs/room01/ply/live_000000.ply
python inference/display_overlay.py <ply> --show surface
```

| button | |
| --- | --- |
| `layers` | solid → see-through cubes over the measurement → same, measurement dimmed |
| `visibility` | the rule the voxels are coloured by: points → surface → zbuffer → all, recomputed live (~0.2 s) |
| `surface` | surface colouring: photo → label → match |
| `voxels` | voxel colouring: photo → class |
| `hide surface` / `hide voxels` | blank either layer outright |
| `reframe` | back to the start view (always the prediction's extent, so both layers are at one scale) |

**Buttons, not key bindings, on purpose.** A `Window`'s `set_on_key` callback is
cast to a C++ type that open3d 0.17 will not accept an `EventCallbackResult`
for, and it raises `RuntimeError: Unable to cast Python instance to C++ type`
out of `Application.run()` the first time any key is pressed. `Button`'s
`set_on_clicked` takes no argument and returns nothing, so there is nothing to
cast. Calling the handler directly from Python does *not* reproduce the crash —
the cast only happens on the way back into C++ — which is why it survived
several rounds of testing here.

**`layers` is how you see both at once.** Displayed together as solids the
cubes swallow the surface running through them — a voxel's cube spans ±0.5 of
the cell centre and the measured surface passes through the middle of it, so
*darkening* the prediction does not help: a dark cube hides the surface exactly
as well as a bright one. The cubes have to let light through. At **0.55 alpha**
the class colours still read — cyan wall, orange bed — with the photo texture
visible underneath. A second click pushes the measurement back to 35 %
brightness, the same comparison from the other side.

Note `defaultUnlitTransparency` is **unusable** in open3d 0.17: it aborts the
process with `uniform named "srgbColor" not found`. `defaultLitTransparency`
works, and being a lit shader it shifts the colours a little — acceptable for a
layer you are looking *through*.

**Three surface colourings.** `photo` is the surface as the camera saw it.
`label` paints each measured point with the class the model gave *its* cell, so
the prediction's labels are read on real geometry — room01 shows the wall
labelled `wall`, the duvet and the bottle `objs`, the bed's foot `bed`, and two
stray `window` patches on a blank wall. `match` is green where the model filled
the measured cell and red where it left it empty, grey outside the 4.8 m volume
(the model was never asked about those, so scoring them as misses would be
unfair).

On load it prints the same comparison as a number — of the grid cells the
measured surface occupies, how many the prediction fills, and under which
labels. room01: **1444 of 1560 (92.6%)**.

**The cells it does not fill are not the model's doing.** On room01, of the 144
measured cells missing from the ply, the model predicted a class in **all 144**
and `prediction/*.npy` still holds them — every one was removed by the display
mask, and specifically by `observed`. That flag is a majority vote over each
cell's 4×4×4 high-res block of "a depth pixel between 0.5 m and 8 m landed
here", so a cell fails it when its block is mostly outside the image (51 %) or
on a depth hole (35 % — the frame is 13.4 % zeros, and this capture has
`hole_fill: false`), or a mix of the two (15 %). Loosening the `|tsdf| < 0.25`
threshold recovers none of them, and neither does `--mask frustum`: they are
unobserved under every mask. The reconstruction has no gap there because it
draws valid depth pixels directly and never asks whether a whole 8 cm cell is
majority-observed. Filling those holes would mean *inferring* depth, which is
a capture-time decision, not a display one.

| flag | meaning |
| --- | --- |
| `--show surface\|voxels\|both` | what is visible at startup (default `surface`) |
| `--surface photo\|label\|match` | the surface colouring to start in |
| `--edge` | metres of range disagreement across a quad before it counts as a depth discontinuity and is left open (default 0.12) |
| `--z-range MIN MAX` | depth values outside this are dropped (default 0.2–8.0 m) |
| `--stride` | use every Nth depth pixel |
| `--shade` | add a Lambert term (off by default — see below) |
| `--smooth` | with `--shade`, smoothing iterations for the shading **normals** only |
| `--colour`, `--hidden`, `--raw-axes`, `--depth`, `--rgb`, `--bin`, `--meta`, `--screenshot` | as in `display_solid` |

**Quads are only closed where the depth agrees across them.** Without the
`--edge` test every depth discontinuity — the gap between the bottle and the
wall behind it — gets bridged by a sheet of triangles that is pure fabrication.

**Shading is off by default.** A Lambert term on raw sensor normals renders the
depth camera's own low-frequency waviness as if it were structure: room01's
wall comes out looking like rippled sand. It isn't. Fitting a plane to the
179,520 measured points that land in cells the model calls `wall` gives a
**22.6 mm RMS** residual (5–95% spread 72 mm) — under a third of one 8 cm
voxel. `--shade` turns it on and smooths the shading normals only; vertex
positions are always the raw measurement.

**One coordinate frame.** The surface is expressed in the prediction's own
voxel-index coordinates (inverting `voxel_centres_world`) and reflected about
the same pivot (`display_solid.view_pivot`), so any offset you can see between
the two is a real disagreement rather than a bookkeeping one. Hovering reports
the cell under the cursor, the **measured** range there in metres, and what the
model put there — e.g. `cell [26, 16,  6]   measured at 0.55 m   model left
this cell EMPTY`, which is the water bottle.

**Picking hits the actual triangles**, one BVH per geometry
(`o3d.t.geometry.RaycastingScene`), not a march over the voxel lattice.
Occupancy is an 8 cm proxy for a continuous surface, and a ray clipping the
corner of a cell whose surface lies elsewhere in it stops early: measured
against a render that colours every vertex by its cell, the lattice march named
the drawn cell for only **32.7%** of pixels. Hitting the triangles scores
**11,999 / 12,000** across room01, room03 and NYU0015 (the one miss grazes a
cell boundary), at 0.04 ms a pick with a BVH that builds in under 10 ms over
half a million triangles.

The depth PNG's encoding is not guessed: NYU frames carry
`((d << 13) | (d >> 3))`, `capture.py` writes plain uint16 millimetres, and
`resolve_frame` reports which this frame is. If the depth frame or the pose
can't be found, it exits with an error rather than falling back — there is no
such thing as a reconstruction from a guess.

---

### `evaluate.py` — per-class metrics

Per-class TP / FP / FN, precision, recall and IoU for the saved predictions,
on either NYU split, written out as markdown. Results live in
**[EVALUATION.md](EVALUATION.md)**.

```bash
python inference/evaluate.py                                  # test tables + train-vs-test comparisons, stdout
python inference/evaluate.py --report inference/document/EVALUATION.md
python inference/evaluate.py --train-tables                   # also the full train tables
python inference/evaluate.py --split test                     # test only, no comparisons
```

**The report is test-first.** Every per-class table is the `test` split — the
held-out frames the paper reports, which it matches exactly (SC IoU 75.0, SSC
mIoU 47.7). Train is evaluated as well, but only shows up where a section
compares the two: *Learned habit, or the data?*, the two error-group sections
and the closing train-vs-test table.

It reads only what is already on disk — `custom_visual_pred/CleanerS/prediction/`
against `data/NYU/{Label,TSDF,Mapping}` — so it needs neither the model nor a
GPU and finishes in about a minute. **Rerun it whenever the predictions are
regenerated.**

The protocol is copied from `examples/segmentation/test_NYU.py:test()`, not
invented here, so the numbers are comparable with the repo's own: **SSC** over
voxels where `label_weight > 0` and `label != 255`, mIoU averaged over classes
1–11; **SC** the same further restricted to `mapping == 307200`, the voxels no
depth pixel reached — the part that must be *completed* rather than merely
labelled — scored occupied-vs-empty. It lands on **SC IoU 75.0 / SSC mIoU
47.7** for the test split, which is where CleanerS's published NYU figures sit.

SC is also broken down **per class**, over the same unseen region, two ways:
*label must match* (the ordinary per-class IoU there — mean 44.0 on test) and
*any class counts* (did the model fill that class's unseen voxels with
anything — mean 73.6). Pooled over the eleven classes the second is exactly the
headline SC IoU, which the script asserts. The gap between them is unseen
volume filled but misnamed: `tvs` 36.5 vs 87.3, `objs` 30.7 vs 77.1.

It breaks the errors down both ways — of everything the model *calls* a class,
how much is wrong and what those voxels really were; and of everything that
*is* a class, how much it missed and what it called them instead — then lists
the largest single confusions.

Beyond the standard tables it prints three the benchmark does not:

- **which classes invent volume** — each class's false positives split into
  *put it where there was nothing* against *called another object by the wrong
  name*. Precision conflates the two, and only the first grows an object past
  its real extent.
- **how much of each prediction is scored at all** — `label_weight` plus the
  `GT != 255` rule hide most of the output. It ranges from 36 % (`floor`) down
  to 1.9 % (`ceiling`). The last column counts voxels the depth frame measured
  as *free space* and the GT calls empty: contradicted by the annotation and
  the sensor at once, and counted by neither.
- **how often each class is asserted at all** — frames predicted against frames
  annotated. A class claimed in every frame that the GT carries in a tenth of
  them is applying a prior, not reading the scene (`tvs` 13.3×, `ceiling` 9×).

It also carries a prose section explaining `label_weight`, and one asking
**learned habit, or the data?** of every class, by comparing train against test
on the diagnostics rather than on IoU alone. Two facts do the work there: the
model has already seen the answers for the train split, and the *training* loss
uses the same mask as the metric (`train_utils.py:83-85`), so nothing outside
`label_weight` is ever penalised in training either. A mistake made on train,
inside the mask, is the model's; one made only outside the mask is the data's.

Finally it groups the eleven classes into **three kinds of error**, because
they are not eleven separate problems and a fix aimed at the wrong one is
wasted. Membership is decided by a rule applied to the confusion matrices, not
by hand:

1. **The data never checked it** — the class transfers intact (train − test
   under 5 IoU points), so inside the scored region there is nothing left to
   learn and the error must be outside it, where neither the metric nor the
   loss ever looked. `floor` alone, at a 1.3-point gap against 16–32 for the
   other ten. *Fix: supervision.*
2. **The model grows things into thin air** — the largest destination of its
   false positives is `empty`: `bed`, `sofa`, `ceiling`, `furn`. The spill
   survives on frames the model was trained on, inside the mask, where the
   loss did charge for it. *Fix: the model.*
3. **The model can't tell the classes apart** — the largest destination is
   another class: `window`→`wall`, `objs`→`furn`, `table`→`furn`, `tvs`→`furn`,
   `wall`→`objs`, `chair`→`table`. The volume is roughly right, the name is
   not. *Fix: the labels.*

The report states plainly that tests 2 and 3 rank by *largest single*
destination rather than a majority — only `bed` sends over half its false
positives to `empty` — so group 2 is a ranking, not a clean threshold.

It does the same for **misses** (false negatives). Every real object voxel is
inside `label_weight`, so every miss is scored and was penalised — the split
that matters is *where* the missed voxel sits. The model says `empty` in only
2% of the space the camera saw through, and 90% of occluded space: it learned
`empty` only where the loss covered. So a miss is never the model erasing a
measured surface; it is either a hidden part left unfilled or the right volume
given the wrong name:

1. **It finds it** — under 10% missed: `floor`.
2. **It doesn't complete what's hidden** — misses mostly go to `empty`, ≥98%
   of them in space the camera could not see: `table`, `chair`, `furn`,
   `wall`, `ceiling`, `sofa`, `bed`. Split by the train column into a habit
   already present on trained frames (`table`, `chair`, `wall`, `ceiling`) and
   a failure to transfer (`furn`, `sofa`, `bed`).
3. **It finds the volume and names it wrong** — `objs`→`furn`, `tvs`→`furn`,
   `window`→`wall`; the same three are in the false-positive group 3.

Every number in that prose is computed at generation time, not typed in, so a
rerun cannot leave it stale.

---

### `colorize_depth.py` — depth PNGs → RealSense colour

Depth PNGs are unreadable by eye — a 2.3 m wall and a 2.9 m wall differ by 600
out of 65535, so the file looks uniformly black. This renders them the way the
RealSense Viewer does, with a scale bar:

```bash
python -m inference.colorize_depth captures/room01/depth --out captures/room01/depth_vis
python -m inference.colorize_depth data/NYU/depth/NYU0001_0000.png --encoding nyu
```

The defaults (`--cmap rs-jet --scaling equalize --invalid black`) are the
Viewer's own **Dynamic** preset. The ten `rs-*` ramps, the cumulative-histogram
equalisation and the float32 truncation in the LUT lookup are ported from
librealsense `src/proc/colorizer.{h,cpp}`. It is a port, not a lookalike: all
10 schemes × {equalize, linear} were compared pixel-for-pixel against
`pyrealsense2`'s own `rs.colorizer()` driven through `rs.software_device`, on
three captures — **60/60 configurations bit-exact**.

| flag | meaning |
| --- | --- |
| `--preset {dynamic,fixed,near,far}` | `RS2_OPTION_VISUAL_PRESET`; sets `--cmap`, `--scaling`, `--min`, `--max` together |
| `--cmap` | `rs-jet` (default), `rs-classic`, `rs-white-to-black`, `rs-black-to-white`, `rs-bio`, `rs-cold`, `rs-warm`, `rs-quantized`, `rs-pattern`, `rs-hue`, plus the OpenCV LUTs `turbo`/`jet`/`viridis`/`magma`/`inferno`/`plasma`/`bone` |
| `--scaling {equalize,linear}` | Viewer histogram equalisation, or plain metres |
| `--min` / `--max` | linear only; default to the 1st/99th percentile of the frame |
| `--fixed` | share one ramp (or one histogram) across a whole folder |
| `--encoding {mm,nyu}` | plain uint16 mm, or NYU's bit-rotated PNGs |
| `--invalid` | `black` (Viewer) / `magenta` / `white` / `grey` |

Three things worth knowing:

**RealSense "Jet" is not OpenCV's `COLORMAP_JET`.** It is a five-stop ramp
ending in dark maroon `(50,0,0)`, which is why far surfaces fade to near-black
in the Viewer but stay bright red under `--cmap jet`.

**Equalisation is per-frame and non-linear.** It spends colour where the pixels
are, which is what makes the Viewer look vivid, but a colour means a different
depth in every frame and equal colour steps are not equal metre steps. The
scale bar is drawn through the inverse CDF so the labels stay truthful — they
are just unevenly spaced. Use `--scaling linear` to compare frames, or
`--fixed` to share one curve across a folder.

**The float32 arithmetic is load-bearing.** The LUT lookup truncates to uint8,
so a value landing on `x.99999` in one precision and `x.00001` in the other is
a whole level apart. Doing the interpolation in numpy's default float64 put
`rs-quantized` off by one across entire plateaus (169 vs 170) and dropped the
match to 37.8%.

---

### `nyu_occlusion_sweep.py` — visible vs occluded recall

Diagnostic, not part of the pipeline. Runs every NYU frame through **our**
encoder + the checkpoint and scores voxels the camera could see separately from
voxels it could not:

```
visible  = weight > 0 and |tsdf| < 0.25    the surface shell
occluded = tsdf < -0.5                     behind the surface, must be completed
```

```bash
python -m inference.nyu_occlusion_sweep --verify_encoder --out /tmp/nyu_sweep.npz
```

~9 min for 1449 frames on a 4 GB GPU. Reads the precomputed `Custom_TSDF` /
`Custom_Mapping` npz for speed; `--verify_encoder` first confirms on one frame
that those are bit-identical to `FrameLoader.build_tsdf_and_mapping()`, so the
sweep cannot silently end up measuring the reference encoder instead.

| flag | default | does |
| --- | --- | --- |
| `--data_root` | `../data/NYU` | NYU root: `depth/`, `Label/`, `{test,train}/RGB/` |
| `--tsdf_dir` | `Custom_TSDF` | which encoder's output feeds the model |
| `--mapping_dir` | `Custom_Mapping` | matching mapping folder |
| `--verify_encoder` | off | check the npz against `FrameLoader` before sweeping |
| `--limit N` | 0 | stop after N frames |
| `--out` | `nyu_sweep.npz` | confusion matrices + per-frame wall recall |

It also computes the benchmark's own masks (`test_NYU.py:206-211`) and
reproduces Table 1 exactly — SC 88.0/83.5/**75.0** and every per-class SSC IoU
to 0.1 — which is what makes the new numbers trustworthy.

Headline result, NYU test split: on the benchmark's **own** occlusion mask,
**wall** IoU is 36.1 occluded against 62.8 on the visible surface (**−26.7**),
while **bed** loses only **−5.4** (61.7 vs 67.1). In recall terms occluded bed
(84.0%) beats *visible* bed (77.3%). Furniture wins the space it cannot see and
planar structure loses it, which is why a bed gets extruded through the wall
behind it. Full write-up, including the room03 capture that raised it, in
[OCCLUSION.md](OCCLUSION.md).

### `run_scannet.py` — Occ-ScanNet runner

Runs the checkpoint on Occ-ScanNet frames. Inputs live under
`data/Scannet/` (one level above the repo):

| path | what | made by |
| --- | --- | --- |
| `<scene>/<frame>.npz` | GT in CleanerS layout: `target`, `target_fov`, `in_fov`, `voxel_origin`, `cam_pose`, `intrinsic_color` | `data/occscannet_gt.py` |
| `posed_images/<scene>/<frame>.{jpg,png,txt}` | colour 1296×968 (640×480 in 12 scenes), depth 640×480 uint16 mm, pose | `data/occscannet_remote.py extract` |
| `scans/<scene>/<scene>.txt` | ScanNet metadata: `fx_depth`, `fy_depth`, `mx_depth`, `my_depth`, `colorToDepthExtrinsics` | fetched from `kaldir.vc.cit.tum.de/scannet/v2/scans/<scene>/<scene>.txt` (the URL `download-scannet.py --type .txt` uses; needs the ScanNet agreement) — all 681 val scenes are on disk |

```bash
python -m inference.run_scannet --pretrained_path ./checkpoint/CleanerS_ckpt.pth --split val --encoder cuda
```

- **Calibration.** Depth intrinsics and the colour→depth transform come only
  from `scans/<scene>/<scene>.txt`; a scene without it is skipped, nothing is
  estimated. Across the 681 val scenes there are 12 distinct depth
  calibrations (fx 570.0–577.9). 336 scenes carry a `colorToDepthExtrinsics`
  (a 4.4 cm offset); the other 345 have none, so ScanNet gives no offset for
  them and identity is used.
- **Grid.** `voxel_origin` and `cam_pose` come straight from the Occ-ScanNet
  GT, so `FrameLoader` needs no change: its `[world X, Z up, world Y]` layout is
  what `occscannet_gt.py` converts the GT to. Unlike NYU the grid follows the
  scan's world axes, not the camera's heading, and it sits ~3.4 m ahead of the
  camera, so 70–85% of it is unobserved (NYU ≈ 34%).
- **Colour.** Every depth pixel with a reading is unprojected with K_depth,
  moved into the colour camera and resampled from the 1296×968 image; pixels
  without depth use the rotation-only mapping. Only pixels with depth reach a
  voxel, so the fill affects 2D context only.
- **Encoder.** `--encoder auto|cuda|numpy`, see `FrameLoader`. `drop_invalid_depth`
  is always on — ScanNet depth has holes.
- Writes `prediction/<scene>/<frame>.npy` (uint8, 60×36×60) and
  `encoded/<scene>/<frame>.npz` (`tsdf`, `weight`, `mapping`); frames already
  written are skipped. About 1.2 s/frame
  on an RTX 3050 Ti laptop GPU (≈ 6.5 h for val), most of it the CUDA TSDF kernel
  sharing the 4 GB GPU with the model.

### `evaluate_scannet.py` — Occ-ScanNet metrics

```bash
python -m inference.evaluate_scannet --pred_dir ./outputs/scannet \
    --report inference/document/EVALUATION_SCANNET.md
```

Reads only what is on disk; no GPU. Also reports CleanerS's rule with **floor
removed** (voxels that are floor in the GT or the prediction are not scored,
mIoU over 10 classes), on ScanNet and on the NYU test split: Occ-ScanNet puts
the grid 5 cm below world z = 0 instead of below the annotated floor, so its
floor sits 1–2 layers above where the model learned it. Scores four ways — all-voxels (every voxel
with GT ≠ 255, the set ISO and MonoScene report) and CleanerS's own
`label_weight` protocol, each on the released GT and on `target_fov` — and
writes the report plus `per_frame.csv`. It also rescores the NYU test
predictions both ways: the cleaners row must reproduce EVALUATION.md
(SC 75.0 / mIoU 47.7, verified), and the all-voxels row isolates what the
protocol alone does to the same predictions. Further sections: per-class
tables, confusions, false positives on `empty` vs another class, recall by
surface / seen-through / hidden, and scores binned by camera heading relative
to the grid axes (pooled and within-scene) and by pitch.

### `export_scannet_ply.py` — best and worst frames as .ply

```bash
python -m inference.evaluate_scannet --pred_dir ./outputs/scannet
python -m inference.export_scannet_ply --pred_dir ./outputs/scannet
python inference/display_overlay.py outputs/scannet/ply/<scene>/<frame>_pred.ply
```

For every scene, the frame with the highest and the lowest per-frame SSC mIoU
under the **NYU protocol** (`ssc_miou_nyu_protocol` in `per_frame.csv`), skipping
frames with under 2,000 scored voxels or under 2 GT classes, whose scores are
noise. Each is written the way `test_NYU.py:visualize_3d_predict` writes NYU's:
`pred[label_weight == 0] = 0`, then `labeled_voxel2ply`, with `label_weight =
GT object | tsdf < -0.5` (verified voxel-for-voxel against an independent
computation). Per frame, in `ply/<scene>/`:

| file | contents |
| --- | --- |
| `<frame>_pred.ply` | prediction, `label_weight`-masked |
| `<frame>_gt.ply` | GT, masked the same way (255 dropped, as `labeled_voxel2ply` does) |
| `<frame>_rgb.png` | the colour image registered onto the 640×480 depth camera — what the model saw |
| `<frame>.json` | `vox_origin`, `cam_pose`, depth `cam_K`, depth path, scores |

`ply/index.csv` lists every exported frame with its role and scores.
`display_solid.resolve_frame` looks for `<frame>.json` next to a `_pred`/`_gt`
ply, so `display_solid.py`, `reconstruct.py` and `display_overlay.py` all open
ScanNet frames with no flags — photo colouring, the measured surface and the
overlay included.

```bash
python inference/display_overlay.py outputs/scannet/ply/<scene>/<frame>_pred.ply
python inference/display_overlay.py outputs/scannet/ply/<scene>/<frame>_gt.ply
```

### `show_nyu.sh` / `show_scannet.sh` — prediction next to GT

Open a frame's prediction and its ground truth in two `display_overlay.py`
windows at once, plus the colour image in `eog` (NYU: `NYU<id>_colors.png`;
ScanNet: `<frame>_rgb.png`, the photo registered onto the depth camera — what the
model saw; `NO_RGB=1` skips it). Ctrl+C in the terminal closes all three. Both `.ply` files are masked the way
`test_NYU.py:visualize_3d_predict` masks the prediction (outside `label_weight`
→ empty), so what you see is what the NYU protocol scores.

```bash
# NYU: CleanerS prediction vs NYU's own GT (data/NYU/Label), not ISO's
inference/show_nyu.sh 0001
# Occ-ScanNet: prediction vs Occ-ScanNet GT
inference/show_scannet.sh scene0702_00                 # the scene's best frame
inference/show_scannet.sh scene0702_00 worst
inference/show_scannet.sh scene0702_00 00004           # any frame; exported on demand
# anything after the frame goes to both display_overlay windows
inference/show_nyu.sh 0001 --show voxels
# which run / which python
PRED_DIR=outputs/scannet PYTHON=python3 inference/show_scannet.sh scene0702_00
# the full prediction instead of the label_weight-masked one
MASK=none inference/show_nyu.sh 0001
```

**`MASK=none` is worth looking at once.** `label_weight` is built from the GT
(GT objects plus occluded space), so the default view hides most of what the
model predicts: 86% of the occupied voxels on NYU0001 (9,029 shown of 63,851),
and 94% across Occ-ScanNet. CleanerS's loss uses the same mask, so it was never
taught what belongs outside it, and it fills the free space in front of the
camera with objects. That hidden part is what ISO's all-voxel scoring counts.
The full prediction goes to `<frame>_pred_full.ply` / `NYU<id>_pred_full.ply`,
leaving the masked files untouched; the GT window is identical either way,
since every GT object voxel is inside `label_weight`.

The `.sh` scripts only launch the viewers; `show_nyu.py` / `show_scannet.py`
write any missing `.ply` pair (`--export_only`) and can also be used directly for
`--list N` (best and worst frames), `--only pred|gt`, and `--screenshot PNG` (the
two side by side in one image).

- NYU plys go to `custom_visual_pred/CleanerS/compare/NYU<id>_{pred,gt}.ply`;
  the `NYU` prefix lets `display_overlay` find the `.bin`, depth and RGB.
- ScanNet plys come from `export_scannet_ply.py`'s `ply/<scene>/` (frames not
  exported yet are written on demand). Needs `per_frame.csv`, so run
  `evaluate_scannet.py` first.
- Each run prints the frame's NYU-protocol SC IoU and SSC mIoU.
- Both windows are opened with **`display_overlay.py --frame-grid`**: the depth
  axis is reflected about the grid (`display_solid.FIXED_PIVOT = 59`) instead of
  about each file's own voxels, and the view frames the whole 60×36×60 grid, so
  the same voxel sits at the same place on screen in both. Without the flag
  `display_overlay` behaves as before.
- Scene-level "best" frames can be trivial: a frame whose score comes from just
  over 2,000 voxels of one wall can top its scene. `--list` shows the voxel
  count; pick frames with more.

---

## The `sample` dict

The contract between `frame_loader` and `model`, returned by
`build_tsdf_and_mapping()`:

| key | shape | dtype | meaning |
| --- | --- | --- | --- |
| `tsdf` | (1, 60, 36, 60) | float32 | signed distance, 0 = unobserved |
| `weight` | (60, 36, 60) | float32 | 1 where observed — the `observed %` source |
| `mapping` | (129600,) | int64 | voxel → pixel index, `307200` = unmapped |
| `mapping2d` | (480, 640) | int64 | pixel → voxel index, `-1` = unmapped |
| `img` | (480, 640, 3) | float32 | BGR, **not yet normalised** |

`mapping2d` is built by inverting `mapping`: for every voxel with a valid pixel,
write the voxel index at that pixel. Note the two sentinels differ — `307200`
for `mapping`, `-1` for `mapping2d`.

`model._build_forward_kwargs` then produces the seven forward kwargs: `img`,
`tsdf`, `tsdf_CAD`, `mapping`, `mapping2d`, `label3d`, `label_weight`.

---

## Coordinate conventions

- Voxel grid: 240×144×240 high-res (2 cm), 60×36×60 low-res (8 cm).
- **Axis 1 (144 / 36 cells = 2.88 m) is room height.**
- NYU pose: `cam +Y (down) → world (0,0,-1)`, so world +Z is up;
  `t = (0, 0, camera_height)`; world Z=0 is the floor.
- `vox_origin = [-2.4, 0.0, -0.05]` — 4.8 m wide (world X, centred on the
  camera), 4.8 m deep (world Y, from the camera plane), 2.88 m tall (world Z,
  from 5 cm below the floor).
- Voxel axis assignment inside the encoder: `gz ← world X`,
  `gx ← world Y` (depth), `gy ← world Z` (height). Grid arrays are indexed
  `occ[gz, gy, gx]`, so **array axis 1 is `gy` — the 144/36-cell height axis**,
  and `vox_origin[0,1,2]` pairs with world `X, Y, Z` respectively.

---

## What to notice

**1. The depth encoding difference is a *silent* failure.**
The encoding itself carries no information: `((d << 13) | (d >> 3))` is a
16-bit rotate-right-by-3, verified bijective over all 65,536 uint16 values with
an exact round trip. Both encodings mean 1 mm quantisation up to 65.535 m, and
depth reaches the model only as float32 metres — decoded at `frame_loader.py:147`,
`camera.py:555` and `run_inference.py:241` — so the storage format cannot affect
a prediction. What it affects is the *failure mode*.
NYU PNGs and capture PNGs are the same container (uint16, 640×480) with
different meanings. Decode a capture with the NYU rotation and a 2.53 m wall
reads as **32.8 m**; decode NYU as plain mm and 3.00 m reads as **24.0 m**. No
crash, no shape error, and the valid-pixel percentage is *unchanged*. All four
decode paths are verified, and the cross-mode guards fire: `--live_dir` at an
NYU folder → `FileNotFoundError` about meta.json; `--data_dir` at a capture
folder → 0 pairs found; an 8-bit depth PNG → `ValueError`.

**2. `inverse_brown_conrady` does not mean invert it.**
The name means the coefficients are *already fitted for deprojection*, so the
polynomial must be applied **forward** when building undistort maps. Inverting
it with `cv2.undistortPoints` measured **6.6637 cm residual — worse than the
3.34 cm uncorrected**. Forward application gives **0.0025 cm**. Caught only by
checking against `rs2_deproject_pixel_to_point` rather than trusting the name.

**3. Don't FOV-match twice.**
`run_live.py` defaults to `--fov nyu`, so replaying an already-matched capture
would run `match_nyu_fov` a second time against intrinsics describing the
*output*. `FolderSource` sets `self.fov = 'none'` when `meta['fov'] == 'nyu'`.

**4. uint16 wraps silently.**
A near-zero-disparity pixel converts to ~65.5 m, and uint16 turns a 70 m reading
into a 4 mm one. `_to_uint16_mm` clamps to 0; `--depth_max` cuts it earlier.

**5. 640×480 is not negotiable.**
`MAPPING_SENTINEL = 480 × 640 = 307200` is baked into both the encoder and the
SC evaluation mask. Nor is the *crop*: keeping the full 90° FOV leaves the
geometry intact but costs 23% of voxel labels, collapsing `window` from 6.0% to
0.5%. Measured in
[CAMERA_TUNING.md](CAMERA_TUNING.md#frame-size-and-fov--do-not-skip-the-crop).

**6. `camera_height` is the most important number you type.**
The grid is floor-anchored. Confirmation that 1.42 m was right on `room01`: all
3,613 floor voxels in the 0.00–0.48 m band, 1,093 ceiling voxels in
1.92–2.88 m, zero bleed between them.

**7. There is no pitch/roll compensation.**
`from_live_camera` assumes the camera is level. A capture pitched upward gave
ceiling 10.3% vs floor 4.2% — backwards. The IMU could supply this.

**8. USB 3 is mandatory.**
On USB 2.1 the SDK *silently removes* high-rate profiles — 1280×720@5 got 0/20
frames. On USB 3.2 the same camera does 30/30 in 1.4 s. Check
`usb_type_descriptor` first; device permissions are not the usual cause.

**9. `fy` is 0.24% off NYU and that's fine.**
D455 `fy/fx = 0.99880` vs NYU `1.00118` — different pixel geometry, which
cropping cannot fix. Harmless, because `K_eff` reports the true value.

**10. Intrinsics drift as the camera warms.**
`fx` 637.7892 @ 32 °C → 638.3094 @ 35 °C, `cx`/`cy` fixed. ~1.6 px, ~4 mm at
3 m. Hence `cam_K` per-capture in `meta.json`.

**11. The mask is display-only.**
`--mask` changes the `.ply` and nothing else. The `.npy` is always the full
unmasked `(60,36,60)` grid.

**12. `observed %` is the health check.**
`room01` gave 45.1% against 46.3% for an NYU reference frame. Below ~25% means
the depth was too sparse — usually the subject is inside the ~0.5 m blind spot.

**13. Image normalisation is not optional.**
Feeding raw BGR to the ImageNet-pretrained `mit_b2` encoder collapses
predictions onto one or two dominant classes. `_normalize_img` handles it.

---

## Verified device numbers

```
D455  serial 037522250930  fw 5.17.0.10  USB 3.2  D400
depth_scale 0.001  laser_power 150  emitter on
IMU: accel 63/250 Hz, gyro 200/400 Hz

color 1280x720: fx 637.79  fy 637.02  cx 638.22  cy 364.37   FOV 90.21 x 58.96
                inverse_brown_conrady
                [-0.054626, 0.065394, 0.000183, 0.000191, -0.020497]
depth 1280x720: fx=fy 648.09  cx 635.97  cy 361.63            FOV 89.28 x 58.10
                coeffs all zero
depth->color translation: (-58.967, -0.068, 0.471) mm

after FOV match:  fx 518.86    fy 518.23    cx 325.21  cy 253.42
NYU reference:    fx 518.8579  fy 519.4696  cx 325.58  cy 253.74
```

The 59 mm depth→colour baseline is why alignment is mandatory: without it,
depth and colour pixel `(u,v)` disagree by 30.6 px at 1 m and 15.3 px at 2 m —
depth-dependent, so no constant offset fixes it. Aligned depth inherits
**colour** intrinsics and **colour** distortion.

---

## Documentation drift

Docstrings that no longer match the code. Worth fixing; listed so nobody
trusts them in the meantime.

| where | says | actually |
| --- | --- | --- |
| `frame_loader.py` module docstring | `from_live_camera` builds a "CAMERA-CENTRIC grid" with "`cam_pose` = identity" | builds a gravity-aligned floor-anchored grid; the method docstring explicitly warns that identity pose rotates the scene 90° |
| `camera.py` module docstring | D455 RGB is "90x65" degrees | measured 90.21 × 58.96 |
| `utils.py` `load_config` | "PLACEHOLDER: config loading" | dead code — both runners use their own `load_cfg` with `EasyConfig` |
| `run_inference.py` "STATUS" block | open question about whether `forward()` ignores the placeholders | resolved — `model.py`'s docstring records the confirmation |

---

## Open items

- ~~Encoding costs 915 ms/frame~~ **done.** `FrameLoader(encoder='torch')`
  does it in 116 ms and `run_live` defaults to it on any machine with a CUDA
  card: 0.75 → 3.16 fps for the whole pipeline. The `'cuda'` extension turned
  out **not** to be the fast path (1101 ms, not the 285 recorded here — that
  figure does not reproduce). `run_inference` still defaults to numpy on
  purpose. See [HARDWARE.md](HARDWARE.md).
- **The model forward is now the floor**, 201 ms of a 317 ms frame, and it is
  not addressable here: fixed-size 3D convolutions, nothing reusable between
  independent single-shot completions. Past ~0.29 s needs TensorRT/ONNX, a
  faster card, or a world-fixed volume — which needs the localisation below.
- IMU gravity alignment — hardware confirmed present, not wired in.
- The volume is camera-fixed, so frames are independent completions rather than
  an accumulating reconstruction. That needs SLAM driving `cam_pose`.
- 3-crop tiling to recover the full 90° FOV without the distribution shift —
  see [CAMERA_TUNING.md](CAMERA_TUNING.md).
- Nothing in `inference/` is committed.
