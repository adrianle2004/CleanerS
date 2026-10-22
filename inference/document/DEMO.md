# Showing the room — a run-book

For putting this in front of someone who does not know the project. What
to type, what they are looking at, what order to show it in, and what the
numbers are when they ask.

The mode is `run_live --accumulate`. Everything it does is documented in
[VIEWERS.md](VIEWERS.md#the-accumulating-room); this file is the short
version, plus the results.

- [What it shows](#what-it-shows)
- [Running it](#running-it)
- [The window](#the-window)
- [The four things it can draw](#the-four-things-it-can-draw)
- [Two minutes, in order](#two-minutes-in-order)
- [What it produced](#what-it-produced)
- [What was built to get here](#what-was-built-to-get-here)
- [Limits, and what to say](#limits-and-what-to-say)

---

## What it shows

A depth camera can only measure what it can see. Point one into a room and
you get the front of the furniture, the near wall, and nothing else — no
floor under the bed, no back of the sofa, no wall behind you. Semantic
scene completion is the model's job of filling that in: given one frame,
say what is in the whole volume, including the parts nothing measured.

The problem with showing that a frame at a time is that there is nothing
to hold on to. The volume is fixed to the camera, so every frame is an
independent answer in its own coordinates, and the picture changes
completely each time the depth frame does. Nobody can tell the model's
output from the sensor's, or judge whether any of it is right.

`--accumulate` fixes one room-sized grid in the world and fills it as the
camera moves. Now the camera walking forward *adds* to the picture instead
of replacing it, and — the point of the whole thing — **when the sweep
reaches somewhere the model had already filled in, the measurement arrives
and settles the claim**. The model guesses first, the sensor arrives
later, and the display keeps score.

---

## Running it

### A recorded sweep

```bash
conda activate CleanerS && cd ~/CleanerS/CleanerS
python -m inference.run_live \
    --cfg ./cfgs/NYU/voxelSSC.yaml \
    --pretrained_path ./checkpoint/CleanerS_ckpt.pth \
    --source bag:captures/room07/scan/scan.bag \
    --accumulate --display --zoom 2.2 \
    --out_dir outputs/room07_world
```

No `--camera_height`: `captures/room07/meta.json` carries the measured
1.0858 m and the bag picks it up. Full sweep is 808 frames, about five
minutes. `captures/room08/scan/scan.bag` is the other one — a green
bedroom, 1065 frames.

Useful variants:

| | |
| --- | --- |
| `--max_frames 150` | about a minute; enough to be past the camera's white balance |
| `--sensor_first` | open on the sensor's reconstruction, press `m` to reveal the model |
| `--world_colour verdict` | open on the green/red comparison instead of the photo |
| `--amp off` | numbers that do not move between runs |
| `--record` | every drawn frame into `<out_dir>/frames/`, for a video |
| `--no_hold` | do not hold the window open at the end (scripted runs) |
| `--zoom 1.2` | a wider opening view; `0` in the window fits whatever is there, and a finished run does it for you |

### A live camera

```bash
python -m inference.run_live \
    --cfg ./cfgs/NYU/voxelSSC.yaml \
    --pretrained_path ./checkpoint/CleanerS_ckpt.pth \
    --source realsense \
    --accumulate --display --zoom 2.2 \
    --camera_height 1.09 \
    --out_dir outputs/live_world
```

**Measure the lens height and pass it.** A live stream has no
`meta.json`, so without it the run warns and assumes 1.25 m, and 15 cm out
moves about 14% of the prediction. `--accumulate` refuses
`--camera_height auto` outright: the world is anchored on the first frame
and a height revised later would move it out from under everything
already in it.

Hold the camera so **furniture or a corner is in view**. The tracker works
on depth geometry alone, and a flat wall filling the frame is degenerate —
the camera can slide along it without the alignment error changing.

---

## The window

| key | |
| --- | --- |
| `space` or `f` | freeze / carry on |
| `m` | flip between the room and the sensor's own reconstruction (needs `--accumulate`) |
| `c` | cycle `photo` → `verdict` → `class` |
| `+` `-` | zoom in and out |
| `0` | fit what has been built so far into the window |
| `s` | screenshot into `--out_dir` |
| drag / scroll | orbit and zoom |
| close the window | end the run there — it still writes everything |

Freezing puts nothing more into the room and holds it exactly as it is,
with the window live so it can be orbited. A **replay** pulls nothing
while it holds and carries on at the frame it stopped at. A **live
camera** keeps being read and *tracked* — nothing is integrated, measured,
predicted or drawn, so the room on screen is the one that was frozen, but
the pose keeps up with the camera in your hand. Without that, letting go
after a few seconds handed the tracker a view that had jumped, and the
room stopped growing for the rest of the run.

Hold a live camera for a few seconds, not a few minutes: nothing is being
integrated while it holds, so the model the pose is tracked against stops
growing, and a hold long enough to walk somewhere new will lose it.

**If the readout says `TRACKING LOST`, point the camera back at what it
was looking at.** It re-acquires in three frames and the room carries on
from where it stopped. The window stays live and keyed throughout.

At the end of a run **the window stays up**, frozen, showing the sensor's
reconstruction — so a replay finishes on something rather than vanishing.

**The reconstruction comes from the tracker, and the tracker only runs
under `--accumulate`.** Without that flag `space` still freezes the frame
on screen, but there is no volume to extract and no amount of freezing
will make one. If `m` says so, restart with the flag.

The counts live in the small camera panel window, not in the 3D window;
the legacy renderer cannot draw text. `--no_panels` turns both off.

---

## The four things it can draw

Three of them are the same geometry asked three different questions, and
`c` cycles them. The fourth is different geometry, and `m` flips to it.

### `photo` — what does this place look like?

The default, and the one to open on. Every voxel is the same size and
wears the camera's own colour: what the pixels that measured it reported,
weighted towards the most recent look. Where the camera never saw a voxel
it borrows from the nearest measured voxel *of the same class*, and a
class with no measured member anywhere keeps a dimmed class colour — that
is the flat green floor you see early on, which is the fallback working,
not a bug.

A claim standing in space the camera **measured as empty** is not drawn at
all here. This mode is answering what is there, and about measured space
the sensor outranks the model.

### `verdict` — did the model get it right?

Photo colours, overpainted where there is something to say:

| | |
| --- | --- |
| **green** | the model filled this in while the sensor knew *nothing* here, and the sweep went on to **prove** it |
| **red** | the model says solid, the camera **saw straight through** that space |
| **blue** | the sensor found it and the model never predicted it |
| **grey** | the model claims it and the sensor still has no evidence either way |

The model's unmeasured claims are drawn smaller here, and they have to be:
faces between two occupied voxels are culled, so at full size you see only
the outermost shell — and that shell is the unverified claims wrapped
around the real room, with every proved voxel hidden inside it.

### `class` — what does the model think each thing is?

The NYU palette: floor, wall, bed, chair, and so on. Useful for the first
few seconds of a run, before any colour is trustworthy.

### `m` — what the sensor alone had

A triangle mesh of everything the depth has measured, at the same
viewpoint, so the only thing that changes when you press `m` is the
geometry. **The mesh stops where the depth stopped**, so every hole in it
is somewhere the sensor could not see — and that is exactly what the room
beside it is filling in.

With `--accumulate` it is free: the tracker already integrates a 2 cm
TSDF to align each frame against, which is the same structure `fuse_scan`
carves ground truth out of, and extracting a mesh from it takes 3–7 ms.
Without `--accumulate` the key still works and shows **this frame's own
surface** — the one depth image integrated into a volume of its own and
thrown away after, 6–7 ms a frame, built only while `m` is showing. It is a thing to **look at**,
not a measurement — no checks are run on it, and
`reconstruction_GT/fuse_scan.py` remains what produces the committed
`room.ply`.

---

## Two minutes, in order

1. **Start it and say nothing for ten seconds.** One frustum of room
   appears. Let them watch it grow as the camera pans.
2. **Freeze early, press `m`.** The sensor's mesh is a single wedge with a
   room-shaped hole around it. Press `m` again: the room already has
   walls, a floor and a bed. *That difference is the model.*
3. **Carry on to the end.** The mesh closes up and the room fills. Say
   that the model guessed those walls hundreds of frames before the camera
   got there.
4. **Press `c` to `verdict`.** Green is where it guessed blind and the
   sweep later proved it right. Red is where it was wrong. Give the
   number: on room07, **86.7% of the completions the sweep could reach
   were right**.
5. **Then give the honest one.** 8 175 of its completions were never
   tested and most never can be — inside the furniture, behind the walls.
   The 86.7% is of the 4 020 that *were* reachable. Say this before
   someone else asks it.

In the first four seconds of any run, look at the geometry and not the
colours: the camera's white balance has not converged, so the mesh comes
out pink and the room is still on class colours. Both are right and both
are ugly.

---

## What it produced

Both rooms, every frameset, none dropped by the tracker.

| | room07 | room08 |
| --- | ---: | ---: |
| framesets | 808 | 1 065 |
| voxels the **sensor measured** | 6 513 | 7 764 |
| voxels the **model claims** | 17 225 | 26 000 |
| of those, **completions** — filled in where the sensor knew nothing | 12 195 | 20 361 |
| completions the sweep **reached** | 4 020 | 4 888 |
| **confirmed** | **3 485 (86.7%)** | **3 716 (76.0%)** |
| contradicted | 535 | 1 172 |
| still untested when the sweep ended | 8 175 | 15 473 |
| measured but never predicted | 322 | 1 108 |
| claims in space the camera measured as empty | 2 805 | 3 790 |
| reconstruction mesh | 168 k triangles | 205 k triangles |

room08 scores lower and that is not noise: it is a longer sweep of a
bigger space, so more of the completion became reachable and more of it
turned out wrong. A short sweep flatters the model by never testing it.

**Checks** — `inference/check_world_map.py`, against things this repo
already committed:

| | room07 | room08 |
| --- | --- | --- |
| `algebra` — the placement arithmetic | 7 889 of 7 889 voxels placed, 0 missing, 0 extra | — |
| `tracker` — against `scan/trajectory.txt` | median **39 mm**, p90 85 mm, max 142 mm | median **53 mm**, p90 83 mm, max 121 mm |
| `measured` — against `scan/room_cloud.ply` | **89.8%** within one voxel, 98.3% within two | **94.3%** within one voxel, 97.8% within two |

Tolerance is 150 mm and
[DRIFT_AND_ALIGNMENT.md](../../reconstruction_GT/document/DRIFT_AND_ALIGNMENT.md)
documents 66–102 mm of
drift for these captures, so both trackers are inside what the capture
itself is worth. The exact-match figure (53.6% / 55.5%) is the wrong
question — both sides are 8 cm shells built from trajectories about a
voxel apart.

**On a live D455** (fw 5.17.0.10, USB 3.2, 1280×720 at 30 cropped to
NYU's FOV), on the 4 GB card: 2.05 fps with the window and 2.41 without,
80 of 80 frames tracked with none dropped, 80 of 80 accelerometer samples
used, reconstruction out in 3 ms. Peak VRAM **3 413 MB of 4 096** with
everything up, and it plateaus rather than creeping — the tracker's volume
is a fixed `--track_blocks` and the map lives in host memory, so a long
sweep does not creep up on the card. `--track_device cpu` takes the peak
to 2 473 MB for 41 ms a frame.

**Cost**, against a 331 ms frame on room07: tracking 13 ms, measuring and
gathering the prediction 24 ms, the room mesh ~38 ms every third frame.
End to end 2.1 fps with the flag against 3.0 without.

**Files**, at `--out_dir`:

| | |
| --- | --- |
| `world.npz` | the whole map — reopen it without re-running the model |
| `world.ply` | occupied voxels, world **metres** |
| `reconstruction.ply` | the sensor's own mesh, same frame, same metres |
| `poses.npz` | the in-loop trajectory, which `check_world_map tracker` reads |
| `frames/` | with `--record`, every drawn frame |

`world.ply` and `reconstruction.ply` load on top of each other in
CloudCompare, beside `reconstruction_GT`'s `room.ply`.

Numbers move by a handful of voxels between runs of the same bag: AMP is
not bitwise reproducible and a voxel sitting on the vote threshold can
fall either way. Use `--amp off` for a figure that must not move.

---

## What was built to get here

In the order the problems turned up.

**One grid in the world, not one per frame** (`inference/world_map.py`).
Pure numpy, so every fusion rule is tested on 4×4×4 grids. Per voxel it
keeps what the sensor measured, what the model claimed, and whether the
sensor knew anything *at the moment the model claimed it* — that last flag
is what makes a completion a completion.

**The pose comes from a tracker in the loop** (`inference/tracker.py`),
the tracking half of `reconstruction_GT/fuse_scan.py`, frame-to-model so
error does not compound — and if someone asks how it knows where the
camera is, [WORLD_BUILDING.md](WORLD_BUILDING.md) answers that from
nothing, with diagrams. Not from the committed `trajectory.txt`: that
file indexes poses by every frameset after a 1 s skip, while the reader
here counts only framesets that survive processing and skips nothing, so
the two disagree by a data-dependent offset and nothing would have
reported it — the room would just have come out quietly smeared.

**A frame the tracker is unsure of contributes nothing at all.** A bad
pose does not degrade the map, it permanently corrupts it.

**Being unsure stopped being fatal.** Frame-to-model tracking compares
every frame with the model as seen from the last believed pose, so one
discontinuity used to end a run: the reference never moved again and every
later frame failed against a viewpoint the camera had left. Two things
cause that discontinuity, and both are now handled. Holding the window on
a live camera no longer throws its frames away — they are tracked while
the room stays frozen, which took a four-second hold from *15 frames and
lost* to 45 of 45 tracked. And when the pose does go, pointing the camera
back at what it was looking at re-acquires it in three agreeing frames,
4 mm from where it stopped on room07. The window used to go grey and stop
answering keys the moment that happened; it now draws every frame, grown
or not.

**And what counts as unsure is measured against the run itself.** Over
both captures end to end, 1 871 frames, the fitness never fell below
0.647 — but resuming room07 at frame 200 after frame 59 reads 0.15–0.23
and `fuse_scan`'s 0.1 floor waves it through, welding twenty frames a
quarter of a metre from where they belong. The floor is now 60% of the
median of the last 30 believed frames, capped at 0.55. Both full sweeps
still drop nothing.

Four silent bugs, each of which produced a plausible-looking wrong answer:

| | |
| --- | --- |
| the network's grid is **not** indexed `(X, Y, Z)` | axis 1 is the 36-cell height axis. Nothing raised; the only symptom was 2 848 of 2 881 measured voxels having no prediction |
| open3d's `trunc` is a voxel **multiplier**, not metres | would have made truncation 0.16 instead of 8 |
| `model.integrate` builds a *coloured* TSDF | the tracking frame needs a colour channel it never reads |
| the tracker was **guessing a frame rate** | room07's bag is 15 fps, the gate assumed 30, ordinary handheld steps were rejected as impossible and the map silently stopped filling 403 frames in |

**The camera's first seconds do not get to say what anything looks like.**
A D4xx opens with auto white balance unconverged — room07's first frame is
7.4% saturated and renders a white wall pink, and the channel means do not
meet until frame 60, four seconds in. A running mean kept that forever, so
one wall stayed pink while the rest of it went white. Now the colour is a
blend weighted to the latest look, and `--colour_settle auto` watches the
frame's chromaticity and holds the colour back until it stops moving —
frame 70 on room07, 49 on room08, 13 and 36 on two live opens. Geometry
never waits; the depth stream has its own exposure.

**The model has to agree with itself before the room shows a claim.** A
voxel one frame glanced at and called a chair was a prediction,
permanently, because a bare majority of a single look is a majority. The
depth is unsteady, so there are plenty of those at every surface edge, and
over a sweep they became spurs on the furniture. A claim now needs three
frames to have seen the voxel and 60% of them to agree: 12% of the claimed
matter goes, 1.2% of the proved goes with it, and the proved share rises
from 85.5% to 86.7%.

**The sensor outranks the model about space it has measured.** Outside
`verdict`, a claim standing in measured-free space is not drawn — 2 805
voxels on room07, including the one-voxel skin that was making every wall
a voxel too thick.

**And the sensor's own reconstruction, beside it** — `m`, above. The room
answers what the model thinks is there; it could not answer what the
sensor actually had to go on, which is the next question anybody asks.

---

## Limits, and what to say

**Most completions are never settled, and that is the honest headline.**
The inside of the sofa and the far side of the wall are unreachable
however long you sweep. Report `untested` as its own number; never fold it
into the score.

**The score is of what the sweep could reach**, not of the model. Quote it
that way.

**Red is not the same question as contradicted.** The score counts
completions the sweep disproved. Red asks the wider question — where has
the model put matter in space the camera saw through — and drops the
one-voxel skin along every surface, which is below the resolution of the
comparison. The exclusion is *not* applied to the score; dropping the
model's near misses from it would simply flatter it.

**The opening camera angle is whatever open3d's fit gives.** In 0.17 the
window's view control returns a copy, so the camera cannot be aimed from
python — the framing is done to the geometry instead (`--zoom`,
`--view_extent`) and the viewpoint is set by hand with the mouse.

**No picking and no on-screen legend in the 3D window.** The legacy
renderer has no mouse-position callback and can draw no text. Open the
saved `.ply` in `display_solid.py` for per-voxel labels.

**The key dispatch itself is not automatically tested.** The handlers are
verified by calling them directly, but this machine has no way to
synthesise a key press. If a key turns out dead, `--world_colour`,
`--sensor_first` and `--zoom` do the same jobs from the command line.

**A pose that cannot be believed stops the room, and that is deliberate.**
Geometry welded into the wrong place never washes out, so an unsure frame
contributes nothing. What is unsure is measured against the run itself —
the floor is 60% of the median fitness of the last 30 believed frames,
where both committed sweeps ran at 0.65 and better throughout — and it is
recoverable: bring the view back and three agreeing frames put the room
back to work. Nothing goes in until all three agree.

**A hold is not a pause for a live camera.** See the note under
[The window](#the-window): the room freezes, the tracking does not.

**The reconstruction carries the camera's early white balance**, unlike
the room — the tracker has to hand `integrate` a colour channel from the
first frame. Over a full sweep it shows as a faint pink cast on whatever
the camera saw only at the start and never came back to.
