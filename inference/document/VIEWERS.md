# Viewers — every button and every mode

Five files show the same 60×36×60 prediction. They differ in what they draw,
what they draw it *against*, and whether the thing on screen is a file or a
camera.

| file | draws | use it to |
| --- | --- | --- |
| [`display.py`](#displaypy) | voxels as screen-space points | the original viewer; kept working, unchanged |
| [`display_solid.py`](#display_solidpy) | voxels as solid 8 cm cubes | look at the prediction |
| [`reconstruct.py`](#reconstructpy) | the measured depth surface, photo-textured | look at what the camera actually got |
| [`display_overlay.py`](#display_overlaypy) | both, cubes made see-through | compare the two |
| [`live_view.py`](#live_viewpy) | the solid cubes, replaced every frame | watch the camera as it runs |
| [`live_view.py --accumulate`](#the-accumulating-room) | ONE room that fills as the camera moves | show someone what the model is actually doing |

The first four open a **file**. `live_view.py` is the odd one out: it is not a
command, it is the window `run_live --display` drives from its own loop.

**If you are showing this to someone who does not know the project, use
`--accumulate`** — and [DEMO.md](DEMO.md) is the run-book for exactly
that: what to type, what to say, in what order, and what the numbers are
when they ask. Frame-by-frame is the honest picture of what the network
computes, and it is also the picture nobody can read: one point of view
changes completely every frame because the depth frame does, so there is
nothing to hold on to and no way to tell the model's output from the
sensor's. The accumulating room answers both at once.

All four of the file viewers use **buttons, not key bindings**. That is not a
style choice — see [Why buttons](#why-buttons-and-not-keys). The live window
is the exception, and the reason is in that section too.

```bash
python inference/display_solid.py    outputs/room03/ply/live_000000.ply
python inference/reconstruct.py      outputs/room03/ply/live_000000.ply
python inference/display_overlay.py  outputs/room03/ply/live_000000.ply

# every voxel coloured by whatever evidence reaches it, labelled per voxel
python inference/display_solid.py    outputs/room07/ply/live_000000.ply --visibility all
python inference/display_overlay.py  outputs/room07/ply/live_000000.ply --visibility all --colour photo
```

---

## `display_solid.py`

Solid cubes, coloured by what the camera saw. Window 1600×1000, readout bar at
the bottom.

### Buttons

| button | label cycles through | what it does |
| --- | --- | --- |
| `colour: …` | `photo` → `class` | which palette the cubes are drawn in |
| `reframe` | — | back to the start view (fits the occupied extent at mid-depth) |

### Mouse

| | |
| --- | --- |
| hover | outlines the voxel under the cursor in yellow and reports it |
| Ctrl+click | prints that same line to stdout |
| drag / scroll | orbit and zoom, as usual |

The readout line looks like

```
voxel [30, 15, 20]   objs (class 11)   photo rgb (110,  98,  87)   measured inside this voxel
```

and the last field names the rule that actually supplied the colour —
`photo_colours` returns a per-voxel `source` code and the readout reads it, so
under `--visibility all`, where three rules are in play at once, the line still
says which one this voxel got:

| readout | means |
| --- | --- |
| `measured inside this voxel` | depth points landed inside this cube; its colour is its own measurement |
| `measured just outside it (within --reach)` | under `--visibility surface` or `all`; the colour came from a neighbouring cell |
| `read from the photo (no depth there to disagree)` | only under `--visibility all`; no depth reached it, but it is front-most on pixels the depth frame does not contradict |
| `read from the photo (the depth there measures something else)` | only under `--visibility all`; the last stage — the pixel is real, but the depth says the surface at it is at another range |
| `hidden -- colour inferred` | no measurement near it; it is showing its class colour |

### Modes (flags, not buttons)

**`--visibility` — where a voxel's photo colour comes from.** Default `points`.

| value | rule | notes |
| --- | --- | --- |
| **`points`** *(default)* | the mean colour of the measured points **inside** the voxel (a Chebyshev half-width of 0.5, i.e. exactly the cube) | exact, parameter-free, and **no voxel is ever coloured from another's measurements**. Leaves flat class-coloured speckles where the camera's coverage has a ragged edge |
| `surface` | the same, with the region grown to `--reach` voxels (0.75 = 6 cm) | fills those speckles by taking a *neighbouring* cell's measurements. That is an inference; the readout labels it separately. Hold-out error 0.6–7.8 of 255 against 18.5–52.9 for two arbitrary voxels |
| `zbuffer` | the front-most voxel over each image pixel | the only rule that needs no depth frame — it is what this file did before `resolve_frame` returned one. Misses 863 of room01's 1615 measured voxels, because on a surface seen edge-on the nearest voxel wins every pixel |
| `all` | all three, each voxel taking the first rule that reaches it | see below |

**`--visibility all` in detail.** Colour every voxel you can, and say for each
one how you did it. The rules are stacked in order of how directly each one
measures the voxel it is colouring, and a voxel takes the first that reaches
it:

| | stage | evidence |
| --- | --- | --- |
| 1 | the measured points **inside** the voxel | `points` |
| 2 | the measured points within `--reach` of it | `surface` |
| 3 | the photo pixels it is front-most on, **where the depth does not contradict them** | the camera's colour where its ranging failed |
| 4 | the photo pixels it is front-most on where the depth *does* say something else | a real pixel, of possibly the wrong surface |

Nothing is ever downgraded — every voxel `points` could colour keeps exactly
the colour `points` gave it, from its own measurements alone, which is not
true of `surface` (there even a voxel full of its own points takes the mean
over the grown region). The later stages only fill in around it, and the
readout says which voxels those are.

**Stage 3 is not plain `zbuffer`.** A pixel counts for a voxel only if

- the depth there agrees with the voxel's own range to within `--agree` (one
  voxel, 8 cm) — the pixel is a measurement *of* this voxel; or
- there is no depth there at all. Dropout, and the photo is then the only
  evidence either way: nothing in the measurement and nothing in the
  prediction puts anything in front of this voxel.

`zbuffer` makes neither test — it resolves occlusion among the *voxels* and
never consults the depth frame, so a voxel standing behind a measured surface
is still front-most if no other predicted voxel is on that ray, and it takes
the colour of whatever the camera really saw there. Of the voxels the ungated
z-buffer adds beyond the two 3D rules on room01's frame, 88% own pixels whose
depth is over 0.2 m away, a median of 1.41 m **in front** of them.

So those pixels are not thrown away, they are demoted: **stage 4** is exactly
them, and it only ever colours a voxel nothing else reached. Held out against
voxels whose own colour is known, over 7 frames (room01/03/04/07,
NYU0015/0284/0515):

| | median error | 90th |
| --- | --- | --- |
| stage 3, gated | 0.8–7.8 of 255 | 6.9–47.5 |
| stage 4, the pixels it threw out | 3.0–27.3 | 30.5–93.5 |

Clearly worse, and clearly better than nothing: `--strict-photo` stops after
stage 3 if you would rather have the flat class colour than a real pixel of
possibly the wrong surface. Stage 3 adds 8–647 voxels a frame, stage 4 another
9–1372, together taking coverage from 14.4–66.1% to 30.1–69.1%.

**`--hidden` defaults to `nearest` here**, and only here: a mode whose point is
to colour everything it can should not then paint half the scene flat. What is
left after all four stages is voxels no pixel ever reached — in room07, 4380 of
7888, most of them completion standing behind what the camera could see — and
`nearest` gives each one the colour of the nearest coloured voxel **of its own
class**. That sibling sits 2.0–7.3 voxels away (median, same 7 frames) and
held out costs 3.0–14.3 of 255; the readout keeps saying *hidden — colour
inferred*, because it is. `--hidden class` puts the flat colours back.

On load the file prints one line per rule that supplied any colour, so you can
see what each stage contributed:

```
2793/7888 voxels (35.4%) hold a measured point of their own
650/7888 voxels (8.2%) take a neighbour's measurement, within 0.75 voxel
42/7888 voxels (0.5%) are front-most on photo pixels the depth frame does not contradict
23/7888 voxels (0.3%) are front-most only on photo pixels whose depth measures something else
3508/7888 voxels (44.5%) end up with a photo colour from the frame
2783 more take the colour of the nearest coloured voxel of their own class (--hidden nearest), leaving 1597 on the flat class colour
```

That last line is room07's real limit: 1597 voxels belong to classes with no
coloured member anywhere in the frame, so there is nothing to borrow from and
they stay flat.

**`--colour` — which palette is drawn.** Default `photo`. Also the `colour` button.

| value | |
| --- | --- |
| **`photo`** *(default)* | the colour the camera saw, from `--visibility` |
| `class` | the flat per-class colours (`wall` cyan, `bed` orange, …), ignoring the photo |

**`--hidden` — what the voxels with no colour look like.** Default `class`,
except under `--visibility all`, where it is `nearest`.

| value | |
| --- | --- |
| **`class`** *(default)* | their class colour dimmed to 0.55, which reads as "unseen" |
| `nearest` | the colour of the nearest coloured voxel **of the same class**, at any distance. This borrows across the scene — a much bigger inference than `--visibility surface` — which is why it is the default only under `--visibility all`, the mode that exists to colour everything it can. The readout still calls these voxels *hidden* |

**Other flags:** `--reach` (region half-width for `--visibility surface` and
`all`), `--agree` (how far the depth may disagree before `all` demotes a pixel
to stage 4, default one voxel), `--strict-photo` (drop stage 4 entirely),
`--raw-axes`, `--flat`, `--no-cull`, `--no-highlight`, `--screenshot PATH`,
`--rgb` / `--bin` / `--meta` / `--camera-height` / `--yaw`.

---

## `reconstruct.py`

The measured depth surface on its own, photo-textured. It knows nothing about
the prediction beyond where to find the frame — the readout gives the cell and
the range the camera measured there, and stops.

### Buttons

| button | label cycles through | what it does |
| --- | --- | --- |
| `surface: …` | `photo` → `label` → `match` | how the measured surface is painted |
| `reframe` | — | back to the start view |

### Modes

**`--surface` — how the measured surface is painted.** Default `photo`. Also
the `surface` button.

| value | |
| --- | --- |
| **`photo`** *(default)* | the surface as the camera saw it |
| `label` | each measured point painted with the class the model gave **its** cell — the prediction's labels read on real geometry |
| `match` | **green** where the model filled the cell you measured, **red** where it left it empty, **grey** outside the 4.8 m volume (the model was never asked about those, so scoring them as misses would be unfair) |

### Mouse

| | |
| --- | --- |
| hover | `cell [30, 15, 20]   measured at 1.68 m` |
| Ctrl+click | prints it |

On load it also prints the coverage: of the grid cells the measured surface
occupies, how many the prediction fills, broken down by label.

**Other flags:** `--edge` (metres of range disagreement across a quad before it
counts as a depth discontinuity and is left open, default 0.12), `--stride`,
`--z-range MIN MAX`, `--shade` / `--smooth`, `--raw-axes`, `--screenshot`.

---

## `display_overlay.py`

Both layers in one window.

### Buttons

| button | label cycles through | what it does |
| --- | --- | --- |
| `layers: …` | `solid` → `measure` → `predict` | how the two layers stack — see below |
| `visibility: …` | `points` → `surface` → `zbuffer` → `all` | the voxel colouring rule from `display_solid`, recomputed live (~0.2 s a click) |
| `surface: …` | `photo` → `label` → `match` | the surface colouring from `reconstruct` |
| `voxels: …` | `photo` → `class` | the voxel palette |
| `hide surface` / `show surface` | toggle | blanks the measured surface outright |
| `hide voxels` / `show voxels` | toggle | blanks the prediction outright |
| `reframe` | — | back to the start view |

### The `layers` button in detail

This is the one that matters. Drawn as solids the cubes **swallow** the surface
running through them — a voxel's cube spans ±0.5 of the cell centre and the
measured surface passes through the middle of it — so *darkening* the
prediction achieves nothing: a dark cube hides the surface exactly as well as a
bright one. The cubes have to let light through.

| value | voxel alpha | surface brightness | what you see |
| --- | --- | --- | --- |
| **`solid`** *(default)* | 1.00 | 1.00 | the prediction only; it hides everything behind it |
| `measure` | 0.55 | 1.00 | see-through cubes over the full-brightness surface. Class colours still read — cyan wall, orange bed — with the photo texture visible underneath. **This is the analysis view** |
| `predict` | 0.55 | 0.35 | the same cubes with the measurement pushed back, so the prediction reads against it — the same comparison from the other side |

### Mouse

| | |
| --- | --- |
| hover | `cell [30, 15, 20]   measured at 1.68 m   model says objs (class 11)` |
| | or `… the model left this cell EMPTY`, or `measured surface outside the 4.8 m volume` |
| Ctrl+click | prints it |

Picking casts the cursor's ray at the actual triangles (one BVH per geometry,
only the visible ones), so the readout always describes something on screen.

**Other flags:** `--show surface|voxels|both` (what is visible at startup,
default `both`), plus everything `reconstruct.py` and `display_solid.py` take.

---

## `live_view.py`

The live window. Not a command of its own:

```bash
python -m inference.run_live --cfg ./cfgs/NYU/voxelSSC.yaml \
    --pretrained_path ./checkpoint/CleanerS_ckpt.pth \
    --source realsense --display --colour zbuffer

# or without a camera, from a recorded sweep of the room
python -m inference.run_live --cfg ./cfgs/NYU/voxelSSC.yaml \
    --pretrained_path ./checkpoint/CleanerS_ckpt.pth \
    --source bag:captures/room07/scan/scan.bag --display --colour all
```

Same solid cubes as `display_solid.py` — it imports that file's
`build_cube_mesh`, `photo_colours` and `fill_hidden` rather than reimplementing
them — but the geometry is **replaced in place** every frame instead of loaded
once, and the renderer is open3d's legacy `Visualizer` rather than the `gui`
toolkit, so every window call is made from `run_live`'s own loop.

The window opens **after** the model and the camera are ready, not before.
It is driven from `run_live`'s loop, so it only answers the desktop while
that loop is running; opening it first left it mute through a ~2 s model
load, the camera's 3 s warm-up and the first frame's cudnn autotune — 5.3 s
measured, against the ~5 s at which GNOME decides a window is hung and offers
to force quit it. That is why the dialog appeared *sometimes*: it depended on
how warm the caches were. It now answers 2 ms after it exists, and at worst
every 0.5 s thereafter.

**The window runs one frame behind, on purpose.** Painting a frame is numpy,
which releases the GIL, so it runs on a worker thread while the main thread
encodes and infers the *next* frame — the paint comes out of the frame budget
almost entirely (`96 of it hidden behind inference`, says the exit line). The
mesh and the camera panels are then drawn together from that same older
frame, so the two halves of the window never disagree. `--sync_display` turns
it off and draws each frame in its own iteration: no lag, and the painting
back on the clock. In `--colour class` the paint is free anyway, so the
overlap switches itself off.

What it buys, concretely: with the window open and the card cool, `--colour
all` now runs at the same fps as `--colour class` — 96 ms a frame of extra
painting for nothing. Before, you paid all of it.

### Colour

`--colour`, or `c` in the window to cycle. Per frame, on the dev box:

| | ms | |
| --- | ---: | --- |
| `class` | ~0 | the semantic colour. What the model *says* |
| `zbuffer` | 11-17 | the photo pixel each voxel is front-most on |
| `all` | ~100 | the three photo rules stacked, best evidence first |

These are `display_solid`'s own `--visibility` rules, and mean exactly what
they mean there. `all` additionally fills the voxels no rule reached with the
nearest visible sibling of the same class (`--hidden nearest`), because at that
coverage borrowing reads better than dimming; `zbuffer` misses too many voxels
for that to be anything but a smear, so those stay dimmed class colour.

`all` cost 424 ms before this window existed. It is ~100 ms now because its
distance query was rewritten as a scatter — see
[`points_in_cubes`](README.md#display_solidpy--solid-voxel-viewer). That is the
difference between 1.3 fps and 2.1 fps end to end.

### The camera panels

A second window, beside the 3D one, holding what the model was actually given:
the colour frame over the depth frame. `--no_panels` turns it off; it costs
about 15 ms a frame.

**It is resizable** — drag it to any size and the image scales into it. The
starting size is `--panel_width` (420), chosen so that 420 + the 3D window's
1280 comes to 1720 × 800: that fits the 1728 × 1080 desktop a 3456 × 2160
panel gives at 2× scaling. `--window WxH` resizes the 3D one to match
whatever screen you are on.

The depth ramp is `colorize_depth`'s rs-jet with per-frame histogram
equalisation — the same function `capture.py --preview` uses, so framing a
shot and watching the result look alike. Per-frame equalisation means the
colours are a **shape readout, not a measurement**: they are not comparable
between frames. The footer carries the numbers that are — frame name, the
fraction of depth pixels the sensor measured, the median range, and the rate.

Both panes show the frame **after** the FOV match, 640×480 with NYU's
intrinsics, because that is what the network sees. A thin prediction is
usually explained by the depth pane before it is explained by the voxels.

### Framing

`--zoom`, as a fraction of the window's height. open3d's own fit frames the
bounding *sphere*, so it sits about twice as far back as it needs to:

| `--zoom` | 1.0 | 1.7 | 2.0 |
| --- | ---: | ---: | ---: |
| a live frame | 31% | 58% | **65%** |
| an NYU frame | 44% | 86% | 99%, at the edges |

**2.0 is the default** because `--display` is for watching a camera, where the
room occupies much less of the 4.8 × 4.8 × 2.88 m volume than NYU's
room-canonical placement does. Replaying NYU frames, `--zoom 1.7` fits better.

The direction needs no correction: the default view looks down −Z with +Y up,
which after `to_view_index` is where the capture camera stood — the same
viewpoint `display_solid` opens with.

### Keys and mouse

| | |
| --- | --- |
| mouse | orbit, pan, zoom — the viewpoint survives every frame update |
| `c` | cycle `class` → `zbuffer` → `all`, repainting the frame on screen |
| `+` / `-` | step the zoom by 15%; `0` fits what has been built so far |
| `s` | screenshot into `--out_dir` |
| close the window | stops the run |

Every key but the zoom pair acts on the **press**, not on the key being
down. X11 repeats a held key and open3d hands on every repeat, so `m`
held for a moment used to flip the window between its two modes dozens
of times a second — each flip a full repaint, which is where the lag
that came with it was from — and even an ordinary press was long enough
to catch one repeat and toggle twice, so the mode looked unchanged.
`+` and `-` are deliberately left repeating, because holding them to go
on zooming is the point.

With `--accumulate` the same keys mean the room instead: `c` cycles
`photo` → `verdict` → `class`, `space` (or `f`) freezes, and `m` flips
between the room and the sensor's own reconstruction.

**`m` works in both modes, and means a slightly different thing in
each.** With `--accumulate` it is the whole sweep as one mesh — the TSDF
the tracker has been building to track against, extracted. Without it
there is no tracker and no sweep, so it is **this frame's own surface**:
the one depth image integrated into a volume of its own, marching-cubed,
and thrown away when the next frame arrives.

That per-frame surface is independent by construction, which is the right
answer in a mode that is not accumulating — no pose, nothing joined to
anything, so a camera that has moved cannot smear one frame into the next
and a bad frame costs exactly itself. It costs a measured 6–7 ms a frame
for about 78 k triangles, and is built only when asked: while `m` is
already showing, each new frame gets its own, and **when `m` is pressed,
one is built there and then for the frame on screen**. That second case
is not an optimisation detail — the moment anyone presses the key is
usually while the window is *frozen*, and then no frame is being
processed at all, so a surface built only in the loop would never
appear. A run that never presses the key pays nothing for it.
`--sensor_first` opens on it.

A surface belongs to **one frame**, and the window treats it that way:
it is cleared as soon as the next frame arrives with `m` off, and
pressing `m` always rebuilds rather than reusing what is lying around.
Both matter for the same sequence — freeze, `m` on, `m` off, play a
hundred frames to somewhere else, freeze, `m` on. Reusing the held mesh
there showed the scene the camera had long since left, with nothing on
screen to say so.

If the frame measured no surface whatever, the key says so and stays on
the prediction rather than blanking the window.

The comparison is the same one either way, and it is the point of the
key: the mesh is what the sensor **measured**, the cubes beside it are
what the model **completed**, and they are drawn at the same viewpoint in
the same coordinates so the only thing that changes is the geometry.

The handlers are verified by calling them directly — two `+` take the zoom to
2.64 and redraw at the larger size — but this machine is a Wayland session
with no way to synthesise a key press, so **the dispatch itself is untested**.
If the keys turn out dead, `--zoom` and `--colour` do the same jobs from the
command line and cannot fail.

### Pacing

A camera does not wait for us. librealsense hands back the newest frameset and
drops what arrived while the model was busy — which is right, because every
frame here is an independent completion and a stale one is worth nothing. The
window therefore shows the most recent frame the pipeline could give, about
one sensor period plus one inference behind reality.

A `.bag` is the opposite by default: it waits, so a replay processes every
recorded frame and two runs are comparable. `--realtime` makes the file behave
like the camera — recorded rate, frames dropped — which is how to see the live
pacing with no camera attached. On room07, recorded at 15 Hz:

```
default     808 of 808 frames processed
--realtime   30 of 159 processed, 129 skipped to keep pace
```

Either way the run prints which it did.

### The accumulating room

`--accumulate`. One room-sized 8 cm grid, in the world frame
`live_cam_pose` builds, filled as the camera moves instead of replaced.
This section is the reference; [DEMO.md](DEMO.md) is the short version
with the results and a running order for showing it to someone.

```bash
python -m inference.run_live --cfg ./cfgs/NYU/voxelSSC.yaml \
    --pretrained_path ./checkpoint/CleanerS_ckpt.pth \
    --source bag:captures/room07/scan/scan.bag \
    --accumulate --display --camera_height 1.0858
```

**What you are looking at, by default.** A room. Every voxel is drawn the
same size and wears the camera's own colour — the colour the pixels that
measured it reported, weighted towards the most recent look — or, where
the camera never saw it, the colour of the nearest measured voxel *of the
same class*, falling back to a dimmed class colour where a class has no
measured member anywhere. That is
`display_solid`'s `--visibility all` rule applied to a room instead of a
frame; `fill_hidden` is imported, not reimplemented, so it means exactly
what it means there.

**The colours answer one of three questions, and `c` cycles them.**
`--world_colour` picks which one a run starts in.

| mode | the question |
| --- | --- |
| **`photo`** *(default)* | what does this place look like? |
| `verdict` | did the model get it right? |
| `class` | what does the model think each thing is? |

Keeping them apart is the point. One picture answering all three at once
is unreadable — which is how this arrived at the default above.

**`verdict` in detail.** Every voxel keeps its photo colour except where
there is something to say about it:

| colour | means |
| --- | --- |
| **green** | the model filled this in while the sensor knew *nothing* here, and the sweep went on to **prove** it |
| **blue** | the sensor found it and the model never predicted it |
| **grey** | the model claims it and the sensor still has no evidence either way |
| **red** | the model says solid, the camera **saw straight through** that space |

Red asks a different question from the score, on purpose. The score below
counts the *completions* the sweep disproved. Red asks where the model put
matter in space the camera saw through: wider in that the voxel need not
have been a completion, since wrong in plain view is still wrong, and
narrower in that it drops the one-voxel skin along every measured surface.

That skin is not an error this comparison can resolve. The model's
surfaces come out about a voxel thicker than the measurement, so the cell
just in front of every wall reads free; at 8 cm, with ~1 voxel of
trajectory drift and a half-voxel hit margin, a disagreement that touches
a surface is below the resolution of the question. On room07 it is 2 325
of 2 805 such voxels, and painting it red turns a whole wall red and says
nothing. The 480 left over are matter floating in open space.

The other modes drop **all** 2 805, skin included — see above. Red is for
the question "did the model get it wrong"; `photo` is answering "what is
there", and there the skin is simply a wall drawn a voxel too thick.

The exclusion is **not** applied to the score. Dropping the model's near
misses from confirmed-versus-contradicted would simply flatter it.

**`--world_shrink`** says how big the voxels only the *model* claims are
drawn. `auto`, the default, decides per mode, and it has to:

- in **`photo`** the room is a *place*, so everything is drawn whole and
  it looks like one;
- in **`verdict`** it is a *comparison*, and whole cubes hide it. Faces
  between two occupied voxels are culled, so you only ever see the
  outermost shell — and that shell is the model's unverified claims
  wrapped around the real room. On room07 all 3 485 proved voxels are
  measured surfaces inside it and **not one of them is visible**. So there
  the model's unmeasured claims are drawn smaller and the room reads
  through the gaps.

A number forces one answer for every mode.

**Stopping half way.** `SPACE` (or `f`) freezes: nothing more goes into
the room and it holds exactly as it is, with the window still live so you
can orbit what the model had at that moment. Press again to carry on, or
close the window to end the run there — it writes the map as usual.

The wait sits *after* a frame rather than before one, so a **replay**
pulls nothing while it holds and carries on at the frame it stopped at
rather than skipping ahead.

A **live camera cannot be held like that**, and pretending it could cost
whole runs. Its frames arrive whether anything is reading them or not, and
the camera keeps moving in the hand of whoever is looking at the window —
so letting go handed the tracker a view that had *jumped*, and a jump is
what costs it the pose. While a live window is held its frames are
therefore taken, and **tracked, and nothing else**: nothing is integrated,
measured, predicted, drawn or recorded, so the room on screen is still the
one that was frozen while the pose keeps up with the camera. Measured by
holding a replay dressed up as a camera — room07, four seconds of
continued sweeping through the hold — **45 of 45 frames tracked** either
side of it and nothing dropped, where the same hold before this left the
tracker with 15 frames and LOST for the rest of the run.

That buys seconds, not minutes. Nothing is being integrated, so the model
the pose is tracked against stops growing, and a hold long enough to sweep
somewhere genuinely new runs out of it — 142 frames, 9.5 s of recorded
motion, on the same test. Then the pose is lost, and getting it back is
*Where the pose comes from*, further down this section.

**And `m` shows what the sensor alone had.** Freezing also extracts the
**sensor's own reconstruction** — a triangle mesh of everything the depth
has measured so far — and `m` flips the window between it and the room,
at the same viewpoint, so the only thing that changes is the geometry.

That is the comparison the mode exists for, and it reads without a legend:
the mesh **stops where the depth stopped**, so every hole in it is
somewhere the sensor could not see, and the room is what the model puts
there. Freeze after the first few frames, before the camera has moved,
and the mesh is one frustum with a room-shaped hole around it; the room
beside it already has walls and a floor. Carry on to the end of the sweep
and the mesh closes up, except where it never can.

In the first four seconds, look at the **geometry** and not the colours:
neither side has a usable one yet. The camera's white balance has not
converged, so the mesh comes out pink and the room, which refuses that
colour outright, is still falling back to class colours. Both are right
and both are ugly; `--sensor_first` opens on the mesh so the geometry is
what gets looked at first.

It costs nothing to build — it is the TSDF the tracker was already
integrating to track against, the same volume `fuse_scan` carves
`room.ply` out of, at 2 cm rather than 1 cm. Extracting it is **3–7 ms**
measured on room07 (57 k triangles at 10 frames, 138 k at 400), so while
`m` is showing it the mesh is rebuilt every frame and fills in as you
sweep; while the room is showing it is not rebuilt at all.

It is a thing to **look at**, not a measurement: no checks are run on it
and `reconstruction_GT/fuse_scan.py` remains what produces the committed
`room.ply`. Its colour is also integrated from frame 0, unlike the room's
— the tracker has to hand `integrate` a colour channel from the first
frame, so the camera's unconverged white balance is in there. Over a full
sweep it shows as a faint pink cast on whatever the camera saw only at
the start and never came back to.

**At the end of a run the window stays up**, frozen, showing that
reconstruction, so a replay finishes on the sensor's own answer rather
than vanishing. `m` flips back to the room, `c` recolours it, closing the
window finishes. `--no_hold` skips this for a scripted run nobody is
watching.

**The room does not keep every claim it is given.** Two rules throw some
away, and both exist because the input to the model is itself unsteady.

*The model has to be consistent before the room shows it.* A voxel one
frame glanced at and called a chair is a coincidence, not a prediction;
the depth the model reads wobbles frame to frame, so there are plenty of
those at the edge of every surface, and accumulated over a sweep they
become permanent spurs — a bed with spikes. So a voxel is claimed only
once **three** frames could see it and **60%** of them agreed
(`--pred_seen_min`, `--pred_agree`). Scoring room07's sweep both ways:

| | rule off | rule on |
| --- | ---: | ---: |
| voxels the model claims | 19 690 | **17 225** |
| of those, confirmed by the sweep | 3 526 | **3 485** |
| contradicted | 596 | **535** |
| claims standing in measured-free space | 3 309 | **2 805** |
| measured voxels the model never claimed | 261 | **322** |
| proved share | 85.5% | **86.7%** |

So 12% of the claimed matter was unstable and 1.2% of the proved went
with it. `missed` rising is the honest cost: some of what goes was a real
surface the model only ever claimed from one viewpoint.

*The sensor outranks the model about space it has measured.* Outside
`verdict`, a claim standing in space the camera measured as empty is not
drawn at all. Those modes are answering what is **there**, and leaving the
claim in puts matter in mid-air and thickens every wall by the voxel in
front of it. `verdict` keeps them — in red — because that mode is asking
how the model did, and the mistakes are the answer.

**The camera's first seconds do not get to say what anything looks like.**
A D4xx opens with its auto white balance unconverged: room07's first frame
is 7.4% saturated and renders a white wall pink, and the channel means do
not meet until frame 60 — 4.0 s in. Colour integrated then is wrong and
stays wrong, which showed as one wall still pink long after the rest of it
had gone white.

So the colour waits. `--colour_settle auto` (the default) watches the
frame's chromaticity and starts colouring when it stops moving — frame 70
on room07, 49 on room08 — or a number of frames, or `off`. **Geometry
never waits**: the depth stream has its own exposure, so every frame is
tracked, measured and predicted from frame 0.

**What a completion is, exactly.** A voxel the model calls occupied *where
the sensor had no information at all* — not one depth ray, ever. It is not
"where the measurement has not settled yet": a state needs two frames to
settle, so that weaker reading counts every surface the camera is looking
straight at on the first frame, and the model ends up taking credit for
predicting what it can plainly see. (That is not hypothetical. It is what
the first version of this did, and the display going almost entirely green
is how it was caught.)

**Most completions are never settled, and the readout says so.** Most of
what a completion model fills in — the inside of the sofa, the far side of
the wall — the sensor can never reach, however long you sweep. Those are
reported as `still untested` on their own line and are never folded into
the confirmed count.

**The fusion rules.**

| | |
| --- | --- |
| measured | each frame projects every voxel in its frustum into the depth image. Within half a voxel of the measured range → hit; in front of it → free; behind it, no depth, or outside the frame → unknown. Occupied once **two frames** have put a surface in it, free once two have measured past it. Hits beat passes |
| predicted | the frame's 60×36×60 is **gathered** into the world, not scattered out of it — at equal resolution a scatter aliases, and the holes read as model failures. A voxel is occupied once **three frames** could see it and **60%** of them said so, and takes the argmax of their votes |
| colour | accumulated as the measurement goes, not recomputed. `--colour`'s per-frame rules answer "which pixel describes this voxel *in this view*"; a room answers it by **blending** each view that saw it into what is there, half-and-half, so the latest look decides |

**Where the pose comes from.** A tracker in the loop
(`inference/tracker.py`), which is the tracking half of
`reconstruction_GT/fuse_scan.py` — frame-to-model, so error does not
compound frame over frame. How that works, rather than what it does:
[WORLD_BUILDING.md](WORLD_BUILDING.md), which is also where the 0.21 m
convergence basin behind every rule below is measured. It is *not* `scan/trajectory.txt`: that file
indexes poses by every frameset the bag reader yields after skipping 1 s,
while `BagSource` counts only framesets that survive `_process` and skips
nothing, so `live_000000` is not bag frame 0 and nothing reports the
mismatch — the room would just come out quietly smeared.

A frame the tracker is not sure of **contributes nothing at all**, neither
measurement nor prediction. A bad pose does not degrade the map, it
permanently corrupts it: geometry welded into the wrong place never washes
out. After a second of consecutive drops the readout says **TRACKING
LOST** and the map stops growing rather than smearing.

*What counts as unsure is measured against the run itself.* Over both
committed captures end to end — 1 871 frames — the ICP fitness was never
below **0.647**, median 0.90. Good tracking is emphatic, not marginal. A
discontinuity is not: resuming room07 at frame 200 after frame 59 reads
**0.15–0.23**, which `fuse_scan`'s 0.1 floor passes happily, and then
tracks on at 0.44–0.51 welding twenty frames of room a quarter of a metre
from where they belong. `fuse_scan` is right not to care — it reads a
file, where frame 200 never follows frame 59. So the floor here is **60%
of the median of the last 30 believed frames**, capped at 0.55, which also
carries over to a sensor or a scene that simply matches worse. On both
full sweeps it still drops nothing. `--track_floor` moves that fraction,
and the run's periodic line now carries the drop rate, so a camera that
is dropping frames it should not shows up while it is happening rather
than in the report at the end.

*One dropped frame is a hiccup, not a relocalisation.* The confirmation
below is asked for only once the pose has actually been declared LOST.
Asking for it after every single drop fed itself on a live camera: the
confirming frames are not integrated, so the model stopped keeping up
with the camera, so the next frame matched worse and dropped too. The
window logged `re-acquired ... after 1 that did not` over and over and
one frame in four reached the room. A frame or two dropped out of a good
run is covered by the jump gate on its own — the pose it is measured
against is a frame old.

*And being lost is no longer the end of the run.* Frame-to-model tracking
compares every frame with a raycast of the model from the last believed
pose. That is what stops drift compounding, and it is also why one jump
used to end everything: the reference never moved again, so every later
frame was compared with a viewpoint the camera had left, failed, and the
room never grew however the operator waved it about. Now **point the
camera back at what it was looking at** and it picks itself up — three
frames in a row, each matching the model at the floor above, each placing
the camera within one plausible step of the last. Nothing is integrated
until all three agree, because one lucky match welds the room into the
wrong place for good. Measured on room07: a 140-frame discontinuity is
dropped frame by frame, and when the view comes back the tracker
re-acquires within three frames, at a pose **4 mm** from where it stopped.

*The window stays alive while that happens.* It used to go grey and stop
answering keys the moment the pose was lost, because the loop skipped
straight to the next frame — no panels to aim the camera back with, no
`space`, no way to close it. The room is now drawn every frame whether or
not it grew, with the live depth and colour beside it and the reason in
the readout.

**On a live camera it is the same code and it has been run on one.**
`--source realsense` against an attached D455 (fw 5.17.0.10, USB 3.2,
1280x720 at 30 cropped to NYU's FOV), on this 4 GB card:

| | |
| --- | --- |
| rate | **2.05 fps** with the window, 2.41 without — the same as a bag |
| tracker | 80 of 80 frames tracked, none dropped; accelerometer 80 of 80 samples used |
| peak VRAM | **3 413 MB of 4 096** with the window up, and it plateaus rather than creeping — see [HARDWARE.md](HARDWARE.md#hardware-gates) |
| colour | settled at frame 13 on one open and 36 on another, on top of the 30 frames `RealSenseSource` already discards |
| reconstruction | 3 ms, 30 k triangles of one frustum — `m` and the end-of-run hold behave exactly as on a replay |

683 MB of headroom is not much. `--track_device cpu` takes the peak to
2 473 MB at the cost of 41 ms a frame, and is the thing to reach for if
something else wants the card.

It is also not enough for the **marching cubes**, which wants a scratch
structure on the same device: with the room up, a live run could not
allocate it for 463 active blocks (about 19 MB of TSDF — the headroom is
the problem, not the volume). So the mesh is rebuilt on the `--mesh_every`
cadence rather than every frame while `m` is showing it, and if the card
still refuses, the volume is copied to the host and meshed there — the
identical mesh, 16 ms instead of 3, and it cannot run out of VRAM. The run
says so once and the final report counts them.

**It needs a measured `--camera_height`.** The world is anchored on the
first frame and everything is fused into it, so a height revised later
would move the world out from under everything already in it.
`--camera_height auto` is refused, with that sentence.

**Framing.** `--world_extent` sizes the MAP — it only has to be big enough
for the camera to walk around inside. `--view_extent` sizes the OPENING
VIEW, and only has to show the room. They are separate because the camera
is fitted once to a constant proxy box and cannot be aimed from python
afterwards (see [What it does not do](#what-it-does-not-do)); fitting it to
the whole 16 m map left the room a smudge in the middle of the window. The
orbit pivot is the map's centre either way, so it never moves as the room
fills.

**What was making the room look far away was not the box, it was where
the box was.** The map is centred on the camera's *starting point*, and
the first frame's volume lies in FRONT of that — `vox_origin` is
`[-2.4, 0, -0.05]`, 4.8 m of depth starting at the camera plane. A proxy
centred on the map therefore put the whole scene in its far half, half
the size and off to one side. Measured on room07, the first five frames'
voxels sit **25.7 cells — 2.06 m — forward of the map centre**. The view
is now anchored half its own depth forward, which is where a frame's grid
sits relative to a frame's camera, and the same anchor is what `--zoom`
scales about. Rendered at 1280 × 800, that took a 120-frame sweep from a
patch in the middle of a black window to filling it.

`--view_extent` itself is a weak lever — 2.4 m and 4.8 m render within 5%
of each other, and 16 m only 30% smaller, because the legacy visualiser's
fit is dominated by the geometry rather than the proxy. It defaults to
**4.8 × 4.8 × 2.88 m, the single-frame volume**, because that is the box
a run without `--accumulate` is framed on and the honest value to start
from. `--zoom` is the strong one: at 1280 × 800 the first 120 frames of
room07 fill 10% of the window at zoom 1.0, 43% at 2.0 and 52% at 2.2.

**`0` fits what has been built so far**, which is how a sweep that has
outgrown its opening framing gets back in frame — room07's whole sweep
comes out at zoom 1.06 against the 2.38 its first 120 frames want. A run
does it for you when it finishes, so the last thing anyone looks at is
the whole room.

**What it costs**, on the dev box, against a 331 ms frame, measured over
room07's 808-frame sweep:

| | ms/frame |
| --- | ---: |
| tracking | 13 |
| measure + gather | 24 |
| room mesh | ~38, every `--mesh_every` frames (default 3) |

End to end that is 2.1 fps against 3.0 for the same run without the flag.

**What it writes**, at `--out_dir`: `world.npz` (the whole map, reopenable
without re-running the model), `world.ply` (occupied voxels in world
**metres**, so it drops straight into CloudCompare beside
`reconstruction_GT`'s `room.ply`), `reconstruction.ply` (the sensor's own
mesh, same frame, same metres — the two load on top of each other) and
`poses.npz`. `--record` also writes every drawn frame to `frames/`, for
the fill-in video.

**What room07's sweep gives**, all 808 framesets, none dropped:

| | |
| --- | ---: |
| voxels the sensor measured | 6 513 |
| voxels the model called occupied | 17 225 |
| of those, completions — filled in where the sensor knew nothing | 12 195 |
| completions the sweep went on to reach | 4 020 |
| **of those, confirmed** | **3 485 (86.7%)** |
| contradicted | 535 |
| still untested when the sweep ended | 8 175 |
| voxels the sensor found that the model never predicted | 322 |
| claims standing in space the camera measured as empty | 2 805, of which 480 are clear of any surface |

`photo` draws 14 742 voxels: the 6 513 measured, plus the 8 229 claims
that are not in measured-free space. `verdict` draws all 11 034 claims,
the measured ones whole and the claims at 0.45.

The 86.7% is the number worth quoting, and the 8 175 is the number that
keeps it honest: the sensor reached under a third of what the model
completed, and it never will reach most of the rest.

These move by a handful of voxels between runs of the same bag — AMP is
not bitwise reproducible, and a voxel sitting on the vote threshold can
fall either way. For a figure that must not move, run with `--amp off`.

**Checking it.** `inference/check_world_map.py` measures all of it against
things this repo already committed. On room07:

| check | result |
| --- | --- |
| `algebra` | 7 889 of 7 889 voxels placed, 0 missing, 0 extra — the composition collapses to the identity on the anchor frame |
| `tracker` | median 39 mm from the committed trajectory, p90 85 mm — inside the 66–102 mm `DRIFT_AND_ALIGNMENT.md` measured |
| `measured` | 89.8% of the map's measured voxels are in the fused sweep within one voxel (98.3% within two; 53.6% exact, which is the wrong question — both sides are 8 cm shells built from trajectories a voxel apart) |

### What it does not do

**No picking and no legend.** The legacy window has no mouse-position
callback, so naming the voxel under the cursor is not a matter of effort — it
is not available. Open the saved `.ply` in `display_solid.py` for that.

**No camera control — the framing is done to the geometry instead.** In open3d
0.17 `Visualizer.get_view_control()` returns a **copy**, so `set_zoom`,
`set_lookat` and `set_front` change an object the window does not own. They
report success and the readback through the same handle agrees; the rendered
pixels are byte-identical at zoom 0.3 and 0.9. `display_solid` aims its camera
with `widget3d.look_at`, which works — but that is the `gui` toolkit, and it
owns the main thread.

So the window fits its camera once to a proxy box covering the whole grid,
then every frame's mesh is **scaled about the grid centre** by `--zoom`. With
the camera fixed, scaling the content is exactly equivalent to moving the
camera, and it cannot be undone by the bug above. The pivot is the grid's
centre, so orbiting behaves the same whatever the model predicts.

---

## `display.py`

The original point-cloud viewer, left working and unchanged. Voxels as
screen-space points, so they never quite tile — that gap is what
`display_solid.py` exists to fix. `--point-size`, `--raw-axes`, `--legacy`.

---

## Why buttons and not keys

A `Window`'s `set_on_key` callback is cast back into C++, and open3d 0.17 will
not accept a `Widget.EventCallbackResult` there. The first key press raises

```
RuntimeError: Unable to cast Python instance to C++ type (compile in debug mode for details)
```

out of `Application.run()`, taking the window down with it. `Button`'s
`set_on_clicked` takes no argument and returns nothing, so there is nothing to
cast.

Worth knowing if you are tempted to add a shortcut back: **calling the handler
directly from Python does not reproduce the crash** — the cast only happens on
the way back into C++ — so a test that pokes `_on_key(...)` will pass while the
real window fails on the first keystroke.

`SceneWidget.set_on_mouse` is a *widget* callback and does take an
`EventCallbackResult`; that is why hovering and Ctrl+click still work.

**None of this applies to `live_view.py`**, which is the legacy
`VisualizerWithKeyCallback`, not the `gui` toolkit.
`register_key_callback(ord('C'), fn)` there takes a plain callable and the
return value is a bool, so there is nothing to cast and `c` and `s` are
ordinary keys. The trade is the other way round: that window has keys but no
widgets at all, hence no buttons, no legend and no readout bar.

---

## Two shared conventions

**Orientation.** All the viewers reflect the ply's depth axis
(`z -> pivot - z`) because the file's axes are left-handed and every renderer
expects the OpenGL convention; without it the scene renders inside-out, with
the eye behind the far wall. Left/right is already correct and is unchanged by
the reflection. `--raw-axes` turns it off to show the difference.

**Framing.** Every *file* viewer frames on the prediction's extent, so the
same frame is the same size in all of them and they can be compared directly.
`live_view.py` cannot (see above), and should not anyway: re-framing on the
occupied extent every frame would re-aim the camera each time the model changed
its mind. It takes open3d's fit once and then leaves the camera to you.
