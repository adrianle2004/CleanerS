# Viewers — every button and every mode

Four files show the same 60×36×60 prediction. They differ in what they draw and
what they draw it *against*.

| file | draws | use it to |
| --- | --- | --- |
| [`display.py`](#displaypy) | voxels as screen-space points | the original viewer; kept working, unchanged |
| [`display_solid.py`](#display_solidpy) | voxels as solid 8 cm cubes | look at the prediction |
| [`reconstruct.py`](#reconstructpy) | the measured depth surface, photo-textured | look at what the camera actually got |
| [`display_overlay.py`](#display_overlaypy) | both, cubes made see-through | compare the two |

All three of the newer ones use **buttons, not key bindings**. That is not a
style choice — see [Why buttons](#why-buttons-and-not-keys).

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

---

## Two shared conventions

**Orientation.** All the viewers reflect the ply's depth axis
(`z -> pivot - z`) because the file's axes are left-handed and every renderer
expects the OpenGL convention; without it the scene renders inside-out, with
the eye behind the far wall. Left/right is already correct and is unchanged by
the reflection. `--raw-axes` turns it off to show the difference.

**Framing.** Every viewer frames on the *prediction's* extent, so the same
frame is the same size in all of them and they can be compared directly.
