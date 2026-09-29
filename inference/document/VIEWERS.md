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

and the last field is one of three, corresponding exactly to the three outcomes
of `photo_colours`:

| readout | means |
| --- | --- |
| `measured inside this voxel` | depth points landed inside this cube; its colour is its own measurement |
| `measured just outside it (within --reach)` | only appears under `--visibility surface`; the colour came from a neighbouring cell |
| `hidden -- colour inferred` | no measurement near it; it is showing its class colour |

### Modes (flags, not buttons)

**`--visibility` — where a voxel's photo colour comes from.** Default `points`.

| value | rule | notes |
| --- | --- | --- |
| **`points`** *(default)* | the mean colour of the measured points **inside** the voxel (a Chebyshev half-width of 0.5, i.e. exactly the cube) | exact, parameter-free, and **no voxel is ever coloured from another's measurements**. Leaves flat class-coloured speckles where the camera's coverage has a ragged edge |
| `surface` | the same, with the region grown to `--reach` voxels (0.75 = 6 cm) | fills those speckles by taking a *neighbouring* cell's measurements. That is an inference; the readout labels it separately. Hold-out error 0.6–7.8 of 255 against 18.5–52.9 for two arbitrary voxels |
| `zbuffer` | the front-most voxel over each image pixel | the only rule that needs no depth frame — it is what this file did before `resolve_frame` returned one. Misses 863 of room01's 1615 measured voxels, because on a surface seen edge-on the nearest voxel wins every pixel |

**`--colour` — which palette is drawn.** Default `photo`. Also the `colour` button.

| value | |
| --- | --- |
| **`photo`** *(default)* | the colour the camera saw, from `--visibility` |
| `class` | the flat per-class colours (`wall` cyan, `bed` orange, …), ignoring the photo |

**`--hidden` — what the voxels with no colour look like.** Default `class`.

| value | |
| --- | --- |
| **`class`** *(default)* | their class colour dimmed to 0.55, which reads as "unseen" |
| `nearest` | the colour of the nearest coloured voxel **of the same class**, at any distance. This borrows across the scene — a much bigger inference than `--visibility surface` — which is why it is not the default |

**Other flags:** `--reach` (region half-width for `--visibility surface`),
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
| `visibility: …` | `points` → `surface` → `zbuffer` | the voxel colouring rule from `display_solid`, recomputed live (~0.2 s a click) |
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
