# VOXEL_TO_SURFACE_WORK.md — what was done to turn the voxels into a surface

[VOXEL_SURFACE_MESH.md](VOXEL_SURFACE_MESH.md) is how the layer **works**.
This page is the **work**: what was there before, what each step was for,
what it was measured at, what was tried and thrown away, and what is
still open. Read it if you want to know why the code looks like this
rather than what it does.

It landed in two commits —
[`e118620`](#step-1-a-mesh-at-all-e118620) (the mesh, pure numpy) and
[`adf1b1d`](#step-2-wire-it-in-and-the-layer-came-out-wrong-adf1b1d)
(wiring it in and giving it the sensor's colour) — plus
[`3282ae8`](#verification) to re-measure the two sweeps afterwards.

---

## Where it started

`m` was a **boolean**: `self.show_mesh`, toggling between the model's
8 cm cubes and the sensor's own reconstruction. Two states, nothing in
between.

The cubes are honest about the resolution and unreadable as a room — a
wall is a staircase and a bed is a pile of bricks. The sensor's mesh is
legible and has holes exactly where the model is supposed to be useful.
So the window could show you the thing with the holes or the thing you
could not read, and nothing that was both.

What was wanted was the **same occupancy** as one continuous surface —
and the hard constraint was that it must not become a way of smuggling
the model's guesses in as measurements. The completion and the
measurement had to stay separable by **colour**, not by geometry.

---

## Step 1: a mesh at all (`e118620`)

411 lines of `inference/voxel_surface.py`, 490 lines of tests, and
**nothing in `live_view.py`**. The placement rule is the whole reason the
layer is worth anything, so it was built and pinned on its own first.

### The one real decision: Surface Nets, not marching cubes

Marching cubes is the obvious reach and it buys nothing here, because of
what our field is. It is **binary** — a voxel is occupied or it is not —
so interpolating to the 0.5 iso-level puts every marching-cubes vertex
exactly at an edge midpoint. It staircases too, just with a different
triangulation. In both cases the smoothness comes from the Laplacian
pass afterwards, not from the isosurface rule, so there was no
smoothness to be had by reaching for the bigger algorithm.

What Surface Nets gives that marching cubes does not is **exactly one
vertex per cell**, and that is what makes `owner` exact: every vertex
names the voxel whose surface it sits on, so it carries that voxel's
colour with no resampling step that has to guess. Which is the whole
constraint above — the measured/completed distinction survives being
meshed because it travels with the vertex.

It also needs no 256-case table and no new dependency. There is no
scikit-image in this repo's py3.7 environment.

### The property it had to have, and the test that holds it

`display_solid` draws a voxel's cube spanning index ±0.5, so a slab
solid up to index `k` has its face at `k + 0.5`. The mesh had to land
there **exactly**, or pressing `m` would move the room instead of
re-describing it. Half a voxel is 4 cm — visibly a mesh floating off its
own floor.

`test_a_flat_face_lands_on_the_cube_face` is that sentence as an
assertion. Several of the other 44 exist for the same reason: that the
normals point away from the matter, that `owner` always names an
**occupied** voxel, that a flat face stays flat under smoothing.

### Built with no renderer on purpose

Pure numpy, no open3d, no torch, no file paths — so the rule is testable
on a 5×5×5 grid with none of that installed. The piece that needs a
renderer stayed in `live_view.py`. That is why there are 45 tests on the
thing most worth pinning.

### What else went in, because it is the same arithmetic

| | what it does |
| --- | --- |
| `pull_to_surface` | steps `tsdf > 0` vertices back along the gradient onto the zero crossing. **Only the front moves**: behind a surface, and out where the TSDF saturates, there is nothing to shrink onto and moving anyway would be inventing a measurement |
| `subdivide` | one midpoint split, with `parents` so any per-vertex array carries over |
| `nearest_colour` | each point's colour from its nearest reference, under a limit that may be one per point |
| `voxelise` | points onto a grid, averaging values per voxel |
| `smooth` | Laplacian. A plane is its own neighbourhood average, so flat faces do not move and the registration above survives it |

`pull_to_surface` was measured on room07's first frame: 2 390 of 10 741
vertices moved (22.3%), 5.3 cm mean for the ones that moved, and their
distance to the sensor's own mesh fell from **6.9 cm to 2.3 cm median**
with 95.8% of them closer.

---

## Step 2: wire it in, and the layer came out wrong (`adf1b1d`)

`m` became a three-way cycle — `LAYERS = ('voxels', 'surface', 'sensor')`
— with the middle layer meshed from the model's voxels and coloured from
them too.

Beside the sensor's own mesh it read as a **dark, soft-focus copy of the
same room**. Two independent causes, both measured on `room07_world`
rather than guessed at.

### Cause 1: the floor had never been photographed

**40.8% of surface vertices wore `fill_hidden`'s dimmed class tint at
luma 68 — and every one of them was the same class: floor.**

The sweep was looking at the walls. So no floor voxel had camera colour
of its own, and no floor voxel had a *photographed sibling of its own
class* either — which is the only borrow `fill_hidden` makes. Both of
its sources were empty, so the whole floor fell through to the palette
tint at 0.55, which reads as unseen and is painted green.

The fix: a voxel in that state now takes the nearest sensor-mesh colour
at **any** range. The two-voxel reach stays for voxels that do have a
colour — it is there to stop them being handed a distant wall's, and a
palette tint has nothing to protect. It is the same borrow `fill_hidden`
already makes across a class, from a better source.

That needed the provenance to travel, so `_world_colours` now returns
`(colours, photographed)` — true for the voxel's own colour and for a
sibling's, false for the tint, which is the one case where the colour is
not a photograph of anything.

### Cause 2: the colour was sampled at 8 cm and then blurred

The surface has one vertex per 8 cm cell. The sensor's mesh carries the
TSDF's 2 cm texture. So the lookup could only be asked once per cell,
throwing three quarters of the texture away — and then the Laplacian
pass, which ran **after** the lookup, blurred what came back.

The fix: split once before the lookup, and smooth nothing after it.
**15 691 vertices → 63 318**, against the sensor mesh's own 62 090, so
the two end up at the same density. The split moves nothing — midpoints
lie on the triangle — it only gives the mesh somewhere to put the other
three quarters of its colour.

### What it bought

| | |
| --- | --- |
| take the mesh's colour within reach | **49.2%** |
| take it from further off | **40.5%** |
| keep their own | **10.3%** |
| median brightness | **75 → 115**, against the sensor layer's own 124 |

The window frame, the pattern on the bed and the flowers on the sill are
legible where they were washed out. Brightness is not accuracy; that
last sentence is what actually settled it.

---

## Step 3: the same rule in both modes

The single-frame layer had the same job and could have had its own copy
of the rule. It does not: `_recolour_from(points, to_world, vox)` is the
rule itself over one mesh, and `_sensor_recolour` (the room, against the
sweep's mesh) and `_frame_recolour` (one frame, against that frame's own)
both call it. They differ only in which mesh they ask and how their grid
reaches metres. **Two copies would drift, and `m` would change the room's
appearance in one mode and not the other** — which is exactly the thing
the layer exists to avoid.

Two consequences had to be handled:

- The frame layer needs a mesh to ask, so `run_live` builds one whenever
  `m` lands on **either** of the two layers that want it, not only on the
  sensor layer; and `_cycle_layer` builds one on the key press, because
  the window is usually **frozen** when anyone presses `m` and no frame
  is being processed at all. Failing to build one **costs the colour and
  never the layer** — the middle layer is meshed from the model's own
  voxels and always has something to show.
- The grids differ, and this is the part that goes quietly wrong. The
  room's is in world axis order. The frame's is indexed `(x, UP, depth)`
  while its metres are `(x, depth, UP)`, so two axes change places on the
  way out: `v[:, [0, 2, 1]]`. A wrong transform does not crash — it
  returns a colour from the wrong part of the room.
  `test_the_axis_swap_is_not_optional` asserts the wrong order lands
  more than two voxels off, and a second test pins the half-voxel.

Measured on the frame layer with `--colour all`: **0.0% dimmed class
tint** — 45.7% photographed, 54.2% borrowed. (Under the `zbuffer`
default 76.7% are dimmed, which is that colour rule's own doing and
documented with it.)

### One thing deliberately not shared

The TSDF shrink is the **frame layer only**. It needs one frame's field,
from the camera that measured it; the accumulated room has no such
thing, and the sweep's own mesh is already the measurement.
`_build_frame_surface` passes a `pull`, `_build_world_surface` does not.

---

## Step 4: two bugs found on the way

**`class` mode was taking the sensor's colour.** The gate was
`world_colour != 'verdict'`, so `verdict` was protected and `class` was
not — the photograph was being smeared over the one view whose whole
point is which class the model called each voxel. It had been wrong
since before this work and nothing had noticed. Now `!= 'photo'`: off
wherever the colour **is** the answer.

**The run report was claiming the room build was free.** It gated on
`_overlapping()`, which is a question about *configuration*, and under
`--accumulate` the answer is always yes and always irrelevant — the room
is built on the loop, `update()` is never called, so `wait_ms` stays 0
and the entire room build was reported as hidden behind inference. It
now counts the frames actually painted on the pool (`self.overlapped`).
Unrelated to the surface, in the same file, and it is the line anyone
reads to find out what the room costs.

---

## Step 5: paying for it

The first working version was nearly twice as expensive as what shipped.
Three passes, each measured:

| pass | what | before | after |
| --- | --- | --- | --- |
| cap the query | ask the tree for the **largest** limit anyone wanted, so a frame — where every voxel has a colour of its own — prunes instead of searching the whole mesh | 108.5 ms | 49.3 ms |
| key edges as one int64 | `e[:,0] * len(verts) + e[:,1]` instead of `np.unique` over a two-column array, which sorts lexicographically row by row | 45.4 ms | 8.5 ms |
| shrink before the split | a quarter as much work, **and** the right order: the colour is then sampled where the surface ends up rather than a voxel proud of it | 21.8 ms | 6.4 ms |

Result:

| | before | after |
| --- | --- | --- |
| frame `mesh+draw` | 33 ms | **153 ms** |
| room mesh, full room07 sweep | 235 ms | **542 ms** |

Final single-frame layers: cubes 33 ms, surface 153 ms, sensor 36 ms.

**During a live sweep none of it runs unless the layer was asked for.**
That was checked rather than assumed: four 400-frame runs put measured
fps unchanged, and the spread that did show up followed running order,
not which layer was on screen (`infer` moved with it).

---

## What was tried and thrown away

| | why not |
| --- | --- |
| marching cubes | binary field — same midpoints, same staircase, bigger table, and a dependency this environment lacks |
| `np.add.at` in `smooth` | unbuffered slow path: **40 ms of a 120 ms room**, more than the surface extraction itself. Three `bincount`s, one per component, are the same sum |
| `np.unique` over the edge array | lexicographic row-by-row sort was most of the cost of splitting a 30 k-triangle surface |
| smoothing after the colour lookup | it is what made the layer soft-focus in the first place (Cause 2) |
| caching the frame's KD-tree on `id()` | the room's mesh is stable and safe to key that way; a frame's is a new object every frame, and an `id()` an old one has released can be handed straight back to a new one |
| borrowing colour from the nearest voxel of **any** class | smears the wall onto the bed standing in front of it. Same-class only, which is why the floor had nothing to borrow and needed Step 2's unlimited reach instead |
| committing the two regenerated sweeps | regenerable from the bag and the checkpoint, and neither git nor HF is the right half for them. `outputs/*_world/` is ignored |

---

## Verification

The surface layer is a **visualisation**. It changes nothing the model
predicts, so the metrics are a control here, not a result — if they had
moved, something would be wrong. Both sweeps were regenerated by hand
and re-measured in `3282ae8`:

| | room07 | room08 |
| --- | --- | --- |
| `check_world_map` algebra | 7 889 / 7 889 | 5 884 / 5 884 |
| tracker median / p90 / max | 38.5 / 85.1 / 142.2 mm | 52.5 / 82.4 / 120.8 mm |
| `evaluate_gt.py` SSC / SC | 39.5 / 72.4 | 35.4 / 59.6 |

`evaluate_gt.py` reproduced both exactly, and the prediction inputs were
untouched, so the per-capture `gt/EVALUATION.md` files were correctly
**not** regenerated. DEMO.md's produced-table was updated to the exact
figures from the new sweeps. Hugging Face needed no push — nothing it
holds (`scan.bag`, `room*.ply`, the checkpoint) changed.

The test suite: **171 tests**, all passing. `tests/test_voxel_surface.py`
has 45; `tests/test_live_view.py` went from 19 to 52.

---

## Still open

- **The two window holes that started this were never reproduced.** The
  full room07 sweep's surface layer shows the window filled. Whatever
  was on screen at the time, nothing since has reproduced it, and it is
  recorded here rather than quietly dropped.
- **The frame surface is softer than the sensor layer until the sensor
  layer has been visited.** Before a mesh exists for the frame there is
  nothing to take the colour from, so the layer falls back to the model's
  own 8 cm colour. Inherent to the ordering, not a defect — and it costs
  the colour, never the layer.
- `smooth` is still the largest single cost inside the room build. The
  `np.add.at` removal took the obvious 40 ms; what is left is the passes
  themselves.

---

## Where the pieces are

| | |
| --- | --- |
| `voxel_surface.surface_nets` | occupancy → `(verts, tris, owner)` |
| `voxel_surface.smooth` / `subdivide` / `pull_to_surface` / `nearest_colour` / `voxelise` | the arithmetic, no renderer |
| `live_view._surface_mesh` | the pipeline: colour out through `owner`, smooth, pull, split, recolour |
| `live_view._build_world_surface` / `_build_frame_surface` | the room's, and one frame's |
| `live_view._world_colours` | one colour per voxel, and which are photographs |
| `live_view._recolour_from` | the colour rule, shared |
| `live_view._sensor_recolour` / `_frame_recolour` | its two callers, and their two transforms |
| `live_view._sensor_colours` | the voxels only the sensor's mesh covers |
| `tests/test_voxel_surface.py`, `tests/test_live_view.py` | 45 and 52 |

Flags, defaults and the layer's behaviour at the keyboard:
[VOXEL_SURFACE_MESH.md](VOXEL_SURFACE_MESH.md#13-flags) and
[VIEWERS.md](VIEWERS.md#live_viewpy).
