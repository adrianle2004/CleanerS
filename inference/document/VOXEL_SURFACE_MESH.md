# VOXEL_SURFACE_MESH.md — the occupancy grid as one surface

`m`'s middle layer. The room is held as 8 cm voxels and drawn as cubes,
which is honest about the resolution and unreadable as a room: a wall is a
staircase and a bed is a pile of bricks. This page is how the same
occupancy becomes one continuous surface that reads as furniture —
Surface Nets in pure numpy, a TSDF shrink onto the measurement, and a
colour lookup against the sensor's own mesh — and, at every step, how it
does that **without pretending the model's guesses were measured**.

The code is [`inference/voxel_surface.py`](../voxel_surface.py) (the
placement rule, pure numpy, no renderer) and `live_view.py`'s
`_surface_mesh`, `_build_world_surface`, `_build_frame_surface` and
`_recolour_from` (the part that needs one). For the keys and what the
layer means to someone looking at the window, see
[VIEWERS.md](VIEWERS.md#live_viewpy).

## 1. What the layer has to do

The input is a semantic occupancy grid: a voxel is occupied or not, and an
occupied one carries a predicted class and, often, a colour. The output
has to be

- geometrically continuous rather than a pile of cubes;
- registered with the cubes, so pressing `m` **re-describes** the room
  instead of moving it;
- shrunk onto the measured depth wherever a measurement exists to shrink
  onto, and left alone where none does;
- coloured from real sensor observations wherever there are any; and
- explicit about which of those a colour came from, so the
  measured/completed distinction the room is read by survives being
  meshed.

The order of operations in `_surface_mesh` is:

1. crop the occupancy to its occupied bounds and pad it with one empty
   layer;
2. extract the mesh with Surface Nets, which also yields `owner`;
3. read one colour per voxel out per vertex through `owner`;
4. smooth the **colour and the geometry together**, same function, same
   connectivity, same number of passes;
5. shrink onto the TSDF (`pull`) — frame layer only, see §4;
6. subdivide every triangle at its edge midpoints;
7. look the colour up against the sensor's mesh.

Three of those orderings are load-bearing and are each argued for below:
the shrink is after the smoothing (§4.5), the colour smoothing is before
the split (§6), and nothing is smoothed after the lookup (§7).

---

## 2. Surface Nets: occupancy to a mesh

### 2.1 Input and output

`surface_nets(occ)` takes a boolean grid and returns

- `verts` — `(V, 3)` float, **in the grid's own index coordinates: a
  voxel centre is at its integer index, so a face lands on x.5**;
- `tris` — `(T, 3)` int, wound counter-clockwise seen from outside the
  matter, so back-face culling and lighting agree;
- `owner` — `(V, 3)` int, the occupied voxel each vertex belongs to.

### 2.2 One vertex per straddling cell

A grid *cell* is the gap between adjacent samples, so its eight corners
are eight voxels. For every cell whose corners are not all in the same
state:

1. find the cell edges whose two endpoints disagree;
2. take each such edge's midpoint;
3. average them — that mean is the cell's one vertex.

### 2.3 Connecting them

One quad per lattice **edge** whose two voxels disagree, joining the four
cells around it, split into two triangles.

`_RING` walks those four cells counter-clockwise as seen from `+axis`.
The code keeps that order when the matter is on the **negative** side of
the axis (`solid_near`), because then the outward normal points along
`+axis`; when the matter is on the positive side the ring is reversed.
Cells that fall outside the padded grid are dropped rather than wrapped.

### 2.4 Why not marching cubes

The field is **binary**, so interpolating to the 0.5 level puts every
marching-cubes vertex exactly at an edge midpoint. It staircases too,
just with a different triangulation; in both cases the smoothness comes
from `smooth` (§3), not from the isosurface rule. So marching cubes buys
nothing here, and it costs a 256-case table and a scikit-image
dependency this repo's py3.7 environment does not have.

What Surface Nets buys is **exactly one vertex per cell**, which is what
makes `owner` exact — see §2.6.

### 2.5 The property the whole layer rests on

A flat face comes out exactly where the cube's face is. `display_solid`
draws a voxel's cube spanning index ±0.5, so a slab solid up to index `k`
produces a surface at `k + 0.5`. Anything else and the two layers
disagree by 4 cm, which looks like a mesh floating off its own floor.
This is why §9 is not a footnote.

### 2.6 `owner`

For each vertex, `owner` is the **nearest occupied corner** of its cell —
not an arbitrary corner. That guarantees the named voxel is occupied, so
it has a class and a colour; a vertex that named empty space would paint
the surface with a voxel nobody predicted. Index any per-voxel array
with `owner` and the surface wears that voxel's value, with no
resampling step that has to guess.

### 2.7 Padding and cropping

The grid is cropped to its occupied bounding box first: the room is a
16 m box because the camera has to walk around in it, the furniture is a
fraction of that, and the cell loop costs the **volume**. On room07 that
is 200×200×38 cropped to about 110×82×36.

The crop is then padded with one empty layer on every side, so occupied
cells at the crop boundary get an explicit empty exterior and the
surface closes. A room whose floor runs to the edge of the map gets an
underside instead of a hole you can see in through.

### 2.8 The known cost: isolated voxels shrink

One voxel alone becomes a cube about a third of its size — roughly 2.7 cm
of an 8 cm voxel — because the mean of its three crossing midpoints sits
inside it. That is the right trade: a lone voxel *is* a speck of
evidence and this draws it as one, while surfaces, where the voxels have
neighbours, land on the cube faces exactly.

---

## 3. Laplacian smoothing

`smooth(verts, tris, passes=2, rate=0.5)` — each vertex steps half the
way towards the mean of its neighbours, twice.

A plane is its own neighbourhood average, so **a flat face stays in its
plane**: the registration with the cubes (§2.5) survives smoothing, and
only corners and specks move. Vertices do drift *along* a flat face,
which is invisible, and that is why the mean is left unweighted — the
quads are split on one diagonal so the stencil is not symmetric, and
correcting for it would cost a pass over the mesh to buy nothing.

The accumulation is three `np.bincount` calls, one per component, rather
than `np.add.at` on the `(V, 3)` array. Same sum: `np.add.at` takes an
unbuffered slow path that cost **40 ms of a 120 ms room**, more than the
surface extraction itself.

---

## 4. Shrinking the surface onto the measurement

### 4.1 Why

The completion comes out about a voxel **proud** of the measured surface.
The frame's TSDF is the one thing in the pipeline that already knows by
how much, per voxel, without being told.

The convention (`tsdf_torch.py`) is `diff = dm - zc` — measured depth
minus the candidate's depth — so **positive means in front of the
measurement**. The magnitude is a distance over `VOX_MARGIN` (0.24 m)
and a voxel is `VOX_UNIT_LOW` (0.08 m), so

    TSDF_VOXELS = 0.24 / 0.08 = 3.0

converts a TSDF value to an offset in voxels.

### 4.2 Sampling it at a vertex

Vertices sit at fractional indices, so the field is read with
`map_coordinates(order=1)` — trilinear — and its gradient the same way,
component by component over `np.gradient`.

### 4.3 Only the front moves

1. sample `t`; if `t <= 0`, leave the vertex alone;
2. take the local gradient and normalise it; if its norm is ~0, leave the
   vertex alone;
3. step back along that direction by `clip(t * 3, 0, max_step)` voxels,
   `max_step` = 1.5 (12 cm).

Behind a surface, and out where nothing was measured, there is nothing to
shrink onto and moving a vertex would be **inventing** a measurement.
Out in open space the TSDF saturates at a constant 1 and its gradient is
noise — `max_step` is what makes that safe, together with the
zero-gradient guard.

### 4.4 Measured effect

On the first room07 frame: 2 390 of 10 741 vertices moved (22.3%, so
77.7% unchanged), mean movement 5.3 cm for the ones that moved, median
distance to the measured surface 6.9 cm → 2.3 cm, and 95.8% **of the
moved vertices** ended closer. Leaving most of the mesh alone is the
point, not a shortfall.

### 4.5 It runs AFTER the smoothing, and BEFORE the split

After the smoothing, because smoothing would otherwise average the
shrunk vertices back out towards the lattice they came from. Before the
split, because the shrink is then done once per cell instead of four
times, and because the colour is then sampled where the surface ends up
rather than a voxel proud of it.

### 4.6 **This is the frame layer only**

`_build_frame_surface` passes a `pull`; `_build_world_surface` does not.
The shrink needs *one* frame's TSDF, taken from the camera that measured
it — the accumulated room has no such field, and the sweep's own mesh
(the third layer) is already the measurement. So §4 describes the
single-frame path. Everything else on this page applies to both.

It is a single-frame correction, not temporal alignment.

---

## 5. Where a voxel's colour comes from

`_world_colours` returns `(colours, photographed)`. The second value is
the whole reason the rest of this page works: it says which colours are
photographs of something and which are not.

| source | `photographed` | what it is |
| --- | --- | --- |
| the voxel's own | yes | the mean of the camera pixels that measured it, accumulated over frames |
| a same-class sibling's | yes | `fill_hidden`'s `nearest`: the closest coloured voxel **of the same class**, at any range |
| the sensor mesh's own | yes | `_sensor_colours`, §5.4 |
| a dimmed class tint | **no** | the class palette at 0.55, which reads as unseen |

**Own colour** is the strongest: actual observation, accumulated.

**A sibling's** is restricted to the same class on purpose — nearest
overall would smear the wall onto the bed standing in front of it. It is
still an inference, but a disciplined one, and it is the same borrow §8
later generalises.

**The dimmed tint** is where a class has no coloured member anywhere in
the room. It is useful for seeing what the model *called* a thing and it
is not a photograph of anything, which is why it is false in
`photographed` and why §8 is allowed to overrule it from any distance.

### 5.4 Mesh-only voxels

`_sensor_colours` handles the voxels that are in the room because the
**sensor's** mesh passes through them, not because the room's hit/pass
vote ever called them a surface. Nothing ever coloured those. Left to
the fallback they would take a class colour, and for the ones the model
never claimed either, that class is `empty` — which paints a window pane
bright teal. They take the mesh's own colour instead, which is as
photographed as it gets, and they count as lit.

---

## 6. Smooth the colour with the geometry

`owner` gives each vertex exactly one voxel's colour, so without this a
smooth surface still wears a blocky 8 cm colour field and every lone
voxel is outlined in its own colour, which made the model's lumps shout.
The colour is therefore smoothed by the same function over the same
connectivity for the same number of passes as the geometry.

It happens **before** the split and the sensor lookup. Smoothing after
the lookup is exactly the bug §7 exists to fix.

---

## 7. Subdivide, so the colour can be asked more than once per cell

The surface has one vertex per 8 cm cell. The sensor's mesh carries the
TSDF's 2 cm texture. A colour sampled off it once per cell throws three
quarters of it away — and then the Laplacian pass, which used to run
*after* the lookup, blurred what came back. The layer arrived as a
soft-focus copy of the room beside it: the window frame, the pattern on
the bed and the flowers on the sill all washed out of it.

`subdivide(verts, tris)` splits every triangle at its edge midpoints —
one triangle becomes four — and returns `(verts, tris, parents)`.
Midpoints lie on the triangle, so **the surface does not move**: only
where it can be asked.

- Each edge is keyed **once**, however many triangles share it;
  splitting a shared edge twice would tear the surface along it.
- The key is one int64, `e[:,0] * len(verts) + e[:,1]`, rather than
  `np.unique` over a two-column array — that sorts lexicographically,
  row by row, and was most of the cost of splitting a 30 k-triangle
  surface (45.4 ms → 8.5 ms).
- `parents` is `(M, 2)`: the two vertices each one sits between, **an
  original naming itself twice**. That is what lets the caller carry
  anything it holds per vertex without `subdivide` knowing what any of it
  means — `arr[parents].mean(axis=1)` for a colour, `.all(axis=1)` for a
  flag, and the originals come through unchanged either way.
- Flags propagate conservatively: a midpoint counts as unphotographed
  only if **both** its parents are.

Room07: 15 691 vertices → 63 318, against the sensor mesh's own 62 090 —
the two are then at the same density.

---

## 8. Take the colour from the sensor's mesh

### 8.1 The lookup

`nearest_colour(points, ref_points, ref_colours, max_dist, base, tree)`
builds a `cKDTree` over the sensor mesh's vertices (or reuses one) and
gives each surface vertex the colour of its nearest sensor vertex. A
point with nothing inside its limit keeps `base` and is false in the
returned `found`.

This is what makes the two layers agree about what the room **looks
like**, not just where it is. They are coloured from different places —
the room blends one colour per 8 cm voxel and falls back to a class tint
where the camera never saw; the mesh carries the TSDF's 2 cm texture — so
without it, flipping between them changes the room's appearance as well
as its geometry.

### 8.2 The limit depends on provenance

| the voxel | limit | why |
| --- | --- | --- |
| photographed (own colour or a sibling's) | `SENSOR_COLOUR_REACH` = **2.0 voxels** (16 cm) | it has real evidence; the cap stops it being handed some wall's colour from across the room |
| never photographed | **`np.inf` — no limit at all** | its only other answer is a dimmed class tint, and a palette green is not what a floor looks like |

In code that is one line — `limit = np.where(unlit, np.inf, reach)` in
`_recolour_from`, with `max_dist` allowed to be one number or one per
point.

The unlimited branch is not a liberty taken lightly; it is what the
measurement forced. **40.8% of room07's surface vertices wore the dimmed
tint at luma 68, and all of them were one class: floor.** The sweep was
looking at the walls, so no floor voxel had camera colour, and no floor
voxel had a photographed sibling of its own class either — which is the
only borrow `fill_hidden` makes. It is the same borrow, from a better
source, across a longer distance. Nothing else moves: a voxel with a
colour of its own keeps the cap.

### 8.3 Query cap and tree cache

The query is capped at `float(np.max(max_dist))` — the largest limit
anyone asked for. That is correct for a per-point limit and much
cheaper when none of them is infinite: a frame, where every voxel has a
photographed colour of its own, then **prunes** instead of searching the
whole mesh (108.5 ms → 49.3 ms). Uncapped is the expensive case — 63 k
vertices against 62 k is 390 ms on one core, 48 ms on twenty, and this
runs between two frames. Hence `workers=-1`, with a `TypeError` fallback
for scipy older than 1.6.

The tree is cached for the **sweep's** mesh, keyed on `id(self.recon)`:
the room is recoloured every `mesh_every` frames against the same 62 k
vertices. The frame path deliberately does not come here — its mesh is a
different one every frame, so there is nothing to cache, and an `id()` an
old mesh has released could be handed straight back to a new one.

---

## 9. Coordinates — the part that goes quietly wrong

A wrong transform still returns a colour. It just comes from the wrong
physical surface, which shows up as smearing around wall corners and
object boundaries and nowhere else.

### 9.1 Indices to metres

`surface_nets` puts a voxel **centre at its integer index** (§2.1), so a
surface vertex converts to metres exactly the way a voxel centre does:

    metres = origin + (v + 0.5) * vox

The half-voxel is **mandatory**, not a double correction to be avoided.
Both call sites carry it — `_sensor_recolour` and `_frame_recolour` —
and without it every lookup lands 4 cm away.

### 9.2 Axis order

The room's grid is in world axis order, so there is nothing to permute.
The **frame's** grid is indexed `(x, UP, depth)` while its metres are
`(x, depth, UP)` — the swap `view_points` inverts — so two axes change
places on the way out:

    loader.vox_origin + (v[:, [0, 2, 1]] + 0.5) * VOX_UNIT_LOW

Two tests pin this, one asserting that the wrong order lands more than
two voxels off.

Everything has to agree: axis order, origin, voxel size, centre-versus-
boundary, and the half-voxel. One of them differing is not a crash.

---

## 10. Where the lookup is off on purpose

Sensor colour is for a view whose question is *what does this look like*.
Where the colour **is** the answer, painting a photograph over it erases
the question the mode exists to ask:

- the room in `verdict` — green for a completion the sweep proved, red
  for one it disproved;
- the room in `class` — which class the model called each voxel;
- a frame under `--colour class`.

So the gate is `world_colour != 'photo'`, not `!= 'verdict'`. That
distinction was a real bug: `class` was taking the sensor's colour within
two voxels, contradicting the rule already written for `verdict`. The
frame's gate is `self.colour == 'class'`.

---

## 11. Both modes, one rule

`_recolour_from(points, to_world, vox)` is the colour rule itself, over
one mesh. `_sensor_recolour` (the room, against the sweep's mesh) and
`_frame_recolour` (one frame, against that frame's own) both call it, and
differ only in which mesh they ask and how their grid reaches metres.
Two copies of the rule would drift, and `m` would change the room's
appearance in one mode and not the other.

For the frame layer to have a mesh to ask, `run_live` builds one whenever
`m` lands on **either** of the two layers that want it —
`view.layer in ('sensor', 'surface')` — and `_cycle_layer` builds one
there and then on the key press, because the window is usually frozen
when anyone presses it and no frame is being processed at all. Failing
to build one **costs the colour and never the layer**: the middle layer is
meshed from the model's own voxels and always has something to show.

Measured on the frame layer with `--colour all`: **0.0% dimmed class
tint** — 45.7% photographed, 54.2% borrowed. Under the `zbuffer` default
76.7% are dimmed, which is that rule's own doing and documented with it
in [VIEWERS.md](VIEWERS.md#display_solidpy).

---

## 12. What it costs

| | before | after |
| --- | --- | --- |
| room mesh, full room07 sweep | 235 ms | 542 ms |
| frame `mesh+draw` | 33 ms | 153 ms |

Three passes took the frame path down from a first cut nearly twice as
expensive: the capped query (108.5 → 49.3 ms, §8.3), the int64 edge key
(45.4 → 8.5 ms, §7), and the shrink before the split (21.8 → 6.4 ms,
§4.5 — a quarter the work *and* the right order).

Final single-frame layers: cubes 33 ms, surface 153 ms, sensor 36 ms.

**During a live sweep none of this runs unless the layer was asked for.**
Four 400-frame runs put measured fps unchanged — the spread followed
running order, not which layer was on screen.

---

## 13. Flags

| flag | default | what it does |
| --- | --- | --- |
| `--surface_smooth N` | 2 | Laplacian passes (§3). 0 leaves the lattice as it comes out |
| `--surface_pull V` | 1.5 | caps the TSDF step in voxels (§4). 0 turns the shrink off |
| `--no_sensor_colour` | on | keeps the room's own colour everywhere; turns §8 off |
| `--no_match_sensor` | on | leaves the room's voxel layers on the room's own evidence instead of also covering what the sensor's mesh passes through |
| `--sensor_first` | off | opens on the sensor's mesh and lets `m` cycle on to the model |
| `--mesh_every N` | 3 | `--accumulate`: rebuild the room mesh every N frames |
| `--world_colour` | `photo` | `photo` / `verdict` / `class` — which question the room's colour answers (§10) |
| `--colour` | `zbuffer` | the single-frame voxel colour rule; `all` is the one with no dimmed tint |

---

## 14. Tests

`tests/test_voxel_surface.py` (45) and `tests/test_live_view.py` (52).
The placement rule is testable on a 5×5×5 grid **with no open3d and no
torch installed**, which is why `voxel_surface.py` is pure numpy with no
file paths: the thing most worth pinning is the one that needs nothing.
Covered there: a slab's face landing on `k + 0.5`, winding, `owner`
always naming an occupied voxel, `subdivide` not moving the surface and
`parents` carrying colours and flags, the TSDF shrink moving only the
front, and both metre conversions in §9 — including the one that asserts
the wrong axis order is more than two voxels off.

---

## 15. Reading the numbers

Room07, the colour lookup: **49.2%** of vertices take the mesh's colour
within reach, **40.5%** from further off, **10.3%** keep their own.
Median brightness goes 75 → 115, against the sensor layer's own 124.

The honest reading is that the layer stopped being dark and that most
vertices now wear a measured colour. Brightness is not accuracy. What
actually settled it was looking: the window frame, the pattern on the bed
and the flowers on the sill are legible where they were washed out.

The two halves of this page answer different questions and neither
answers the other's:

- the TSDF numbers (§4.4) say whether vertices moved **closer to the
  measured depth**;
- the colour numbers say **which source** each vertex's colour came from
  and how the appearance changed.

Neither says the completion is right. That is what
[EVALUATION.md](EVALUATION.md) and the per-capture `gt/EVALUATION.md`
are for.

---

## 16. The pipeline, in one list

1. **Crop and pad.** Occupied bounding box, one empty layer around it.
2. **Extract.** One vertex per straddling cell, one quad per
   disagreeing lattice edge, wound outward.
3. **Keep `owner`.** Every vertex names the occupied voxel whose surface
   it sits on — exact, not resampled.
4. **Read the colour out per vertex**, with the `photographed` flag it
   came with.
5. **Smooth colour and geometry together.** A flat face stays in its
   plane; corners and specks round off.
6. **Shrink onto the TSDF** — frame layer only, front only, clamped.
7. **Subdivide.** Edge midpoints, parents recorded; the surface does not
   move, only where it can be asked.
8. **Look up the sensor's colour.** Two voxels for anything
   photographed, no limit for anything wearing a class tint.
9. **Nothing is smoothed after that**, so what the sensor measured
   arrives as measured.
10. **Off entirely** where the colour is the answer: `verdict`, `class`,
    `--colour class`.

The idea is to use each representation for what it is good for: the
occupancy grid supplies the semantic completion, Surface Nets makes it
continuous, the TSDF pulls it onto what was measured, and the sensor's
mesh supplies the fine appearance. `owner` and the `photographed` flag
are what connect them without letting the completion quietly dress up as
a measurement.
