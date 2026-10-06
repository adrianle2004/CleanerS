# Box editor — every key

The annotation tool: place solids in the fused room cloud, tag each with a
class, save `gt/solids.json`. Step 5 of [MAKING_GT.md](MAKING_GT.md).

```bash
python -m reconstruction_GT.box_editor captures/room07
```

It reloads whatever is already in that capture's `gt/solids.json`, so you can
stop and come back.

---

## Mouse

| | |
| --- | --- |
| left-drag | rotate the view |
| right-drag / middle-drag | pan |
| scroll | zoom |

The mouse only moves the camera. Solids are moved with the keyboard, so you
can never nudge one by accident while looking around.

## Solids

| key | what it does |
| --- | --- |
| `N` | new solid, dropped in the middle of the room (0.6 × 0.6 × 0.8 box) |
| `TAB` or `.` | select the **next** solid |
| `,` | select the **previous** solid |
| `L` | list every solid in the terminal, `->` marking the selected one |
| `BACKSPACE` | delete the selected solid |

The selected solid is **red**, the others green, so you can always see which one
the keys will move. `L` is the quickest way to find a particular one: read the
list, then press `.` until the arrow reaches it.

## Moving

| key | what it does |
| --- | --- |
| `←` `→` | move along world X |
| `↑` `↓` | move along world Y |
| `R` / `V` | move up / down |
| `PAGE UP` / `PAGE DOWN` | same as `R` / `V`, if your keyboard has them |
| `K` | pick the rotation axis: `z (yaw)` → `y (pitch)` → `x (roll)` |
| `Q` / `E` | rotate −5° / +5° about that axis |

Movement is in world axes; rotation and resizing are in the solid's own frame,
so rotations compose — lay a cylinder down with `K`,`K`,`Q`×18, then yaw it
into place. `solids.json` stores the full 3×3 (`rotation`), with `yaw_deg` kept
alongside for reading.

## Resizing — one face at a time

| key | what it does |
| --- | --- |
| `A` | pick the face: `+x` → `-x` → `+y` → `-y` → `+z(top)` → `-z(bottom)` |
| `W` | push that face **outward** (grow) |
| `S` | pull that face **inward** (shrink) |

**The selected face is drawn in yellow.** The opposite face never moves, so you
can extend a bed towards the wall without its front edge shifting. Minimum size
is 5 cm on any axis.

## Shape

| key | what it does |
| --- | --- |
| `H` | cycle `box` → `cylinder` → `wedge` |

| shape | fills | what `size` means |
| --- | --- | --- |
| `box` | the whole box | width, depth, height |
| `cylinder` | elliptic cylinder about the solid's own z — bin, stool, bottle | two diameters, length |
| `wedge` | a ramp: full height at the `-x` face, nothing at `+x` | the enclosing box |
| `mesh` | a CAD model, scaled into the box (`C`) | the box it is scaled into |

`C` cycles the CAD library — `<capture>/gt/cad/`, then `reconstruction_GT/cad/`
(`--cad_dir` for anywhere else). The mesh is drawn in place and its filename is
printed, so you always know which model you are placing, and the box stays the
thing you move and resize. `H` returns the solid to a plain box.

**When a mesh is worth it:** measured on NYU's own CAD objects, `furn` fills
87% of its bounding box and `objs` ~100% — a box is already right for those.
But a chair fills **15%** and a table **14%**, so boxing them caps their IoU
near 0.15 however well the model predicts. The two seeded models
(`chair_simple`, `table_simple`, built from a seat/back/legs and a top/legs)
fill 18% and 15% — NYU's own proportions.

All three fill **solid**, which is what ground truth has to be. A 1 m box
voxelizes to 1.02 m³, a cylinder to 0.81 m³ (π/4 = 0.785), a wedge to 0.51 m³.

## Helpers

| key | what it does |
| --- | --- |
| `F` | **fit**: snap the solid to the cloud points currently inside it |
| `G` | **floor**: drop it so its bottom sits at z = 0, keeping its top |
| `I` | **into the wall**: take the room's yaw, centre on the nearest wall plane, 12 cm thick |
| `M` | edit the **room shell** itself — walls move with `A`/`W`/`S` like any solid |
| `[` / `]` | step size: 1 → 2 → 5 → 10 → 20 cm (applies to moving and resizing) |

`I` is for windows and doorways, which are openings **in** a wall rather than
objects in the room. NYU models them as polygons on the wall plane, which
voxelize to a thin planar patch — its windows measure 6–14 voxels (12–28 cm)
across the wall normal. The voxelizer paints the shell first and your solids
after, so a snapped window **overwrites** the wall it sits in, as NYU's does.

`M` exists because some walls cannot be fitted. IR goes straight through glass,
so the cloud carries points metres beyond a window wall and the shell overshoots
it. Press `M`, pick that wall with `A`, pull it back with `S` — which is what
Guo & Hoiem's annotators did for every wall, drawing each as a line segment in
an overhead view. Your edit is saved and reloaded rather than re-fitted
(`--refit_room` fits again).

`F` only sees what the camera saw — the front and top of a bed, never the side
against the wall. Use it to get close, then extend the hidden faces by hand.
`G` matters for the same reason: a bed's ground truth is solid from the floor
up, not a duvet-thick slab floating at 0.5 m.

## Class

| key | id | class | what goes in it |
| --- | --- | --- | --- |
| `1` | 1 | ceiling | *(the room shell writes this)* |
| `2` | 2 | floor | *(shell)* |
| `3` | 3 | wall | *(shell)* |
| `4` | 4 | window | glass, frame, blinds |
| `5` | 5 | chair | chairs, stools, desk chairs |
| `6` | 6 | bed | mattress, duvet, frame |
| `7` | 7 | sofa | sofas, armchairs |
| `8` | 8 | table | desks, tables, nightstands |
| `9` | 9 | tvs | TVs, monitors |
| `U` | 10 | furn | large storage: wardrobe, shelves, drawers, cabinets |
| `O` | 11 | objs | portable things on top: pillows, bottles, books, bags |

`0` also sets furn, but `U` is the one to remember — `0` and `O` are
indistinguishable in a key list.

You do **not** annotate floor, ceiling or walls: the room shell is all three.
Keep the `furn` vs `objs` line consistent across rooms, or mIoU measures your
labelling policy rather than the model.

## Finishing

| key | what it does |
| --- | --- |
| `ENTER` | save `gt/solids.json` and close |
| `ESC` | close **without saving** |

`ENTER` refuses while any solid still has no class, and names the offenders.
The previous file is kept as `solids.json.bak`.

---

## What the colours mean

| colour | what |
| --- | --- |
| **red** | the solid you are editing |
| **green** | your other solids |
| **yellow** | the face `W`/`S` will move |
| **blue** | the room shell — floor, ceiling and walls come from this |

The blue shell is fitted the way Guo & Hoiem annotated NYU's rooms
([ICCV 2013](https://openaccess.thecvf.com/content_iccv_2013/papers/Guo_Support_Surface_Prediction_2013_ICCV_paper.pdf),
sec. 2), in two steps:

1. **Manhattan alignment** — vote every wall-like surface normal (|n_z| < 0.3)
   into one dominant direction. Gravity is already fixed by the floor-anchored
   grid, so only the yaw is unknown, and a circular mean over 4·θ finds it.
2. **One plane per wall** — in that rotated frame, take the strongest plane
   near each edge of the cloud. Near the *edge*, because the biggest flat thing
   in a bedroom is usually the bed or a wardrobe front, and a global peak
   search snaps the room onto the furniture.

The editor prints how many points support each wall. A wall with little support
was barely scanned, and the shell falls back to the outermost points there —
NYU's own rooms are often annotated with only two or three walls.

For room07: **2.91 × 4.34 m at +40.3°**, 93.4% of the cloud inside. The
smallest-rectangle-round-the-footprint method it replaced gave 4.90 × 2.98 —
56 cm too long, stretched by stray points — where the fitted walls match the
floor's own 4.39 m extent.

## The terminal line

After every keypress the terminal reprints the selected solid:

```
  2/5 box      bed      size 1.42 x 1.98 x 0.55  centre 0.41 2.16 0.28  yaw -140  [face +y, step 5 cm]
```

That size in metres is the thing to check against the real furniture. `(class?)`
instead of a class name means the key did not register.

## A typical object

1. `N` — a box appears in the middle of the room
2. arrows, `R`/`V` — drive it roughly over the object
3. `F` — snap it to the points
4. `G` — sit it on the floor
5. `A` then `W` — push out the faces the camera never saw
6. `6` — tag it (bed)
7. `TAB` or `N` — next

Then `ENTER`, and:

```bash
python -m reconstruction_GT.voxelize_gt captures/room07
```
