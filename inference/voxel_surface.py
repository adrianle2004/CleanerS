"""
inference/voxel_surface.py

A binary occupancy grid -> ONE continuous surface, in the grid's own index
coordinates.

This is what the middle of live_view's `m` cycle draws. The room is held as
8 cm voxels and drawn as cubes, which is honest about the resolution and
unreadable as a room: a wall is a staircase and a bed is a pile of bricks.
The same occupancy meshed as a surface reads as furniture, and -- because
the completion and the measurement stay separable by COLOUR rather than by
geometry (see live_view._world_colours) -- it does so without pretending the
model's guesses were measured.

*** PURE NUMPY, ON PURPOSE ***
No open3d, no torch, no file paths. The placement rule below is the whole
reason the layer is worth anything, and it is testable on a 5x5x5 grid with
none of those installed -- see tests/test_voxel_surface.py. The piece that
needs a renderer lives in live_view.py.

*** SURFACE NETS, AND WHY NOT MARCHING CUBES ***
The algorithm is naive Surface Nets (dual contouring with the simplest
vertex rule): one vertex per grid CELL that straddles the boundary, placed
at the mean of its crossing edge midpoints, and one quad per lattice EDGE
whose two voxels disagree.

Marching cubes is the obvious reach and it buys nothing here. Our field is
BINARY, so interpolating to the 0.5 level puts every marching-cubes vertex
exactly at an edge midpoint -- it staircases too, just with a different
triangulation. In both cases the smoothness comes from `smooth` below, not
from the isosurface rule. Surface Nets additionally gives exactly one vertex
per cell, which is what makes `owner` exact: every vertex names the voxel
whose surface it sits on, so it can carry that voxel's colour with no
resampling. It also needs no 256-case table and no new dependency -- there
is no scikit-image in this repo's py3.7 environment.

*** THE PROPERTY THE WHOLE LAYER RESTS ON ***
A flat face comes out exactly where the cube's face is. display_solid draws
a voxel's cube spanning index +/- 0.5 (its `_C` corner table), and a slab
solid up to index k produces a surface at k + 0.5 -- so pressing `m`
re-describes the room without moving it. Anything else and the two layers
disagree by 4 cm, which is a mesh floating off its own floor.

Isolated voxels are the known cost: one voxel alone becomes a cube a third
of its size, because the mean of three crossing midpoints sits inside it.
That is the right trade -- a lone voxel IS a speck of evidence, and this
draws it as one, while surfaces (where the voxels have neighbours) land on
the cube faces exactly.
"""

import numpy as np

# corner k of a cell, as an offset from the cell's base corner. The bit
# order matters below: flipping bit (2 - axis) steps along that axis, which
# is how _EDGES is built.
_CORNERS = np.array([(a, b, c) for a in (0, 1) for b in (0, 1)
                     for c in (0, 1)], dtype=np.int64)

# the cell's 12 edges, as pairs of corner ids differing in exactly one axis
_EDGES = [(i, i ^ (1 << b)) for i in range(8) for b in (2, 1, 0)
          if i < (i ^ (1 << b))]

# where each edge's midpoint sits, in cell-local coordinates
_MIDS = np.array([(_CORNERS[i] + _CORNERS[j]) / 2.0 for i, j in _EDGES])

# The four cells around a lattice edge, as offsets in the edge's two
# PERPENDICULAR axes, wound so the quad comes out counter-clockwise seen
# from +axis. With (axis, u, v) a cyclic permutation of (x, y, z), going
# (u-,v-) -> (u+,v-) -> (u+,v+) -> (u-,v+) puts the normal along +axis.
_RING = ((0, 0), (1, 0), (1, 1), (0, 1))


def surface_nets(occ):
    """Occupancy -> (verts, tris, owner).

    occ:    (nx, ny, nz) boolean. True = there is matter in this voxel.
    verts:  (V, 3) float64, in the grid's own index coordinates -- a voxel
            centre is at its integer index, so a face lands on x.5.
    tris:   (T, 3) int64, wound counter-clockwise seen from OUTSIDE the
            matter, so a renderer's back-face culling and lighting agree.
    owner:  (V, 3) int64, the voxel index each vertex belongs to -- the
            occupied corner nearest it. Index any per-voxel array with this
            and the surface wears that voxel's colour.

    The grid is padded with one layer of empty space first, so a room whose
    floor runs to the edge of the map still gets an underside instead of a
    hole you can see in through.
    """
    occ = np.ascontiguousarray(occ, dtype=bool)
    if occ.ndim != 3:
        raise ValueError('occ must be a 3D grid, not %r' % (occ.shape,))
    if not occ.any():
        return (np.zeros((0, 3)), np.zeros((0, 3), np.int64),
                np.zeros((0, 3), np.int64))
    # Work on the occupied bounding box, not the whole map. The room is a
    # 16 m box because the camera has to be able to walk around in it; the
    # furniture is a fraction of that, and the cell loop below costs the
    # VOLUME. On room07 this is 200x200x38 cropped to about 110x82x36.
    # The padding below is what keeps this exact: a crop is still given an
    # empty layer on every side, so the border closes the same way.
    filled = np.argwhere(occ)
    crop = filled.min(axis=0)
    far = filled.max(axis=0) + 1
    occ = occ[crop[0]:far[0], crop[1]:far[1], crop[2]:far[2]]
    pad = np.zeros(np.asarray(occ.shape) + 2, dtype=bool)
    pad[1:-1, 1:-1, 1:-1] = occ
    # one cell per gap between adjacent samples
    cells = tuple(int(n) - 1 for n in pad.shape)

    def corner(k):
        """Cell-aligned view of the grid at corner k -- no copy."""
        a, b, c = _CORNERS[k]
        return pad[a:a + cells[0], b:b + cells[1], c:c + cells[2]]

    # ---- one vertex per straddling cell ----
    # Summed midpoint and crossing count per cell. float32 keeps this to
    # 20 MB on the 202x202x40 room; the mean is taken in float64 below.
    acc = np.zeros(cells + (3,), np.float32)
    cnt = np.zeros(cells, np.uint8)
    for (i, j), mid in zip(_EDGES, _MIDS):
        crossing = corner(i) != corner(j)
        for axis in range(3):
            if mid[axis]:
                acc[..., axis] += crossing * np.float32(mid[axis])
        cnt += crossing

    active = cnt > 0
    base = np.argwhere(active)                      # (V,3) cell base corner
    local = (acc[active].astype(np.float64)
             / cnt[active][:, None].astype(np.float64))
    # padded corner coordinates -> the caller's voxel indices, crop undone
    verts = base.astype(np.float64) + local - 1.0 + crop

    vid = np.full(cells, -1, np.int64)
    vid[active] = np.arange(len(base))

    # ---- one quad per lattice edge whose two voxels disagree ----
    quads = []
    for axis in range(3):
        u, v = (axis + 1) % 3, (axis + 2) % 3
        lo, hi = [slice(None)] * 3, [slice(None)] * 3
        lo[axis], hi[axis] = slice(0, -1), slice(1, None)
        near, far = pad[tuple(lo)], pad[tuple(hi)]
        crossing = near != far
        if not crossing.any():
            continue
        p = np.argwhere(crossing)       # the edge's lower endpoint
        solid_near = near[crossing]     # is the matter on the -axis side?

        ring = np.empty((len(p), 4), np.int64)
        inside = np.ones(len(p), bool)
        for col, (du, dv) in enumerate(_RING):
            q = p.copy()
            q[:, u] += du - 1           # the cell is one back along u, v
            q[:, v] += dv - 1
            ok = ((q[:, u] >= 0) & (q[:, u] < cells[u])
                  & (q[:, v] >= 0) & (q[:, v] < cells[v]))
            inside &= ok
            # clip before indexing: a negative index would silently wrap
            np.clip(q[:, u], 0, cells[u] - 1, out=q[:, u])
            np.clip(q[:, v], 0, cells[v] - 1, out=q[:, v])
            ring[:, col] = vid[q[:, 0], q[:, 1], q[:, 2]]
        keep = inside & np.all(ring >= 0, axis=1)
        ring, solid_near = ring[keep], solid_near[keep]
        # matter on the +axis side means the outward normal is -axis
        ring[~solid_near] = ring[~solid_near][:, ::-1]
        quads.append(ring)

    if not quads:
        return (np.zeros((0, 3)), np.zeros((0, 3), np.int64),
                np.zeros((0, 3), np.int64))
    quad = np.concatenate(quads)
    tris = np.concatenate([quad[:, [0, 1, 2]], quad[:, [0, 2, 3]]])

    # ---- which voxel does each vertex belong to ----
    # the nearest OCCUPIED corner of its cell: a vertex that named empty
    # space would paint the surface with a voxel nobody predicted
    occupied = np.stack([corner(k)[active] for k in range(8)], axis=1)
    d2 = ((local[:, None, :] - _CORNERS[None, :, :]) ** 2).sum(axis=2)
    pick = np.argmin(np.where(occupied, d2, np.inf), axis=1)
    owner = base + _CORNERS[pick] - 1 + crop

    return verts, tris, owner


# The TSDF is a signed distance over VOX_MARGIN (0.24 m), and a voxel is
# VOX_UNIT_LOW (0.08 m), so a TSDF of t is 3t voxels from the measured
# surface. Both constants belong to frame_loader; the ratio is repeated
# here rather than imported for the reason the module header gives --
# this file needs nothing but numpy.
TSDF_VOXELS = 3.0


def nearest_colour(points, ref_points, ref_colours, max_dist, base,
                   tree=None):
    """Each point's colour taken from the nearest reference point.

    -> (colours, found). A point with no reference within `max_dist`
    keeps its `base` colour and is false in `found`. `max_dist` may be
    one number or one per point, and `tree` a cKDTree over `ref_points`
    built earlier -- the room is recoloured every few frames against
    the same mesh, and rebuilding a tree over 62 k vertices each time
    is most of the cost.

    This is how the surface layer is made to agree with the sensor's own
    mesh about what the room LOOKS like, not just where it is. The two
    are coloured from different places -- the room blends one colour per
    8 cm voxel and falls back to a class tint where the camera never saw,
    the mesh carries the TSDF's 2 cm texture -- so without this, flipping
    between them changes the room's appearance as well as its geometry.

    The distance limit is what stops a vertex that already has a colour
    of its own being handed some wall's from across the room. It is the
    caller's to set: live_view keeps it at two voxels for everything the
    camera coloured, and takes it off entirely for the voxels no
    photograph ever reached, where the alternative is not the model's
    own colour but a dimmed class tint -- and a palette green is not
    what a floor looks like. Pass np.inf for that.
    """
    points = np.asarray(points, np.float64).reshape(-1, 3)
    base = np.array(base, dtype=np.float64, copy=True).reshape(-1, 3)
    found = np.zeros(len(points), bool)
    ref_points = np.asarray(ref_points, np.float64).reshape(-1, 3)
    if not len(points) or not len(ref_points):
        return base, found
    if tree is None:
        from scipy.spatial import cKDTree
        tree = cKDTree(ref_points)
    # Capped at the LARGEST limit anyone asked for, which is both
    # correct for a per-point limit and much cheaper when none of them
    # is infinite: a frame, where every voxel has a photographed colour
    # of its own, then prunes instead of searching the whole mesh.
    cap = float(np.max(max_dist))
    try:
        # 63 k vertices against 62 k, uncapped: 390 ms on one core, 48
        # on twenty, and this runs between two frames
        d, i = tree.query(points, distance_upper_bound=cap, workers=-1)
    except TypeError:                 # scipy older than 1.6 has no workers
        d, i = tree.query(points, distance_upper_bound=cap)
    found = (i < len(ref_points)) & (d <= max_dist)
    if found.any():
        base[found] = np.asarray(ref_colours, np.float64)[i[found]]
    return base, found


def subdivide(verts, tris):
    """Split every triangle at its edge midpoints. -> (verts, tris,
    parents).

    `parents` is (M,2): the two vertices each one sits between, an
    original naming itself twice. It is what carries anything else the
    caller holds per vertex -- `arr[parents].mean(axis=1)` for a colour,
    `.all(axis=1)` for a flag -- without this function having to know
    what any of it means.

    The surface has one vertex per 8 cm cell and the sensor's mesh
    carries the TSDF's 2 cm texture, so a colour sampled off it once per
    cell throws three quarters of it away. One split puts the two at the
    same density. Midpoints lie on the triangle, so the surface does not
    move: only where it can be asked about.
    """
    verts = np.asarray(verts, np.float64).reshape(-1, 3)
    tris = np.asarray(tris, np.int64).reshape(-1, 3)
    n = len(verts)
    if not len(tris):
        return verts, tris, np.repeat(np.arange(n)[:, None], 2, axis=1)
    # each edge once, however many triangles share it -- splitting a
    # shared edge twice would tear the surface along it
    e = np.sort(np.concatenate([tris[:, [0, 1]], tris[:, [1, 2]],
                                tris[:, [2, 0]]]), axis=1)
    # one int64 per edge rather than np.unique over a 2-column array:
    # that sorts lexicographically, row by row, and is most of the cost
    # of splitting a 30 k-triangle surface
    key, inv = np.unique(e[:, 0] * len(verts) + e[:, 1],
                         return_inverse=True)
    uniq = np.stack([key // len(verts), key % len(verts)], axis=1)
    mid = n + inv.reshape(3, -1).T
    a, b, c = tris[:, 0], tris[:, 1], tris[:, 2]
    ab, bc, ca = mid[:, 0], mid[:, 1], mid[:, 2]
    out = np.concatenate([np.stack([a, ab, ca], axis=1),
                          np.stack([ab, b, bc], axis=1),
                          np.stack([ca, bc, c], axis=1),
                          np.stack([ab, bc, ca], axis=1)])
    parents = np.concatenate([
        np.repeat(np.arange(n)[:, None], 2, axis=1), uniq])
    new = (verts[uniq[:, 0]] + verts[uniq[:, 1]]) * 0.5
    return np.concatenate([verts, new]), out, parents


def voxelise(points, shape, values=None):
    """Put points onto a grid. -> (covered, mean) with mean None if no
    values were given.

    `points` are FRACTIONAL voxel indices in the caller's own axis order,
    because only the caller knows how its grid maps to metres. Points
    outside are dropped rather than wrapped.

    This is how the sensor's own mesh gets onto the room's grid. The two
    are built from the same depth frames and still disagree -- the TSDF
    resolves 2 cm cells and raises a surface wherever a weighted average
    of signed distance crosses zero, while the room samples ONE point per
    8 cm voxel and needs two hits to outvote two passes -- so at a window
    one has a sheet and the other a hole. `covered` is the mesh's own
    verdict in the room's coordinates.

    `values` (a colour per point) are AVERAGED per voxel, because a 2 cm
    mesh puts many vertices in one 8 cm voxel and the colour that voxel
    should wear is their mean rather than whichever landed last.
    """
    shape = tuple(int(n) for n in shape)
    cov = np.zeros(shape, bool)
    mean = None if values is None else np.zeros(shape + (3,), np.float64)
    p = np.asarray(points, np.float64).reshape(-1, 3)
    if not len(p):
        return cov, mean
    idx = np.floor(p).astype(np.int64)
    inside = np.all((idx >= 0) & (idx < np.asarray(shape)), axis=1)
    idx = idx[inside]
    if not len(idx):
        return cov, mean
    cov[idx[:, 0], idx[:, 1], idx[:, 2]] = True
    if values is None:
        return cov, mean
    v = np.asarray(values, np.float64).reshape(-1, 3)[inside]
    flat = np.ravel_multi_index((idx[:, 0], idx[:, 1], idx[:, 2]), shape)
    n = np.bincount(flat, minlength=int(np.prod(shape))).astype(np.float64)
    acc = np.stack([np.bincount(flat, weights=v[:, k],
                                minlength=int(np.prod(shape)))
                    for k in range(3)], axis=1)
    hit = n > 0
    out = np.zeros_like(acc)
    out[hit] = acc[hit] / n[hit, None]
    return cov, out.reshape(shape + (3,))


def pull_to_surface(verts, tsdf, scale=TSDF_VOXELS, max_step=1.5):
    """Shrink the surface onto the measurement, on the camera's side only.

    verts are FRACTIONAL grid indices in `tsdf`'s own axis order. Each one
    in front of a measured surface (tsdf > 0) is stepped back along the
    TSDF's gradient by the distance the TSDF says it is out, which lands
    it on the zero crossing -- the surface the depth actually measured.

    The model's completion comes out about a voxel proud of the
    measurement, and this is the one place in the pipeline that already
    knows by how much, per voxel, without being told. Anything NOT in
    front -- behind a surface, or out where nothing was measured and the
    TSDF saturates -- is left exactly where it was: there is no
    measurement there to shrink onto, and moving it would be inventing
    one.

    `max_step` is why saturation is safe. Out in open space the TSDF is a
    constant 1 and its gradient is noise, so the step is clamped to a
    voxel and a half rather than trusted.
    """
    from scipy import ndimage
    verts = np.array(verts, dtype=np.float64, copy=True)
    if not len(verts):
        return verts
    tsdf = np.asarray(tsdf, np.float64)
    co = verts.T
    t = ndimage.map_coordinates(tsdf, co, order=1, mode='nearest')
    front = t > 0
    if not front.any():
        return verts

    g = np.stack([ndimage.map_coordinates(d, co, order=1, mode='nearest')
                  for d in np.gradient(tsdf)], axis=1)
    n = np.linalg.norm(g, axis=1)
    # a constant TSDF has no direction to move in -- nothing was measured
    move = front & (n > 1e-9)
    if not move.any():
        return verts
    step = np.clip(t[move] * scale, 0.0, max_step)
    verts[move] -= g[move] / n[move, None] * step[:, None]
    return verts


def smooth(verts, tris, passes=2, rate=0.5):
    """Laplacian smoothing: each vertex steps `rate` of the way towards the
    mean of its neighbours, `passes` times.

    This is what turns the lattice into something that reads as a surface.
    It leaves a FLAT face where it was -- a plane is its own neighbourhood
    average -- so the registration with the cubes survives it, and only
    corners and specks move. Vertices drift along a flat face, which is
    invisible, and that is why the mean is left unweighted: the quads are
    split into triangles on one diagonal, so the stencil is not symmetric,
    and correcting for it would cost a pass over the mesh to buy nothing.
    """
    verts = np.array(verts, dtype=np.float64, copy=True)
    tris = np.asarray(tris)
    if passes <= 0 or not len(tris) or not len(verts):
        return verts

    e = np.concatenate([tris[:, [0, 1]], tris[:, [1, 2]], tris[:, [2, 0]]])
    e = np.concatenate([e, e[:, ::-1]])         # both directions
    src, dst = e[:, 0], e[:, 1]
    deg = np.bincount(src, minlength=len(verts)).astype(np.float64)
    movable = deg > 0

    # bincount per component rather than np.add.at on the (V,3) array:
    # same sum, and np.add.at takes an unbuffered slow path that cost more
    # than the surface extraction itself -- 40 ms of a 120 ms room.
    n = len(verts)
    for _ in range(int(passes)):
        acc = np.stack([np.bincount(src, weights=verts[dst, k], minlength=n)
                        for k in range(3)], axis=1)
        verts[movable] += rate * (acc[movable] / deg[movable, None]
                                  - verts[movable])
    return verts
