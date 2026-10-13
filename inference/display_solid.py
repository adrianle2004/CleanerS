"""
inference/display_solid.py

Solid-voxel viewer for the coloured .ply files written by inference/visualize.py.
A companion to display.py, which is left untouched -- that one draws the voxels
as screen-space POINTS, so they never quite tile: at any zoom the background
shows through the gaps and a wall reads as a cloud of dots rather than a
surface. This one draws each voxel as an actual 1x1x1 cube, so neighbouring
voxels share faces and surfaces come out solid at every zoom, and it can paint
the cubes with the colour the CAMERA saw instead of the flat
per-class colour -- which is what makes one chair distinguishable from the next.

    python inference/display_solid.py outputs/room01/ply/live_000000.ply
    python inference/display_solid.py visual_pred/CleanerS/NYU0015.ply
    python inference/display_solid.py <ply> --colour class        # flat labels
    python inference/display_solid.py <ply> --visibility points   # strict
    python inference/display_solid.py <ply> --visibility all       # everything
    python inference/display_solid.py <ply> --screenshot out.png

Hover the cursor over the geometry: the voxel under it is outlined in yellow
and the bar below the 3D view reports its grid index, predicted class and photo
RGB. Buttons under the view cycle the colouring and reframe; Ctrl+click prints
the readout to stdout. (Buttons rather than key bindings on purpose -- see the
comment in Viewer.__init__.)

The pick is a ray march through the voxel lattice, not a nearest-centre
lookup -- see raycast() for why that distinction is what makes the readout
agree with what you can see.

*** THE THREE MODE SWITCHES ***

Three independent choices, applied in the order the pixels flow through them.
Defaults are marked (*); each is a --flag, and C cycles the second one live.

1. --visibility  WHERE A VOXEL'S COLOUR COMES FROM  (see photo_colours)

   Which measured points belong to a given cube. Two rules that differ by a
   single number, a fallback for having no depth frame at all, and the three
   of them stacked.

   points(*)  Unproject every valid depth pixel and give each voxel the mean
              colour of the points INSIDE it. All in 3D -- no projection, no
              z-buffer, no tolerance. This is the exact answer to "did the
              camera see this voxel": a point is in a voxel or it is not, and
              the voxel size is known.

              A voxel you can plainly see -- front-most along its ray -- can
              still hold no point (275 of room01's 1027 front-most voxels),
              because the camera's coverage has a ragged edge that the model's
              surface does not: see `surface` below for the measurement. Those
              voxels get their class colour and the render speckles there.
              That is the honest picture, and it is the default: a voxel is
              coloured from its OWN measurements or not at all.

   surface    the same rule with the region grown to --reach voxels (0.75 =
              6 cm), which fills the speckles -- but by taking a NEIGHBOUR's
              measurements, so it is an inference and is no longer the
              default. Opt in with --visibility surface.

              What it recovers are voxels at the FRINGE of the measured
              region: the model says the surface continues, the camera's
              samples stop just short of them. Measured over 80 NYU frames
              (30,861 such voxels), the nearest measured point sits at the
              same distance from the camera -- offset ALONG the view averages
              +0.07 voxels, 55% further vs 45% nearer, i.e. no bias -- and
              about 0.8 voxels SIDEWAYS. So they sit beside the measured
              patch, not in front of or behind it.

              They are at a fringe because the camera has no data there:
              26.5% of their image footprints straddle a depth step over
              0.3 m (a silhouette), their dropout rate is 12.5% against 5.5%
              for voxels that did get points, and 4.4% touch the image border.

              The model is not misplacing them. NYU's ground truth calls the
              model's own cube occupied in 99.99% of cases (57.5% occupies
              both it and the neighbour the points fell in, 42.5% only the
              model's); just 3 of 30,861 are the "model picked the wrong cell"
              case. Reaching 6 cm therefore colours voxels that really are on
              the surface, from real measured pixels.

              Voxels with no measurement anywhere near them still get nothing.
              That is the honest thing to show: they are completion the model
              placed in FRONT of a wall, a median 1.5 voxels from any measured
              surface and 26 at the 90th percentile.

   zbuffer    Paint the voxels into the image far-to-near and colour whichever
              is front-most over each pixel. Needs no depth frame, which is
              why it was the original rule -- resolve_frame did not return
              depth_path until reconstruct.py was written. Right for a wall
              facing the camera, wrong for one seen edge-on: along a bed top
              seen from standing height the ray travels a long way
              horizontally per 8 cm of drop, so consecutive bed voxels sit far
              apart in depth, the nearest wins every pixel, and the rest fall
              back to the flat class colour.

   all        All three, in that order, each voxel taking its colour from the
              first rule that reaches it -- so every voxel `points` could
              colour keeps the colour `points` gave it, from its own
              measurements alone, and the later stages only fill in around it.

              The photo stage is NOT plain `zbuffer`. A pixel counts for a
              voxel only if the depth there agrees with the voxel's own range
              to within --agree, or there is no depth there at all -- dropout,
              where the photo is the only evidence and nothing contradicts it.
              Ungated, 88% of the voxels the z-buffer adds here own pixels
              whose depth is over 0.2 m away, a median of 1.41 m IN FRONT of
              them: the colour of the foreground object. Held out against
              voxels whose own colour is known, the gate wins on all 8 frames
              measured, 0.8-7.8 of 255 against 0.9-10.8, and still lifts
              coverage from 14.4-66.1% to 17.8-68.8%. See photo_from_image.

   The readout names whichever rule the voxel under the cursor was coloured
   by: "measured inside this voxel", "measured just outside it (within
   --reach)", "read from the photo (no depth there to disagree)", or "hidden
   -- colour inferred" for the ones nothing reached.

2. --colour / the C key   WHICH PALETTE IS DRAWN

   photo(*)   the colour the camera saw, from rule 1.
   class      display.py's flat per-class colours, ignoring the photo.

3. --hidden   WHAT THE VOXELS WITH NO COLOUR LOOK LIKE

   Whatever rule 1 could not colour -- behind a surface, outside the frustum,
   in a depth hole, or completion the camera never saw.

   class(*)   their class colour dimmed to 0.55, which reads as "unseen".
   nearest    the colour of the nearest visible voxel OF THE SAME CLASS, for a
              continuous surface. Nearest overall would smear the wall's
              colour onto the bed standing in front of it.

The rest are plain on/off switches, not modes: --raw-axes (skip the handedness
fix), --flat (no face shading), --no-cull (keep interior faces), --no-highlight
(no outline under the cursor), --screenshot PATH (render offscreen and exit).

*** ORIENTATION -- the same fix display.py makes, for the same reason ***

The .ply stores voxel-grid indices, and the grid's axis order is

    ply x = grid axis 0 -> world X, increasing to the CAMERA'S RIGHT
    ply y = grid axis 1 -> world Z, increasing UP
    ply z = grid axis 2 -> world Y, increasing AWAY from the camera

(right, up, forward) is LEFT-handed, and every viewer expects the OpenGL
(right, up, TOWARD the viewer) convention, so the raw file renders inside-out:
the eye ends up behind the far wall, which is then drawn over the furniture in
front of it. Reflecting the depth axis (z -> zmax + zmin - z) makes the basis
right-handed and puts the eye where the capture camera stood. Left/right is
ALREADY correct and is unchanged by the reflection -- reflecting x instead, the
obvious guess, breaks it. --raw-axes turns the fix off to show the difference.

The reflection is applied to the voxel INDICES before the cubes are built, so
the cubes keep their outward face winding and the neighbour test that culls
hidden faces stays exact. Everything the readout reports, and every projection
into the camera image, uses the ORIGINAL index, which still indexes the
prediction arrays directly.

*** WHERE THE CAMERA GEOMETRY COMES FROM ***

Both directions of the same transform, and the file has one implementation of
each so they cannot drift apart:

    voxel_centres_world   index -> world metres.  Used by the zbuffer rule and
                          by reconstruct.py.
    depth_to_grid_index   depth pixel -> world -> fractional index; the exact
                          inverse. This is the one the default rule uses, and
                          it never goes near the image plane.

Both use the same vox_origin / cam_pose / cam_K the frame went through on the
way IN (frame_loader.py). Nothing here is estimated: the pose comes from the
.bin header for NYU frames and from the capture's own meta.json (camera_height,
yaw, cam_K) for live ones, and the file refuses to guess if it can't find them.
--visibility points, surface and all additionally need the frame's depth,
found the same way; zbuffer is the one rule that works from the colour image
alone.
"""

import argparse
import json
import os
import sys

import numpy as np
import open3d as o3d

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from visualize import DEFAULT_COLORMAP  # noqa: E402

from utils import CLASSES  # noqa: E402

CLASS_NAMES = list(CLASSES) + ['ignore'] * (len(DEFAULT_COLORMAP) - len(CLASSES))
COLOR_TO_CLASS = {tuple(int(c) for c in rgb): (i, CLASS_NAMES[i])
                  for i, rgb in enumerate(DEFAULT_COLORMAP)}

VOX_UNIT = 0.08          # low-res voxel edge, metres (0.02 * 4)
GRID = (60, 36, 60)      # ply x, ply y, ply z

# --visibility all, third stage: how far the depth frame may disagree with a
# voxel's own range before the pixel it is front-most on is thrown out. One
# voxel; see photo_from_image for what it is protecting against.
IMAGE_AGREE = 0.08

# Where a voxel's colour came from. --visibility all is the only rule that can
# produce more than one of these in a frame, and the readout names whichever
# one the voxel under the cursor got.
SRC_NONE, SRC_INSIDE, SRC_REACH, SRC_IMAGE, SRC_IMAGE_FAR = 0, 1, 2, 3, 4
SOURCE_TEXT = {
    SRC_NONE: 'hidden -- colour inferred',
    SRC_INSIDE: 'measured inside this voxel',
    SRC_REACH: 'measured just outside it (within --reach)',
    SRC_IMAGE: 'read from the photo (no depth there to disagree)',
    SRC_IMAGE_FAR: 'read from the photo (the depth there measures something '
                   'else)',
}


# ------------------------------------------------------------------ ply input

def load_voxels(ply_path):
    """-> (N,3) int32 grid indices, (N,3) int32 class colour, (N,) class id."""
    pcd = o3d.io.read_point_cloud(ply_path)
    n = len(pcd.points)
    if n == 0:
        raise SystemExit(f"{ply_path} loaded but contains 0 points.")
    idx = np.rint(np.asarray(pcd.points)).astype(np.int32)
    rgb = np.rint(np.asarray(pcd.colors) * 255.0).astype(np.int32)
    cls = np.full(n, -1, dtype=np.int32)
    for colour, (cid, _) in COLOR_TO_CLASS.items():
        cls[np.all(rgb == np.asarray(colour, dtype=np.int32), axis=1)] = cid
    return idx, rgb, cls


# Set by display_overlay.py --frame-grid: reflect about the grid itself rather
# than this file's voxels, so two files of the same frame (prediction and GT)
# land in the same view coordinates.
FIXED_PIVOT = None


def view_pivot(idx):
    """The constant to_view_index reflects the depth axis about.

    Anything else drawn in the same window -- reconstruct.py's measured
    surface, say -- has to be reflected about this SAME value or it will sit a
    few voxels off the prediction it is meant to be compared with.
    """
    if FIXED_PIVOT is not None:
        return FIXED_PIVOT
    return int(idx[:, 2].max()) + int(idx[:, 2].min())


def to_view_index(idx, pivot=None):
    """Left-handed (right, up, forward) -> right-handed (right, up, backward).

    Identical to display.py's to_view_frame, kept on integers so the result is
    still a grid index and the neighbour lookups below stay exact.
    """
    out = idx.copy()
    out[:, 2] = (view_pivot(idx) if pivot is None else pivot) - idx[:, 2]
    return out


def describe(idx, cls):
    lines = ["%d voxels" % len(idx),
             "bounds  x %d..%d   y %d..%d   z %d..%d"
             % (idx[:, 0].min(), idx[:, 0].max(), idx[:, 1].min(),
                idx[:, 1].max(), idx[:, 2].min(), idx[:, 2].max())]
    ids, counts = np.unique(cls, return_counts=True)
    for cid, n in sorted(zip(ids.tolist(), counts.tolist()), key=lambda kv: -kv[1]):
        name = CLASS_NAMES[cid] if cid >= 0 else 'unknown colour'
        lines.append("  %-8s %6d  (%4.1f%%)" % (name, n, 100.0 * n / len(idx)))
    return "\n".join(lines)


# ------------------------------------------------------- geometry: solid cubes

# corners of a unit cube, offsets from its centre
_C = np.array([[-.5, -.5, -.5], [+.5, -.5, -.5], [+.5, +.5, -.5], [-.5, +.5, -.5],
               [-.5, -.5, +.5], [+.5, -.5, +.5], [+.5, +.5, +.5], [-.5, +.5, +.5]])

# (neighbour direction, the face's four corners wound CCW seen from OUTSIDE,
#  shade factor). The shade is baked into the vertex colours and rendered
# unlit: real lighting would tone-map the class colours and make the readout
# disagree with the screen, but with no shading at all a cube grid is a
# silhouette -- you cannot tell a step from a flat wall.
FACES = [
    ((+1, 0, 0), [1, 2, 6, 5], 0.86),   # +x  right
    ((-1, 0, 0), [0, 4, 7, 3], 0.66),   # -x  left
    ((0, +1, 0), [3, 7, 6, 2], 1.00),   # +y  up
    ((0, -1, 0), [0, 1, 5, 4], 0.44),   # -y  down
    ((0, 0, +1), [4, 5, 6, 7], 0.92),   # +z  toward the viewer
    ((0, 0, -1), [0, 3, 2, 1], 0.58),   # -z  away
]


def build_cube_mesh(view_idx, colours, shaded=True):
    """One cube per voxel, sharing faces with its neighbours.

    Faces between two occupied voxels are dropped: they are invisible from
    outside the solid and are most of the geometry (a filled wall is nearly all
    interior). Returns the mesh plus, per vertex, which voxel it came from and
    its face's shade -- enough to recolour without rebuilding.
    """
    lo = view_idx.min(axis=0)
    span = view_idx.max(axis=0) - lo + 1
    occ = np.zeros(span, dtype=bool)
    local = view_idx - lo
    occ[local[:, 0], local[:, 1], local[:, 2]] = True

    verts, tris, vox_of_vert, shade_of_vert = [], [], [], []
    base = 0
    centres = view_idx.astype(np.float64)
    for direction, corners, shade in FACES:
        nb = local + np.asarray(direction)
        inside = np.all((nb >= 0) & (nb < span), axis=1)
        hidden = np.zeros(len(view_idx), dtype=bool)
        hidden[inside] = occ[nb[inside, 0], nb[inside, 1], nb[inside, 2]]
        keep = np.nonzero(~hidden)[0]
        if len(keep) == 0:
            continue
        # (M,4,3): each kept voxel's four corners for this face
        quad = centres[keep][:, None, :] + _C[corners][None, :, :]
        verts.append(quad.reshape(-1, 3))
        off = base + 4 * np.arange(len(keep))[:, None]
        tris.append(np.concatenate([off + [0, 1, 2], off + [0, 2, 3]], axis=0))
        vox_of_vert.append(np.repeat(keep, 4))
        shade_of_vert.append(np.full(4 * len(keep), shade if shaded else 1.0))
        base += 4 * len(keep)

    verts = np.concatenate(verts)
    mesh = o3d.geometry.TriangleMesh(
        o3d.utility.Vector3dVector(verts),
        o3d.utility.Vector3iVector(np.concatenate(tris).astype(np.int32)))
    vox_of_vert = np.concatenate(vox_of_vert)
    shade_of_vert = np.concatenate(shade_of_vert)
    paint_mesh(mesh, colours, vox_of_vert, shade_of_vert)
    return mesh, vox_of_vert, shade_of_vert


def paint_mesh(mesh, colours, vox_of_vert, shade_of_vert):
    c = colours[vox_of_vert].astype(np.float64) / 255.0 * shade_of_vert[:, None]
    mesh.vertex_colors = o3d.utility.Vector3dVector(np.clip(c, 0.0, 1.0))
    return mesh


# ------------------------------------------------- geometry: the camera's view

def voxel_centres_world(idx, vox_origin):
    """Grid index -> world metres, inverting frame_loader's axis order:
    it wrote occ[gz, gy, gx] with gz from world X, gy from world Z, gx from
    world Y, so ply (x, y, z) is world (X, Z, Y)."""
    w = np.empty((len(idx), 3), dtype=np.float64)
    w[:, 0] = vox_origin[0] + (idx[:, 0] + 0.5) * VOX_UNIT   # world X
    w[:, 1] = vox_origin[1] + (idx[:, 2] + 0.5) * VOX_UNIT   # world Y
    w[:, 2] = vox_origin[2] + (idx[:, 1] + 0.5) * VOX_UNIT   # world Z
    return w


def depth_to_grid_index(depth, cam_K, cam_pose, vox_origin):
    """Every depth pixel -> a FRACTIONAL grid index, (H, W, 3).

    The exact inverse of voxel_centres_world: unproject the pixel with cam_K,
    put it in the world with cam_pose, then undo `origin + (i + 0.5) * unit`
    with the same world (X, Y, Z) = ply (x, z, y) permutation. Pixels with no
    depth come back at the camera's own position; callers mask them.
    """
    H, W = depth.shape
    fx, fy = cam_K[0, 0], cam_K[1, 1]
    cx, cy = cam_K[0, 2], cam_K[1, 2]
    v, u = np.meshgrid(np.arange(H, dtype=np.float64),
                       np.arange(W, dtype=np.float64), indexing='ij')
    z = depth.astype(np.float64)
    cam = np.stack([(u - cx) * z / fx, (v - cy) * z / fy, z], axis=-1)
    world = cam @ cam_pose[:3, :3].T + cam_pose[:3, 3][None, None, :]

    g = np.empty_like(world)
    g[..., 0] = (world[..., 0] - vox_origin[0]) / VOX_UNIT - 0.5
    g[..., 1] = (world[..., 2] - vox_origin[2]) / VOX_UNIT - 0.5
    g[..., 2] = (world[..., 1] - vox_origin[1]) / VOX_UNIT - 0.5
    return g


def photo_from_points(idx, vox_origin, cam_pose, cam_K, image, depth,
                      reach=0.5, z_min=0.2, z_max=8.0):
    """Colour each voxel from the measured points at its own coordinates.

    Unproject every valid depth pixel, then give each voxel the mean colour of
    the points within `reach` of its centre, Chebyshev -- that is, inside the
    voxel's cube grown to half-width `reach`. All in 3D: no projection, no
    z-buffer, no image-space reasoning at any point.

    reach = 0.5 is exactly the voxel, so it answers "did the camera see THIS
    voxel" with no slack at all. Larger values answer the more useful question
    "what did the camera measure at this voxel's coordinates", which matters
    at the FRINGE of the measured region -- silhouettes, dropout edges, the
    image border -- where the model continues a surface the camera stopped
    sampling. Those voxels' nearest measured point is at the same range (the
    offset along the view averages +0.07 voxels, with no bias either way) and
    about 0.8 voxels sideways, so reach = 0.75 (6 cm) reaches them while
    staying local. See photo_colours for the full measurement.

    Above reach 0.5 this IS an inference, and worth being precise about: every
    pixel used is a real measurement, but it was measured in a neighbouring
    cell, not this one. What bounds it is that the query box is 6 cm on ALL
    THREE axes, depth included, so it can only reach the same local surface
    patch -- a foreground object at a silhouette is far too distant to
    contribute (measured: the borrowed points never sit more than 9 cm from
    the voxel in depth).

    How good an inference: hold out the voxels that DO have their own points,
    colour them from outside-only points, and compare. Median error 0.6-7.8
    of 255 across room01, room03, NYU0015/0284/0515, against 18.5-52.9 for
    two arbitrary voxels in the same scene. So it is 4-20x better than chance,
    not exact. --visibility points refuses the inference entirely.

    (The reason display_solid did not always do this: it used to resolve only
    the frame's COLOUR image, so the geometry itself was the only occlusion
    evidence available. resolve_frame returns depth_path now.)
    """
    H, W = depth.shape
    if image.shape[:2] != (H, W):
        import cv2
        image = cv2.resize(image, (W, H), interpolation=cv2.INTER_AREA)

    from scipy.spatial import cKDTree

    g = depth_to_grid_index(depth, cam_K, cam_pose, vox_origin)
    ok = (depth >= z_min) & (depth <= z_max)
    pts, px = g[ok], image[ok]
    n = len(idx)
    if len(pts) == 0:
        return np.zeros((n, 3), np.int32), np.zeros(n, bool)

    # p=inf is the Chebyshev metric, so the query region is the voxel's own
    # cube grown to half-width `reach` -- not a sphere, which would clip the
    # cube's corners at reach = 0.5 and stop being "inside the voxel"
    tree = cKDTree(pts)
    hits = tree.query_ball_point(idx.astype(np.float64), r=reach, p=np.inf)

    lens = np.fromiter((len(h) for h in hits), dtype=np.int64, count=n)
    visible = lens > 0
    photo = np.zeros((n, 3), dtype=np.float64)
    if visible.any():
        flat = np.concatenate([np.asarray(h, dtype=np.int64)
                               for h in hits if len(h)])
        owner = np.repeat(np.arange(n)[visible], lens[visible])
        got = px[flat]
        sums = np.stack([np.bincount(owner, weights=got[:, k], minlength=n)
                         for k in range(3)], axis=1)
        photo[visible] = sums[visible] / lens[visible, None]
    return np.rint(photo).astype(np.int32), visible


def photo_colours(idx, vox_origin, cam_pose, cam_K, image, depth,
                  mode='points', reach=0.75, agree=IMAGE_AGREE, strict=False):
    """Colour the voxels from the frame, and say how well each one is known.

    Returns (colours, visible, measured, source). `measured` is the strict
    statement -- a measured point lies inside this voxel; `visible` also
    includes voxels the surface merely passes close to, or that only the photo
    reaches. `source` is the per-voxel SRC_* code saying which rule actually
    supplied the colour, which is the only honest readout once more than one
    rule is in play.

    Three rules and a combination of them, plus a fallback for when there is
    no depth frame:

    points   the measured points inside the voxel itself (reach 0.5). Exact,
             parameter-free, and the right answer to "did the camera see this
             voxel". Not the best answer to "what colour is this cube": a
             voxel you can plainly see -- front-most along its ray -- can end
             up with nothing in it (275 of room01's 1027 front-most voxels),
             and rendered strictly the result speckles.

    surface  (default) the same rule with the region grown to `reach` voxels
             (0.75, i.e. 6 cm). That is the whole difference -- one number, no
             second stage, no projection.

             What it recovers are voxels at the FRINGE of the measured region,
             where the model continues a surface the camera stopped sampling:
             a silhouette, a dropout edge, the image border. Over 80 NYU
             frames those voxels' nearest measured point is at the SAME range
             (offset along the view averages +0.07 voxels, 55% further vs 45%
             nearer -- no bias) and about 0.8 voxels SIDEWAYS. Not a rounding
             in depth, which is what it looks like at first.

             Their footprints show why the camera had nothing there: 26.5%
             straddle a depth step over 0.3 m, dropout runs 12.5% against 5.5%
             for voxels that did get points, 4.4% touch the image border. And
             the model is not misplacing them -- NYU's GT calls the model's
             own cube occupied in 99.99% of cases, with only 3 of 30,861 being
             "the model picked the wrong cell".

             Voxels with no measurement anywhere near them still get nothing,
             which is the honest thing to show: that is completion the model
             placed in front of a wall, median 1.5 voxels from any measured
             surface and 26 at the 90th percentile.

    zbuffer  paint the voxels into the image far-to-near and colour whichever
             is front-most over each pixel. The only rule here that needs no
             depth frame, which is why it was the original one -- resolve_frame
             did not return depth_path until reconstruct.py was written. Right
             for a wall facing the camera, wrong for one seen edge-on: along a
             bed top seen from standing height the ray travels a long way
             horizontally per 8 cm of drop, so consecutive bed voxels sit far
             apart in depth, the nearest wins every pixel, and the rest fall
             back to the flat class colour. It misses 863 of room01's 1615
             measured voxels.

    all      all three, each voxel taking its colour from the first rule that
             reaches it: its own measured points, else a neighbour's within
             `reach`, else the photo pixels it is front-most on -- those split
             into the pixels the depth frame does not contradict and, last of
             all, the ones it does (photo_from_image says what each is worth,
             and `strict` drops the second).

             The order is by how directly the rule measures THAT voxel, so
             nothing is ever downgraded: every voxel `points` could colour
             keeps exactly the colour `points` gave it, from its own
             measurements alone -- which is not true of `surface`, where even
             a voxel full of its own points takes the mean over the grown
             region. What the later stages do is fill in around it, and
             `source` says which voxels those are, so the picture stays
             readable as evidence rather than becoming one flat claim.

             The last stage is deliberately included: a voxel nothing else
             reached is otherwise drawn in its flat class colour, which is not
             a more honest picture than a real pixel of the scene, only a less
             useful one -- and the readout says which it is either way. Pass
             strict=True to stop at the gated stage.
    """
    n = len(idx)
    if mode == 'zbuffer':          # must work with depth=None; that is the point
        c, v = project_photo_colours(idx, vox_origin, cam_pose, cam_K, image)
        src = np.where(v, SRC_IMAGE, SRC_NONE)
        if depth is None:
            return c, v, np.zeros(n, dtype=bool), src
        _, measured = photo_from_points(idx, vox_origin, cam_pose, cam_K,
                                        image, depth, reach=0.5)
        return c, v, measured & v, src

    own, measured = photo_from_points(idx, vox_origin, cam_pose, cam_K,
                                      image, depth, reach=0.5)
    if mode == 'points':
        return own, measured, measured, np.where(measured, SRC_INSIDE,
                                                 SRC_NONE)
    # the same rule, reaching a little past the voxel's face. `measured` above
    # stays the strict statement, so the readout can still say which voxels
    # actually contain measurement and which merely sit against it.
    out, near = photo_from_points(idx, vox_origin, cam_pose, cam_K, image,
                                  depth, reach=reach)
    if mode == 'surface':
        return out, near, measured, np.where(
            measured, SRC_INSIDE, np.where(near, SRC_REACH, SRC_NONE))

    # 'all': the rules stacked, best source first. A voxel with its own points
    # keeps `own`, its strict colour, rather than `out`'s grown mean -- the
    # whole point of stacking is that no voxel is coloured by a rule weaker
    # than the one that already reached it.
    image_c, image_v, far_c, far_v = photo_from_image(
        idx, vox_origin, cam_pose, cam_K, image, depth, agree=agree)
    third = image_v & ~near
    fourth = (np.zeros(n, dtype=bool) if strict
              else far_v & ~near & ~third)
    colour = out.copy()
    colour[measured] = own[measured]
    colour[third] = image_c[third]
    colour[fourth] = far_c[fourth]
    source = np.where(measured, SRC_INSIDE,
                      np.where(near, SRC_REACH,
                               np.where(third, SRC_IMAGE,
                                        np.where(fourth, SRC_IMAGE_FAR,
                                                 SRC_NONE))))
    # measured is in near for any reach >= 0.5, but a caller is free to pass
    # less than that, and `visible` must stay exactly "source says something"
    return colour, measured | near | third | fourth, measured, source


def paint_owners(idx, vox_origin, cam_pose, cam_K, shape):
    """Paint the voxels into an image of `shape`, far to near, and say which
    one ends up on top of each pixel.

    Returns (who, z): who[v, u] is the row of the voxel front-most over that
    pixel, or -1 where no voxel covers it, and z holds every voxel's range
    from the camera. Each voxel covers the perspective-correct square its 8 cm
    cube subtends, not a single sample, so a one-pixel misalignment cannot
    take its colour off the wrong object.

    Shared by the two rules that work in image space -- project_photo_colours
    (--visibility zbuffer) and photo_from_image (the third stage of
    --visibility all) -- so they cannot come to disagree about which voxel
    owns a pixel.
    """
    R, t = cam_pose[:3, :3], cam_pose[:3, 3]
    fx, fy = cam_K[0, 0], cam_K[1, 1]
    cx, cy = cam_K[0, 2], cam_K[1, 2]
    H, W = shape

    cam = (voxel_centres_world(idx, vox_origin) - t[None, :]) @ R
    z = cam[:, 2]
    front = z > 1e-3
    zz = np.where(front, z, 1.0)
    u = fx * cam[:, 0] / zz + cx
    v = fy * cam[:, 1] / zz + cy
    ru = fx * (VOX_UNIT / 2.0) / zz
    rv = fy * (VOX_UNIT / 2.0) / zz

    n = len(idx)
    box = np.empty((n, 4), dtype=np.int32)
    box[:, 0] = np.clip(np.floor(u - ru), 0, W)          # u0
    box[:, 1] = np.clip(np.ceil(u + ru) + 1, 0, W)       # u1
    box[:, 2] = np.clip(np.floor(v - rv), 0, H)          # v0
    box[:, 3] = np.clip(np.ceil(v + rv) + 1, 0, H)       # v1
    ok = front & (box[:, 0] < box[:, 1]) & (box[:, 2] < box[:, 3])

    who = np.full((H, W), -1, dtype=np.int32)
    for k in np.argsort(-z):                    # far first, near overwrites
        if not ok[k]:
            continue
        u0, u1, v0, v1 = box[k]
        who[v0:v1, u0:u1] = k
    return who, z


def mean_by_owner(who, image, n):
    """Mean image colour per owning voxel, over the pixels where who >= 0.

    Pass a `who` with the unwanted pixels set to -1 to average over a subset;
    that is how photo_from_image drops the pixels the depth frame contradicts.
    """
    flat = who.reshape(-1)
    owned = flat >= 0
    ids = flat[owned]
    count = np.bincount(ids, minlength=n)
    sums = np.stack([np.bincount(ids, weights=image[..., c].reshape(-1)[owned],
                                 minlength=n) for c in range(3)], axis=1)
    visible = count > 0
    photo = np.zeros((n, 3), dtype=np.float64)
    photo[visible] = sums[visible] / count[visible, None]
    return np.rint(photo).astype(np.int32), visible


def photo_from_image(idx, vox_origin, cam_pose, cam_K, image, depth,
                     agree=IMAGE_AGREE, z_min=0.2, z_max=8.0):
    """Colour each voxel from the photo pixels it is front-most on, keeping
    only the pixels the depth frame does not contradict.

    The z-buffer rule with one extra test, and the test is what makes it
    usable as a last stage rather than a mode of its own. project_photo_colours
    decides visibility among the VOXELS and never consults the depth frame, so
    a voxel standing behind a measured surface -- or buried inside one -- is
    still front-most if no other predicted voxel lies on that ray, and it
    takes the colour of whatever the camera actually saw there. Measured on
    room01's own frame: of the voxels the plain z-buffer adds beyond the two 3D
    rules, 88% own pixels whose depth is more than 0.2 m away from them, the
    median being 1.41 m IN FRONT. That colour belongs to the foreground
    object, not to the voxel.

    So a pixel counts for a voxel only if

      * the depth there is within `agree` (one voxel) of the voxel's own
        range -- the pixel is a measurement OF this voxel; or
      * there is no depth there at all. Dropout, and the photo is then the
        only evidence either way: nothing in the measurement and nothing in
        the prediction puts anything in front of this voxel, so its pixel is
        the best the frame has to offer.

    The first case photo_from_points has mostly already covered. The second is
    the new information, and the reason this stage earns a place: depth
    dropout and colour loss are not co-located, so the camera's colour
    survives in plenty of places its ranging does not.

    Returns (colours, visible, far_colours, far_visible): the gated answer,
    then the SAME painting read from the pixels the gate threw out -- the ones
    whose depth measures something at a different range. photo_colours uses
    that as a last stage, below everything else, because it is the weakest
    evidence here and should only ever colour a voxel nothing else reached.

    How good an inference: hold out the voxels that DO hold their own measured
    points, so the right colour is known, and colour them by this rule
    instead. Over 8 frames (room01/03/04 and NYU0001/0002/0015/0284/0515) the
    median error is 0.8-7.8 of 255 gated against 0.9-10.8 ungated, compared on
    the same voxels within each frame -- the gate wins in all 8, and at the
    90th percentile in 7 of 8. It still adds 8-647 voxels a frame beyond
    points + surface, taking coverage from 14.4-66.1% to 17.8-68.8%.
    """
    if depth is None:
        raise ValueError("photo_from_image needs the frame's depth to know "
                         "which pixels contradict a voxel; --visibility "
                         "zbuffer is the rule that works without it")
    H, W = depth.shape
    if image.shape[:2] != (H, W):
        import cv2
        image = cv2.resize(image, (W, H), interpolation=cv2.INTER_AREA)

    who, z = paint_owners(idx, vox_origin, cam_pose, cam_K, (H, W))
    owned = who >= 0
    # the range of whichever voxel owns the pixel, so the test below is made
    # per PIXEL: one voxel can own pixels with and without depth at once
    own_z = np.where(owned, z[np.clip(who, 0, None)], 0.0)
    valid = (depth >= z_min) & (depth <= z_max)
    agrees = owned & (~valid | (np.abs(depth - own_z) <= agree))
    n = len(idx)
    near_c, near_v = mean_by_owner(np.where(agrees, who, -1), image, n)
    far_c, far_v = mean_by_owner(np.where(owned & ~agrees, who, -1), image, n)
    return near_c, near_v, far_c, far_v


def project_photo_colours(idx, vox_origin, cam_pose, cam_K, image):
    """The z-buffer rule: colour each voxel from the pixels it is front-most on.

    Kept because it is the only rule that needs no depth frame. Each voxel is
    painted into the image as the perspective-correct square it covers, far to
    near, and takes the mean of the pixels it still owns -- not a single sample
    that a one-pixel misalignment could take off the wrong object.

    Its weakness is structural, not a matter of tuning: it decides visibility
    in IMAGE space, so on a surface seen edge-on -- a bed top, a desk -- the
    consecutive voxels along the ray project almost on top of each other, the
    nearest wins every pixel, and the rest are declared unseen although they
    are the same visible surface. photo_from_points has no such problem
    because it never leaves 3D.
    """
    who, _ = paint_owners(idx, vox_origin, cam_pose, cam_K, image.shape[:2])
    return mean_by_owner(who, image, len(idx))


def fill_hidden(photo, visible, class_rgb, view_idx, mode):
    """Colour for the voxels the camera never saw.

    Returns (colours, borrowed): `borrowed` marks the hidden voxels that got a
    sibling's colour under `nearest`, which is never all of them -- a class
    with no coloured member anywhere in the frame has nothing to borrow from,
    and those voxels keep the dimmed class colour. Under `class` it is empty.
    """
    out = photo.copy()
    hidden = ~visible
    done = np.zeros(len(out), dtype=bool)
    if not hidden.any():
        return out, done
    if mode == 'nearest':
        from scipy.spatial import cKDTree
        # nearest visible voxel OF THE SAME CLASS: nearest overall would
        # smear the wall's colour onto the bed it stands behind
        key = (class_rgb[:, 0].astype(np.int64) << 16
               | class_rgb[:, 1].astype(np.int64) << 8 | class_rgb[:, 2])
        for k in np.unique(key[hidden]):
            src = np.nonzero(visible & (key == k))[0]
            dst = np.nonzero(hidden & (key == k))[0]
            if len(src) == 0 or len(dst) == 0:
                continue
            _, j = cKDTree(view_idx[src].astype(np.float64)).query(
                view_idx[dst].astype(np.float64))
            out[dst] = photo[src[j]]
            done[dst] = True
        hidden = hidden & ~done          # no visible sibling -> class colour
    out[hidden] = np.rint(class_rgb[hidden] * 0.55)   # dimmed: reads as unseen
    return out, done


# ------------------------------------------------------------ frame discovery

def load_depth(path, nyu):
    """Metres. NYU's PNGs are bit-rotated; capture.py's are plain millimetres."""
    import cv2
    raw = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if raw is None:
        raise SystemExit("cv2.imread returned None for %r" % path)
    raw = raw.astype(np.uint16)
    if nyu:
        return (((raw << 13) | (raw >> 3)).astype(np.float32) / 1000.0)
    return raw.astype(np.float32) / 1000.0


def resolve_frame(ply_path, args):
    """Find the pose/intrinsics/photo that belong to this ply.

    NYU frames carry their own vox_origin/cam_pose in the .bin header; live
    captures get theirs from capture.py's meta.json through the same
    FrameLoader constructor run_inference.py used, so the projection cannot
    drift from the pipeline's own geometry.
    """
    import cv2
    from frame_loader import CAM_K, FrameLoader

    stem = os.path.splitext(os.path.basename(ply_path))[0]
    # a ground-truth or prediction ply is named <frame>_gt.ply, so the frame --
    # and the photo, depth and per-frame camera under it -- is the stem without
    # that suffix
    frame = stem
    for _suffix in ('_pred_full', '_pred', '_gt_scored', '_gt_fov', '_gt'):
        if frame.endswith(_suffix):
            frame = frame[:-len(_suffix)]
            break
    repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    bin_path, rgb_path = args.bin, args.rgb
    depth_path = getattr(args, 'depth', None)
    if bin_path is None and stem.startswith('NYU'):
        base = stem.split('_')[0]
        for cand in (os.path.join(repo, '..', 'data', 'NYU', 'depth',
                                  base + '_0000.bin'),
                     os.path.join(repo, 'data', 'NYU', 'depth', base + '_0000.bin')):
            if os.path.exists(cand):
                bin_path = cand
                break
        if bin_path is not None and depth_path is None:
            cand = os.path.splitext(bin_path)[0] + '.png'
            depth_path = cand if os.path.exists(cand) else None
        if rgb_path is None:
            for cand in (os.path.join(repo, '..', 'data', 'NYU', 'RGB',
                                      base + '_colors.png'),
                         os.path.join(repo, 'data', 'NYU', 'RGB', base + '_colors.png')):
                if os.path.exists(cand):
                    rgb_path = cand
                    break

    # Occ-ScanNet frames from export_scannet_ply.py: <frame>_pred.ply / _gt.ply
    # sit next to <frame>.json, which holds the exact vox_origin / cam_pose /
    # depth cam_K run_scannet.py encoded with, and the colour image already
    # registered onto the depth camera.
    if bin_path is None and args.meta is None:
        side_path = os.path.join(os.path.dirname(os.path.abspath(ply_path)), frame + '.json')
        if os.path.exists(side_path):
            with open(side_path) as f:
                side = json.load(f)
            if side.get('dataset') == 'occ-scannet':
                here = os.path.dirname(side_path)
                rgb_path = rgb_path or os.path.join(here, side['rgb'])
                depth_path = depth_path or os.path.normpath(os.path.join(here, side['depth']))
                image = cv2.imread(rgb_path)
                if image is None:
                    raise SystemExit("cv2.imread returned None for %r" % rgb_path)
                return dict(vox_origin=np.asarray(side['vox_origin'], dtype=np.float64),
                            cam_pose=np.asarray(side['cam_pose'], dtype=np.float64),
                            cam_K=np.asarray(side['cam_K'], dtype=np.float64),
                            image=image[:, :, ::-1].astype(np.float64), rgb_path=rgb_path,
                            source='%s (%s, %s frame)' % (
                                os.path.relpath(side_path, repo), side['scene'], side['role']),
                            depth_path=depth_path, nyu=False)   # ScanNet depth: plain millimetres

    meta_path = args.meta
    if bin_path is None and meta_path is None:
        # outputs/<scene>/ply/<frame>.ply  ->  captures/<scene>/
        scene = os.path.dirname(os.path.dirname(os.path.abspath(ply_path)))
        cand = os.path.join(repo, 'captures', os.path.basename(scene), 'meta.json')
        if os.path.exists(cand):
            meta_path = cand
            if rgb_path is None:
                cand = os.path.join(os.path.dirname(meta_path), 'rgb', frame + '.png')
                rgb_path = cand if os.path.exists(cand) else None
            if depth_path is None:
                cand = os.path.join(os.path.dirname(meta_path), 'depth', frame + '.png')
                depth_path = cand if os.path.exists(cand) else None

    if bin_path is not None:
        vox_origin, cam_pose = FrameLoader._load_bin_header(bin_path)
        cam_K = CAM_K            # NYU's fixed intrinsics, same as FrameLoader's
        source = os.path.relpath(bin_path, repo)
    elif meta_path is not None:
        from frame_loader import frame_meta
        with open(meta_path) as f:
            # the frame's own camera, where it has one: a capture may hold
            # frames lifted out of the sweep (frame_loader.frame_meta)
            meta = frame_meta(json.load(f), frame)
        cam_K = np.asarray(meta['cam_K'], dtype=np.float32)
        height = args.camera_height if args.camera_height is not None \
            else float(meta.get('camera_height', 1.25))
        yaw = args.yaw if args.yaw is not None else float(meta.get('yaw', 0.0))
        loader = FrameLoader.from_live_camera(
            np.zeros((480, 640), np.float32), cam_K=cam_K,
            camera_height=height, yaw=yaw, up_camera=meta.get('up_camera'))
        vox_origin, cam_pose = loader.vox_origin, loader.cam_pose
        source = "%s [%s] (camera_height=%.2f m, yaw=%.3f rad%s)" % (
            os.path.relpath(meta_path, repo), frame, height, yaw,
            ', measured tilt' if meta.get('up_camera') else '')
    else:
        return None

    if rgb_path is None or not os.path.exists(rgb_path):
        return None
    image = cv2.imread(rgb_path)
    if image is None:
        raise SystemExit("cv2.imread returned None for %r" % rgb_path)
    image = image[:, :, ::-1].astype(np.float64)          # cv2 gives BGR
    return dict(vox_origin=np.asarray(vox_origin, dtype=np.float64),
                cam_pose=np.asarray(cam_pose, dtype=np.float64),
                cam_K=np.asarray(cam_K, dtype=np.float64),
                image=image, rgb_path=rgb_path, source=source,
                # NYU's PNGs carry the ((d << 13) | (d >> 3)) bit rotation;
                # capture.py's are plain millimetres. reconstruct.py needs to
                # know which, and getting it wrong is not subtle.
                depth_path=depth_path, nyu=bin_path is not None)


# ------------------------------------------------------------- picking a voxel

def build_lookup(view_idx):
    """(lookup, lo): lookup[cell - lo] is the row of the voxel at that cell, or
    -1. The grid is small (60x36x60 at most), so a dense array beats a dict."""
    lo = view_idx.min(axis=0)
    span = view_idx.max(axis=0) - lo + 1
    lookup = np.full(span, -1, dtype=np.int32)
    lookup[view_idx[:, 0] - lo[0], view_idx[:, 1] - lo[1],
           view_idx[:, 2] - lo[2]] = np.arange(len(view_idx), dtype=np.int32)
    return lookup, lo


def raycast(lookup, lo, origin, direction):
    """First SOLID voxel the ray enters, or None.

    A 3D-DDA march (Amanatides & Woo) over the voxel lattice: voxel `c` owns
    the unit box [c - 0.5, c + 0.5], so the ray is walked cell by cell in the
    order it actually crosses them and the first occupied one is the answer.

    This replaces snapping to the nearest voxel CENTRE, which is only right
    when you look straight at a face. On a surface seen edge-on -- a floor, or
    a wall from a shallow angle -- the hit point sits far from the centre of
    the voxel it belongs to and nearer the centre of the voxel next door, so
    the nearest-centre pick reported a neighbour: hovering a wall could name
    the floor voxel in front of it. Marching the ray cannot make that mistake,
    and it agrees with what is drawn on screen by construction, because the
    cubes drawn ARE these unit boxes.
    """
    o = np.asarray(origin, dtype=np.float64)
    d = np.asarray(direction, dtype=np.float64)
    n = np.linalg.norm(d)
    if n < 1e-12:
        return None
    d = d / n

    # clip to the occupied volume first, so a ray that starts far outside
    # doesn't burn its step budget crossing empty space
    box_lo = lo - 0.5
    box_hi = lo + np.asarray(lookup.shape) - 0.5
    with np.errstate(divide='ignore', invalid='ignore'):
        t1 = (box_lo - o) / d
        t2 = (box_hi - o) / d
    t_near = np.nanmax(np.minimum(t1, t2))
    t_far = np.nanmin(np.maximum(t1, t2))
    if not (t_far > max(t_near, 0.0)):
        return None

    t = max(t_near, 0.0) + 1e-6
    cell = np.floor(o + t * d + 0.5).astype(np.int64)
    step = np.where(d > 0, 1, -1)
    t_max = np.empty(3)
    t_delta = np.empty(3)
    for i in range(3):
        if abs(d[i]) < 1e-12:
            t_max[i] = np.inf
            t_delta[i] = np.inf
        else:
            boundary = cell[i] + 0.5 * step[i]
            t_max[i] = (boundary - o[i]) / d[i]
            t_delta[i] = 1.0 / abs(d[i])

    shape = np.asarray(lookup.shape)
    for _ in range(int(shape.sum()) + 3):        # a ray crosses at most this many
        local = cell - lo
        if np.all((local >= 0) & (local < shape)):
            row = int(lookup[local[0], local[1], local[2]])
            if row >= 0:
                return row
        elif t_max.min() > t_far:
            return None
        i = int(np.argmin(t_max))
        cell[i] += step[i]
        t_max[i] += t_delta[i]
    return None


_PICK_FAILED = False


def guard_pick(fn, *a, **kw):
    """Run a pick, and never let it take the window down with it.

    A mouse handler fires on every motion event, so an exception here is not
    one traceback, it is one per frame and an unusable viewer. The first
    failure is reported in full -- that is the one worth reading -- and after
    that picking is quietly disabled while the camera controls keep working.
    """
    global _PICK_FAILED
    try:
        return fn(*a, **kw)
    except Exception:
        if not _PICK_FAILED:
            _PICK_FAILED = True
            import traceback
            print("\n--- picking failed, and is now disabled for this session."
                  "\n--- The window still works; please report this traceback:",
                  file=sys.stderr)
            traceback.print_exc()
            print("--- open3d %s, python %s"
                  % (o3d.__version__, sys.version.split()[0]), file=sys.stderr)
        return None


def pick_ray(camera, x, y, width, height):
    """Screen pixel -> (origin, direction) in world coordinates.

    unproject's far plane is at infinity (z=1 returns nan/-inf), so the
    direction is taken from the near plane to the midpoint instead.
    """
    near = np.asarray(camera.unproject(x, y, 0.0, width, height), dtype=np.float64)
    mid = np.asarray(camera.unproject(x, y, 0.5, width, height), dtype=np.float64)
    if not (np.all(np.isfinite(near)) and np.all(np.isfinite(mid))):
        return None, None
    return near, mid - near


# ------------------------------------------------------------------- readouts

def readout(i, idx, cls, photo, visible, measured=None, source=None):
    """One line for the voxel under the cursor.

    The last field is how well its colour is known, and the cases are exactly
    photo_colours' outcomes: measured points inside the voxel, measured points
    within --reach of it, the photo alone, or no evidence at all. Given
    photo_colours' `source` it names the rule that actually supplied the
    colour -- which is the only way to read --visibility all, where three
    rules are in play at once; without it the old two-way split stands.
    """
    x, y, z = (int(v) for v in idx[i])
    name = CLASS_NAMES[cls[i]] if cls[i] >= 0 else 'unknown colour'
    r, g, b = (int(v) for v in photo[i])
    if source is not None:
        seen = SOURCE_TEXT[int(source[i])]
    elif measured is not None and measured[i]:
        seen = SOURCE_TEXT[SRC_INSIDE]
    elif visible[i]:
        seen = SOURCE_TEXT[SRC_REACH]
    else:
        seen = SOURCE_TEXT[SRC_NONE]
    return ("voxel [%2d, %2d, %2d]   %s (class %d)   photo rgb (%3d, %3d, %3d)   %s"
            % (x, y, z, name, cls[i], r, g, b, seen))


# --------------------------------------------------------------- gui viewer

# the three independent switches, in the order the pixels flow through them.
# See "THE THREE MODE SWITCHES" at the top of this file.
VISIBILITY_MODES = ['points', 'surface', 'zbuffer', 'all']  # colour source
COLOUR_MODES = ['photo', 'class']                     # which palette is drawn
HIDDEN_MODES = ['class', 'nearest']                   # what the rest look like


class Viewer:
    """SceneWidget with the readout in a bar UNDER the 3D view, so it never
    sits on top of the geometry being inspected, and a wireframe box marking
    the voxel the cursor is on.

    Picking casts the cursor's ray straight through the voxel lattice (see
    raycast). That is both exact and synchronous, so the outline and the
    readout keep up with the mouse -- the earlier depth-buffer round trip had
    to wait on a render callback before it could report anything.
    """

    FOV = 60.0   # vertical, degrees

    def __init__(self, mesh, vox_of_vert, shade_of_vert, view_idx, idx, cls,
                 palette, visible, title, mode, highlight=True, measured=None,
                 source=None):
        from open3d.visualization import gui, rendering
        self.gui = gui
        self.mesh, self.vox_of_vert, self.shade = mesh, vox_of_vert, shade_of_vert
        self.view_idx, self.idx, self.cls = view_idx, idx, cls
        self.palette, self.visible = palette, visible
        self.measured, self.source = measured, source
        self.mode = mode
        self._framed = False
        self._picked = None
        self.highlight = highlight
        self.lookup, self.lookup_lo = build_lookup(view_idx)

        self.window = gui.Application.instance.create_window(title, 1600, 1000)
        em = self.window.theme.font_size

        self.widget3d = gui.SceneWidget()
        self.widget3d.scene = rendering.Open3DScene(self.window.renderer)
        self.widget3d.scene.set_background([0.05, 0.05, 0.08, 1.0])
        # filament tone-maps by default, which pulls the rendered colours away
        # from the ones the readout reports; the face shading is already baked
        # into the vertex colours, so unlit + no post-processing is exact
        self.widget3d.scene.view.set_post_processing(False)
        self.mat = rendering.MaterialRecord()
        self.mat.shader = "defaultUnlit"
        self.widget3d.scene.add_geometry("vox", self.mesh, self.mat)

        # the cursor's voxel, outlined. Added once and moved with a transform:
        # re-adding geometry on every mouse move is visibly slower.
        if self.highlight:
            self.hi_mat = rendering.MaterialRecord()
            self.hi_mat.shader = "unlitLine"
            self.hi_mat.line_width = 3.0
            self.widget3d.scene.add_geometry("pick", highlight_box(), self.hi_mat)
            self.widget3d.scene.show_geometry("pick", False)
        self.window.add_child(self.widget3d)

        self.panel = gui.Vert(0.25 * em, gui.Margins(0.5 * em, 0.35 * em,
                                                     0.5 * em, 0.35 * em))
        self.panel.background_color = gui.Color(0.05, 0.05, 0.08, 1.0)
        self.info = gui.Label("hover the voxels   |   Ctrl+click prints the line")
        self.idle_text = self.info.text
        self.info.text_color = gui.Color(1.0, 1.0, 1.0)
        self.panel.add_child(self.info)

        # Buttons rather than key bindings: a Window's set_on_key callback is
        # cast to a C++ type open3d 0.17 would not accept an
        # EventCallbackResult for, and it raises out of Application.run() the
        # first time a key is pressed. set_on_clicked takes no argument and
        # returns nothing, so there is nothing to cast.
        row = gui.Horiz(0.4 * em)
        self.colour_button = gui.Button("colour: %s" % self.mode)
        self.colour_button.horizontal_padding_em = 0.6
        self.colour_button.vertical_padding_em = 0.2
        self.colour_button.set_on_clicked(self.cycle_colour)
        row.add_child(self.colour_button)
        reframe = gui.Button("reframe")
        reframe.horizontal_padding_em = 0.6
        reframe.vertical_padding_em = 0.2
        reframe.set_on_clicked(self.reframe)
        row.add_child(reframe)
        row.add_stretch()
        self.panel.add_child(row)

        self.legend = gui.Horiz(0.75 * em)
        # the class list never changes, so the legend is built once; only the
        # mode chip's text is updated (gui.Horiz has no remove_all_children)
        self.mode_label = gui.Label("[%s]" % self.mode)
        self.mode_label.text_color = gui.Color(0.6, 0.75, 1.0)
        self.legend.add_child(self.mode_label)
        for cid in sorted(set(int(c) for c in cls if c >= 0)):
            r, g, b = (int(v) for v in DEFAULT_COLORMAP[cid])
            swatch = gui.Label(CLASS_NAMES[cid])
            swatch.text_color = gui.Color(r / 255.0, g / 255.0, b / 255.0)
            self.legend.add_child(swatch)
        self.legend.add_stretch()
        self.panel.add_child(self.legend)
        self.window.add_child(self.panel)

        self.window.set_on_layout(self._on_layout)
        self.widget3d.set_on_mouse(self._on_mouse)

    # -- colouring -------------------------------------------------------
    def _repaint(self):
        paint_mesh(self.mesh, self.palette[self.mode], self.vox_of_vert, self.shade)
        # the geometry is unchanged, but a MaterialRecord carries no vertex
        # colours -- re-adding is how open3d picks the new ones up
        self.widget3d.scene.remove_geometry("vox")
        self.widget3d.scene.add_geometry("vox", self.mesh, self.mat)
        self.mode_label.text = "[%s]" % self.mode
        self.colour_button.text = "colour: %s" % self.mode
        self.window.post_redraw()

    # -- layout / camera -------------------------------------------------
    def _tan(self):
        return float(np.tan(np.radians(self.FOV / 2.0)))

    def reset_view(self):
        """Stand where the capture camera stood, looking into the scene.

        setup_camera fits the volume's bounding SPHERE, which backs off about
        twice as far as it needs to; this fits the occupied extent to the
        scene rect at the volume's mid-depth instead.
        """
        frame = self.widget3d.frame
        pts = self.view_idx.astype(np.float64)
        lo, hi = pts.min(axis=0) - 0.5, pts.max(axis=0) + 0.5
        ext = hi - lo
        aspect = (frame.width / frame.height) if frame.height else 1.6
        tan = self._tan()
        dist = 1.05 * max(ext[1] / 2.0 / tan, ext[0] / 2.0 / (tan * aspect))
        centre = 0.5 * (lo + hi)
        self.widget3d.look_at(centre, centre + np.array([0.0, 0.0, dist]),
                              [0.0, 1.0, 0.0])

    def _on_layout(self, ctx):
        r = self.window.content_rect
        c = self.gui.Widget.Constraints()
        c.width, c.height = r.width, r.height
        bar = min(self.panel.calc_preferred_size(ctx, c).height, r.height // 3)
        self.widget3d.frame = self.gui.Rect(r.x, r.y, r.width, r.height - bar)
        self.panel.frame = self.gui.Rect(r.x, r.y + r.height - bar, r.width, bar)
        if not self._framed:      # first layout: now the scene rect is real
            self._framed = True
            self.reset_view()

    # -- input -----------------------------------------------------------
    # -- actions (wired to the buttons above) ----------------------------
    def cycle_colour(self):
        self.mode = COLOUR_MODES[(COLOUR_MODES.index(self.mode) + 1)
                                 % len(COLOUR_MODES)]
        self._repaint()

    def reframe(self):
        self.reset_view()
        self.window.post_redraw()

    def _show_pick(self, row):
        """Outline the picked voxel and report it. row=None clears both."""
        if row == self._picked:
            return
        self._picked = row
        if row is None:
            if self.highlight:
                self.widget3d.scene.show_geometry("pick", False)
            self.info.text = self.idle_text
        else:
            if self.highlight:
                t = np.eye(4)
                t[:3, 3] = self.view_idx[row]
                self.widget3d.scene.set_geometry_transform("pick", t)
                self.widget3d.scene.show_geometry("pick", True)
            self.info.text = readout(row, self.idx, self.cls,
                                     self.palette['photo'], self.visible,
                                     self.measured, self.source)
        self.window.post_redraw()

    def _pick_at(self, event):
        if _PICK_FAILED:
            return None
        frame = self.widget3d.frame
        # int(): open3d's own mouse_and_point_coord example does this, and the
        # window's coordinates are not guaranteed to arrive as ints on every
        # platform / display scale
        x, y = int(event.x) - frame.x, int(event.y) - frame.y
        if not (0 <= x < frame.width and 0 <= y < frame.height):
            return None
        ray = guard_pick(pick_ray, self.widget3d.scene.camera,
                         x, y, frame.width, frame.height)
        if ray is None:            # guard_pick returns None, not a pair
            return None
        origin, direction = ray
        if origin is None:
            return None
        return guard_pick(raycast, self.lookup, self.lookup_lo, origin, direction)

    def _on_mouse(self, event):
        gui = self.gui
        moved = event.type == gui.MouseEvent.Type.MOVE
        clicked = (event.type == gui.MouseEvent.Type.BUTTON_DOWN
                   and event.is_modifier_down(gui.KeyModifier.CTRL))
        if not (moved or clicked):
            return gui.Widget.EventCallbackResult.IGNORED

        row = self._pick_at(event)
        self._show_pick(row)
        if clicked and row is not None:
            print(self.info.text, flush=True)
        # MOVE must stay IGNORED or the camera controls stop seeing drags
        return (gui.Widget.EventCallbackResult.HANDLED if clicked
                else gui.Widget.EventCallbackResult.IGNORED)


def highlight_box(half=0.52):
    """Wireframe cube marking the picked voxel. Slightly larger than the voxel
    so its edges clear the cube faces instead of z-fighting them."""
    c = _C * (2.0 * half)
    lines = [[0, 1], [1, 2], [2, 3], [3, 0], [4, 5], [5, 6], [6, 7], [7, 4],
             [0, 4], [1, 5], [2, 6], [3, 7]]
    ls = o3d.geometry.LineSet(o3d.utility.Vector3dVector(c),
                              o3d.utility.Vector2iVector(lines))
    ls.colors = o3d.utility.Vector3dVector(np.tile([1.0, 1.0, 0.25], (len(lines), 1)))
    return ls


def run_gui(*a, **kw):
    from open3d.visualization import gui
    app = gui.Application.instance
    app.initialize()
    Viewer(*a, **kw)
    app.run()


def screenshot(mesh, view_idx, path, width=1600, height=1000):
    """Offscreen render of the same view -- for checking the result without a
    display, and for pasting into a report."""
    from open3d.visualization import rendering
    ren = rendering.OffscreenRenderer(width, height)
    ren.scene.set_background([0.05, 0.05, 0.08, 1.0])
    ren.scene.view.set_post_processing(False)
    mat = rendering.MaterialRecord()
    mat.shader = "defaultUnlit"
    ren.scene.add_geometry("vox", mesh, mat)

    pts = view_idx.astype(np.float64)
    lo, hi = pts.min(axis=0) - 0.5, pts.max(axis=0) + 0.5
    ext, tan = hi - lo, float(np.tan(np.radians(30.0)))
    dist = 1.05 * max(ext[1] / 2.0 / tan, ext[0] / 2.0 / (tan * width / height))
    centre = 0.5 * (lo + hi)
    ren.setup_camera(60.0, centre, centre + np.array([0.0, 0.0, dist]),
                     [0.0, 1.0, 0.0])
    o3d.io.write_image(path, ren.render_to_image())
    print("wrote %s" % path)


# ------------------------------------------------------------------- driver

def main():
    ap = argparse.ArgumentParser(
        description=__doc__.split('*** ORIENTATION')[0],
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('ply_path')
    ap.add_argument('--colour', '--color', dest='colour', default='photo',
                    choices=COLOUR_MODES,
                    help="photo: the colour the camera saw; class: "
                         "display.py's flat per-class colours (default: photo)")
    ap.add_argument('--hidden', default=None, choices=HIDDEN_MODES,
                    help="voxels that got no photo colour: their dimmed class "
                         "colour, or the colour of the nearest coloured voxel "
                         "of the same class. Default: class, except under "
                         "--visibility all, whose whole point is to colour "
                         "every voxel it can, where it is nearest")
    ap.add_argument('--visibility', default='points',
                    choices=VISIBILITY_MODES,
                    help="how a voxel's colour is sourced -- see "
                         "photo_colours. points (default): only the measured "
                         "points INSIDE the voxel, so no voxel is ever "
                         "coloured from another's measurements. surface: also "
                         "takes points within --reach, which fills the fringe "
                         "but is an inference. zbuffer: front-most voxel over "
                         "each pixel, the one rule needing no depth frame. "
                         "all: the three stacked, each voxel taking the first "
                         "rule that reaches it, with the photo stage gated on "
                         "--agree")
    ap.add_argument('--reach', type=float, default=0.75, metavar='VOXELS',
                    help="--visibility surface: half-width of the region a "
                         "voxel collects measured points from, in voxels. 0.5 "
                         "is exactly the voxel (= --visibility points); 0.75, "
                         "the default, is 6 cm and picks up a surface running "
                         "along the voxel's face")
    ap.add_argument('--strict-photo', dest='strict', action='store_true',
                    help="--visibility all: stop after the photo pixels the "
                         "depth frame does not contradict, instead of also "
                         "taking the ones it does. Fewer voxels coloured, "
                         "every colour better evidenced")
    ap.add_argument('--agree', type=float, default=IMAGE_AGREE, metavar='METRES',
                    help="--visibility all, photo stage: how far the depth "
                         "frame may disagree with a voxel's own range before "
                         "the pixel it is front-most on is thrown out "
                         "(default one voxel, %.2f m). Pixels with no depth "
                         "are kept either way -- there the photo is the only "
                         "evidence" % IMAGE_AGREE)
    ap.add_argument('--flat', action='store_true',
                    help="no face shading -- flat cubes, harder to read")
    ap.add_argument('--no-cull', dest='cull', action='store_false',
                    help="keep the faces between adjacent voxels (slower, and "
                         "they are invisible from outside anyway)")
    ap.add_argument('--raw-axes', dest='fix_axes', action='store_false',
                    help="show the file's left-handed axes unchanged -- the "
                         "scene is then rendered inside-out, from behind")
    ap.add_argument('--rgb', help="the frame's colour image (default: found "
                                  "next to the capture / NYU frame)")
    ap.add_argument('--bin', help="NYU .bin holding vox_origin + cam_pose")
    ap.add_argument('--meta', help="capture.py meta.json for a live frame")
    ap.add_argument('--camera-height', type=float, default=None,
                    help="override meta.json's camera_height (metres)")
    ap.add_argument('--yaw', type=float, default=None,
                    help="override meta.json's yaw (radians)")
    ap.add_argument('--screenshot', help="render offscreen to this PNG and exit")
    ap.add_argument('--no-highlight', dest='highlight', action='store_false',
                    help="don't outline the voxel under the cursor")
    args = ap.parse_args()
    if args.hidden is None:
        # the switches stay independent -- this only picks a default, and
        # --hidden class puts it back
        args.hidden = 'nearest' if args.visibility == 'all' else 'class'

    idx, class_rgb, cls = load_voxels(args.ply_path)
    print("Loaded %s" % args.ply_path)
    print(describe(idx, cls))

    view_idx = to_view_index(idx) if args.fix_axes else idx

    frame = resolve_frame(args.ply_path, args)
    if frame is None:
        if args.colour != 'class':
            print("No colour image / camera pose found for this ply -- falling "
                  "back to --colour class. Pass --rgb with --bin or --meta to "
                  "paint the voxels with what the camera saw.")
        args.colour = 'class'
        photo, visible = class_rgb, np.ones(len(idx), dtype=bool)
        measured = np.zeros(len(idx), dtype=bool)
        source = np.full(len(idx), SRC_NONE)
    else:
        print("Camera geometry from %s" % frame['source'])
        print("Photo from %s" % os.path.relpath(frame['rgb_path'],
                                                os.path.dirname(os.path.abspath(args.ply_path))))
        dep = None
        if args.visibility != 'zbuffer':
            if not frame.get('depth_path'):
                raise SystemExit(
                    "--visibility %s needs the frame's depth, which was not "
                    "found. Pass --depth, or use --visibility zbuffer, which "
                    "works from the colour image alone." % args.visibility)
            dep = load_depth(frame['depth_path'], frame['nyu'])
            print("Depth       %s" % frame['depth_path'])
        photo, visible, measured, source = photo_colours(
            idx, frame['vox_origin'], frame['cam_pose'], frame['cam_K'],
            frame['image'], dep, mode=args.visibility,
            reach=args.reach, agree=args.agree, strict=args.strict)
        if args.visibility == 'zbuffer':
            print("%d/%d voxels (%.1f%%) are front-most over some pixel and "
                  "take the photo colour" % (visible.sum(), len(idx),
                                             100.0 * visible.mean()))
        else:
            # one line per rule that supplied any colour, in the order the
            # rules are tried, so --visibility all shows what each stage added
            for code, why in (
                    (SRC_INSIDE, "hold a measured point of their own"),
                    (SRC_REACH, "take a neighbour's measurement, within "
                                "%.3g voxel" % args.reach),
                    (SRC_IMAGE, "are front-most on photo pixels the depth "
                                "frame does not contradict"),
                    (SRC_IMAGE_FAR, "are front-most only on photo pixels whose "
                                    "depth measures something else")):
                k = int((source == code).sum())
                if k:
                    print("%d/%d voxels (%.1f%%) %s"
                          % (k, len(idx), 100.0 * k / len(idx), why))
        print("%d/%d voxels (%.1f%%) end up with a photo colour from the "
              "frame" % (visible.sum(), len(idx), 100.0 * visible.mean()))
        photo, borrowed = fill_hidden(photo, visible, class_rgb, view_idx,
                                      args.hidden)
        if borrowed.any():
            print("%d more take the colour of the nearest coloured voxel of "
                  "their own class (--hidden nearest), leaving %d on the flat "
                  "class colour"
                  % (int(borrowed.sum()),
                     int((~visible & ~borrowed).sum())))
        else:
            print("the other %d are coloured --hidden %s"
                  % (int((~visible).sum()), args.hidden))

    palette = {'photo': photo, 'class': class_rgb}

    build = build_cube_mesh if args.cull else build_cube_mesh_all
    mesh, vox_of_vert, shade = build(view_idx, palette[args.colour],
                                     shaded=not args.flat)
    print("%d triangles" % len(np.asarray(mesh.triangles)))

    if args.screenshot:
        screenshot(mesh, view_idx, args.screenshot)
        return

    title = (os.path.basename(args.ply_path)
             + ('' if args.fix_axes else '  [raw axes]'))
    run_gui(mesh, vox_of_vert, shade, view_idx, idx, cls, palette, visible,
            title, args.colour, highlight=args.highlight, measured=measured,
            source=source)


def build_cube_mesh_all(view_idx, colours, shaded=True):
    """--no-cull: every face of every cube, interior included."""
    verts, tris, vox_of_vert, shade_of_vert = [], [], [], []
    base = 0
    centres = view_idx.astype(np.float64)
    n = len(view_idx)
    for _, corners, shade in FACES:
        quad = centres[:, None, :] + _C[corners][None, :, :]
        verts.append(quad.reshape(-1, 3))
        off = base + 4 * np.arange(n)[:, None]
        tris.append(np.concatenate([off + [0, 1, 2], off + [0, 2, 3]], axis=0))
        vox_of_vert.append(np.repeat(np.arange(n), 4))
        shade_of_vert.append(np.full(4 * n, shade if shaded else 1.0))
        base += 4 * n
    mesh = o3d.geometry.TriangleMesh(
        o3d.utility.Vector3dVector(np.concatenate(verts)),
        o3d.utility.Vector3iVector(np.concatenate(tris).astype(np.int32)))
    vox_of_vert = np.concatenate(vox_of_vert)
    shade_of_vert = np.concatenate(shade_of_vert)
    paint_mesh(mesh, colours, vox_of_vert, shade_of_vert)
    return mesh, vox_of_vert, shade_of_vert


if __name__ == '__main__':
    main()
