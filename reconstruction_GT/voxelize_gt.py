"""
reconstruction_GT/voxelize_gt.py

Turn the solids you placed against a room mesh into the two arrays the
evaluator reads: `label3d` (60x36x60) and `label_weight`. MAKING_GT.md steps
6-9; verify_voxelizer.py proves this file reproduces NYU's own files.

    python -m reconstruction_GT.voxelize_gt captures/room06 \
        --solids captures/room06/gt/solids.json

The chain, and where each piece comes from:

    solids.json (yours, per ROOM)  --+
                                     +--> 240x144x240 labels --+
    vox_origin (this FRAME's grid) --+                         |
                                                               +--> label3d
    depth PNG -> FrameLoader -> high-res TSDF ------------------+    (60x36x60)
                                                               |
                                                               +--> label_weight

`label3d` never touches depth: it is your annotation, voxelized. The TSDF is
the model's own input, and enters only through `label_weight`'s occluded term
(`tsdf_low < -0.5`) -- which is why GT is per FRAME even though the solids are
per ROOM. See MAKING_GT.md, "The frame contract".

*** THE GRID ***

Axis-aligned with the world, so only vox_origin places it (cam_pose does not
enter here). Array (240,144,240) is indexed [gz, gy, gx] and packs C-order to
SSCNet's `flat = gz*240*144 + gy*240 + gx`, with

    world X = vox_origin[0] + (gz + 0.5) * unit      axis 0
    world Z = vox_origin[2] + (gy + 0.5) * unit      axis 1  <- HEIGHT
    world Y = vox_origin[1] + (gx + 0.5) * unit      axis 2  <- depth

Get this wrong and nothing downstream complains; it just scores badly. That is
what verify_voxelizer.py exists to rule out.

*** SOLIDS FILE ***

One per room, in the world frame (Z up, Z=0 the floor). Classes by name from
CLASSES, or by id.

    {
      "room": {"center": [0.2, 2.1, 1.28], "size": [4.2, 4.3, 2.55],
               "yaw_deg": 40.0, "thickness": 0.04},
      "solids": [
        {"class": "furn",  "center": [0.4, 2.1, 0.48], "size": [0.94, 0.55, 0.96],
         "yaw_deg": 12.0},
        {"class": "chair", "center": [-0.6, 1.7, 0.45], "size": [0.5, 0.5, 0.9]},
        {"class": "objs",  "shape": "cylinder", "center": [1.0, 2.0, 0.2],
         "size": [0.3, 0.3, 0.4]},
        {"class": "sofa",  "mesh": "sofa_01.ply"}
      ]
    }

`shape` is "box" (the default), "cylinder" (vertical, `size` = the two
diameters and the height), "wedge" (a ramp filling the box from full height at
its -x face to nothing at +x) or "mesh". All but mesh use center/size/yaw_deg.

`room` writes the floor, ceiling and wall slabs (classes 2, 1, 3) of
`thickness` metres, marks its interior 0 (empty) and everything outside it 255.
NYU's floor is 2 voxels (4 cm) thick and straddles z = 0; that is the default.
Give it `center`/`size`/`yaw_deg` so the walls follow the room rather than the
world axes (the older `min`/`max` form still works, unrotated). You do NOT
annotate floor, ceiling or walls yourself -- the shell is all three.

`"ceiling": false` leaves the ceiling out, for a sweep that never looked up:
the walls and the empty interior then run to the top of the grid, which is what
NYU does on the third of its frames that carry no ceiling.

A `mesh` solid is filled by ray-casting occupancy, so the mesh must be closed.
An open scan surface will come out hollow -- the failure MAKING_GT.md's
"Annotate solids, not surfaces" is about.

Solids are written in order, so a later one overwrites an earlier one where
they overlap; the room slabs go down first.
"""

import os
import sys
import json
import logging
import argparse

import numpy as np

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_THIS_DIR)
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

log = logging.getLogger(__name__)

CLASSES = ['empty', 'ceiling', 'floor', 'wall', 'window', 'chair', 'bed',
           'sofa', 'table', 'tvs', 'furn', 'objs']
NAME_TO_ID = {n: i for i, n in enumerate(CLASSES)}
IGNORE = 255

VOX_UNIT_HI = 0.02
VOX_SIZE_HI = (240, 144, 240)        # (gz, gy, gx) = (world X, world Z, world Y)
SAMPLE_RATIO = 4
VOX_UNIT_LOW = VOX_UNIT_HI * SAMPLE_RATIO
VOX_SIZE_LOW = tuple(n // SAMPLE_RATIO for n in VOX_SIZE_HI)
IMG_H, IMG_W = 480, 640              # what capture.py writes and the model gets


# ---------------------------------------------------------------------------
class Grid(object):
    """A voxel grid axis-aligned with the world.

    `shape` is the array shape (gz, gy, gx); `vox_origin` is the world (X, Y, Z)
    of the grid's low corner, as SSCNet's .bin header stores it.
    """

    def __init__(self, vox_origin, shape=VOX_SIZE_HI, unit=VOX_UNIT_HI):
        self.vox_origin = np.asarray(vox_origin, np.float64)
        self.shape = tuple(int(n) for n in shape)
        self.unit = float(unit)

    def __repr__(self):
        return 'Grid(origin=%s, shape=%s, unit=%g)' % (
            np.round(self.vox_origin, 3).tolist(), self.shape, self.unit)

    # -- axis order lives in these two methods and nowhere else --
    def axis_coords(self):
        """1-D world coordinates of voxel centres along each ARRAY axis:
        (axis0 = world X, axis1 = world Z, axis2 = world Y)."""
        nz, ny, nx = self.shape
        return (self.vox_origin[0] + (np.arange(nz) + 0.5) * self.unit,
                self.vox_origin[2] + (np.arange(ny) + 0.5) * self.unit,
                self.vox_origin[1] + (np.arange(nx) + 0.5) * self.unit)

    def world_to_index(self, pts):
        """World (N,3) XYZ -> integer (N,3) array indices (gz, gy, gx).
        Indices may fall outside the grid; the caller clips."""
        pts = np.asarray(pts, np.float64)
        g = (pts - self.vox_origin) / self.unit
        return np.stack([np.floor(g[:, 0]), np.floor(g[:, 2]),
                         np.floor(g[:, 1])], axis=1).astype(np.int64)

    def index_to_world(self, idx):
        """Array indices (N,3) (gz, gy, gx) -> world (N,3) XYZ voxel centres."""
        idx = np.asarray(idx, np.float64)
        o, u = self.vox_origin, self.unit
        return np.stack([o[0] + (idx[:, 0] + 0.5) * u,
                         o[1] + (idx[:, 2] + 0.5) * u,
                         o[2] + (idx[:, 1] + 0.5) * u], axis=1)

    def aabb_slices(self, lo, hi):
        """World AABB -> array slices covering every voxel whose CENTRE could
        lie inside, clipped to the grid. Returns None if the box misses it."""
        lo, hi = np.asarray(lo, np.float64), np.asarray(hi, np.float64)
        i0 = self.world_to_index(lo[None])[0]
        i1 = self.world_to_index(hi[None])[0] + 1
        i0 = np.maximum(i0, 0)
        i1 = np.minimum(i1, self.shape)
        if np.any(i1 <= i0):
            return None
        return tuple(slice(int(a), int(b)) for a, b in zip(i0, i1))

    def block_centres(self, slices):
        """World coordinates of the voxel centres in a sub-block: (n0,n1,n2,3)."""
        cz, cy, cx = self.axis_coords()
        X = cz[slices[0]]
        Z = cy[slices[1]]
        Y = cx[slices[2]]
        out = np.empty((X.size, Z.size, Y.size, 3), np.float64)
        out[..., 0] = X[:, None, None]
        out[..., 1] = Y[None, None, :]
        out[..., 2] = Z[None, :, None]
        return out


def _yaw_matrix(deg):
    c, s = np.cos(np.radians(deg)), np.sin(np.radians(deg))
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def solid_rotation(spec):
    """A solid's 3x3 rotation.

    `rotation` (a 3x3 list) if present, else `yaw_deg` about the vertical. Full
    rotations are stored as a matrix rather than Euler angles: a cylinder laid
    on its side and then turned is awkward to write as angles and trivial to
    write as the matrix the editor already has.
    """
    if 'rotation' in spec:
        R = np.asarray(spec['rotation'], np.float64).reshape(3, 3)
        if abs(np.linalg.det(R) - 1.0) > 1e-3:
            raise SystemExit('rotation is not a rotation matrix (det %.3f)'
                             % np.linalg.det(R))
        return R
    return _yaw_matrix(float(spec.get('yaw_deg', 0.0)))


def _class_id(spec):
    c = spec.get('class', spec.get('label'))
    if isinstance(c, str):
        if c not in NAME_TO_ID:
            raise SystemExit('unknown class %r; expected one of %s'
                             % (c, ', '.join(CLASSES[1:])))
        return NAME_TO_ID[c]
    c = int(c)
    if not (1 <= c <= 11):
        raise SystemExit('class id %d out of range 1..11' % c)
    return c


# ---------------------------------------------------------------------------
def paint_box(vol, grid, centre, size, rot=0.0, value=1):
    """Fill an oriented box with `value`. `rot` is a 3x3 matrix or a yaw."""
    sl, local, half = _local_coords(grid, centre, size, rot)
    if sl is None:
        return 0
    inside = np.all(np.abs(local) <= half + 1e-9, axis=-1)
    vol[sl][inside] = value
    return int(inside.sum())


def _local_coords(grid, centre, size, rot):
    """-> (slices, local coords in the solid's own frame, half extents).

    Shared by every primitive: each one is a test on `local`. `rot` is a 3x3
    matrix or a yaw in degrees.
    """
    centre = np.asarray(centre, np.float64)
    half = np.asarray(size, np.float64) / 2.0
    R = rot if isinstance(rot, np.ndarray) else _yaw_matrix(float(rot))
    reach = np.abs(R) @ half
    sl = grid.aabb_slices(centre - reach, centre + reach)
    if sl is None:
        return None, None, half
    local = (grid.block_centres(sl) - centre) @ R
    return sl, local, half


def paint_cylinder(vol, grid, centre, size, rot=0.0, value=1):
    """Elliptic cylinder about the solid's own z: size is (diameter x,
    diameter y, length). Rotate it to lay it on its side."""
    sl, local, half = _local_coords(grid, centre, size, rot)
    if sl is None:
        return 0
    r2 = (local[..., 0] / max(half[0], 1e-9)) ** 2 + \
         (local[..., 1] / max(half[1], 1e-9)) ** 2
    inside = (r2 <= 1.0) & (np.abs(local[..., 2]) <= half[2] + 1e-9)
    vol[sl][inside] = value
    return int(inside.sum())


def paint_wedge(vol, grid, centre, size, rot=0.0, value=1):
    """Right triangular prism inside the same box: full height at local -x,
    tapering to nothing at +x. A ramp, for sloped things like a duvet edge or
    a desk return."""
    sl, local, half = _local_coords(grid, centre, size, rot)
    if sl is None:
        return 0
    t = (local[..., 0] + half[0]) / max(2.0 * half[0], 1e-9)      # 0 at -x, 1 at +x
    top = half[2] - 2.0 * half[2] * t
    inside = ((np.abs(local[..., 0]) <= half[0] + 1e-9) &
              (np.abs(local[..., 1]) <= half[1] + 1e-9) &
              (local[..., 2] >= -half[2] - 1e-9) & (local[..., 2] <= top + 1e-9))
    vol[sl][inside] = value
    return int(inside.sum())


def mesh_transform(mesh, centre, size, rot):
    """Put a CAD mesh inside the box the editor placed: scale its bounding box
    to `size`, rotate, then move it to `centre`."""
    R = rot if isinstance(rot, np.ndarray) else _yaw_matrix(float(rot))
    b = mesh.get_axis_aligned_bounding_box()
    extent = np.maximum(b.get_extent(), 1e-6)
    S = np.diag(np.asarray(size, np.float64) / extent)
    T = np.eye(4)
    T[:3, :3] = R @ S
    T[:3, 3] = np.asarray(centre, np.float64) - (R @ S) @ b.get_center()
    return T


def paint_mesh(vol, grid, path, value, transform=None, centre=None,
               size=None, rot=0.0):
    """Fill a CLOSED mesh by ray-cast occupancy."""
    import open3d as o3d
    mesh = o3d.io.read_triangle_mesh(path)
    if len(mesh.vertices) == 0:
        raise SystemExit('empty or unreadable mesh: %s' % path)
    if transform is None and centre is not None:
        transform = mesh_transform(mesh, centre, size, rot)
    if transform is not None:
        mesh.transform(np.asarray(transform, np.float64))
    if not mesh.is_watertight():
        log.warning('%s is not watertight: occupancy may leak, leaving the '
                    'solid hollow or the room flooded. Prefer a box.', path)
    b = mesh.get_axis_aligned_bounding_box()
    sl = grid.aabb_slices(b.min_bound - grid.unit, b.max_bound + grid.unit)
    if sl is None:
        return 0
    pts = grid.block_centres(sl).reshape(-1, 3).astype(np.float32)
    scene = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(mesh))
    occ = scene.compute_occupancy(o3d.core.Tensor(pts)).numpy().astype(bool)
    occ = occ.reshape([s.stop - s.start for s in sl])
    vol[sl][occ] = value
    return int(occ.sum())


PAINTERS = {'box': lambda *a: paint_box(*a), 'cylinder': paint_cylinder,
            'wedge': paint_wedge}


def build_label_volume(spec, grid):
    """solids spec + grid -> uint8 volume of class ids (255 = unannotated)."""
    vol = np.full(grid.shape, IGNORE, np.uint8)
    room = spec.get('room')
    if room is None:
        log.warning('no "room" in the solids file: everything outside your '
                    'objects stays 255, so the metric sees almost no empty '
                    'space. See MAKING_GT.md, "What to mark 255".')
    else:
        th = float(room.get('thickness', 2 * grid.unit))
        # the same rule every other solid follows: a 3x3 `rotation` if the
        # file has one, else `yaw_deg`. A room carried into another frame's
        # world (frame_meta's world_from_room) arrives as a matrix.
        R = solid_rotation(room)
        if 'center' in room:                       # yawed form
            c = np.asarray(room['center'], np.float64)
            size = np.asarray(room['size'], np.float64)
        else:                                      # axis-aligned min/max form
            lo = np.asarray(room['min'], np.float64)
            hi = np.asarray(room['max'], np.float64)
            c, size = (lo + hi) / 2.0, hi - lo
        # A sweep that never saw the ceiling should not invent one. NYU does
        # the same on a third of its frames: measured over 24 frames, the space
        # above 2.3 m is ~44% empty and ~55% 255 whether or not a ceiling is
        # annotated, and empty runs to the top of the grid. So 255 comes from
        # being outside the WALLS, never from a missing ceiling -- with no
        # ceiling the interior simply continues upward.
        want_ceiling = room.get('ceiling', True)
        if not want_ceiling:
            top_of_grid = grid.vox_origin[2] + grid.shape[1] * grid.unit
            bottom = c[2] - size[2] / 2.0
            size = size.copy()
            size[2] = top_of_grid - bottom
            c = c.copy()
            c[2] = bottom + size[2] / 2.0

        # interior first: everything inside the shell is empty unless something
        # is placed in it. Outside stays 255.
        paint_box(vol, grid, c, size, R, 0)
        # then the shell itself, as slabs of `thickness` on each face, matching
        # NYU's 2-voxel floor straddling z = 0. Walls follow the room's own yaw,
        # because a room is rarely square to the camera that captured it.
        faces = [(2, -1, 'floor'), (0, -1, 'wall'), (0, +1, 'wall'),
                 (1, -1, 'wall'), (1, +1, 'wall')]
        if want_ceiling:
            faces.insert(1, (2, +1, 'ceiling'))
        for axis, sign, cls in faces:
            cc = c + R @ (np.eye(3)[axis] * sign * size[axis] / 2.0)
            ss = size.copy()
            ss[axis] = th
            paint_box(vol, grid, cc, ss, R, NAME_TO_ID[cls])
    for s in spec.get('solids', []):
        cid = _class_id(s)
        shape = s.get('shape', 'mesh' if 'mesh' in s else 'box')
        R = solid_rotation(s)
        if shape == 'mesh':
            path = s['mesh']
            if not os.path.isabs(path):
                path = os.path.join(spec.get('_dir', '.'), path)
            n = paint_mesh(vol, grid, path, cid, s.get('transform'),
                           s.get('center'), s.get('size'), R)
        elif shape in PAINTERS:
            n = PAINTERS[shape](vol, grid, s['center'], s['size'], R, cid)
        else:
            raise SystemExit('unknown shape %r; expected one of %s, mesh'
                             % (shape, ', '.join(PAINTERS)))
        log.info('  %-8s %-8s %7d voxels %s', CLASSES[cid], shape, n,
                 '(MISSES THE GRID)' if n == 0 else '')
    return vol


# ---------------------------------------------------------------------------
def downsample_label(hi_label, tsdf_hi, ratio=SAMPLE_RATIO):
    """SSCNet's downSample__ (../data/utils/datautil.cu:166) in numpy.

    Per 4x4x4 block of 64: mean TSDF, and a class vote that is NOT a plain
    mode --

      if empty + ignore > 0.95*64 (>= 61):  argmax over all 13 bins
      else:                                 argmax over classes 1..11 only

    so ~4 occupied sub-voxels already force the output voxel occupied. Ties go
    to the lowest class id, as the kernel's strict `<` does.

    Returns (label int32 flat, tsdf_downsample float32 flat).
    """
    hi = np.asarray(hi_label).reshape(VOX_SIZE_HI)
    t = np.asarray(tsdf_hi, np.float32).reshape(VOX_SIZE_HI)
    r = ratio
    out_shape = tuple(n // r for n in VOX_SIZE_HI)
    blocks = hi.reshape(out_shape[0], r, out_shape[1], r, out_shape[2], r)
    blocks = blocks.transpose(0, 2, 4, 1, 3, 5).reshape(-1, r ** 3)
    tb = t.reshape(out_shape[0], r, out_shape[1], r, out_shape[2], r)
    tsdf_ds = tb.transpose(0, 2, 4, 1, 3, 5).reshape(-1, r ** 3).mean(axis=1)

    b = blocks.astype(np.int64).copy()
    b[b == IGNORE] = 12                                  # bin 12 == ignore
    counts = np.zeros((b.shape[0], 13), np.int64)
    np.add.at(counts, (np.arange(b.shape[0])[:, None], b), 1)

    mostly_void = (counts[:, 0] + counts[:, 12]) > 0.95 * (r ** 3)
    lab_all = counts.argmax(axis=1)                      # ties -> lowest index
    lab_obj = counts[:, 1:12].argmax(axis=1) + 1
    label = np.where(mostly_void, lab_all, lab_obj)
    label = np.where(label == 12, IGNORE, label)
    return label.astype(np.int32), tsdf_ds.astype(np.float32)


def compute_label_weight(label, tsdf_ds):
    """NYU's scoring mask: annotated (1..253) OR observed-empty (tsdf < -0.5).

    Deterministic. NOT `reprojection.getLabelWeight`, which samples background
    at random -- see MAKING_GT.md Gotcha 7.
    """
    lw = np.zeros(np.shape(label), np.float32)
    lw[np.abs(127 - np.asarray(label)) < 127] = 1.0
    lw[np.asarray(tsdf_ds) < -0.5] = 1.0
    return lw


# ---------------------------------------------------------------------------
def frustum_mask(grid, cam_K, cam_pose, img_hw=(IMG_H, IMG_W)):
    """True where a voxel centre projects inside the image the model is given.

    The grid is 4.8 m wide whatever the lens does, so its corners sit outside a
    63 x 49 degree crop. NYU gets away with ignoring this -- 95-99.6% of its
    SCORED voxels land in frame anyway -- but a camera at desk height in a
    small room does not: the floor only enters view beyond ~2.3 m and the
    ceiling beyond ~3.3 m, so both are annotated and neither was ever seen.
    Occ-ScanNet ships the same idea as its `target_fov`.
    """
    H, W = img_hw
    cz, cy, cx = grid.axis_coords()
    X = cz[:, None, None] * np.ones((1, len(cy), len(cx)))
    Z = cy[None, :, None] * np.ones((len(cz), 1, len(cx)))
    Y = cx[None, None, :] * np.ones((len(cz), len(cy), 1))
    w = np.stack([X, Y, Z], axis=-1).reshape(-1, 3)
    cam = (w - np.asarray(cam_pose)[:3, 3]) @ np.asarray(cam_pose)[:3, :3]
    z = cam[:, 2]
    ok = z > 1e-6
    u = np.where(ok, cam_K[0, 0] * cam[:, 0] / np.where(ok, z, 1) + cam_K[0, 2], -1)
    v = np.where(ok, cam_K[1, 1] * cam[:, 1] / np.where(ok, z, 1) + cam_K[1, 2], -1)
    return (ok & (u >= 0) & (u < W) & (v >= 0) & (v < H)).reshape(grid.shape)


def transform_solids(spec, A):
    """Move a solids spec into another world frame. A is 4x4, new <- old.

    The annotation is made once, against the fused room, so it is expressed in
    the ROOM's world. A frame exported from the sweep sits in its own
    gravity-aligned world (origin under its own camera), and its
    `world_from_room` in meta.json is the A that carries the room's solids
    there -- see frame_loader.frame_meta.
    """
    A = np.asarray(A, np.float64)
    R, t = A[:3, :3], A[:3, 3]
    out = {k: v for k, v in spec.items() if k not in ('room', 'solids')}
    out['solids'] = []
    for key, src in (('room', [spec['room']] if 'room' in spec else []),
                     ('solids', spec.get('solids', []))):
        moved = []
        for sld in src:
            s2 = dict(sld)
            s2['center'] = (R @ np.asarray(sld['center'], np.float64) + t).tolist()
            s2['rotation'] = (R @ solid_rotation(sld)).tolist()
            s2.pop('yaw_deg', None)
            if 'transform' in sld:               # an explicit CAD placement
                s2['transform'] = (A @ np.asarray(sld['transform'],
                                                  np.float64)).tolist()
            moved.append(s2)
        if key == 'room':
            if moved:
                out['room'] = moved[0]
        else:
            out['solids'] = moved
    return out


def load_solids(path):
    with open(path) as f:
        spec = json.load(f)
    spec['_dir'] = os.path.dirname(os.path.abspath(path))
    return spec


def main():
    from inference.utils import setup_logging
    setup_logging()
    p = argparse.ArgumentParser(
        description='Voxelize annotation solids into CleanerS ground truth.')
    p.add_argument('capture_dir', help='capture folder with meta.json and depth/')
    p.add_argument('--solids', default=None,
                   help='solids JSON (default <capture_dir>/gt/solids.json)')
    p.add_argument('--frame', default=None,
                   help='frame stem (default: every frame in meta.json)')
    p.add_argument('--out_dir', default=None,
                   help='where Label/ and TSDF/ go (default <capture_dir>/gt)')
    p.add_argument('--fov', choices=['all', 'frustum'], default='all',
                   help="'all' (default) keeps every annotated voxel, like "
                        "Occ-ScanNet's `target`. 'frustum' bakes the "
                        "camera-view mask into the file, like its `target_fov`. "
                        "Leave it on 'all': the file then holds everything you "
                        "annotated, and the EVALUATOR applies the frustum, so "
                        "one ground truth serves both protocols. "
                        "`frustum_mask()` here is what it calls")
    args = p.parse_args()

    import cv2
    from inference.frame_loader import (FrameLoader, frame_meta,
                                        load_cuda_encoder)

    cap = args.capture_dir
    solids_path = args.solids or os.path.join(cap, 'gt', 'solids.json')
    if not os.path.exists(solids_path):
        raise SystemExit('no solids file at %s -- place solids first '
                         '(MAKING_GT.md step 5)' % solids_path)
    with open(os.path.join(cap, 'meta.json')) as f:
        meta = json.load(f)
    spec = load_solids(solids_path)
    out_dir = args.out_dir or os.path.join(cap, 'gt')
    for sub in ('Label', 'TSDF', 'Mapping'):
        os.makedirs(os.path.join(out_dir, sub), exist_ok=True)

    frames = [args.frame] if args.frame else meta['frames']
    for stem in frames:
        depth_path = os.path.join(cap, 'depth', stem + '.png')
        depth = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
        if depth is None:
            raise SystemExit('cannot read %s' % depth_path)
        # the capture's record, overridden by whatever this frame states --
        # a frame exported from the sweep was shot from its own height and
        # tilt, and reaches the room's solids through `world_from_room`
        fm = frame_meta(meta, stem)
        fspec = (transform_solids(spec, fm['world_from_room'])
                 if 'world_from_room' in fm else spec)
        fspec['_dir'] = spec.get('_dir')
        loader = FrameLoader.from_live_camera(
            depth.astype(np.float32) / 1000.0,          # capture.py: plain mm
            cam_K=np.asarray(fm['cam_K'], np.float32),
            camera_height=fm['camera_height'], yaw=fm.get('yaw', 0.0),
            drop_invalid_depth=True, up_camera=fm.get('up_camera'))
        loader.encoder = 'cuda' if load_cuda_encoder() is not None else 'numpy'
        sample = loader.build_tsdf_and_mapping()
        # build_tsdf_and_mapping returns only the low-res TSDF; label_weight's
        # occluded term needs the high-res one, block-mean-ed by the same rule
        # SSCNet's downSample__ uses, so take it from the builder directly.
        tsdf_hi, _ = (loader._build_high_res_tsdf_cuda() if loader.encoder == 'cuda'
                      else loader._build_high_res_tsdf())
        tsdf_hi = np.asarray(tsdf_hi, np.float32).reshape(-1)

        grid = Grid(loader.vox_origin, VOX_SIZE_HI, VOX_UNIT_HI)
        log.info('%s: %s', stem, grid)
        hi = build_label_volume(fspec, grid)
        label, tsdf_ds = downsample_label(hi, tsdf_hi)
        # Normally left off: masking here would throw away annotation the
        # file could keep. The evaluator calls frustum_mask() instead, so GT
        # and model input are compared over the same view without the GT
        # itself being cut down to one frame's lens.
        if args.fov == 'frustum':
            low = Grid(loader.vox_origin, VOX_SIZE_LOW, VOX_UNIT_LOW)
            seen = frustum_mask(low, np.asarray(fm['cam_K'], np.float64),
                                loader.cam_pose).reshape(-1)
            hidden = (label != IGNORE) & ~seen
            label = np.where(seen, label, IGNORE).astype(np.int32)
            log.info('%s: %d voxels outside the camera view -> 255 (%.0f%% of '
                     'the grid)', stem, int(hidden.sum()), 100 * (~seen).mean())
        # label_weight AFTER the frustum mask: a voxel the camera never saw
        # must not enter the scored set through the "annotated" term
        lw = compute_label_weight(label, tsdf_ds)

        np.savez_compressed(os.path.join(out_dir, 'Label', stem + '.npz'),
                            arr_0=label)
        np.savez_compressed(os.path.join(out_dir, 'TSDF', stem + '.npz'),
                            arr_0=np.asarray(sample['tsdf'], np.float32).reshape(-1),
                            arr_1=lw)
        # SC scores only voxels no depth pixel reached (mapping == 307200), so
        # the evaluator needs this frame's mapping too -- same layout as NYU's
        # Mapping/ folder.
        np.savez_compressed(os.path.join(out_dir, 'Mapping', stem + '.npz'),
                            arr_0=np.asarray(sample['mapping'], np.int32).reshape(-1))
        occupied = int(((label > 0) & (label != IGNORE)).sum())
        log.info('%s: %d scored voxels (%.1f%%), %d occupied, %d ignore',
                 stem, int(lw.sum()), 100.0 * lw.mean(), occupied,
                 int((label == IGNORE).sum()))
    log.info('wrote %s/{Label,TSDF}', out_dir)


if __name__ == '__main__':
    main()
