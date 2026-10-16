"""
inference/world_map.py

ONE room-sized voxel grid that fills as the camera moves, holding what the
model COMPLETED and what the sensor MEASURED as separate evidence, so the
two can be compared voxel by voxel.

This is the answer to run_live's old limit 1: "the volume is fixed relative
to the camera, so each frame is an independent single-shot completion, not
an accumulating reconstruction". It is no longer fixed to the camera. The
grid is the room, the camera moves through it, and every frame adds to what
is already there instead of replacing it.

*** PURE NUMPY, ON PURPOSE ***
No open3d, no torch, no file paths, no model. Every fusion rule below is
testable on a 4x4x4 grid with none of those installed -- see
tests/test_world_map.py. The pieces that need a camera live in run_live.py;
the pieces that need a renderer live in live_view.py.

*** THE THREE QUESTIONS IT ANSWERS ***
  what did the sensor measure?      measured_state() -> UNKNOWN/FREE/OCCUPIED
  what did the model claim?         predicted()      -> (occupied, class)
  did the sweep settle the claim?   verdicts()       -> confirmed /
                                                        contradicted /
                                                        untested / missed

The third is the whole point. A voxel the model fills in while nothing has
been measured there is a COMPLETION; walking the camera round until the
depth frame reaches it is what turns that completion into a result.
"""

import numpy as np

VOX = 0.08                 # metres; the network's own 4.8 m / 60
N_CLASS = 12               # NYU: 0 empty, 1..11 named, 12 ignore
UNKNOWN, FREE, OCCUPIED = 0, 1, 2


class WorldMap:
    """An 8 cm grid of a whole room, in the world frame live_cam_pose
    builds: gravity-aligned, world Z up, floor at z = 0.

    hit_min / pass_min are in FRAMES, not rays -- see add_measurement.
    """

    def __init__(self, origin, shape, vox=VOX, hit_min=2, pass_min=2):
        self.origin = np.asarray(origin, np.float64).reshape(3)
        self.shape = tuple(int(s) for s in shape)
        self.vox = float(vox)
        self.hit_min, self.pass_min = int(hit_min), int(pass_min)
        n = int(np.prod(self.shape))
        # Counts are per FRAME, so they are bounded by the frame count and
        # uint16 cannot overflow -- and the thresholds read as "how many
        # frames said so", which is a sentence rather than a tuning knob.
        self.n_hit = np.zeros(n, np.uint16)
        self.n_pass = np.zeros(n, np.uint16)
        self.colour_sum = np.zeros((n, 3), np.uint32)
        self.colour_n = np.zeros(n, np.uint32)
        self.votes = np.zeros((n, N_CLASS), np.uint16)
        self.n_seen = np.zeros(n, np.uint16)
        self.first_meas = np.full(n, -1, np.int32)
        self.first_pred = np.full(n, -1, np.int32)
        self.blind = np.zeros(n, bool)

    # NYU's own volume starts here, 5 cm below the floor, and every frame
    # grid shares it (frame_loader.from_live_camera). Gravity never rotates
    # the vertical axis, so putting the world's z origin a whole number of
    # voxels from this one makes the two grids share their z layers exactly
    # -- no resampling at all on the axis the encoder depends on, where
    # floor and ceiling live. x and y cannot be aligned the same way: the
    # camera translates continuously, so there is no constant offset.
    NYU_Z0 = -0.05

    @classmethod
    def around(cls, centre_xy, extent=(16.0, 16.0, 3.04), vox=VOX, **kw):
        """A box centred horizontally on a point, floor one voxel down.

        Fixed rather than growing on purpose: 16 x 16 x 3.04 m at 8 cm is
        200 x 200 x 38 voxels and about 80 MB of channels, which is cheaper
        than owning repad logic that only ever runs at the edges.
        """
        ex, ey, ez = extent
        origin = [centre_xy[0] - ex / 2.0, centre_xy[1] - ey / 2.0,
                  cls.NYU_Z0 - vox]
        shape = tuple(int(round(e / vox)) for e in (ex, ey, ez))
        return cls(origin, shape, vox=vox, **kw)

    # ---- geometry ------------------------------------------------------

    def index_of(self, p_world):
        """World metres -> (grid index, is it inside the box)."""
        p = np.asarray(p_world, np.float64).reshape(-1, 3)
        idx = np.floor((p - self.origin) / self.vox).astype(np.int64)
        inside = np.all((idx >= 0) & (idx < np.asarray(self.shape)), axis=1)
        return idx, inside

    def centres(self, idx):
        """Grid index -> the world coordinate of that voxel's centre."""
        return self.origin + (np.asarray(idx, np.float64) + 0.5) * self.vox

    def _flat(self, idx):
        """(N,3) -> (flat index of the ones inside, the inside mask)."""
        idx = np.asarray(idx, np.int64).reshape(-1, 3)
        inside = np.all((idx >= 0) & (idx < np.asarray(self.shape)), axis=1)
        k = idx[inside]
        _, ny, nz = self.shape
        return k[:, 0] * (ny * nz) + k[:, 1] * nz + k[:, 2], inside

    # ---- what the sensor measured --------------------------------------

    def add_measurement(self, hit_idx, pass_idx, frame, rgb=None):
        """One frame's evidence: voxels it found a surface in, and voxels
        it measured straight past.

        Deduplicated within the call, so n_hit and n_pass count FRAMES. A
        frame that hit a voxel does not also report passing through it --
        otherwise a surface grazed at the edge of one ray bundle would
        argue with itself.
        """
        hf, hin = self._flat(hit_idx)
        pf, _ = self._flat(pass_idx)
        if rgb is not None and len(hf):
            c = np.asarray(rgb, np.uint32).reshape(-1, 3)[hin]
            np.add.at(self.colour_sum, hf, c)
            np.add.at(self.colour_n, hf, 1)
        hu = np.unique(hf)
        pu = np.unique(pf)
        if len(hu) and len(pu):
            pu = pu[~np.isin(pu, hu)]
        self.n_hit[hu] += 1
        self.n_pass[pu] += 1
        touched = np.concatenate([hu, pu]).astype(np.int64)
        if not len(touched):
            return
        settled = touched[(self.first_meas[touched] < 0)
                          & (self._state()[touched] != UNKNOWN)]
        self.first_meas[settled] = frame

    def _state(self):
        st = np.zeros(self.n_hit.shape, np.uint8)
        st[self.n_pass >= self.pass_min] = FREE
        st[self.n_hit >= self.hit_min] = OCCUPIED
        return st

    def measured_state(self):
        """UNKNOWN / FREE / OCCUPIED per voxel.

        Occupied when hit_min frames put a surface in it, else free when
        pass_min frames measured past it. Hits beat passes: a hit measures
        a surface directly, while a pass can be a ray that grazed it.
        """
        return self._state().reshape(self.shape)

    def colours(self):
        """The mean colour of the pixels that measured each voxel, RGB."""
        n = np.maximum(self.colour_n, 1)[:, None]
        return (self.colour_sum // n).astype(np.uint8).reshape(
            self.shape + (3,))

    # ---- what the model claimed ----------------------------------------

    def add_prediction(self, idx, cls, frame):
        """One frame's answer for the voxels it can speak about.

        cls: the NYU class per voxel, 0 meaning empty. A voxel counts as
        SEEN whatever the answer -- that is the denominator of the vote.
        """
        cls = np.asarray(cls, np.int64).reshape(-1)
        f, inside = self._flat(idx)
        if not len(f):
            return
        c = cls[inside]
        self.n_seen[np.unique(f)] += 1
        occ = c > 0
        if not occ.any():
            return
        np.add.at(self.votes, (f[occ], c[occ]), 1)
        fo = np.unique(f[occ])
        new = fo[self.first_pred[fo] < 0]
        self.first_pred[new] = frame
        # The flag the whole comparison rests on: had the sensor ANY
        # information about this voxel when the model filled it in?
        #
        # Not "was its state still UNKNOWN" -- that is a different and
        # much weaker question, because a state needs hit_min frames to
        # settle. On the first frame nothing has settled, so every
        # surface the camera is looking straight at would count as a
        # completion and be confirmed two frames later. The display went
        # almost entirely green, which is how this was caught: the model
        # was taking credit for predicting what the camera could see.
        #
        # So: no hit and no pass, not one ray, ever. run_live integrates
        # the measurement BEFORE the prediction within a frame, so a
        # surface this very frame measured is already excluded.
        self.blind[new] = ((self.n_hit[new] == 0)
                           & (self.n_pass[new] == 0))

    def predicted(self):
        """-> (occupied, class), both shaped like the grid.

        Occupied when MORE THAN HALF the frames that could see this voxel
        called it something, and the class is the argmax of those votes.
        """
        total = self.votes[:, 1:].sum(axis=1, dtype=np.int64)
        occ = total * 2 > self.n_seen.astype(np.int64)
        cls = (self.votes[:, 1:].argmax(axis=1) + 1).astype(np.uint8)
        cls[~occ] = 0
        return occ.reshape(self.shape), cls.reshape(self.shape)

    # ---- did the sweep settle it ---------------------------------------

    def verdicts(self):
        """Six boolean grids. Read in ONE order: the vote decides which
        voxels are predicted occupied, and first_pred / blind are then
        consulted only for those. A voxel called occupied by a single frame
        and later outvoted is not a prediction and gets no verdict.

        A COMPLETION is a voxel the model filled in where the sensor had
        no information at all -- not merely where its state had not
        settled. See add_prediction.

        'untested' is normally the MAJORITY and must be reported as its own
        number: most of what a completion model fills in is permanently
        unreachable by the sensor, and that is not a win.
        """
        st = self._state()
        occ = self.predicted()[0].reshape(-1)
        blind = self.blind & occ
        out = {'measured': st == OCCUPIED,
               'predicted': occ,
               'confirmed': blind & (st == OCCUPIED),
               'contradicted': blind & (st == FREE),
               'untested': blind & (st == UNKNOWN),
               'missed': (st == OCCUPIED) & ~occ}
        return dict((k, v.reshape(self.shape)) for k, v in out.items())

    def counts(self):
        return dict((k, int(v.sum())) for k, v in self.verdicts().items())

    # ---- persistence ----------------------------------------------------

    _ARRAYS = ('n_hit', 'n_pass', 'colour_sum', 'colour_n', 'votes',
               'n_seen', 'first_meas', 'first_pred', 'blind')

    def save(self, path):
        kw = dict((k, getattr(self, k)) for k in self._ARRAYS)
        np.savez_compressed(path, origin=self.origin,
                            shape=np.asarray(self.shape), vox=self.vox,
                            hit_min=self.hit_min, pass_min=self.pass_min,
                            **kw)

    @classmethod
    def load(cls, path):
        z = np.load(path)
        m = cls(z['origin'], tuple(int(s) for s in z['shape']),
                vox=float(z['vox']), hit_min=int(z['hit_min']),
                pass_min=int(z['pass_min']))
        for k in cls._ARRAYS:
            setattr(m, k, z[k])
        return m


# ---- one frame of depth -> what it measures -----------------------------

def frustum_box(world, cam_K, world_T_cam, depth_shape, max_range):
    """The grid index range covering the camera's view out to max_range."""
    h, w = depth_shape
    fx, fy = float(cam_K[0, 0]), float(cam_K[1, 1])
    cx, cy = float(cam_K[0, 2]), float(cam_K[1, 2])
    us = np.array([0.0, w, 0.0, w])
    vs = np.array([0.0, 0.0, h, h])
    dirs = np.stack([(us - cx) / fx, (vs - cy) / fy, np.ones(4)], axis=1)
    pts = np.vstack([np.zeros((1, 3)), dirs * max_range])
    T = np.asarray(world_T_cam, np.float64)
    pts = pts @ T[:3, :3].T + T[:3, 3]
    bound = np.asarray(world.shape) - 1
    lo = world.index_of(pts.min(axis=0) - world.vox)[0][0]
    hi = world.index_of(pts.max(axis=0) + world.vox)[0][0]
    return np.clip(lo, 0, bound), np.clip(hi, 0, bound)


def observe(world, depth, cam_K, world_T_cam, max_range=3.5, margin=None,
            rgb=None, z_near=0.15):
    """What one depth frame measures -> (hit_idx, hit_rgb, pass_idx).

    By PROJECTION, not by marching rays. Every world voxel in the frustum's
    bounding box is transformed into camera coordinates and projected into
    the depth image, and its own range compared with what was measured at
    that pixel:

        z    = (cam_T_world @ p_world).z     the voxel's own range
        d    = depth[v, u]                   what the sensor measured
        hit  = |d - z| <= margin             the surface is in this voxel
        free = d - z  >  margin              the voxel is in front of it
        else   unknown                       no depth, out of frame, behind

    Marching 77 k rays at 4 cm out to 3.5 m is about 22 M sample points a
    frame. This is ~300 k voxels in one vectorised pass and yields the
    identical three states, because every voxel between the camera and the
    surface projects to a pixel whose measured depth exceeds its own range.
    It is also the arithmetic the TSDF encoder already does -- see
    document/CONCEPTS.md on turning a world position into camera
    coordinates.
    """
    empty = np.zeros((0, 3), np.int64)
    none_rgb = None if rgb is None else np.zeros((0, 3), np.uint8)
    margin = world.vox / 2.0 if margin is None else float(margin)
    h, w = depth.shape
    lo, hi = frustum_box(world, cam_K, world_T_cam, depth.shape, max_range)
    if np.any(hi < lo):
        return empty, none_rgb, empty

    ax = [np.arange(lo[i], hi[i] + 1) for i in range(3)]
    gx, gy, gz = np.meshgrid(ax[0], ax[1], ax[2], indexing='ij')
    idx = np.stack([gx.ravel(), gy.ravel(), gz.ravel()], axis=1)

    T = np.linalg.inv(np.asarray(world_T_cam, np.float64))
    p = world.centres(idx) @ T[:3, :3].T + T[:3, 3]
    z = p[:, 2]
    with np.errstate(divide='ignore', invalid='ignore'):
        u = np.round(float(cam_K[0, 0]) * p[:, 0] / z + float(cam_K[0, 2]))
        v = np.round(float(cam_K[1, 1]) * p[:, 1] / z + float(cam_K[1, 2]))
    seen = ((z > z_near) & (z <= max_range) & np.isfinite(u) & np.isfinite(v)
            & (u >= 0) & (u < w) & (v >= 0) & (v < h))
    if not seen.any():
        return empty, none_rgb, empty

    idx = idx[seen]
    ui, vi = u[seen].astype(np.int64), v[seen].astype(np.int64)
    d = np.asarray(depth, np.float64)[vi, ui]
    diff = d - z[seen]
    good = (d > 0) & (d <= max_range)
    hit = good & (np.abs(diff) <= margin)
    free = good & (diff > margin)

    hit_rgb = None
    if rgb is not None:
        # the loader hands us BGR; the map stores RGB
        hit_rgb = np.asarray(rgb)[vi[hit], ui[hit]][:, ::-1].copy()
    return idx[hit], hit_rgb, idx[free]


# ---- one frame of prediction -> where it goes in the world --------------

# The network's grid is NOT indexed (X, Y, Z), and assuming it is puts the
# room on its side. frame_loader builds it as [z_g, y_g, x_g], where axis 0
# walks world X, axis 1 walks world Z -- the 36-cell HEIGHT axis, which is
# why the shape is (60, 36, 60) -- and axis 2 walks world Y, the depth away
# from the camera (frame_loader.py:503-507, and document/README.md's axis
# table). So a world-order index (iX, iY, iZ) addresses the array at
# [iX, iZ, iY].
WORLD_TO_ARRAY = [0, 2, 1]


def gather_prediction(world, pred_label, keep, vox_origin, cam_pose,
                      world_T_cam, grid=(60, 36, 60), vox=VOX):
    """The frame's 60x36x60 -> (world voxel indices, class).

    A GATHER: for every world voxel in the frame grid's world bounding box,
    find the frame voxel containing its centre and take its class.
    Scattering the frame grid outward instead aliases at equal resolution --
    some world voxels would take two votes and others none, and the holes
    would read as model failures rather than as arithmetic.

        p_local = cam_pose @ inv(world_T_cam) @ p_world

    `cam_pose` maps camera -> that frame's local world (frame_loader's NYU
    convention) and `world_T_cam` maps camera -> the room's world, so the
    pair carries a world point into the frame's own grid. With
    world_T_cam == cam_pose -- the FIRST frame, which fuse_scan anchors the
    world on -- this collapses to the identity, and the result must land
    exactly where run_inference already puts that frame. That is the test
    that catches a transposed rotation or an inverted pose.

    keep: which frame voxels this frame may speak about at all (in-frustum,
    or occluded behind a measured surface).
    """
    M = np.asarray(cam_pose, np.float64) @ np.linalg.inv(
        np.asarray(world_T_cam, np.float64))
    Mi = np.linalg.inv(M)
    # the grid's extent along world X, Y, Z -- 60, 60, 36, not 60, 36, 60
    dims = np.asarray(grid, np.int64)[WORLD_TO_ARRAY]
    g = dims.astype(np.float64)
    c = np.array(np.meshgrid([0.0, g[0]], [0.0, g[1]], [0.0, g[2]],
                             indexing='ij')).reshape(3, -1).T
    corners = np.asarray(vox_origin, np.float64) + c * vox
    corners = corners @ Mi[:3, :3].T + Mi[:3, 3]
    bound = np.asarray(world.shape) - 1
    lo = np.clip(world.index_of(corners.min(axis=0) - vox)[0][0], 0, bound)
    hi = np.clip(world.index_of(corners.max(axis=0) + vox)[0][0], 0, bound)
    empty = (np.zeros((0, 3), np.int64), np.zeros(0, np.int64))
    if np.any(hi < lo):
        return empty

    ax = [np.arange(lo[i], hi[i] + 1) for i in range(3)]
    gx, gy, gz = np.meshgrid(ax[0], ax[1], ax[2], indexing='ij')
    idx = np.stack([gx.ravel(), gy.ravel(), gz.ravel()], axis=1)

    p = world.centres(idx) @ M[:3, :3].T + M[:3, 3]
    fw = np.floor((p - np.asarray(vox_origin, np.float64)) / vox).astype(
        np.int64)
    ok = np.all((fw >= 0) & (fw < dims), axis=1)
    idx, fw = idx[ok], fw[ok]
    if not len(fw):
        return empty
    f = fw[:, WORLD_TO_ARRAY]
    k = np.asarray(keep)[f[:, 0], f[:, 1], f[:, 2]]
    idx, f = idx[k], f[k]
    return idx, np.asarray(pred_label)[f[:, 0], f[:, 1], f[:, 2]]
