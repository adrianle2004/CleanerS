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

# How much the LATEST look at a voxel counts for when it is coloured.
#
# A running mean over every frame that ever measured a voxel is the wrong
# average, because the frames are not equivalent samples: the camera's
# auto white balance moves for the first seconds of a stream, so the
# earliest looks are the WRONG colour and a mean keeps them forever. On
# room07 that showed as a wall still pink long after the camera had
# settled and the rest of the same wall had gone white.
#
# So each frame's look is blended in with this weight rather than
# averaged. At 0.5 the newest frame is half the colour and anything four
# frames old is under 7% of it -- recent, without being one frame's noise.
# ColourSettle below deals with the other half: the early frames are not
# integrated at all.
COLOUR_ALPHA = 0.5

# How sure the model has to be before the room shows its claim.
#
# A voxel two frames glanced at and one of them called a chair is not a
# prediction, it is a coincidence, and accumulating those is what makes a
# bed grow spikes. SEEN_MIN is how many frames must have been able to
# speak about a voxel at all, and AGREE is the fraction of them that must
# have called it occupied.
#
# Measured on room07's 808-frame sweep, the same map scored both ways:
# the model's claimed matter goes from 20021 voxels to 17624 while the
# CONFIRMED count moves 3604 -> 3561. So 12% of what the model claimed
# was unstable, 1.2% of what the sweep had proved went with it, and the
# proved share rose from 86.0% to 87.1%. What leaves is what no two
# viewpoints ever agreed on -- including 353 of the 805 voxels the
# sensor had measured straight through.
SEEN_MIN = 3
AGREE = 0.6


class WorldMap:
    """An 8 cm grid of a whole room, in the world frame live_cam_pose
    builds: gravity-aligned, world Z up, floor at z = 0.

    hit_min / pass_min are in FRAMES, not rays -- see add_measurement.
    """

    def __init__(self, origin, shape, vox=VOX, hit_min=2, pass_min=2,
                 seen_min=SEEN_MIN, agree=AGREE, alpha=COLOUR_ALPHA):
        self.origin = np.asarray(origin, np.float64).reshape(3)
        self.shape = tuple(int(s) for s in shape)
        self.vox = float(vox)
        self.hit_min, self.pass_min = int(hit_min), int(pass_min)
        self.seen_min, self.agree = int(seen_min), float(agree)
        self.alpha = float(alpha)
        n = int(np.prod(self.shape))
        # Counts are per FRAME, so they are bounded by the frame count and
        # uint16 cannot overflow -- and the thresholds read as "how many
        # frames said so", which is a sentence rather than a tuning knob.
        self.n_hit = np.zeros(n, np.uint16)
        self.n_pass = np.zeros(n, np.uint16)
        # the blend, not a sum: see COLOUR_ALPHA. colour_n counts the
        # FRAMES that have coloured this voxel, and is only ever asked
        # whether it is zero -- has the camera seen this place at all.
        self.colour_ema = np.zeros((n, 3), np.float32)
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

        rgb: the colour of the pixel that measured each hit voxel, RGB.
        Pass None while the camera's white balance is still moving --
        ColourSettle says when -- because an early colour blended in is
        wrong, and blending is not a vote it can lose.
        """
        hf, hin = self._flat(hit_idx)
        pf, _ = self._flat(pass_idx)
        if rgb is not None and len(hf):
            self._add_colour(hf, np.asarray(rgb, np.float64)
                             .reshape(-1, 3)[hin])
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

    def _add_colour(self, hf, c):
        """Blend one frame's look at these voxels into their colour.

        The frame speaks ONCE per voxel however many of its pixels landed
        in it: those pixels are averaged first, and that average is then
        blended with COLOUR_ALPHA. Blending per pixel instead would make
        a voxel the camera happened to see at an angle -- hundreds of
        pixels -- forget everything before it, and one seen edge-on keep
        a colour it should have lost.
        """
        u, inv = np.unique(hf, return_inverse=True)
        tot = np.zeros((len(u), 3), np.float64)
        np.add.at(tot, inv, c)
        mean = tot / np.bincount(inv, minlength=len(u))[:, None]
        first = self.colour_n[u] == 0
        e = self.colour_ema[u].astype(np.float64)
        e[first] = mean[first]
        e[~first] += self.alpha * (mean[~first] - e[~first])
        self.colour_ema[u] = e
        self.colour_n[u] += 1

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
        """Each voxel's colour, RGB, weighted towards the LATEST look.

        See COLOUR_ALPHA: a running mean would keep the colour the camera
        reported before its white balance had converged.
        """
        return np.clip(np.rint(self.colour_ema), 0, 255).astype(
            np.uint8).reshape(self.shape + (3,))

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

        Occupied when the frames that could see this voxel AGREE it is
        occupied, and enough of them looked to make that agreement mean
        something -- `agree` of at least `seen_min` frames. The class is
        the argmax of those votes.

        The second condition is what keeps the room from growing spikes.
        The depth the model reads is itself unsteady, so a voxel at the
        edge of one frame's surface is claimed by that frame and by no
        other; with a bare majority of a single look, every one of those
        is a permanent spur on the furniture. See SEEN_MIN / AGREE.
        """
        total = self.votes[:, 1:].sum(axis=1, dtype=np.int64)
        seen = self.n_seen.astype(np.int64)
        occ = (total > self.agree * seen) & (seen >= self.seen_min)
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

        'contradicted' is the completions the sweep disproved, and is
        what the confirmed/contradicted score is computed from. It counts
        every one of them: excluding the model's near misses from the
        score would simply flatter it.

        'in_free' is a different question, asked for the DISPLAY: where
        has the model put matter in space the camera MEASURED as empty?
        It is wider than 'contradicted' in that the voxel need not have
        been a completion -- wrong in plain view is still wrong. The room
        drops these when it is showing what is there rather than how the
        model did: the sensor outranks the model about space it has
        measured, so a claim it has refuted is not drawn at all.

        'disproved' is 'in_free' minus the one-voxel skin along every
        surface, which is below the resolution of the comparison. It is
        what the verdict mode paints red. See _disproved.

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
               'missed': (st == OCCUPIED) & ~occ,
               'in_free': occ & (st == FREE),
               'disproved': self._disproved(occ, st)}
        return dict((k, v.reshape(self.shape)) for k, v in out.items())

    def _disproved(self, occ, st):
        """Matter the model put in space the sensor measured as EMPTY.

        What the display paints red, and deliberately not what the score
        is computed from -- see verdicts().

        Voxels touching a measured surface are excluded, because that is
        not an error this comparison can resolve. The model's surfaces
        come out about a voxel thicker than the measurement, so the cell
        just in front of every wall reads free; at 8 cm, with ~1 voxel of
        trajectory drift and a half-voxel hit margin, a disagreement that
        touches a surface is below the resolution of the question. On
        room07 that skin is 2469 of 3276 such voxels, and painting it red
        turns an entire wall red and says nothing. The 807 that remain are
        matter floating in open space the camera saw through.
        """
        from scipy import ndimage
        free = (occ & (st == FREE)).reshape(self.shape)
        surface = (st == OCCUPIED).reshape(self.shape)
        return (free & ~ndimage.binary_dilation(surface)).reshape(-1)

    def counts(self):
        return dict((k, int(v.sum())) for k, v in self.verdicts().items())

    # ---- persistence ----------------------------------------------------

    _ARRAYS = ('n_hit', 'n_pass', 'colour_ema', 'colour_n', 'votes',
               'n_seen', 'first_meas', 'first_pred', 'blind')

    def save(self, path):
        kw = dict((k, getattr(self, k)) for k in self._ARRAYS)
        np.savez_compressed(path, origin=self.origin,
                            shape=np.asarray(self.shape), vox=self.vox,
                            hit_min=self.hit_min, pass_min=self.pass_min,
                            seen_min=self.seen_min, agree=self.agree,
                            alpha=self.alpha, **kw)

    @classmethod
    def load(cls, path):
        z = np.load(path)
        f = z.files

        def opt(k, d):
            return type(d)(z[k]) if k in f else d

        m = cls(z['origin'], tuple(int(s) for s in z['shape']),
                vox=float(z['vox']), hit_min=int(z['hit_min']),
                pass_min=int(z['pass_min']), seen_min=opt('seen_min', SEEN_MIN),
                agree=opt('agree', AGREE), alpha=opt('alpha', COLOUR_ALPHA))
        for k in cls._ARRAYS:
            if k in f:
                setattr(m, k, z[k])
        if 'colour_ema' not in f and 'colour_sum' in f:
            # a map written before the colour was a blend: its colour_n
            # counted PIXELS, so recover the mean it stood for and treat
            # that as where the blend had got to
            n = np.maximum(z['colour_n'], 1)[:, None]
            m.colour_ema = (z['colour_sum'] / n).astype(np.float32)
            m.colour_n = (z['colour_n'] > 0).astype(np.uint32)
        return m



# ---- when has the camera finished getting ready ------------------------

class ColourSettle:
    """True once the camera's auto white balance has stopped moving.

    A D4xx opens with auto-exposure and auto-white-balance unconverged and
    takes SECONDS to settle, not frames. Measured by reading the recorded
    streams back (both at 15 fps):

        room07  frame 0 is 7.4% saturated and reads R137 G120 B187 -- a
                white wall as pink. The blue channel decays frame after
                frame and the three means do not meet until frame 60,
                4.0 s in, after which they are flat to 0.1 of a count.
        room08  the same shape, converged by about frame 65.

    capture.py's own --warmup discards 30 frames before it starts, which
    is 1 s at 30 fps and nowhere near enough; and it discards them from
    the CAPTURE, not from the recording, so a scan.bag replays the
    unconverged frames whatever that was set to.

    Geometry is not affected -- the depth stream has its own exposure --
    so this gates the COLOUR only. Every frame is still tracked, measured
    and predicted from frame 0; the early ones simply do not get to say
    what anything looks like.

    *** WHY NOT A FIXED NUMBER OF FRAMES ***
    Because 60 is room07's answer, not the camera's. The rate differs
    (15 fps recorded, 30 live), the convergence depends on the scene --
    room08's took half as long -- and a number tuned on one bag applied
    to a live camera is a guess wearing a measurement's clothes.

    What is observable is the drift itself. White balance moves the
    frame's CHROMATICITY, the channel means as fractions of their sum,
    and when it has finished, that stops moving. Measured per frame on
    room07: 0.0012-0.0031 while converging, 0.0001-0.0004 afterwards
    while the camera is being panned round the room. An order of
    magnitude apart, which is what makes the test possible -- it is the
    white balance that is being watched, not the scene.

    So: settled when the chromaticity has moved less than `drift` per
    frame for `hold` frames running. At 8e-4 over 8 frames that is
    frame 70 on room07, 71 on room06 and 49 on room08, against hand
    reading of the channel means of about 60, 70 and 65. `cap` is the
    escape for a stream where it never settles -- a camera carried
    through changing light -- after which the colour is taken as it
    comes, which is still better than no colour at all.
    """

    def __init__(self, drift=8e-4, hold=8, cap=150):
        self.drift, self.hold, self.cap = float(drift), int(hold), int(cap)
        self.settled = False
        self.n = 0            # frames offered
        self.at = None        # the frame it settled on
        self.capped = False
        self._prev = None
        self._run = 0

    def update(self, rgb):
        """One frame -> may its colour be used? Channel order does not
        matter: the test is that the three stop moving relative to
        each other."""
        if self.settled:
            return True
        self.n += 1
        if rgb is None:
            return False
        a = np.asarray(rgb)[::8, ::8].reshape(-1, 3).astype(np.float64)
        m = a.mean(axis=0)
        t = m.sum()
        if t <= 0:
            return False
        c = m / t
        if self._prev is not None:
            self._run = (self._run + 1
                         if np.abs(c - self._prev).max() < self.drift else 0)
        self._prev = c
        if self._run >= self.hold:
            self.settled, self.at = True, self.n
        elif self.n >= self.cap:
            self.settled, self.at, self.capped = True, self.n, True
        return self.settled


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
