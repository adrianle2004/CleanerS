"""
inference/tracker.py

Where the accumulating grid gets its pose: the TRACKING half of
reconstruction_GT/fuse_scan.py and nothing else -- no mesh, no files, no
report. Frame-to-model, so error does not compound frame over frame.

fuse_scan keeps its own copy. It produced committed ground truth as a side
effect of running, and refactoring it is not worth doing for a display
change; the constants below are therefore repeated rather than imported,
so changing one here cannot silently move the annotation there.

*** WHY NOT JUST READ trajectory.txt ***
Because the frame indices do not mean the same thing and nothing says so.
fuse_scan indexes its poses by `read - 1`, counting EVERY frameset the bag
reader yields and skipping --skip_s (1 s by default). BagSource names
frames `live_{i:06d}` where i only advances on framesets that survive
_process, and skips nothing. So live_000000 is not bag frame 0, the offset
depends on how many framesets happened to be incomplete, and a mismatch
would not raise -- the room would just come out quietly smeared. Tracking
in the loop removes the question: same reader, same frames, no index.

*** IT TRACKS ON DEPTH GEOMETRY ALONE ***
A view of nothing but one flat wall is degenerate: the camera can slide
along it without the error changing. Keep furniture or a corner in view.

*** A LIVE RUN IS NOT A FILE, AND CAN BE INTERRUPTED ***
fuse_scan reads a recording: every frame follows the one before it, so a
pose that cannot be believed is a blow-up and the only sane answer is to
stop. A window someone is driving is different. They hold it to look at
the room, and their hand keeps moving; they point the camera at the floor
while they talk. When the frames come back, the view has JUMPED, and what
happens then is this file's problem rather than fuse_scan's:

    Belief   decides what a frame must show before it may touch the map,
             and -- the part fuse_scan has no need of -- how a tracker
             that has stopped believing the pose gets back to believing
             it, instead of the run being over.

See Belief's docstring for the measurements behind the numbers.
"""

import logging

import numpy as np

log = logging.getLogger('tracker')

# fuse_scan.py:88-94, repeated on purpose -- see the module docstring
MIN_FITNESS = 0.1
MAX_SPEED_M_S = 1.0
MAX_TURN_DEG_S = 120.0
LOST_AFTER_S = 1.0

# The frame interval the per-frame allowance assumes when the real one is
# not known. fuse_scan divides a SPEED by the bag's fps, which is right
# there because it reads every frameset of a bag whose rate it has
# measured. Here the rate is not knowable: a live camera drops framesets
# under load, and a replay delivers whatever the loop keeps up with.
#
# Guessing it is worse than not having it. Room07's bag is 15 fps and the
# first cut of this file assumed 30, so the allowance came out 0.033 m and
# steps of 0.036-0.157 m -- an ordinary handheld sweep -- were rejected as
# physically impossible. The tracker declared itself lost half way through
# the room and the display simply stopped filling.
#
# So the default allowance is derived from a frame interval long enough to
# be safe for any plausible stream. It is a blow-up detector, not a speed
# limit on the operator's hands: half a metre between consecutive frames is
# implausible at any frame rate, 3.6 cm is not. Pass --track_fps when you
# do know the rate and want the tighter gate.
DT_FALLBACK_S = 0.5
LOST_AFTER_FRAMES = 15

# How much worse than this run's own tracking a frame may be and still be
# believed, how many frames that reference is taken over, and the most it
# is ever allowed to demand. See Belief's docstring for the measurements.
FITNESS_FRACTION = 0.6
FITNESS_WINDOW = 30
FITNESS_CAP = 0.55
FITNESS_SAMPLES = 5

# Frames that must agree, one after another, before a tracker that lost
# the pose is believed again.
CONFIRM_FRAMES = 3


def reject(fitness, step_T, fps=None, min_fitness=MIN_FITNESS,
           max_speed=MAX_SPEED_M_S, max_turn_deg=MAX_TURN_DEG_S,
           dt=None):
    """-> why this step is not believable, or None if it is.

    Read as "integrate nothing this frame", not as a warning. A bad pose
    does not degrade the map, it permanently corrupts it: geometry welded
    into the wrong place never washes out, however many good frames follow.

    fps: frames per second, when it is genuinely known. Without it the
    allowance comes from DT_FALLBACK_S -- see the note there.
    """
    if fitness < min_fitness:
        return 'fitness %.3f' % fitness
    if step_T is None:              # nothing to gate: fitness alone decides
        return None
    if dt is None:
        dt = DT_FALLBACK_S if not fps else 1.0 / float(fps)
    S = np.asarray(step_T, np.float64)
    d = float(np.linalg.norm(S[:3, 3]))
    turn = float(np.degrees(np.arccos(np.clip(
        (np.trace(S[:3, :3]) - 1.0) / 2.0, -1.0, 1.0))))
    if d > max_speed * dt or turn > max_turn_deg * dt:
        return 'jump %.3f m / %.1f deg in one frame' % (d, turn)
    return None


class Belief:
    """What a frame must show before it is allowed into the map, and how a
    tracker that has stopped believing the pose gets back to believing it.

    Pure bookkeeping -- no open3d, no GPU, no bag -- so the rule can be
    tested directly. Tracker owns one of these and asks it about every
    frame; the answer is a verb:

        integrate  believed. Put it in the map.
        confirm    believable, but the pose was lost and one frame is not
                   evidence that it is found. Nothing goes in yet.
        drop       not believed. Nothing goes in, and nothing is lost by
                   waiting -- a frame costs 1/30 s, a smeared map costs
                   the run.

    *** WHY THE FITNESS FLOOR IS NOT fuse_scan's 0.1 ***
    Measured over both committed captures, end to end: room07 (807 frames)
    ran at ICP fitness 0.712 at worst, median 0.903; room08 (1064 frames)
    0.647 at worst, median 0.895. The worst frame of 1871 was 0.647 and not
    one came near 0.6. Good tracking is not marginal -- it is emphatic.

    A discontinuity is not. Holding room07's window at frame 59 and
    resuming at frame 200 -- what a hand that keeps moving while the
    display is held does to a live camera -- ICP converged on a WRONG
    place at fitness 0.15-0.23, and then tracked on quite happily at
    0.44-0.51, welding twenty frames of room a quarter of a metre from
    where they belonged. fuse_scan's 0.1 floor passes every one of those,
    and fuse_scan is right not to care: it reads a file, where frame 200
    never follows frame 59.

    So the floor is relative to what this run is ACHIEVING -- 60% of the
    median of the last 30 believed frames -- which also carries over to a
    sensor or a scene that simply matches worse, where a fixed floor would
    either kill the run or let the jump through. It is capped at 0.55, a
    tenth below the worst good frame ever measured, so it can never become
    stricter than the evidence supports.

    *** AND WHY LOST IS NO LONGER FINAL ***
    Frame-to-model tracking compares every frame with a raycast of the
    model from the last believed pose. That is what stops drift
    compounding, and it is also why ONE discontinuity used to end a run:
    the reference never moves again, so every later frame is compared with
    a viewpoint the camera has left, fails, and the map never grows --
    whatever the operator does. Recovery is therefore not a nicety. It
    only has to be earned: CONFIRM_FRAMES frames in a row, each matching
    the model at the floor above, each placing the camera within one
    plausible step of the last. The jump gate is dropped for the first of
    them, because how far the camera went while nothing was believed is
    exactly what is not knowable -- the agreement replaces it.
    """

    def __init__(self, fps=None, dt=None, fraction=FITNESS_FRACTION,
                 window=FITNESS_WINDOW, cap=FITNESS_CAP,
                 samples=FITNESS_SAMPLES, confirm=CONFIRM_FRAMES,
                 lost_after=None):
        self.fps = float(fps) if fps else None
        self.dt = float(dt) if dt else (
            (1.0 / self.fps) if self.fps else DT_FALLBACK_S)
        self.fraction, self.window = float(fraction), int(window)
        self.cap, self.samples = float(cap), int(samples)
        self.confirm = max(1, int(confirm))
        self.lost_after = int(lost_after) if lost_after else (
            max(1, int(round(LOST_AFTER_S * self.fps))) if self.fps
            else LOST_AFTER_FRAMES)
        self.fits = []        # fitness of the believed frames, most recent last
        self.good = 0         # believable frames in a row, since the last one in
        self.run_lost = 0     # frames in a row not believed
        self.dropped = 0      # frames not believed, all run
        self.held = 0         # frames that touched nothing, all run
        self.reacquired = 0   # times the pose was found again
        self.lost = False
        self.reasons = []

    @property
    def confirming(self):
        """Is this a tracker trying to find itself again?"""
        return bool(self.run_lost)

    def floor(self):
        """The fitness this frame has to reach. Until there is a run to
        compare with, fuse_scan's floor is all there is."""
        if len(self.fits) < self.samples:
            return MIN_FITNESS
        ref = float(np.median(self.fits))
        return max(MIN_FITNESS, min(self.cap, self.fraction * ref))

    def see(self, fitness, step_T, error=None, where=None):
        """One frame's evidence -> ('integrate'|'confirm'|'drop', why).

        step_T is this frame's motion from the last pose that was
        BELIEVABLE -- which while confirming is the previous candidate,
        not the last pose integrated. Pass None to gate on fitness alone.
        error: a tracking failure instead of a result, as a string.
        where: which frame this was, for the report.
        """
        reason = error
        if reason is None:
            gate = None if (self.confirming and not self.good) else step_T
            reason = reject(fitness, gate, dt=self.dt,
                            min_fitness=self.floor())
        if reason is not None:
            self.good = 0
            self.dropped += 1
            self.held += 1
            self.run_lost += 1
            if len(self.reasons) < 5:
                self.reasons.append('%s: %s' % (where, reason)
                                    if where else reason)
            if not self.lost and self.run_lost >= self.lost_after:
                self.lost = True
            return 'drop', reason
        self.good += 1
        if self.good < (self.confirm if self.confirming else 1):
            self.held += 1
            return 'confirm', None
        if self.confirming:
            self.reacquired += 1
            log.info('tracking re-acquired: %d frames in a row agree on '
                     'where the camera is, after %d that did not. The room '
                     'carries on from where it stopped', self.good,
                     self.run_lost)
        self.fits.append(float(fitness))
        del self.fits[:-self.window]
        self.good, self.run_lost, self.lost = 0, 0, False
        return 'integrate', None


class Tracker:
    """world_T_cam per frame, in the frame live_cam_pose builds.

    The composition is fuse_scan's, verbatim (fuse_scan.py:484,542): the
    tracker accumulates T from the identity and the world pose is W @ T,
    where W = live_cam_pose(camera_height, yaw, up_camera) of the FIRST
    frame. That is what makes world_T_cam of frame 0 equal to that frame's
    own cam_pose, and the whole placement collapse to the identity there --
    the property inference/check_world_map.py checks the algebra with.
    """

    def __init__(self, cam_K, width, height, initial_pose, fps=None,
                 voxel=0.02, block_count=8000, depth_scale=1000.0,
                 depth_max=3.0, depth_diff=0.07, trunc=8.0, device='auto',
                 belief=None):
        import open3d as o3d
        import open3d.core as o3c
        self._o3d, self._o3c = o3d, o3c
        if device == 'auto':
            device = 'CUDA:0' if o3c.cuda.is_available() else 'CPU:0'
        elif device in ('cpu', 'cuda'):
            device = {'cpu': 'CPU:0', 'cuda': 'CUDA:0'}[device]
        self.device = o3c.Device(device)
        self.scale = float(depth_scale)
        self.depth_max, self.depth_diff = float(depth_max), float(depth_diff)
        # a MULTIPLIER of the voxel size, not metres: open3d's
        # trunc_voxel_multiplier. 8 at 2 cm is a 16 cm truncation
        self.trunc = float(trunc)
        self.fps = float(fps) if fps else None
        self.dt = (1.0 / self.fps) if self.fps else DT_FALLBACK_S
        self.belief = belief if belief is not None else Belief(fps=fps)
        self.W = np.asarray(initial_pose, np.float64).copy()
        self.T = np.identity(4)
        # where the frames being confirmed say the camera is. Kept apart
        # from T, which is only ever a pose the map was built from.
        self._cand = None
        Kt = o3c.Tensor(np.asarray(cam_K, np.float64), o3c.Dtype.Float64)
        # 2 cm rather than fuse_scan's 1 cm, and a third of the blocks:
        # this one only has to TRACK, it is not producing the mesh, and the
        # pipeline it shares the card with already wants ~1.1 GB
        self.model = o3d.t.pipelines.slam.Model(
            float(voxel), 16, int(block_count),
            o3c.Tensor(np.identity(4)), self.device)
        self.frame = o3d.t.pipelines.slam.Frame(int(height), int(width), Kt,
                                                self.device)
        self.raycast = o3d.t.pipelines.slam.Frame(int(height), int(width),
                                                  Kt, self.device)
        self.n = 0            # frames integrated: the map's own frame count
        self.id = 0           # steps the model holds a pose for
        self.seen = 0         # frames handed to step(), believed or not
        self.pose_only = 0    # steps tracked but deliberately not integrated

    # the counters run_live and the window read. They live on Belief, which
    # owns the rule, and are surfaced here because the tracker is what the
    # rest of the code has a handle on.
    @property
    def lost(self):
        return self.belief.lost

    @property
    def dropped(self):
        return self.belief.dropped

    @property
    def confirming(self):
        return self.belief.confirming

    def step(self, depth_m, rgb=None, integrate=True):
        """depth in METRES -> world_T_cam, or None when this frame must not
        be integrated.

        integrate=False tracks the frame and keeps the pose current without
        putting anything in the map -- what a HELD window wants. The room
        stays exactly as it was while someone looks at it, and the camera
        can still be moving when they let it go, which is the whole reason
        this argument exists: without it, holding the window hands the
        tracker a discontinuity, and a discontinuity is what ends a run.
        The first frame ignores it -- that frame IS the map.

        rgb is not used for TRACKING -- that is depth geometry alone --
        but model.integrate builds a coloured TSDF and fails without a
        colour channel, so one is always set, black if none is given.
        """
        o3c, o3d = self._o3c, self._o3d
        self.seen += 1
        if not self.n:
            integrate = True
        mm = np.clip(np.nan_to_num(depth_m) * self.scale, 0, 65535)
        mm = np.ascontiguousarray(mm.astype(np.uint16)[:, :, None])
        self.frame.set_data_from_image(
            'depth', o3d.t.geometry.Image(o3c.Tensor(mm)).to(self.device))
        if rgb is None:
            col = np.zeros(mm.shape[:2] + (3,), np.uint8)
        else:
            col = np.ascontiguousarray(np.asarray(rgb, np.uint8))
        self.frame.set_data_from_image(
            'color', o3d.t.geometry.Image(o3c.Tensor(col)).to(self.device))
        if self.n:
            cand, fitness, error = None, 0.0, None
            try:
                res = self.model.track_frame_to_model(
                    self.frame, self.raycast, self.scale, self.depth_max,
                    self.depth_diff)
                fitness = float(res.fitness)
                # the raycast still holds the last BELIEVED pose, so the
                # ICP result is this frame's place in the world, absolutely
                cand = self.T @ res.transformation.cpu().numpy()
            except RuntimeError as e:
                error = 'tracking error: %s' % e
            if error is None:
                prev = self._cand if self._cand is not None else self.T
                rel = np.linalg.inv(prev) @ cand
            else:
                rel = None
            verb, why = self.belief.see(fitness, rel, error=error,
                                        where='frame %d' % self.seen)
            if verb == 'drop':
                self._cand = None
                if self.belief.lost and self.belief.run_lost == \
                        self.belief.lost_after:
                    log.warning('tracking lost after %d frames in a row the '
                                'pose could not be believed (%s). The map is '
                                'frozen where it is rather than smeared -- '
                                'point the camera back at what it was '
                                'looking at and it will pick itself up',
                                self.belief.run_lost, why)
                return None
            if verb == 'confirm':
                self._cand = cand
                return None
            self.T, self._cand = cand, None
        self.model.update_frame_pose(self.id, o3c.Tensor(self.T))
        if integrate:
            self.model.integrate(self.frame, self.scale, self.depth_max,
                                 self.trunc)
            self.n += 1
        else:
            self.pose_only += 1
        # from the pose this frame is at, whether or not it went in: it is
        # the reference the NEXT frame is tracked against
        self.model.synthesize_model_frame(self.raycast, self.scale, 0.1,
                                          self.depth_max, self.trunc, False)
        self.id += 1
        return self.W @ self.T

    # fuse_scan.py:133 -- surfaces seen in fewer frames than this are not
    # meshed, which is what drops one-frame depth noise from the surface.
    MIN_WEIGHT = 3.0

    def mesh(self, min_weight=MIN_WEIGHT):
        """The sensor's OWN reconstruction of what it has seen, in world
        metres -- or None if nothing has been integrated yet.

        Free, in the sense that it is already there: the TSDF this builds
        to track against is the same volume fuse_scan carves room.ply out
        of, so extracting it costs one call and no second integration. It
        is the same call fuse_scan makes (fuse_scan.py:434), at 2 cm
        rather than 1, and with no checks run on it -- this is a thing to
        LOOK at, not a measurement. room.ply remains the committed one.

        Extracting does not disturb the volume: measured on room07, 60
        frames with a mesh pulled out every 10 ends at 61 203 triangles
        against 61 193 for the same 60 frames extracted once at the end,
        a ten-triangle marching-cubes tie-break, and the ICP fitness of
        the frames after an extraction matches to three decimals. So it
        can be pulled whenever someone wants to look, as often as they
        like, and the run carries on growing it.

        What it is for: it is the half of the picture the model is not.
        The mesh stops where the depth stopped, so every hole in it is a
        place the sensor could not see, and that is exactly what the
        completion is claiming to fill.

        *** THE COLOURS COME OUT BGR ***
        step() hands model.integrate the loader's frame, which is BGR,
        because it only has to be a colour channel for tracking's sake.
        Here it is being looked at, so the channels are swapped back.
        """
        if not self.n:
            return None
        import numpy as np
        m = self.model.extract_trianglemesh(min_weight).to_legacy()
        if not len(m.vertices):
            return None
        m.transform(self.W)
        if len(m.vertex_colors):
            c = np.asarray(m.vertex_colors)[:, ::-1].copy()
            m.vertex_colors = self._o3d.utility.Vector3dVector(c)
        return m

    def report(self):
        b = self.belief
        bits = ['tracker: %d frames tracked' % self.n]
        if self.pose_only:
            bits.append('%d tracked without being integrated (the window '
                        'was held)' % self.pose_only)
        if b.dropped:
            bits.append('%d dropped (%s)' % (b.dropped,
                                             '; '.join(b.reasons)))
        if b.reacquired:
            bits.append('re-acquired %d time%s' % (b.reacquired,
                                                   's' * (b.reacquired != 1)))
        if b.lost:
            bits.append('LOST at the end -- the map stopped growing there')
        return ', '.join(bits)
