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
"""

import logging

import numpy as np

log = logging.getLogger('tracker')

# fuse_scan.py:88-94, repeated on purpose -- see the module docstring
MIN_FITNESS = 0.1
MAX_SPEED_M_S = 1.0
MAX_TURN_DEG_S = 120.0
LOST_AFTER_S = 1.0


def reject(fitness, step_T, fps, min_fitness=MIN_FITNESS,
           max_speed=MAX_SPEED_M_S, max_turn_deg=MAX_TURN_DEG_S):
    """-> why this step is not believable, or None if it is.

    Read as "integrate nothing this frame", not as a warning. A bad pose
    does not degrade the map, it permanently corrupts it: geometry welded
    into the wrong place never washes out, however many good frames follow.
    """
    if fitness < min_fitness:
        return 'fitness %.3f' % fitness
    S = np.asarray(step_T, np.float64)
    d = float(np.linalg.norm(S[:3, 3]))
    turn = float(np.degrees(np.arccos(np.clip(
        (np.trace(S[:3, :3]) - 1.0) / 2.0, -1.0, 1.0))))
    if d > max_speed / fps or turn > max_turn_deg / fps:
        return 'jump %.3f m / %.1f deg' % (d, turn)
    return None


class Tracker:
    """world_T_cam per frame, in the frame live_cam_pose builds.

    The composition is fuse_scan's, verbatim (fuse_scan.py:484,542): the
    tracker accumulates T from the identity and the world pose is W @ T,
    where W = live_cam_pose(camera_height, yaw, up_camera) of the FIRST
    frame. That is what makes world_T_cam of frame 0 equal to that frame's
    own cam_pose, and the whole placement collapse to the identity there --
    the property inference/check_world_map.py checks the algebra with.
    """

    def __init__(self, cam_K, width, height, initial_pose, fps=30.0,
                 voxel=0.02, block_count=8000, depth_scale=1000.0,
                 depth_max=3.0, depth_diff=0.07, trunc=8.0, device='auto'):
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
        self.fps = float(fps)
        self.W = np.asarray(initial_pose, np.float64).copy()
        self.T = np.identity(4)
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
        self.n = 0
        self.dropped = 0
        self.run_lost = 0
        self.reasons = []
        self.lost = False
        self.lost_after = max(1, int(round(LOST_AFTER_S * self.fps)))

    def step(self, depth_m, rgb=None):
        """depth in METRES -> world_T_cam, or None when this frame must not
        be integrated. Once lost, it stays lost and always returns None.

        rgb is not used for TRACKING -- that is depth geometry alone --
        but model.integrate builds a coloured TSDF and fails without a
        colour channel, so one is always set, black if none is given.
        """
        if self.lost:
            return None
        o3c, o3d = self._o3c, self._o3d
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
            reason = None
            try:
                res = self.model.track_frame_to_model(
                    self.frame, self.raycast, self.scale, self.depth_max,
                    self.depth_diff)
                step = res.transformation.cpu().numpy()
                reason = reject(res.fitness, step, self.fps)
            except RuntimeError as e:
                reason = 'tracking error: %s' % e
            if reason is not None:
                self.dropped += 1
                self.run_lost += 1
                if len(self.reasons) < 5:
                    self.reasons.append('frame %d: %s' % (self.n, reason))
                if self.run_lost >= self.lost_after:
                    self.lost = True
                    log.warning('tracking lost after %d consecutive dropped '
                                'frames (%s); the map is frozen where it is '
                                'rather than smeared', self.run_lost, reason)
                return None
            self.run_lost = 0
            self.T = self.T @ step
        self.model.update_frame_pose(self.n, o3c.Tensor(self.T))
        self.model.integrate(self.frame, self.scale, self.depth_max,
                             self.trunc)
        self.model.synthesize_model_frame(self.raycast, self.scale, 0.1,
                                          self.depth_max, self.trunc, False)
        self.n += 1
        return self.W @ self.T

    def report(self):
        bits = ['tracker: %d frames tracked' % self.n]
        if self.dropped:
            bits.append('%d dropped (%s)' % (self.dropped,
                                             '; '.join(self.reasons)))
        if self.lost:
            bits.append('LOST -- the map stopped growing there')
        return ', '.join(bits)
