"""
inference/frame_loader.py

Loads a single depth (+RGB) frame and produces the (tsdf, mapping, mapping2d,
img) tensors the model needs -- reusing the validated geometry from
generate_cleaners_data.py (0.96 correlation, 12,313/12,668 unique TSDF values
against the CleanerS reference for NYU0001).

Two entry points:
  - FrameLoader.from_nyu_bin(bin_path, png_path)   -- for NYU-format test data,
    where vox_origin/cam_pose come from the .bin header.
  - FrameLoader.from_live_camera(depth, cam_pose=None, vox_origin=None)
    -- for a real depth camera feed, where you supply pose/origin yourself
    (see the note below -- this is the part that changes once you wire up
    an actual camera).

*** NOTE ON LIVE CAMERA USE ***
NYU's vox_origin/cam_pose come from a dataset-specific "room-canonical"
registration step (floor detection + a fixed room-sized bounding box) that
doesn't exist for a live camera out of the box. For a first working version,
FrameLoader.from_live_camera defaults to a CAMERA-CENTRIC grid: it places the
voxel volume directly in front of the camera at a fixed offset, with
cam_pose = identity (the grid is defined in the camera's own coordinate
frame, not world-fixed). This is a reasonable placeholder but will need
replacing once you add real localization (e.g. a SLAM/registration step) if
you want a world-fixed volume across multiple frames.
"""

import ctypes
import os
import struct
import sys
import numpy as np
import cv2
from scipy.ndimage import distance_transform_edt

CAM_K = np.array([[518.8579, 0, 325.58],
                   [0, 519.4696, 253.74],
                   [0, 0, 1]], dtype=np.float32)

VOX_UNIT_HI = 0.02
VOX_SIZE_HI = np.array([240, 144, 240])
VOX_MARGIN = 0.24
SAMPLE_RATIO = 4
VOX_SIZE_LOW = (VOX_SIZE_HI // SAMPLE_RATIO).astype(np.int64)   # [60, 36, 60]
VOX_UNIT_LOW = VOX_UNIT_HI * SAMPLE_RATIO

IMG_H, IMG_W = 480, 640
MAPPING_SENTINEL = IMG_H * IMG_W

# SSCNet's own CUDA kernels (depth2Grid__ + SquaredDistanceTransform__), built
# one directory above the repo. ~4-5x faster than _build_high_res_tsdf.
CUDA_UTILS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))), 'data', 'utils')
_dataprocess = None


def load_cuda_encoder():
    """Import DataProcess, or return None if it is not usable here.

    DataProcess links against build/libdatautil.so without an rpath, so the
    library is preloaded by absolute path first (no LD_LIBRARY_PATH needed)."""
    global _dataprocess
    if _dataprocess is None:
        try:
            ctypes.CDLL(os.path.join(CUDA_UTILS_DIR, 'build', 'libdatautil.so'), mode=ctypes.RTLD_GLOBAL)
            if CUDA_UTILS_DIR not in sys.path:
                sys.path.insert(0, CUDA_UTILS_DIR)
            import DataProcess
            _dataprocess = DataProcess
        except (OSError, ImportError):
            _dataprocess = False
    return _dataprocess or None


def live_cam_pose(camera_height, yaw=0.0, up_camera=None):
    """Camera-to-world in NYU's gravity-aligned, floor-anchored convention.

    `up_camera` is the measured gravity direction in CAMERA coordinates -- the
    accelerometer vector, which at rest points up. Given it, the world frame is
    built around real gravity, so a camera that is pitched or rolled still gets
    a level grid: world Z is up, world Z = 0 is the floor, and the camera's
    heading (its optical axis flattened into the horizontal) becomes world +Y.

    Without it the camera is assumed level, which NYU's own frames are not --
    their tilt runs 0.31 to 7.65 degrees. A camera pitched 4 degrees puts a
    surface 3 m away 21 cm from where a level grid draws it.

    `yaw` then turns the grid about the vertical, as before.

    Shared with reconstruction_GT/*, which must place meshes and solids in
    exactly this frame for them to line up with the frame's voxel grid.
    """
    if up_camera is None:
        up = np.array([0.0, -1.0, 0.0])        # camera -Y: assume level
    else:
        up = np.asarray(up_camera, np.float64)
        n = np.linalg.norm(up)
        if n < 1e-6:
            raise ValueError('up_camera is a zero vector')
        up = up / n
    # world axes, written in camera coordinates:
    #   Z = up (gravity)
    #   Y = where the camera looks, flattened into the horizontal plane
    #   X = Y x Z, to keep it right-handed
    fwd = np.array([0.0, 0.0, 1.0])
    e_y = fwd - np.dot(fwd, up) * up
    if np.linalg.norm(e_y) < 1e-6:             # aimed straight up or down
        alt = np.array([0.0, -1.0, 0.0])
        e_y = alt - np.dot(alt, up) * up
    e_y = e_y / np.linalg.norm(e_y)
    e_x = np.cross(e_y, up)
    R = np.stack([e_x, e_y, up])               # rows: world axes in cam coords
    c, sn = np.cos(yaw), np.sin(yaw)
    Rz = np.array([[c, -sn, 0.0], [sn, c, 0.0], [0.0, 0.0, 1.0]])
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = Rz @ R
    T[2, 3] = camera_height
    return T.astype(np.float32)


def frame_meta(meta, stem=None):
    """The camera record for ONE frame of a capture.

    A capture folder describes a room: one meta.json, one set of intrinsics,
    one measured camera height. Frames lifted out of the sweep afterwards
    (reconstruction_GT/export_frame.py) live in the same folder but were shot
    from somewhere else, so meta.json carries what differs under

        "frame_meta": {"live_000445": {"camera_height": 1.079,
                                       "up_camera": [...],
                                       "world_from_room": [[4x4]]}}

    and everything unstated stays the capture's own value. `world_from_room`
    is how the room's solids reach that frame's world -- the annotation is
    made once, against the fused room, and every frame borrows it.
    """
    out = {k: v for k, v in meta.items() if k != 'frame_meta'}
    if stem:
        out.update(meta.get('frame_meta', {}).get(stem, {}))
    return out


def cam_pose_from_meta(meta, stem=None):
    """live_cam_pose for a capture, using its measured tilt when it has one."""
    m = frame_meta(meta, stem)
    return live_cam_pose(m['camera_height'], m.get('yaw', 0.0),
                         m.get('up_camera'))


class FrameLoader:
    def __init__(self, depth, vox_origin, cam_pose, rgb=None, cam_K=None,
                 drop_invalid_depth=False, encoder='numpy'):
        """drop_invalid_depth: exclude depth<=0 pixels from the surface stamp.

        Leave False to reproduce the NYU reference exactly. Set True for live
        sensors. Every zero-depth pixel unprojects to (0,0,0) in the camera
        frame, so they ALL land in the single voxel containing the camera
        itself -- the effect of this flag is exactly one voxel, nothing else.
        NYU tolerates the quirk because its grid usually excludes the camera;
        a floor-anchored live grid contains it, and a D455 reports 0 for every
        out-of-range or low-confidence pixel, so the blob would be permanent.

        encoder: 'numpy' (validated against the NYU reference), 'cuda'
        (SSCNet's datautil.cu kernels; raises if unavailable) or 'auto'
        (cuda when importable, else numpy). On ScanNet frames the two differ on
        0-247 of 8.29 M high-res voxels -- surface voxels on a voxel boundary,
        float32 vs float64 -- and by at most 0.019 after the 4x block mean.
        """
        self.depth = depth
        self.vox_origin = vox_origin.astype(np.float32)
        self.cam_pose = cam_pose.astype(np.float32)
        self.rgb = rgb
        self.cam_K = cam_K if cam_K is not None else CAM_K
        self.drop_invalid_depth = drop_invalid_depth
        if encoder not in ('numpy', 'cuda', 'auto'):
            raise ValueError('encoder must be numpy, cuda or auto, not %r' % encoder)
        if encoder != 'numpy' and load_cuda_encoder() is None:
            if encoder == 'cuda':
                raise ImportError('DataProcess CUDA encoder not importable from %s' % CUDA_UTILS_DIR)
            encoder = 'numpy'
        self.encoder = encoder if encoder != 'auto' else 'cuda'

    # -- constructors --
    @classmethod
    def from_nyu_bin(cls, bin_path, png_path, rgb_path=None):
        vox_origin, cam_pose = cls._load_bin_header(bin_path)
        depth = cls._load_depth(png_path)
        rgb = None
        if rgb_path is not None:
            rgb = cv2.imread(rgb_path)
            if rgb is None:
                raise FileNotFoundError(
                    f"cv2.imread returned None for rgb_path={rgb_path!r} -- "
                    f"either the file doesn't exist or isn't a readable image. "
                    f"(cv2.imread fails silently instead of raising, hence this check.)"
                )
            rgb = rgb.astype(np.float32)
        return cls(depth, vox_origin, cam_pose, rgb=rgb)

    @classmethod
    def from_live_camera(cls, depth, rgb=None, cam_pose=None, vox_origin=None,
                         cam_K=None, camera_height=1.25, yaw=0.0,
                         drop_invalid_depth=True, up_camera=None):
        """depth: (480,640) float32 array in METRES, already decoded (no
        bit-shift -- that trick is specific to NYU's raw Kinect PNG encoding).

        Builds a GRAVITY-ALIGNED, FLOOR-ANCHORED grid in NYU's convention,
        which was read off the dataset's own .bin headers:

            cam +Y (down) -> world (0,0,-1)     i.e. world +Z is UP
            t             =  (0, 0, height)     i.e. world Z=0 is the FLOOR
            vox_origin[2] = -0.05               volume starts 5cm below it

        This matters because the encoder sends world Z to the 144-cell axis --
        the 2.88 m one, which is room HEIGHT. An identity cam_pose sends world
        Z along the camera's viewing direction instead, producing a 4.8 m-tall
        by 2.88 m-deep box and handing the network a scene rotated 90 degrees.
        `floor` and `ceiling` are the first predictions to collapse.

        camera_height: metres above the floor. NYU's own frames sit at
            1.18-1.32 m, so the 1.25 default is mid-range; measure yours.
        yaw: rotation about the vertical axis, radians. Only matters if you
            want the volume aligned to walls rather than to the camera.

        STILL A HEURISTIC: the volume is fixed relative to the camera, so
        consecutive frames are independent single-shot completions, not an
        accumulating reconstruction, and a pitched/rolled camera is not
        compensated. A D455's IMU can supply real gravity alignment --
        see the note in run_live.py.
        """
        if cam_pose is None:
            cam_pose = live_cam_pose(camera_height, yaw, up_camera)
        if vox_origin is None:
            # 4.8 m wide (world X, centred on the camera)
            # 4.8 m deep  (world Y, starting at the camera plane)
            # 2.88 m tall (world Z, from 5 cm below the floor -- NYU's value)
            vox_origin = np.array([-2.4, 0.0, -0.05], dtype=np.float32)
        return cls(depth, vox_origin, cam_pose, rgb=rgb, cam_K=cam_K,
                   drop_invalid_depth=drop_invalid_depth)

    @staticmethod
    def _load_bin_header(bin_path):
        with open(bin_path, 'rb') as f:
            vox_origin = np.array(struct.unpack('3f', f.read(12)), dtype=np.float32)
            cam_pose = np.array(struct.unpack('16f', f.read(64)), dtype=np.float32).reshape(4, 4)
        return vox_origin, cam_pose

    @staticmethod
    def _load_depth(png_path):
        depth_raw = cv2.imread(png_path, -1).astype(np.uint16)
        depth = ((depth_raw << 13) | (depth_raw >> 3)).astype(np.float32) / 1000.0
        return depth

    # -- geometry pipeline (validated) --
    def build_tsdf_and_mapping(self):
        if self.encoder == 'cuda':
            tsdf_hi, weight_hi = self._build_high_res_tsdf_cuda()
        else:
            tsdf_hi, weight_hi = self._build_high_res_tsdf()
        tsdf_low = self._downsample_block_mean(tsdf_hi).astype(np.float32)
        weight_low = (self._downsample_block_mean(weight_hi) >= 0.5).astype(np.float32)
        mapping_low = self._build_low_res_mapping()

        mapping2d = (np.ones((IMG_H, IMG_W)) * -1).reshape(-1).astype(np.int64)
        valid = mapping_low != MAPPING_SENTINEL
        mapping2d[mapping_low[valid]] = np.nonzero(valid)[0]
        mapping2d = mapping2d.reshape(IMG_H, IMG_W)

        return {
            'tsdf': tsdf_low.reshape(1, *VOX_SIZE_LOW),      # (1,60,36,60)
            'weight': weight_low,
            'mapping': mapping_low,
            'mapping2d': mapping2d,
            'img': self.rgb,
        }

    def _build_high_res_tsdf(self):
        R = self.cam_pose[:3, :3]
        t = self.cam_pose[:3, 3]
        fx, fy, cx, cy = self.cam_K[0, 0], self.cam_K[1, 1], self.cam_K[0, 2], self.cam_K[1, 2]
        depth = self.depth
        H, W = depth.shape
        search_region = int(round(VOX_MARGIN / VOX_UNIT_HI))

        # No `depth > 0` mask here, matching depth2Grid__ (datautil.cu:39-62),
        # which has no depth-validity check: zero-depth pixels unproject to the
        # camera origin and stamp that one voxel as "surface". An SSCNet quirk,
        # but the reference data and the pretrained checkpoint carry it.
        # For live cameras set drop_invalid_depth=True: a floor-anchored grid
        # DOES contain the camera (verified: the origin lands at high-res voxel
        # x=0,y=65,z=120, in bounds on all three axes), and a D455 reports 0
        # for every out-of-range or low-confidence pixel.
        vs, us = np.meshgrid(np.arange(H), np.arange(W), indexing='ij')
        pc0 = (us - cx) * depth / fx
        pc1 = (vs - cy) * depth / fy
        pc2 = depth
        point_cam = np.stack([pc0, pc1, pc2], axis=-1).reshape(-1, 3)
        point_world = point_cam @ R.T + t[None, :]

        gz = np.floor((point_world[:, 0] - self.vox_origin[0]) / VOX_UNIT_HI).astype(np.int64)
        gx = np.floor((point_world[:, 1] - self.vox_origin[1]) / VOX_UNIT_HI).astype(np.int64)
        gy = np.floor((point_world[:, 2] - self.vox_origin[2]) / VOX_UNIT_HI).astype(np.int64)

        occ = np.zeros((VOX_SIZE_HI[2], VOX_SIZE_HI[1], VOX_SIZE_HI[0]), dtype=bool)
        inb = ((gx >= 0) & (gx < VOX_SIZE_HI[0])
               & (gy >= 0) & (gy < VOX_SIZE_HI[1]) & (gz >= 0) & (gz < VOX_SIZE_HI[2]))
        if self.drop_invalid_depth:
            inb &= depth.reshape(-1) > 0
        occ[gz[inb], gy[inb], gx[inb]] = True

        z_g, y_g, x_g = np.meshgrid(np.arange(VOX_SIZE_HI[2]), np.arange(VOX_SIZE_HI[1]),
                                     np.arange(VOX_SIZE_HI[0]), indexing='ij')
        w0 = self.vox_origin[0] + z_g * VOX_UNIT_HI
        w1 = self.vox_origin[1] + x_g * VOX_UNIT_HI
        w2 = self.vox_origin[2] + y_g * VOX_UNIT_HI
        world_pts = np.stack([w0, w1, w2], axis=-1).reshape(-1, 3).astype(np.float32)
        del w0, w1, w2, z_g, y_g, x_g

        cam_pts = (world_pts - t[None, :]) @ R
        del world_pts
        zc = cam_pts[:, 2]
        valid_z = zc > 1e-6
        u = np.zeros_like(zc); v = np.zeros_like(zc)
        u[valid_z] = fx * cam_pts[valid_z, 0] / zc[valid_z] + cx
        v[valid_z] = fy * cam_pts[valid_z, 1] / zc[valid_z] + cy
        u_pix = np.round(u).astype(np.int32)
        v_pix = np.round(v).astype(np.int32)
        in_bounds = valid_z & (u_pix >= 0) & (u_pix < W) & (v_pix >= 0) & (v_pix < H)

        depth_measured = np.zeros_like(zc)
        depth_measured[in_bounds] = depth[v_pix[in_bounds], u_pix[in_bounds]]

        known = in_bounds & (depth_measured >= 0.5) & (depth_measured <= 8.0)
        diff = depth_measured - zc
        sign = np.ones_like(zc)
        sign[known] = np.where(np.abs(diff[known]) < 1e-4, 1.0, np.sign(diff[known]))

        tsdf = np.ones(zc.shape, dtype=np.float32)
        tsdf[known] = sign[known]
        weight = np.zeros(zc.shape, dtype=np.float32)
        weight[known] = 1.0

        unsigned_dist = distance_transform_edt(~occ).reshape(-1)
        refined = np.minimum(unsigned_dist / search_region, 1.0).astype(np.float32)
        better = known & (refined < np.abs(tsdf))
        tsdf[better] = refined[better] * sign[better]

        # Unobserved voxels are 0, and this MUST happen at high resolution,
        # before the 4x4x4 block-mean below. SSCNet's SquaredDistanceTransform__
        # (datautil.cu:111-124) bare-returns for behind-camera / outside-FOV /
        # out-of-depth-range voxels, leaving them at their np.zeros init
        # (preget_for_nyu.py:101); downSample__ (datautil.cu:194) then averages
        # the raw 240^3 volume. Zeroing after the downsample instead gives a
        # low-res mask that cannot represent partial frustum coverage, so
        # boundary blocks lose their fractional values (corr 0.996 vs 1.000).
        tsdf[~known] = 0.0

        tsdf = tsdf.reshape(VOX_SIZE_HI[2], VOX_SIZE_HI[1], VOX_SIZE_HI[0])
        weight = weight.reshape(VOX_SIZE_HI[2], VOX_SIZE_HI[1], VOX_SIZE_HI[0])
        return tsdf, weight

    def _build_high_res_tsdf_cuda(self):
        """Same output as _build_high_res_tsdf, from datautil.cu.

        drop_invalid_depth is reproduced without touching the kernel: a
        depth <= 0 pixel is replaced by 100 m, which unprojects outside the
        grid (no surface stamp) and fails the kernel's 0.5-8 m test (no known
        voxel) -- exactly what the flag does in the numpy path.

        The kernel returns no observed mask. `weight` is rebuilt as
        tsdf != 0 or surface: every known voxel is non-zero except a surface
        voxel, whose distance is 0. It is used for display and diagnostics
        only, never by the model or the evaluation.
        """
        H, W = self.depth.shape
        depth = np.ascontiguousarray(self.depth, dtype=np.float32).reshape(-1).copy()
        if self.drop_invalid_depth:
            depth[depth <= 0] = 100.0
        n = int(np.prod(VOX_SIZE_HI))
        tsdf = np.zeros(n, dtype=np.float32)
        pix2vox = np.zeros(H * W, dtype=np.float32)
        args = (depth,
                np.ascontiguousarray(self.cam_K, dtype=np.float32).reshape(-1).copy(),
                np.ascontiguousarray(self.cam_pose, dtype=np.float32).reshape(-1).copy(),
                np.ascontiguousarray(self.vox_origin, dtype=np.float32).copy(),
                np.float32(VOX_UNIT_HI), VOX_SIZE_HI.astype(np.float32), np.float32(VOX_MARGIN),
                H, W, tsdf, pix2vox)
        # the extension dlopens "./build/libdatautil.so" relative to the cwd
        cwd = os.getcwd()
        os.chdir(CUDA_UTILS_DIR)
        try:
            rc = load_cuda_encoder().TSDF(*args)
        finally:
            os.chdir(cwd)
        if rc != 0:
            raise RuntimeError('DataProcess.TSDF returned %r' % rc)

        # depth2Grid__ writes the voxel index only for pixels that hit the grid
        # and leaves the zero initialisation elsewhere, so voxel 0 is ambiguous;
        # recompute the surface stamp from the pixels instead.
        occ = np.zeros(n, dtype=bool)
        hit = pix2vox > 0
        occ[pix2vox[hit].astype(np.int64)] = True
        weight = ((tsdf != 0) | occ).astype(np.float32)
        shape = (VOX_SIZE_HI[2], VOX_SIZE_HI[1], VOX_SIZE_HI[0])
        return tsdf.reshape(shape), weight.reshape(shape)

    @staticmethod
    def _downsample_block_mean(vol, r=SAMPLE_RATIO):
        Z, Y, X = vol.shape
        return vol.reshape(Z // r, r, Y // r, r, X // r, r).mean(axis=(1, 3, 5))

    def _build_low_res_mapping(self):
        """Per-low-res-voxel -> pixel index, or MAPPING_SENTINEL (307200).

        INVERSE of SSCNet's depth2Grid__ (datautil.cu:39-62): unproject each
        PIXEL and record it in the voxel it falls into. NOT a projection of
        voxel centres into the image -- that marks ~70% of the grid where the
        reference marks ~2%, and this array drives the model's 2D->3D feature
        reprojection, so the difference is not cosmetic. Verified exact against
        Mapping/*.npz; the voxel-centre version agreed on only ~30% of voxels.

        float32 throughout to match the CUDA kernel, and no `depth > 0` mask,
        same as _build_high_res_tsdf's occupancy grid.
        """
        R = self.cam_pose[:3, :3].astype(np.float32)
        t = self.cam_pose[:3, 3].astype(np.float32)
        fx, fy, cx, cy = self.cam_K[0, 0], self.cam_K[1, 1], self.cam_K[0, 2], self.cam_K[1, 2]
        H, W = self.depth.shape
        unit = np.float32(VOX_UNIT_LOW)

        vs, us = np.meshgrid(np.arange(H, dtype=np.float32),
                              np.arange(W, dtype=np.float32), indexing='ij')
        point_cam = np.stack([(us - cx) * self.depth / fx,
                              (vs - cy) * self.depth / fy,
                              self.depth], axis=-1).reshape(-1, 3).astype(np.float32)
        point_world = (point_cam @ R.T + t[None, :]).astype(np.float32)

        gz = np.floor((point_world[:, 0] - self.vox_origin[0]) / unit).astype(np.int64)
        gx = np.floor((point_world[:, 1] - self.vox_origin[1]) / unit).astype(np.int64)
        gy = np.floor((point_world[:, 2] - self.vox_origin[2]) / unit).astype(np.int64)

        in_bounds = ((gx >= 0) & (gx < VOX_SIZE_LOW[0])
                     & (gy >= 0) & (gy < VOX_SIZE_LOW[1])
                     & (gz >= 0) & (gz < VOX_SIZE_LOW[2]))
        if self.drop_invalid_depth:
            # same one-voxel effect as in _build_high_res_tsdf; kept in sync so
            # the mapping never claims the camera-origin voxel either
            in_bounds &= self.depth.reshape(-1) > 0

        mapping = np.full(int(np.prod(VOX_SIZE_LOW)), MAPPING_SENTINEL, dtype=np.int64)
        vox_idx = gz * VOX_SIZE_LOW[0] * VOX_SIZE_LOW[1] + gy * VOX_SIZE_LOW[0] + gx
        # many pixels can fall in one voxel; numpy keeps the last, matching the ref
        mapping[vox_idx[in_bounds]] = np.arange(H * W)[in_bounds]
        return mapping