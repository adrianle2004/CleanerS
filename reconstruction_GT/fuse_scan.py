"""
reconstruction_GT/fuse_scan.py

Fuse a hand-held D455 sweep (record_scan.py) into one room mesh, in the
SAME world frame as the capture's still evaluation frame, so the mesh can be
loaded into CloudCompare or Blender and used to place the solids that become
ground truth (MAKING_GT.md, "Build something to annotate against").

    python -m reconstruction_GT.record_scan --preview --out_dir captures/room02
    python -m reconstruction_GT.fuse_scan captures/room02

Writes, all under <capture>/scan/:

    room.ply            fused, coloured triangle mesh, world frame, metres
    trajectory.txt      per fused frame: bag frame index, 4x4 world_T_cam
    fuse_report.json    what was fused, what was dropped, and the checks below

*** WHAT THE MESH IS FOR, AND WHAT IT IS NOT ***

It is scaffolding for placing solids. It is not ground truth, and it must not
be traced into ground truth: it is built from the same sensor as the model
input, so GT derived from it is the NYUCAD condition MAKING_GT.md warns about.
Nothing here writes to meta.json either -- that file records what was measured
at capture time, and a fusion result is not a measurement of the capture.

*** THE WORLD FRAME ***

The mesh is anchored on the first fused frame, which record_scan takes with the
camera still sitting where the still capture left it. So that frame's pose IS
the evaluation frame's pose, and world_T_cam0 is the same live_cam_pose()
matrix the voxel grid is built with, from the MEASURED camera_height and yaw in
meta.json. Nothing about the frame is estimated.

Three checks confirm the premise, and all of them only warn:

  still-frame match   first fused depth vs the still depth frame -- did the
                      camera move between the two captures?
  IMU tilt            gravity from the accelerometer during the hold -- was
                      the camera actually level, as live_cam_pose assumes?
  floor height        lowest large up-facing surface in the mesh -- does it
                      sit at world z = 0, i.e. was camera_height right? Only
                      meaningful if the sweep saw the floor: otherwise the
                      lowest surface is a tabletop and the warning says so.

Without a meta.json the mesh is written in first-frame camera coordinates as
room_camera_frame.ply, and the checks that need a still frame are skipped.

*** TRACKING ***

Open3D's dense SLAM: every frame is tracked against a raycast of the model
built so far (KinectFusion-style), not against the previous frame, so drift
does not compound frame over frame. It tracks on depth geometry alone, which
means a view of nothing but one flat wall is degenerate -- the camera can slide
along it without the error changing. Keep furniture or a corner in view.

Frames whose tracking fails, or whose motion is physically implausible for a
hand-held sweep, are dropped rather than fused; a run of them long enough to
mean the tracker is lost ends fusion there, and the report says where.

Colour: the bag's depth is aligned to the colour camera, whose lens distortion
this pinhole fusion ignores -- up to ~3 cm at the frame corners (see
_build_undistort_maps in inference/camera.py). Fine for placing boxes; not a survey.
"""

import os
import sys
import json
import time
import logging
import argparse

import cv2
import numpy as np

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_THIS_DIR)
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from inference.frame_loader import live_cam_pose, cam_pose_from_meta
from inference.utils import setup_logging

log = logging.getLogger(__name__)

# Motion a hand-held sweep cannot produce between consecutive frames. Tracking
# that reports more has latched onto a wrong alignment, and fusing that frame
# smears a ghost copy of the room into the volume.
MAX_SPEED_M_S = 1.0
MAX_TURN_DEG_S = 120.0
# Frame-to-model fitness below this is a failed track, not a noisy one.
MIN_FITNESS = 0.1
# Consecutive dropped frames (in seconds of recording) after which the tracker
# is treated as lost and fusion stops.
LOST_AFTER_S = 1.0

# check thresholds -- all warnings, see the module docstring
# median |depth difference| vs the still frame. 5 cm, not 3: an unmoved
# camera still reads 2.9-3.1 cm on every good capture so far, because the
# still frame is undistorted and NYU-cropped while the bag frames are raw.
# 3 cm was flagging the method's own noise floor as camera motion.
STILL_MATCH_WARN_M = 0.05
TILT_WARN_DEG = 2.0
FLOOR_WARN_M = 0.04             # ~ half a low-res GT voxel (8 cm)


def parse_args():
    p = argparse.ArgumentParser(
        description='Fuse a record_scan.py sweep into a room mesh.')
    p.add_argument('capture_dir',
                   help='capture folder holding scan/scan.bag (and meta.json '
                        'from the still capture, for the world frame)')
    p.add_argument('--bag', default=None,
                   help='bag to fuse (default <capture_dir>/scan/scan.bag)')
    p.add_argument('--voxel', type=float, default=0.01,
                   help='fusion voxel size in metres. 1 cm is half the GT '
                        'grid\'s 2 cm, which is all annotation needs')
    p.add_argument('--depth_max', type=float, default=3.0,
                   help='ignore depth beyond this many metres. D455 error '
                        'grows with range (~2%% at 4 m); fusing far readings '
                        'blurs every surface they touch')
    p.add_argument('--depth_diff', type=float, default=0.07,
                   help='max point distance for a tracking correspondence')
    p.add_argument('--trunc', type=float, default=8.0,
                   help='TSDF truncation, in voxels')
    p.add_argument('--block_count', type=int, default=30000,
                   help='voxel blocks to reserve. Each 16^3 block is ~50 KB '
                        'on the GPU; raise it if a large room runs out')
    p.add_argument('--skip_s', type=float, default=1.0,
                   help='seconds at the start of the bag to skip while auto-'
                        'exposure settles. Must be shorter than the hold '
                        'record_scan asked for, so the first fused frame is '
                        'still the evaluation pose')
    p.add_argument('--min_weight', type=float, default=3.0,
                   help='mesh only surfaces seen in at least this many frames; '
                        'drops one-frame noise. Lower it for a very short bag')
    p.add_argument('--max_frames', type=int, default=0,
                   help='stop after this many fused frames (0 = all)')
    p.add_argument('--cpu', action='store_true',
                   help='run on the CPU even if CUDA is available (slow)')
    return p.parse_args()


# ---------------------------------------------------------------------------
def _load_json(path):
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


def _check_still_match(depth_m, K_bag, meta, capture_dir):
    """First fused frame vs the still evaluation frame.

    Both are in the colour camera's frame, so if the camera did not move,
    reprojecting the bag's depth into the still frame's intrinsics lands every
    point on the same depth. Compared over the central half of the still image
    only: the still was undistorted and the bag was not, and the two only agree
    where distortion is negligible.
    """
    name = meta['frames'][0]
    still_path = os.path.join(capture_dir, 'depth', name + '.png')
    still = cv2.imread(still_path, cv2.IMREAD_UNCHANGED)
    if still is None:
        return {'status': 'skipped', 'reason': 'cannot read %s' % still_path}
    still = still.astype(np.float32) / 1000.0      # capture.py: plain mm
    H, W = still.shape
    K = np.asarray(meta['cam_K'], np.float64)

    v, u = np.nonzero(depth_m > 0)
    z = depth_m[v, u].astype(np.float64)
    x = (u - K_bag[0, 2]) * z / K_bag[0, 0]
    y = (v - K_bag[1, 2]) * z / K_bag[1, 1]
    u2 = np.round(K[0, 0] * x / z + K[0, 2]).astype(np.int64)
    v2 = np.round(K[1, 1] * y / z + K[1, 2]).astype(np.int64)
    ok = (u2 >= W // 4) & (u2 < 3 * W // 4) & (v2 >= H // 4) & (v2 < 3 * H // 4)
    s = np.zeros_like(z)
    s[ok] = still[v2[ok], u2[ok]]
    ok &= s > 0
    n = int(ok.sum())
    if n < 1000:
        return {'status': 'skipped', 'reason': 'only %d overlapping pixels' % n,
                'still_frame': name}
    diff = z[ok] - s[ok]
    med_abs = float(np.median(np.abs(diff)))
    return {'status': 'warn' if med_abs > STILL_MATCH_WARN_M else 'ok',
            'still_frame': name, 'pixels': n,
            'median_abs_diff_m': round(med_abs, 4),
            'median_diff_m': round(float(np.median(diff)), 4)}


def _check_imu_tilt(bag, skip_s, hold_s):
    """Gravity during the hold, expressed in the colour camera's frame.

    At rest an accelerometer reads the reaction to gravity, i.e. it points UP.
    live_cam_pose assumes up is camera -Y exactly; the angle between the two is
    how far the camera was from level when the grid was placed.
    """
    try:
        import pyrealsense2 as rs
    except ImportError:
        return {'status': 'skipped', 'reason': 'pyrealsense2 not installed'}
    pipe, cfg = rs.pipeline(), rs.config()
    rs.config.enable_device_from_file(cfg, bag, repeat_playback=False)
    cfg.enable_stream(rs.stream.accel)
    cfg.enable_stream(rs.stream.color)
    try:
        profile = pipe.start(cfg)
    except RuntimeError as e:
        return {'status': 'skipped', 'reason': 'no accelerometer stream in the '
                'bag (%s)' % e}
    samples = []
    try:
        profile.get_device().as_playback().set_real_time(False)
        ext = profile.get_stream(rs.stream.accel).get_extrinsics_to(
            profile.get_stream(rs.stream.color))
        # librealsense stores extrinsic rotations column-major
        R = np.asarray(ext.rotation, np.float64).reshape(3, 3).T
        t0 = None
        while True:
            try:
                fs = pipe.wait_for_frames(2000)
            except RuntimeError:
                break                                   # end of file
            for f in fs:
                ts = f.get_timestamp() / 1000.0
                t0 = ts if t0 is None else t0
                if not f.is_motion_frame():
                    continue
                if skip_s <= ts - t0 <= hold_s:
                    m = f.as_motion_frame().get_motion_data()
                    samples.append((m.x, m.y, m.z))
            if t0 is not None and ts - t0 > hold_s:
                break
    finally:
        pipe.stop()

    if len(samples) < 20:
        return {'status': 'skipped',
                'reason': 'only %d accelerometer samples in the hold' % len(samples)}
    a = np.asarray(samples)
    mag = np.linalg.norm(a, axis=1)
    if mag.std() > 0.05 * 9.81:
        return {'status': 'skipped', 'reason': 'camera was moving during the '
                'hold (|a| std %.2f m/s^2)' % mag.std()}
    up = R @ (a.mean(axis=0) / np.linalg.norm(a.mean(axis=0)))
    tilt = float(np.degrees(np.arccos(np.clip(-up[1], -1.0, 1.0))))
    if tilt > 45.0:
        # nobody holds a camera 45 degrees off level for a still frame; this is
        # an IMU-to-camera extrinsic this code has misread, not a real tilt
        return {'status': 'skipped', 'reason': 'implausible %.0f deg tilt -- '
                'IMU extrinsics look inconsistent' % tilt}
    return {'status': 'warn' if tilt > TILT_WARN_DEG else 'ok',
            'samples': len(samples),
            'tilt_deg': round(tilt, 2),
            # what the tilt costs where it matters: the grid assumes level, so
            # this is how far a surface 3 m away sits from where it is drawn
            'offset_at_3m_cm': round(300.0 * np.tan(np.radians(tilt)), 1),
            # positive = camera looking down
            'pitch_down_deg': round(float(np.degrees(np.arctan2(-up[2], -up[1]))), 2),
            'roll_deg': round(float(np.degrees(np.arctan2(up[0], -up[1]))), 2),
            'up_in_camera': [round(float(c), 4) for c in up]}


def _check_floor(mesh, min_area=1.5):
    """Lowest room-sized horizontal plane of the WORLD-frame mesh.

    RANSAC planes, not a histogram of heights: a floor that comes out tilted
    (a camera that was not level does exactly that) smears across height bins
    until no bin looks like a floor, and the tallest bin is then a bed. On
    room08 the bin rule reported the bed at +0.46 while the floor plane sits at
    -0.01, tilted 4.9 degrees by a camera pitched 4.2 degrees up.

    Lowest rather than largest, for the same reason: in a furnished room the
    biggest horizontal surface is often a bed.
    """
    import open3d as o3d
    mesh.compute_vertex_normals()
    V = np.asarray(mesh.vertices)
    N = np.asarray(mesh.vertex_normals)
    up = N[:, 2] > 0.9
    if up.sum() < 2000:
        return {'status': 'skipped', 'reason': 'no large up-facing surface in '
                'the mesh -- point the camera at the floor during the sweep'}
    pc = o3d.geometry.PointCloud()
    pc.points = o3d.utility.Vector3dVector(V[up])
    best = None
    for _ in range(6):
        if len(pc.points) < 2000:
            break
        model, inliers = pc.segment_plane(0.03, 3, 300)
        P = np.asarray(pc.points)[inliers]
        n = np.asarray(model[:3])
        n = n / max(np.linalg.norm(n), 1e-9)
        tilt = float(np.degrees(np.arccos(min(1.0, abs(n[2])))))
        area = 0.04 * len(set(map(tuple, (P[:, :2] // 0.2).astype(int))))
        z = float(np.median(P[:, 2]))
        if tilt < 15.0 and area >= min_area and (best is None or z < best[0]):
            best = (z, len(inliers), area, tilt)
            best_n = n if n[2] > 0 else -n
        pc = pc.select_by_index(inliers, invert=True)
    if best is None:
        return {'status': 'skipped',
                'reason': 'no horizontal plane of at least %.1f m2 -- the sweep '
                          'probably never saw the floor' % min_area}
    z_floor, n_in, area, tilt = best
    best_n = best_n / max(np.linalg.norm(best_n), 1e-9)
    bad = abs(z_floor) > FLOOR_WARN_M or tilt > TILT_WARN_DEG
    out = {'status': 'warn' if bad else 'ok',
           'lowest_up_surface_z_m': round(z_floor, 4),
           'surface_tilt_deg': round(tilt, 2),
           'area_m2': round(area, 1),
           'vertices': int(n_in),
           # refine_tilt.sh --from floor turns this into the camera's true
           # gravity: 30k points on a floor beat an accelerometer's bias
           'normal_world': [round(float(v), 6) for v in best_n]}
    if bad:
        out['meaning'] = ('if this surface is the floor, camera_height or level '
                          'is off; a tilted plane means the camera was not '
                          'level, which the grid cannot represent')
    return out


# ---------------------------------------------------------------------------
def fuse(args, bag, still_meta, rec_meta, capture_dir):
    import open3d as o3d
    import open3d.core as o3c

    device = o3c.Device('CPU:0' if args.cpu or not o3c.cuda.is_available()
                        else 'CUDA:0')
    # bag_reader, not o3d's RSBagReader: that one is a playback device and
    # drops frames when the consumer is slow, which is exactly what a SLAM loop
    # is. The indices written to trajectory.txt below have to mean the same
    # frames when something reads the bag again -- MAKING_GT.md, gotcha 10.
    from reconstruction_GT.bag_reader import bag_metadata, iter_bag
    md = bag_metadata(bag)
    fps = fps_nominal = md['fps']
    scale = md['depth_scale_o3d']        # raw units per metre, Open3D's sense
    K = md['K']
    log.info('%s: %s %dx%d @ %g fps, %.1f s, depth scale %g, on %s',
             os.path.basename(bag), md['device_name'], md['width'],
             md['height'], fps, md['duration_s'], scale, device)

    # Every frame-count rule below -- how much to skip, what motion is
    # possible between frames, when the tracker counts as lost -- is a RATE
    # times a number of frames, so it has to use the rate the recording
    # ACHIEVED. A bandwidth-starved link (a D455 on USB 2 manages ~5 fps at
    # 848x480) still labels the bag 15 fps, and trusting that skips three
    # times too much of it and calls normal hand motion impossible.
    seen = rec_meta.get('frames_seen')
    dur = rec_meta.get('duration_s')
    if seen and dur and dur > 0:
        measured = seen / dur
        if measured < 0.8 * fps_nominal:
            log.warning('recorded at %.1f fps, not the %.0f fps the bag claims '
                        '-- dropped frames at capture time. Using the measured '
                        'rate. If this was USB 2, re-record on USB 3.',
                        measured, fps_nominal)
            fps = measured

    Kt = o3c.Tensor(K, o3c.Dtype.Float64)
    T = np.identity(4)
    model = o3d.t.pipelines.slam.Model(args.voxel, 16, args.block_count,
                                       o3c.Tensor(T), device)
    frame = o3d.t.pipelines.slam.Frame(md['height'], md['width'], Kt, device)
    raycast = o3d.t.pipelines.slam.Frame(md['height'], md['width'], Kt, device)

    max_step = MAX_SPEED_M_S / fps
    max_turn = MAX_TURN_DEG_S / fps
    lost_after = max(1, int(round(LOST_AFTER_S * fps)))
    skip = int(round(args.skip_s * fps))

    poses, fitness, dropped = [], [], []
    still_check = {'status': 'skipped', 'reason': 'no meta.json (no still frame)'}
    read, fused, run_lost, stop_reason = 0, 0, 0, 'end of bag'
    t_start = time.time()
    for bf in iter_bag(bag):
        read += 1
        if read <= skip:
            continue

        if fused == 0 and still_meta is not None:
            still_check = _check_still_match(bf.depth, K, still_meta, capture_dir)

        frame.set_data_from_image('depth', o3d.t.geometry.Image(
            o3c.Tensor(bf.depth_raw[:, :, None])).to(device))
        frame.set_data_from_image('color', o3d.t.geometry.Image(
            o3c.Tensor(np.ascontiguousarray(bf.rgb))).to(device))
        if fused > 0:
            reason = None
            try:
                res = model.track_frame_to_model(frame, raycast, scale,
                                                 args.depth_max, args.depth_diff)
                step = res.transformation.cpu().numpy()
                turn = np.degrees(np.arccos(np.clip(
                    (np.trace(step[:3, :3]) - 1.0) / 2.0, -1.0, 1.0)))
                if res.fitness < MIN_FITNESS:
                    reason = 'fitness %.3f' % res.fitness
                elif np.linalg.norm(step[:3, 3]) > max_step or turn > max_turn:
                    reason = 'jump %.3f m / %.1f deg' % (
                        np.linalg.norm(step[:3, 3]), turn)
            except RuntimeError as e:
                reason = 'tracking error: %s' % e
            if reason is not None:
                dropped.append({'frame': read - 1, 'reason': reason})
                run_lost += 1
                if run_lost >= lost_after:
                    stop_reason = ('tracking lost at %.1f s into the bag: %d '
                                   'consecutive frames dropped' %
                                   ((read - 1) / fps, run_lost))
                    log.warning(stop_reason)
                    break
                continue
            run_lost = 0
            T = T @ step
            fitness.append(float(res.fitness))

        model.update_frame_pose(fused, o3c.Tensor(T))
        model.integrate(frame, scale, args.depth_max, args.trunc)
        model.synthesize_model_frame(raycast, scale, 0.1, args.depth_max,
                                     args.trunc, False)
        poses.append((read - 1, T.copy()))
        fused += 1
        if fused % 100 == 0:
            log.info('fused %d frames (%.1f s of bag, %.1f fps)', fused,
                     (read - 1) / fps, fused / (time.time() - t_start))
        if args.max_frames and fused >= args.max_frames:
            stop_reason = '--max_frames reached'
            break

    if fused == 0:
        raise SystemExit('no frames fused from %s' % bag)
    log.info('fused %d frames, dropped %d; stopped: %s',
             fused, len(dropped), stop_reason)
    mesh = model.extract_trianglemesh(args.min_weight).to_legacy()
    if len(mesh.vertices) == 0:
        raise SystemExit('fused %d frames but no surface was seen in %g or more '
                         'of them -- the bag is too short, or --min_weight too '
                         'high for it' % (fused, args.min_weight))
    return mesh, poses, {
        'bag': os.path.relpath(bag, capture_dir),
        'device': str(device),
        'stream': {'width': md['width'], 'height': md['height'],
                   'fps_nominal': fps_nominal, 'fps_measured': round(fps, 2),
                   'depth_scale': md['depth_scale'], 'cam_K': K.tolist()},
        'params': {'voxel': args.voxel, 'depth_max': args.depth_max,
                   'depth_diff': args.depth_diff, 'trunc': args.trunc,
                   'skip_s': args.skip_s, 'min_weight': args.min_weight},
        'reader': 'reconstruction_GT/bag_reader.py (pyrealsense2, '
                  'real_time=False): frame indices are reproducible',
        'frames_read': read, 'frames_fused': fused,
        'frames_dropped': len(dropped), 'stop_reason': stop_reason,
        'fitness': ({'min': round(min(fitness), 4),
                     'median': round(float(np.median(fitness)), 4)}
                    if fitness else None),
        'dropped': dropped[:200],
        'checks': {'still_match': still_check},
    }


def main():
    setup_logging()
    args = parse_args()
    cap = args.capture_dir
    scan_dir = os.path.join(cap, 'scan')
    bag = args.bag or os.path.join(scan_dir, 'scan.bag')
    if not os.path.exists(bag):
        raise SystemExit('no bag at %s -- record one with '
                         'python -m reconstruction_GT.record_scan --out_dir %s'
                         % (bag, cap))
    os.makedirs(scan_dir, exist_ok=True)

    still_meta = _load_json(os.path.join(cap, 'meta.json'))
    rec_meta = _load_json(os.path.join(scan_dir, 'record.json')) or {}
    hold_s = rec_meta.get('hold_s')
    if hold_s is not None and args.skip_s >= hold_s:
        raise SystemExit('--skip_s %.1f is not shorter than the %.1f s hold '
                         'recorded in record.json: the first fused frame would '
                         'be mid-sweep, not the evaluation pose'
                         % (args.skip_s, hold_s))

    mesh, poses, report = fuse(args, bag, still_meta, rec_meta, cap)

    if still_meta is not None:
        W = cam_pose_from_meta(still_meta).astype(np.float64)
        frame_desc = ('world: live_cam_pose(camera_height=%g, yaw=%g%s) from '
                      'meta.json' % (still_meta['camera_height'],
                                     still_meta.get('yaw', 0.0),
                                     ', measured tilt'
                                     if still_meta.get('up_camera') else
                                     ', assumed level'))
        out_name = 'room.ply'
    else:
        W = np.identity(4)
        frame_desc = ('first fused frame\'s colour camera (x right, y down, '
                      'z forward) -- no meta.json, so no world frame')
        out_name = 'room_camera_frame.ply'
        log.warning('no meta.json in %s: writing the mesh in camera '
                    'coordinates. Take the still frame first (capture.py '
                    '--preview) to get a mesh in the grid\'s world frame.', cap)
    mesh.transform(W)

    report['frame'] = frame_desc
    if hold_s is not None:
        report['checks']['imu_tilt'] = _check_imu_tilt(bag, args.skip_s, hold_s)
        if still_meta is not None and still_meta.get('up_camera') is not None:
            # the grid is built around this measurement now, so the tilt is a
            # fact about the capture rather than an error in it
            t = report['checks']['imu_tilt']
            if t.get('status') == 'warn':
                t['status'] = 'ok'
                t['compensated'] = ('meta.json carries up_camera, so the grid '
                                    'follows gravity rather than assuming level')
    else:
        report['checks']['imu_tilt'] = {'status': 'skipped',
                                        'reason': 'no record.json (hold unknown)'}
    report['checks']['floor'] = (
        _check_floor(mesh) if still_meta is not None else
        {'status': 'skipped', 'reason': 'no world frame'})
    b = mesh.get_axis_aligned_bounding_box()
    report['bounds_m'] = {'min': np.round(b.min_bound, 3).tolist(),
                          'max': np.round(b.max_bound, 3).tolist()}
    cloud_name = out_name.replace('.ply', '_cloud.ply')
    report['mesh'] = {'file': out_name, 'cloud': cloud_name,
                      'vertices': len(mesh.vertices),
                      'triangles': len(mesh.triangles)}

    import open3d as o3d
    o3d.io.write_triangle_mesh(os.path.join(scan_dir, out_name), mesh)
    # the same geometry as points: what box_editor annotates against, and what
    # CloudCompare opens far faster (no triangles to build)
    cloud = o3d.geometry.PointCloud()
    cloud.points = mesh.vertices
    cloud.colors = mesh.vertex_colors
    o3d.io.write_point_cloud(os.path.join(scan_dir, cloud_name), cloud)
    with open(os.path.join(scan_dir, 'trajectory.txt'), 'w') as f:
        # bag frame index only: RSBagReader.get_timestamp() reads 0 on
        # every frame of the bags it was tested on, and a column of zeros
        # would look like data
        f.write('# bag_frame  world_T_cam (4x4, row-major)  -- %s\n'
                % frame_desc)
        for idx, T in poses:
            f.write('%d %s\n' % (idx, ' '.join(
                '%.6f' % v for v in (W @ T).ravel())))
    with open(os.path.join(scan_dir, 'fuse_report.json'), 'w') as f:
        json.dump(report, f, indent=2)

    log.info('wrote %s and %s (%d vertices) to %s', out_name, cloud_name,
             len(mesh.vertices), scan_dir)
    for name, c in report['checks'].items():
        detail = {k: v for k, v in c.items() if k != 'status'}
        (log.warning if c['status'] == 'warn' else log.info)(
            'check %-12s %-7s %s', name, c['status'], detail)
    if report['checks']['still_match']['status'] == 'warn':
        log.warning('the camera moved between the still frame and the start '
                    'of the recording: the mesh is offset from the grid by '
                    'that motion. Re-record starting from the still pose.')


if __name__ == '__main__':
    main()
