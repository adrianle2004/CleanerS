"""
reconstruction_GT/refine_frame_pose.py

Correct one swept frame's pose against the room, the way ROS 2 corrects odometry.

    python -m reconstruction_GT.refine_frame_pose captures/room08            # report
    python -m reconstruction_GT.refine_frame_pose captures/room08 --write    # apply

*** WHY THIS EXISTS ***

`fuse_scan.py` tracks frame-to-model with no loop closure, so its trajectory
drifts: `drift_check.py` measures 175 mm over room07's 7.2 m sweep and 156 mm
over room08's 14.4 m, which is 0.8-1.3 voxels at the frames that carry ground
truth. Perturbing a frame's annotation by that much moves SSC by up to 9 points
and SC by up to 15, so it is not a rounding concern.

The fix here is REP-105's, not KinectFusion's. ROS 2 does not try to make
odometry drift-free -- it keeps `odom -> base_link` continuous but drifting and
lets a localisation node publish `map -> odom` as a correction. We have the
same split already, because only a handful of frames per capture are ever
scored: the trajectory stays as it is (the mesh and the annotation are built on
it), and each EVALUATED frame gets its own correction.

`frame_meta[stem]['world_from_room']` is the slot for it -- a per-frame 4x4,
structurally the same thing as `map -> odom`.

*** WHAT IT ALIGNS TO, AND WHY NOT THE ANNOTATION ***

The target is the fused room cloud by default: measured geometry. Aligning a
frame to the ANNOTATION instead would make the ground truth self-consistent by
construction -- it would place observed surfaces inside annotated solids, which
is exactly what SSC's surface voxels score, so the number would improve whether
or not the pose got better. `--target solids` exists for diagnosis; do not
report a score from a capture refined that way without saying so.

The mesh carries the sweep's drift too, but drift is slow: locally, around one
frame, the mesh is self-consistent, and the annotation was drawn on that mesh.
So frame-to-mesh alignment is as good as the annotation is, with no circularity.

*** FULL WRITE-UP ***

document/DRIFT_AND_ALIGNMENT.md -- the measurement, the two biased ways of
measuring it that both pointed the wrong way, and what is left for the mesh.

*** THE MATH ***

export_frame.py stores, for a frame whose tracked pose is T (world_from_camera,
colour frame, in the ROOM's world):

    P = live_cam_pose(T[2,3], 0, up_from(T))     the canonical evaluation pose
    A = P @ inv(T)                               = world_from_room

so a room point reaches the frame's world at `A p`, with the camera at `P`, and
`inv(P) A p == inv(T) p` -- the point in the camera frame, as tracking says. T
comes back out of the record as `T = inv(A) @ P`, and the frame's observed
points land in room coordinates at `T p_cam`.

ICP returns the correction C that takes those observed points onto the room, so

    T_new = C @ T

and height, `up_camera` and `world_from_room` are then rebuilt from T_new by
exactly the lines export_frame.py uses, keeping the record internally
consistent. Nothing here invents a value: the correction is measured from this
frame's own depth against the fused room, with its fitness and RMSE recorded
beside it -- the same standing as `camera_height` from a floor-plane fit.
"""

import os
import sys
import json
import shutil
import argparse
import logging

import numpy as np

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_THIS_DIR)
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from reconstruction_GT.voxelize_gt import (                      # noqa: E402
    solid_rotation, IGNORE)
from inference.frame_loader import live_cam_pose, frame_meta     # noqa: E402

log = logging.getLogger(__name__)

MAX_CORR = 0.25          # ICP correspondence distance: well above the drift
MIN_FITNESS = 0.3        # below this the frame sees too little of the room
SAMPLE = 400000          # target points


def observed_in_room(cap, stem, fm, T, max_depth=3.0):
    """This frame's depth as a point cloud in ROOM coordinates."""
    import cv2
    import open3d as o3d
    K = np.asarray(fm['cam_K'], np.float64)
    d = cv2.imread(os.path.join(cap, 'depth', stem + '.png'), cv2.IMREAD_UNCHANGED)
    if d is None:
        raise SystemExit('no depth for %s' % stem)
    z = d.astype(np.float32).reshape(-1) / 1000.0
    H, W = d.shape
    vs, us = np.meshgrid(np.arange(H, dtype=np.float32),
                         np.arange(W, dtype=np.float32), indexing='ij')
    ok = (z > 0.3) & (z < max_depth)
    pt = np.stack([(us.reshape(-1) - K[0, 2]) * z / K[0, 0],
                   (vs.reshape(-1) - K[1, 2]) * z / K[1, 1], z], 1)[ok]
    pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(
        pt @ T[:3, :3].T + T[:3, 3]))
    return pcd.voxel_down_sample(0.02)


def target_from_mesh(cap, cam_pos):
    """The fused room cloud, normals oriented toward this camera."""
    import open3d as o3d
    path = os.path.join(cap, 'scan', 'room_cloud.ply')
    if not os.path.exists(path):
        path = os.path.join(cap, 'scan', 'room.ply')
        if not os.path.exists(path):
            raise SystemExit('no scan/room_cloud.ply or room.ply in %s -- '
                             'pull it with tools/hf_data.py' % cap)
    pcd = o3d.io.read_point_cloud(path)
    pcd = pcd.voxel_down_sample(0.02)
    pcd.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(0.1, 30))
    pcd.orient_normals_towards_camera_location(cam_pos)
    return pcd


def target_from_solids(cap, cam_pos):
    """The annotation's visible surfaces. Diagnosis only -- see the header."""
    import open3d as o3d
    spec = json.load(open(os.path.join(cap, 'gt', 'solids.json')))
    meshes = []
    room = spec.get('room')
    if room is not None:
        # the shell's slabs straddle each nominal face, so the surface the
        # camera actually sees is half a thickness inside it
        th = float(room.get('thickness', 0.04))
        c = np.asarray(room['center'], np.float64)
        size = np.asarray(room['size'], np.float64) - th
        m = o3d.geometry.TriangleMesh.create_box(*size)
        m.translate(-size / 2.0)
        m.rotate(solid_rotation(room), center=(0, 0, 0))
        m.translate(c)
        meshes.append(m)
    for s in spec.get('solids', []):
        if s.get('shape', 'box') != 'box':
            continue                      # cylinders/meshes: skip, boxes carry it
        c = np.asarray(s['center'], np.float64)
        size = np.asarray(s['size'], np.float64)
        m = o3d.geometry.TriangleMesh.create_box(*size)
        m.translate(-size / 2.0)
        m.rotate(solid_rotation(s), center=(0, 0, 0))
        m.translate(c)
        meshes.append(m)
    merged = meshes[0]
    for m in meshes[1:]:
        merged += m
    merged.compute_vertex_normals()
    pcd = merged.sample_points_uniformly(SAMPLE, use_triangle_normal=True)
    p = np.asarray(pcd.points)
    n = np.asarray(pcd.normals)
    # keep only surfaces facing the camera: this is what drops the shell's
    # outer faces, which the camera never sees and which sit a wall thickness
    # behind the ones it does
    face = ((cam_pos - p) * n).sum(1) > 0
    out = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(p[face]))
    out.normals = o3d.utility.Vector3dVector(n[face])
    return out


def hit_rate(cap, stem):
    """Share of observed-surface voxels the annotation calls solid."""
    lab = np.load(os.path.join(cap, 'gt', 'Label', stem + '.npz'))['arr_0'].reshape(-1)
    mp = np.load(os.path.join(cap, 'gt', 'Mapping', stem + '.npz'))['arr_0'].reshape(-1)
    seen = (mp != 307200) & (lab != IGNORE)
    if seen.sum() == 0:
        return float('nan')
    return 100.0 * float((lab[seen] > 0).mean())


def refine_one(cap, stem, meta, target_kind):
    """-> dict with the correction, or None if the frame cannot be refined."""
    import open3d as o3d
    fm = frame_meta(meta, stem)
    if 'world_from_room' not in fm:
        log.info('%s: no world_from_room (the still frame) -- skipped', stem)
        return None
    A = np.asarray(fm['world_from_room'], np.float64)
    P = live_cam_pose(fm['camera_height'], fm.get('yaw', 0.0),
                      fm.get('up_camera')).astype(np.float64)
    T = np.linalg.inv(A) @ P                    # the tracked pose, recovered

    src = observed_in_room(cap, stem, fm, T)
    cam_pos = T[:3, 3]
    tgt = (target_from_solids(cap, cam_pos) if target_kind == 'solids'
           else target_from_mesh(cap, cam_pos))
    src.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(0.1, 30))
    res = o3d.pipelines.registration.registration_icp(
        src, tgt, MAX_CORR, np.eye(4),
        o3d.pipelines.registration.TransformationEstimationPointToPlane(),
        o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=60))
    C = np.asarray(res.transformation, np.float64)
    shift = float(np.linalg.norm(C[:3, 3]))
    turn = float(np.degrees(np.arccos(np.clip(
        (np.trace(C[:3, :3]) - 1.0) / 2.0, -1.0, 1.0))))
    T_new = C @ T
    h = float(T_new[2, 3])
    up = (T_new[:3, :3].T @ np.array([0.0, 0.0, 1.0])).tolist()
    A_new = live_cam_pose(h, 0.0, up).astype(np.float64) @ np.linalg.inv(T_new)
    return {'stem': stem, 'fitness': float(res.fitness),
            'rmse_mm': 1000 * float(res.inlier_rmse),
            'shift_mm': 1000 * shift, 'turn_deg': turn,
            'camera_height': round(h, 4),
            'up_camera': [round(float(v), 6) for v in up],
            'world_from_room': [[round(float(v), 6) for v in r] for r in A_new],
            'prev': {'camera_height': fm['camera_height'],
                     'up_camera': fm.get('up_camera'),
                     'world_from_room': A.tolist()},
            'target': target_kind}


def main():
    from inference.utils import setup_logging
    p = argparse.ArgumentParser(description=__doc__.strip().splitlines()[2])
    p.add_argument('capture_dir')
    p.add_argument('--frame', nargs='+', default=None,
                   help='stems to refine (default: every frame with a pose)')
    p.add_argument('--target', choices=['mesh', 'solids'], default='mesh',
                   help='mesh (measured, the default) or solids (diagnosis)')
    p.add_argument('--write', action='store_true',
                   help='apply it; meta.json is backed up to .preicp first')
    p.add_argument('--max_shift', type=float, default=0.30,
                   help='refuse a correction larger than this, in metres: ICP '
                        'has found something other than the same room')
    a = p.parse_args()
    setup_logging()
    cap = a.capture_dir.rstrip('/')
    meta = json.load(open(os.path.join(cap, 'meta.json')))
    stems = a.frame or list(meta.get('frames', []))

    out, refused = [], []
    for stem in stems:
        r = refine_one(cap, stem, meta, a.target)
        if r is None:
            continue
        why = None
        if r['fitness'] < MIN_FITNESS:
            why = 'fitness %.2f below %.2f' % (r['fitness'], MIN_FITNESS)
        elif r['shift_mm'] / 1000.0 > a.max_shift:
            why = 'shift %.0f mm over the %.0f mm limit' % (
                r['shift_mm'], 1000 * a.max_shift)
        (refused if why else out).append((r, why))

    print('\n%-14s %8s %9s %9s %9s %10s' %
          ('frame', 'fitness', 'rmse', 'shift', 'turn', 'verdict'))
    for r, why in sorted(out + refused, key=lambda x: x[0]['stem']):
        print('%-14s %8.3f %6.1f mm %6.1f mm %7.3f° %10s'
              % (r['stem'], r['fitness'], r['rmse_mm'], r['shift_mm'],
                 r['turn_deg'], 'refuse' if why else 'apply'))
        if why:
            print('%-14s   %s' % ('', why))
    print('\ntarget: %s   %d to apply, %d refused'
          % (a.target, len(out), len(refused)))

    if not a.write:
        print('\nnothing written. --write applies it (meta.json backed up to '
              'meta.json.preicp), then re-run voxelize_gt and evaluate_gt for '
              'the frames it touched.')
        return 0
    if not out:
        print('\nnothing to apply.')
        return 0

    bak = os.path.join(cap, 'meta.json.preicp')
    if not os.path.exists(bak):
        shutil.copy(os.path.join(cap, 'meta.json'), bak)
        log.info('backed up meta.json -> %s', os.path.basename(bak))
    for r, _ in out:
        e = meta['frame_meta'][r['stem']]
        e['camera_height'] = r['camera_height']
        e['up_camera'] = r['up_camera']
        e['world_from_room'] = r['world_from_room']
        e['pose_refined'] = {
            'method': 'point-to-plane ICP of this frame\'s depth against the '
                      'fused room (%s), correcting the frame-to-model drift '
                      'measured by drift_check.py' % r['target'],
            'fitness': round(r['fitness'], 4),
            'inlier_rmse_mm': round(r['rmse_mm'], 2),
            'shift_mm': round(r['shift_mm'], 2),
            'turn_deg': round(r['turn_deg'], 4),
            'previous': r['prev'],
        }
        e['camera_height_source'] = (
            '%s; then ICP-refined against the fused room (shift %.0f mm)'
            % (r['prev'].get('camera_height') is not None and
               e.get('camera_height_source', 'tracked pose') or 'tracked pose',
               r['shift_mm']))
    json.dump(meta, open(os.path.join(cap, 'meta.json'), 'w'), indent=2)
    print('\nwrote %s for %d frame(s). Now:' % (
        os.path.join(cap, 'meta.json'), len(out)))
    # voxelize_gt takes one --frame at a time
    for r, _ in out:
        print('  python -m reconstruction_GT.voxelize_gt %s --frame %s'
              % (cap, r['stem']))
    print('  python -m reconstruction_GT.evaluate_gt %s --report' % cap)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
