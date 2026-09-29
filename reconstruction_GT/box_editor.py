"""
reconstruction_GT/box_editor.py

Put 3D boxes into the room cloud and shape them with the keyboard, watching
them in the cloud as you go. Writes gt/solids.json for voxelize_gt.py.

    python -m reconstruction_GT.box_editor captures/room07

KEYS

    N           new solid, dropped at the middle of the room
    TAB or .    next solid          , (comma)  previous solid
    L           list every solid in the terminal, with the current one marked
    M           edit the ROOM SHELL itself -- move its walls with A/W/S, the
                same as any solid. Use it where a fit cannot work: a wall
                behind a window, where the depth goes through the glass.
    BACKSPACE   delete this one
    arrows      move X / Y          R / V  move up / down  (or PAGE UP/DN)
    A           pick a FACE: +x -x +y -y +z(top) -z(bottom)
                the one you picked is drawn in YELLOW
    W / S       push that face out / pull it in -- the opposite face stays put
    Q / E       rotate (yaw) -5 / +5 degrees
    H           shape: box -> cylinder -> wedge   (clears a CAD mesh)
    C           put the next CAD mesh from the library into this solid
    K           pick the rotation axis: z (yaw) -> y (pitch) -> x (roll),
                then Q / E turn about it -- a cylinder has to lie down to be
                a bottle or a pipe
    F           FIT: snap it to the points inside it
    G           drop it to the floor (bottom to z = 0)
    I           snap it INTO the nearest wall: the room's yaw, centred on the
                wall plane, 12 cm thick -- for windows and doorways
    [ / ]       step size: 1, 2, 5, 10, 20 cm
    1..9        set the class 1..9  (ceiling floor wall window chair
                bed sofa table tvs)
    U           furn (10)          O    objs (11)
    ENTER       save gt/solids.json           ESC  quit without saving

The blue wireframe is the room shell, fitted to the room's own walls. It
writes floor, ceiling and walls for you -- do not box those yourself.

SHAPES: a box for most furniture; a cylinder for a bin, a lamp base or a stool
(size is the two diameters and the height); a wedge for a ramp, full height at
its -x face tapering to nothing at +x. All three fill solid, as ground truth
must.

THE USUAL LOOP: N, drive it roughly over the object with the arrows, press F to
snap it to the points, G to sit it on the floor, then a digit for the class.
The terminal prints the box's size in metres after every change, which is what
you check against the real furniture.

*** F ONLY SEES WHAT THE CAMERA SAW ***

The fit uses the points inside the box, and the sensor only ever saw the front
and top of a bed. So F gives you the visible extent, and the far side against a
wall is yours to extend with the arrows -- MAKING_GT.md, "Annotate solids, not
surfaces". G matters for the same reason: a bed's ground truth is solid from
the floor up, not a duvet-thick shell.
"""

import os
import sys
import glob
import json
import logging
import argparse

import numpy as np

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_THIS_DIR)
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from reconstruction_GT.voxelize_gt import CLASSES, mesh_transform

log = logging.getLogger(__name__)

# GLFW key codes the legacy visualiser passes through
K_ENTER, K_TAB, K_BACKSPACE, K_ESC = 257, 258, 259, 256
K_RIGHT, K_LEFT, K_DOWN, K_UP = 262, 263, 264, 265
K_PGUP, K_PGDN = 266, 267
STEPS = [0.01, 0.02, 0.05, 0.10, 0.20]
# the six faces W/S push and pull, as (axis, sign). A cycles through them, so
# either side of the solid can be moved on its own -- a bed grows towards the
# wall without its front edge shifting.
FACES = [(0, +1), (0, -1), (1, +1), (1, -1), (2, +1), (2, -1)]
FACE_NAMES = ['+x', '-x', '+y', '-y', '+z(top)', '-z(bottom)']
# corners of each face, in the order shape_lines() emits box corners
# (index = 4*sx + 2*sy + sz, each 0 for -1 and 1 for +1)
FACE_CORNERS = [[4, 5, 7, 6], [0, 1, 3, 2], [2, 3, 7, 6],
                [0, 1, 5, 4], [1, 3, 7, 5], [0, 2, 6, 4]]
SHAPES = ['box', 'cylinder', 'wedge']
ROT_AXES = ['z (yaw)', 'y (pitch)', 'x (roll)']
ROT_IDX = [2, 1, 0]


def fit_box_to_points(P, pad=0.0):
    """Points (N,3) -> (centre, size, yaw_deg), box kept upright.

    Yaw is the principal horizontal direction of the points; a box and its
    90-degree rotation are the same box, so yaw folds into [-45, 45).
    """
    xy = P[:, :2] - P[:, :2].mean(axis=0)
    vec = np.linalg.svd(xy, full_matrices=False)[2][0]
    yaw = np.degrees(np.arctan2(vec[1], vec[0]))
    yaw = (yaw + 45) % 90 - 45
    c, s = np.cos(np.radians(yaw)), np.sin(np.radians(yaw))
    R = np.array([[c, s], [-s, c]])                  # world -> box frame
    local = np.column_stack([P[:, :2] @ R.T, P[:, 2]])
    lo, hi = local.min(axis=0) - pad, local.max(axis=0) + pad
    size = hi - lo
    centre_local = (lo + hi) / 2.0
    centre = np.array([*(R.T @ centre_local[:2]), centre_local[2]])
    return centre, size, yaw


# facing walls must be at least this far apart, so the two peaks picked for an
# axis cannot both land on the same wall
MIN_ROOM_SIZE = 1.2
# how far from the cloud's edge to look for each wall plane
SEARCH_NEAR_EDGE = 0.40
# thickness the I key gives a solid snapped onto a wall. NYU's windows are 6-14
# voxels across the wall normal (12-28 cm), a planar patch lying IN the wall
# plane rather than a slab poking into the room.
WALL_SNAP_THICKNESS = 0.12


def manhattan_yaw(N_horizontal):
    """Dominant wall direction, as Guo & Hoiem align a room.

    They vote surface normals onto the x/y/z axes and re-solve the rotation
    until it stops moving (ICCV 2013, sec. 2.2). Gravity is already fixed here
    -- the grid is floor-anchored -- so only the yaw is unknown, and a circular
    mean over 4*theta finds it in closed form: multiplying by four makes the
    four wall directions of a Manhattan room coincide, so they reinforce
    instead of cancelling.
    """
    theta = np.arctan2(N_horizontal[:, 1], N_horizontal[:, 0])
    mean4 = np.mean(np.exp(4j * theta))
    return np.degrees(np.angle(mean4) / 4.0)


def fit_room_shell(P, N, height, min_support=300, bin_m=0.02):
    """Walls from the dominant planes in each Manhattan direction.

    -> (room dict, per-wall support). Each wall is the strongest plane facing
    that way, not the outermost point, so a wardrobe or a sliver of cloud
    through a doorway cannot push a wall outwards. A direction with too few
    votes falls back to the extreme point and is reported, which is how you
    find out a wall was never really seen -- NYU's own rooms are often
    annotated with only two or three walls.
    """
    horiz = np.abs(N[:, 2]) < 0.3
    if horiz.sum() < 4 * min_support:
        raise SystemExit('too few wall-like normals (%d) to align the room'
                         % int(horiz.sum()))
    yaw = manhattan_yaw(N[horiz])
    R = _yaw_R(-yaw)                       # world -> room frame
    Q, Nr = P @ R.T, N @ R.T

    # Where the room ENDS is a question about its boundary, so each wall is
    # looked for near the edge of the cloud, not among the strongest planes
    # anywhere: the biggest flat thing in a bedroom is usually the bed or a
    # wardrobe front, and picking that collapses the room onto the furniture.
    # Percentiles rather than min/max, so a few stray points cannot drag an
    # edge outwards.
    bounds, support = {}, {}
    for axis in (0, 1):
        face_on = np.abs(Nr[:, axis]) > 0.8
        coords = Q[face_on, axis]
        lo_prior, hi_prior = np.percentile(Q[:, axis], [0.5, 99.5])
        got = []
        for prior in (lo_prior, hi_prior):
            near = coords[np.abs(coords - prior) < SEARCH_NEAR_EDGE]
            if near.size < min_support:
                got.append((float(prior), 0))
                continue
            hist, edges = np.histogram(near, bins=np.arange(
                near.min(), near.max() + bin_m, bin_m))
            peak = float(edges[int(np.argmax(hist))] + bin_m / 2)
            got.append((peak, int(np.sum(np.abs(coords - peak) < 0.05))))
        if got[1][0] - got[0][0] < MIN_ROOM_SIZE:      # implausible room
            bounds[axis] = (float(lo_prior), float(hi_prior))
            support[axis] = (0, 0)
        else:
            bounds[axis] = (got[0][0], got[1][0])
            support[axis] = (got[0][1], got[1][1])

    cx = (bounds[0][0] + bounds[0][1]) / 2.0
    cy = (bounds[1][0] + bounds[1][1]) / 2.0
    centre_world = _yaw_R(yaw) @ np.array([cx, cy, 0.0])
    room = {'center': [round(float(centre_world[0]), 3),
                       round(float(centre_world[1]), 3), round(height / 2, 3)],
            'size': [round(float(bounds[0][1] - bounds[0][0]), 3),
                     round(float(bounds[1][1] - bounds[1][0]), 3),
                     round(float(height), 3)],
            'yaw_deg': round(float(yaw), 1), 'thickness': 0.04}
    return room, {'-x': support[0][0], '+x': support[0][1],
                  '-y': support[1][0], '+y': support[1][1]}


def _yaw_R(yaw_deg):
    c, s = np.cos(np.radians(yaw_deg)), np.sin(np.radians(yaw_deg))
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def _axis_R(axis, deg):
    """Rotation about one of the solid's OWN axes."""
    c, s = np.cos(np.radians(deg)), np.sin(np.radians(deg))
    if axis == 0:
        return np.array([[1.0, 0, 0], [0, c, -s], [0, s, c]])
    if axis == 1:
        return np.array([[c, 0, s], [0, 1.0, 0], [-s, 0, c]])
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1.0]])


def yaw_of(R):
    """The yaw part of a rotation, for printing."""
    return float(np.degrees(np.arctan2(R[1, 0], R[0, 0])))


def shape_lines(shape, centre, size, rot):
    """Wireframe for a solid, matching what voxelize_gt will fill."""
    R = rot if isinstance(rot, np.ndarray) else _yaw_R(float(rot))
    h = np.asarray(size) / 2.0
    if shape == 'cylinder':
        n = 24
        a = np.linspace(0, 2 * np.pi, n, endpoint=False)
        ring = np.stack([h[0] * np.cos(a), h[1] * np.sin(a)], axis=1)
        local = np.vstack([np.column_stack([ring, np.full(n, -h[2])]),
                           np.column_stack([ring, np.full(n, h[2])])])
        edges = ([[i, (i + 1) % n] for i in range(n)] +
                 [[n + i, n + (i + 1) % n] for i in range(n)] +
                 [[i, n + i] for i in range(0, n, 6)])
    elif shape == 'wedge':
        # full height at -x, nothing at +x
        local = np.array([[-h[0], -h[1], -h[2]], [-h[0], h[1], -h[2]],
                          [-h[0], -h[1], h[2]], [-h[0], h[1], h[2]],
                          [h[0], -h[1], -h[2]], [h[0], h[1], -h[2]]])
        edges = [[0, 1], [0, 2], [1, 3], [2, 3], [0, 4], [1, 5],
                 [4, 5], [2, 4], [3, 5]]
    else:                                    # box, and the cage round a mesh
        local = np.array([[sx * h[0], sy * h[1], sz * h[2]]
                          for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)])
        edges = [[0, 1], [0, 2], [0, 4], [1, 3], [1, 5], [2, 3],
                 [2, 6], [3, 7], [4, 5], [4, 6], [5, 7], [6, 7]]
    return local @ R.T + np.asarray(centre), edges


class Box(object):
    def __init__(self, centre, size, rot=None, cid=None, shape='box', mesh=None):
        self.centre = np.asarray(centre, float)
        self.size = np.asarray(size, float)
        # a full 3x3, not a yaw: a cylinder is only useful for a bottle or a
        # pipe if it can lie down, and that is a rotation about x or y
        if rot is None:
            self.R = np.eye(3)
        elif np.ndim(rot) == 0:
            self.R = _yaw_R(float(rot))
        else:
            self.R = np.asarray(rot, float).reshape(3, 3)
        self.cid = cid
        self.shape = shape
        self.mesh = mesh

    @property
    def yaw(self):
        return yaw_of(self.R)

    def to_json(self):
        d = {'class': CLASSES[self.cid], 'shape': self.shape,
             'center': [round(float(v), 3) for v in self.centre],
             'size': [round(float(v), 3) for v in self.size],
             'rotation': [[round(float(v), 6) for v in row] for row in self.R],
             'yaw_deg': round(self.yaw, 1)}      # readable, not authoritative
        if self.mesh:
            d['mesh'] = self.mesh
        return d

    def corners(self):
        h = self.size / 2.0
        local = np.array([[sx * h[0], sy * h[1], sz * h[2]]
                          for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)])
        return local @ self.R.T + self.centre

    def contains(self, P):
        local = (P - self.centre) @ self.R
        return np.all(np.abs(local) <= self.size / 2.0, axis=1)


def main():
    from inference.utils import setup_logging
    setup_logging()
    p = argparse.ArgumentParser(description='Place 3D boxes in a room cloud.')
    p.add_argument('capture_dir')
    p.add_argument('--cloud', default=None)
    p.add_argument('--voxel', type=float, default=0.02,
                   help='downsample the displayed cloud, metres')
    p.add_argument('--cad_dir', default=None,
                   help='folder of CAD meshes to cycle with C (default '
                        '<capture>/gt/cad, then reconstruction_GT/cad)')
    p.add_argument('--refit_room', action='store_true',
                   help='fit the room shell again, discarding the one saved in '
                        'solids.json (which may have been edited by hand)')
    p.add_argument('--no_ceiling', action='store_true',
                   help='the sweep never saw the ceiling: leave it out, and let '
                        'the walls and the empty interior run to the top of the '
                        'grid, as NYU does when a frame has no ceiling')
    p.add_argument('--room_height', type=float, default=None,
                   help='ceiling height in metres. The default is the cloud\'s '
                        'highest point, which is stray points as often as it is '
                        'ceiling -- measure it, or read it off the fused mesh')
    args = p.parse_args()
    import open3d as o3d

    cap = args.capture_dir
    cloud_path = args.cloud
    if cloud_path is None:
        for name in ('room_cloud.ply', 'room.ply',           # a level capture
                     'room_camera_frame_cloud.ply', 'room_camera_frame.ply'):
            cand = os.path.join(cap, 'scan', name)
            if os.path.exists(cand):
                cloud_path = cand
                break
    if cloud_path is None or not os.path.exists(cloud_path):
        raise SystemExit('no fused cloud or mesh in %s -- run 5_fuse_scan.sh '
                         'first' % os.path.join(cap, 'scan'))
    pcd = o3d.io.read_point_cloud(cloud_path)
    if args.voxel > 0:
        pcd = pcd.voxel_down_sample(args.voxel)
    P = np.asarray(pcd.points)
    lo, hi = P.min(axis=0), P.max(axis=0)
    # The room shell: walls are the dominant plane in each Manhattan
    # direction, the way Guo & Hoiem annotated NYU -- not a rectangle round
    # the outermost points, which a wardrobe or a doorway can push outwards.
    # This writes floor, ceiling and walls, so you never annotate those.
    if not pcd.has_normals():
        log.info('estimating normals for the wall fit...')
        pcd.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(0.12, 30))
    height = args.room_height if args.room_height else float(hi[2])
    # A sweep that never pointed up has no ceiling in it, and the cloud's top
    # is then a wardrobe or a stray point rather than the ceiling. Say so.
    top = P[P[:, 2] > height - 0.15]
    if args.room_height is None and (len(top) < 500 or
                                     top[:, 0].ptp() < 1.5 or top[:, 1].ptp() < 1.5):
        log.warning('the top of the cloud (%.2f m) spans only %.1f x %.1f m -- '
                    'that is probably not the ceiling. Pass --room_height with '
                    'the measured height or --no_ceiling, or the shell puts '
                    'the ceiling there.',
                    height, top[:, 0].ptp() if len(top) else 0,
                    top[:, 1].ptp() if len(top) else 0)
    out_path = os.path.join(cap, 'gt', 'solids.json')
    saved_room = None
    if os.path.exists(out_path):
        saved_room = json.load(open(out_path)).get('room')
        if saved_room is not None and 'center' not in saved_room:
            saved_room = None                  # the old min/max form: refit
    if saved_room is not None and not args.refit_room:
        room, wall_support = saved_room, {}
        log.info('using the room shell saved in solids.json (--refit_room to '
                 'fit it again)')
    else:
        room, wall_support = fit_room_shell(P, np.asarray(pcd.normals), height)
    weak = [k for k, v in wall_support.items() if v < 300]
    log.info('wall support: %s', '  '.join('%s %d' % kv for kv in
                                           sorted(wall_support.items())))
    if weak:
        log.warning('walls %s have little or no support in the cloud -- the '
                    'shell falls back to the outermost points there. NYU often '
                    'annotates only the walls it saw; tell me if one of these '
                    'is a doorway rather than a wall.', ', '.join(weak))
    if args.no_ceiling:
        room['ceiling'] = False
        log.info('no ceiling will be written; walls run to the top of the grid')
    log.info('%s: %d points | room shell %.2f x %.2f x %.2f m at %.1f deg',
             os.path.basename(cloud_path), len(P), room['size'][0],
             room['size'][1], height, room['yaw_deg'])

    boxes, state = [], {'i': -1, 'step': 2, 'face': 0, 'rot': 0, 'cad': -1,
                        'save': False}
    SHELL = -2                     # state['i'] == SHELL -> editing the room

    # reload anything already annotated, so this is editable across sessions
    if os.path.exists(out_path):
        old = json.load(open(out_path))
        for s in old.get('solids', []):
            cid = CLASSES.index(s['class']) if s['class'] in CLASSES else 11
            boxes.append(Box(s['center'], s['size'],
                             s.get('rotation', s.get('yaw_deg', 0.0)), cid,
                             s.get('shape', 'box'), s.get('mesh')))
        if boxes:
            state['i'] = 0
            log.info('loaded %d boxes from the existing solids.json', len(boxes))

    shell = Box(room['center'], room['size'], room['yaw_deg'])

    # the CAD library. NYU used 30 SketchUp models for 6 furniture categories;
    # a chair or a table fills only ~15% of its bounding box, so a real mesh is
    # worth far more there than anywhere else.
    cad_dirs = ([args.cad_dir] if args.cad_dir else
                [os.path.join(cap, 'gt', 'cad'), os.path.join(_THIS_DIR, 'cad')])
    cad_files = []
    for d in cad_dirs:
        for ext in ('*.ply', '*.obj', '*.stl', '*.off'):
            cad_files.extend(sorted(glob.glob(os.path.join(d, ext))))
    if cad_files:
        log.info('%d CAD meshes available (C cycles them): %s', len(cad_files),
                 ', '.join(os.path.basename(f) for f in cad_files[:6]) +
                 (' ...' if len(cad_files) > 6 else ''))
    mesh_cache = {}

    vis = o3d.visualization.VisualizerWithKeyCallback()
    vis.create_window('box editor -- N new  F fit  G floor  digits class  '
                      'ENTER save  ESC quit', 1500, 950)
    vis.add_geometry(pcd)
    lines, meshes = [], []

    def add_lineset(pts, edges, colour):
        ls = o3d.geometry.LineSet(o3d.utility.Vector3dVector(pts),
                                  o3d.utility.Vector2iVector(edges))
        ls.colors = o3d.utility.Vector3dVector(np.tile(colour, (len(edges), 1)))
        vis.add_geometry(ls, reset_bounding_box=False)
        lines.append(ls)

    def load_mesh(path):
        if path not in mesh_cache:
            m = o3d.io.read_triangle_mesh(path)
            if len(m.vertices) == 0:
                print('\n  cannot read %s' % path)
                mesh_cache[path] = None
            else:
                m.compute_vertex_normals()
                mesh_cache[path] = m
        return mesh_cache[path]

    def rebuild():
        for ls in lines:
            vis.remove_geometry(ls, reset_bounding_box=False)
        lines.clear()
        for m in meshes:
            vis.remove_geometry(m, reset_bounding_box=False)
        meshes.clear()
        # the room shell, dim blue: floor, ceiling and walls come from this
        rp, re = shape_lines('box', shell.centre, shell.size, shell.R)
        add_lineset(rp, re, [1.0, 0.2, 0.1] if state['i'] == SHELL
                    else [0.25, 0.45, 0.9])
        b = cur()
        if b is not None:
            # the face W/S will move, in yellow -- always the enclosing box's
            # face, whatever the shape
            cp, _ = shape_lines('box', b.centre, b.size, b.R)
            ring = FACE_CORNERS[state['face']]
            add_lineset(cp[ring], [[0, 1], [1, 2], [2, 3], [3, 0]],
                        [1.0, 0.95, 0.1])
        for j, b in enumerate(boxes):
            hot = (j == state['i'])
            # a mesh solid shows the mesh AND the cage you resize
            pts, edges = shape_lines('box' if b.mesh else b.shape,
                                     b.centre, b.size, b.R)
            add_lineset(pts, edges, [1.0, 0.2, 0.1] if hot else [0.1, 0.8, 0.3])
            if b.mesh:
                src = load_mesh(b.mesh)
                if src is not None:
                    m = o3d.geometry.TriangleMesh(src)
                    m.transform(mesh_transform(src, b.centre, b.size, b.R))
                    m.paint_uniform_color([1.0, 0.45, 0.35] if hot
                                          else [0.35, 0.75, 0.45])
                    m.compute_vertex_normals()
                    vis.add_geometry(m, reset_bounding_box=False)
                    meshes.append(m)
        status()

    def status():
        if state['i'] != SHELL and (state['i'] < 0 or not boxes):
            print('\r  no box -- press N to add one', end='', flush=True)
            return
        b = cur()
        tag = ('ROOM SHELL' if b is shell else '%d/%d %-8s'
               % (state['i'] + 1, len(boxes),
                  os.path.basename(b.mesh) if b.mesh else b.shape))
        # kept under 80 columns: this is reprinted with \r after every
        # keypress, and a line that wraps smears down the terminal instead
        print('\r%-14s %-7s %4.2fx%4.2fx%4.2f @%5.2f,%5.2f,%5.2f y%+4.0f '
              '%s %s %.0fcm  '
              % (tag[:14], '' if b is shell else
                 (CLASSES[b.cid] if b.cid else '?')[:7],
                 b.size[0], b.size[1], b.size[2],
                 b.centre[0], b.centre[1], b.centre[2], b.yaw,
                 FACE_NAMES[state['face']], ROT_AXES[state['rot']][0],
                 STEPS[state['step']] * 100),
              end='', flush=True)

    def cur():
        if state['i'] == SHELL:
            return shell
        return boxes[state['i']] if 0 <= state['i'] < len(boxes) else None

    # things that make no sense for the room shell
    SOLID_ONLY = ('set_class', 'delete', 'cycle_shape')

    def act(fn):
        def cb(vis_):
            b = cur()
            if b is None and fn.__name__ not in ('new_box',):
                return False
            if b is shell and fn.__name__ in SOLID_ONLY:
                print('\n  that is the room shell: no class, no shape, cannot '
                      'be deleted. M returns to your solids.')
                return False
            fn(b)
            rebuild()
            return True
        return cb

    def new_box(_b):
        centre = [(lo[0] + hi[0]) / 2, (lo[1] + hi[1]) / 2, 0.4]
        boxes.append(Box(centre, [0.6, 0.6, 0.8]))
        state['i'] = len(boxes) - 1

    def toggle_shell(_b):
        """Edit the room itself. A wall behind a window is the reason this
        exists: IR goes through glass, so the cloud carries points metres
        past the wall and no fit can tell them from the room. Guo & Hoiem's
        annotators drew each wall by hand in the overhead view for the same
        reason."""
        state['i'] = 0 if state['i'] == SHELL else SHELL

    def move(dx, dy, dz):
        def fn(b):
            b.centre += np.array([dx, dy, dz]) * STEPS[state['step']]
        fn.__name__ = 'move'
        return fn

    def resize(sign):
        """Move ONE face: the opposite face stays where it is."""
        def fn(b):
            axis, face_sign = FACES[state['face']]
            step = sign * STEPS[state['step']]
            new = b.size[axis] + step
            if new < 0.05:
                return
            b.size[axis] = new
            # shift the centre by half, along the face's direction in the
            # solid's own (yawed) frame, so only that face moves
            d = b.R @ (np.eye(3)[axis] * face_sign)
            b.centre += d * step / 2.0
        fn.__name__ = 'resize'
        return fn

    def rotate(d):
        def fn(b):
            b.R = b.R @ _axis_R(ROT_IDX[state['rot']], d)
        fn.__name__ = 'rotate'
        return fn

    def cycle_rot_axis(_b):
        state['rot'] = (state['rot'] + 1) % len(ROT_AXES)

    def fit(b):
        if b is shell:
            new, sup = fit_room_shell(P, np.asarray(pcd.normals), b.size[2])
            b.centre = np.asarray(new['center'], float)
            b.size = np.asarray(new['size'], float)
            b.yaw = new['yaw_deg']
            print('\n  walls re-fitted: %s' % sup)
            return
        inside = b.contains(P)
        if inside.sum() < 30:
            print('\n  only %d points inside -- move the box over the object '
                  'first' % int(inside.sum()))
            return
        c, sz, yaw = fit_box_to_points(P[inside], 0.01)
        b.centre, b.size, b.R = c, sz, _yaw_R(yaw)
        print('\n  fitted to %d points' % int(inside.sum()))

    def snap_to_wall(b):
        """Lay the solid flat in the nearest wall of the room shell.

        Windows and doorways are openings IN a wall: NYU models them as
        polygons on the wall plane, which voxelize to a thin planar patch. The
        voxelizer paints the shell first and solids after, so a window snapped
        here overwrites the wall it sits in, exactly as NYU's does.
        """
        if b is shell:
            print('\n  the shell is the walls -- nothing to snap it to')
            return
        local = shell.R.T @ (b.centre - shell.centre)
        best = None
        for axis in (0, 1):
            for sign in (-1, +1):
                plane = sign * shell.size[axis] / 2.0
                d = abs(local[axis] - plane)
                if best is None or d < best[0]:
                    best = (d, axis, plane)
        d, axis, plane = best
        local[axis] = plane
        b.centre = shell.centre + shell.R @ local
        b.R = shell.R.copy()
        b.size[axis] = WALL_SNAP_THICKNESS
        print('\n  snapped onto the %s%s wall (moved %.2f m), %.0f cm thick'
              % ('-+'[int(plane > 0)], 'xy'[axis], d, WALL_SNAP_THICKNESS * 100))

    def to_floor(b):
        """Sit it on the floor. With a full rotation the bottom is the lowest
        CORNER, not centre minus half the height, so a tilted solid still lands
        on z = 0 instead of sinking through it."""
        z = b.corners()[:, 2]
        if b.shape == 'box' and abs(b.R[2, 2] - 1.0) < 1e-6:
            top = z.max()                       # upright: keep the top, grow down
            b.size[2] = max(0.05, top)
            b.centre[2] = b.size[2] / 2.0
        else:
            b.centre[2] -= z.min()              # tilted or a mesh: just drop it

    def set_class(cid):
        def fn(b):
            b.cid = cid
        fn.__name__ = 'set_class'
        return fn

    def delete(_b):
        boxes.pop(state['i'])
        state['i'] = min(state['i'], len(boxes) - 1)

    def next_box(_b):
        state['i'] = (state['i'] + 1) % len(boxes)

    def prev_box(_b):
        state['i'] = (state['i'] - 1) % len(boxes)

    def list_boxes(_b):
        print('\n  all solids in this scene:')
        for j, bb in enumerate(boxes):
            print('   %s %2d  %-8s %-8s  %.2f x %.2f x %.2f  at %.2f %.2f %.2f'
                  % ('->' if j == state['i'] else '  ', j + 1,
                     CLASSES[bb.cid] if bb.cid else '(class?)', bb.shape,
                     bb.size[0], bb.size[1], bb.size[2],
                     bb.centre[0], bb.centre[1], bb.centre[2]))

    def cycle_face(_b):
        state['face'] = (state['face'] + 1) % len(FACES)

    def cycle_shape(b):
        b.shape = SHAPES[(SHAPES.index(b.shape) % len(SHAPES) + 1) % len(SHAPES)] \
            if b.shape in SHAPES else SHAPES[0]
        b.mesh = None                     # H leaves a CAD mesh behind

    def cycle_cad(b):
        """Put the next CAD mesh from the library into this solid.

        The mesh is scaled into the box you already placed, so the box stays
        the thing you position and resize and the mesh only decides which
        voxels inside it are filled.
        """
        if not cad_files:
            print('\n  no CAD meshes found. Put .ply/.obj/.stl files in %s'
                  % cad_dirs[0])
            return
        state['cad'] = (state['cad'] + 1) % len(cad_files)
        b.mesh = cad_files[state['cad']]
        b.shape = 'mesh'
        print('\n  mesh: %s' % os.path.basename(b.mesh))

    def step_up(_b):
        state['step'] = min(state['step'] + 1, len(STEPS) - 1)

    def step_down(_b):
        state['step'] = max(state['step'] - 1, 0)

    def save(_vis):
        missing = [j + 1 for j, b in enumerate(boxes) if b.cid is None]
        if missing:
            print('\n  boxes %s have no class -- press a digit on each first'
                  % ', '.join(map(str, missing)))
            return False
        state['save'] = True
        _vis.close()
        return False

    def quit_(_vis):
        _vis.close()
        return False

    vis.register_key_callback(ord('N'), act(new_box))
    vis.register_key_callback(K_TAB, act(next_box))
    # TAB is easy to miss and some window managers eat it, so the bracket-free
    # comma/period pair does the same, and L prints the whole list
    vis.register_key_callback(ord('.'), act(next_box))
    vis.register_key_callback(ord(','), act(prev_box))
    vis.register_key_callback(ord('L'), act(list_boxes))
    vis.register_key_callback(ord('M'), act(toggle_shell))
    vis.register_key_callback(K_BACKSPACE, act(delete))
    vis.register_key_callback(K_LEFT, act(move(-1, 0, 0)))
    vis.register_key_callback(K_RIGHT, act(move(1, 0, 0)))
    vis.register_key_callback(K_UP, act(move(0, 1, 0)))
    vis.register_key_callback(K_DOWN, act(move(0, -1, 0)))
    vis.register_key_callback(K_PGUP, act(move(0, 0, 1)))
    vis.register_key_callback(K_PGDN, act(move(0, 0, -1)))
    vis.register_key_callback(ord('W'), act(resize(1)))
    vis.register_key_callback(ord('S'), act(resize(-1)))
    vis.register_key_callback(ord('A'), act(cycle_face))
    vis.register_key_callback(ord('H'), act(cycle_shape))
    # laptops without PgUp/PgDn: R and V move the solid vertically too
    vis.register_key_callback(ord('R'), act(move(0, 0, 1)))
    vis.register_key_callback(ord('V'), act(move(0, 0, -1)))
    vis.register_key_callback(ord('Q'), act(rotate(-5)))
    vis.register_key_callback(ord('E'), act(rotate(5)))
    vis.register_key_callback(ord('K'), act(cycle_rot_axis))
    vis.register_key_callback(ord('C'), act(cycle_cad))
    vis.register_key_callback(ord('F'), act(fit))
    vis.register_key_callback(ord('G'), act(to_floor))
    vis.register_key_callback(ord('I'), act(snap_to_wall))
    vis.register_key_callback(ord('['), act(step_down))
    vis.register_key_callback(ord(']'), act(step_up))
    for d in range(1, 10):
        vis.register_key_callback(ord(str(d)), act(set_class(d)))
    # 10 and 11 need non-digit keys. U and O are the ones to remember; zero is
    # kept as an alias for furn because people try it, but '0' and 'O' side by
    # side in a key list are unreadable, so the docs say U.
    vis.register_key_callback(ord('U'), act(set_class(10)))
    vis.register_key_callback(ord('0'), act(set_class(10)))
    vis.register_key_callback(ord('O'), act(set_class(11)))
    vis.register_key_callback(K_ENTER, save)
    vis.register_key_callback(K_ESC, quit_)

    print(__doc__[__doc__.index('KEYS'):__doc__.index('***')].rstrip())
    rebuild()
    vis.run()
    vis.destroy_window()
    print()

    if not state['save']:
        log.warning('quit without saving -- solids.json untouched')
        return
    room_out = dict(room)
    room_out['center'] = [round(float(v), 3) for v in shell.centre]
    room_out['size'] = [round(float(v), 3) for v in shell.size]
    room_out['yaw_deg'] = round(float(shell.yaw), 1)
    room_out['rotation'] = [[round(float(v), 6) for v in row] for row in shell.R]
    spec = {'room': room_out, 'solids': [b.to_json() for b in boxes]}
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    if os.path.exists(out_path):
        os.replace(out_path, out_path + '.bak')
    with open(out_path, 'w') as f:
        json.dump(spec, f, indent=2)
    for b in boxes:
        print('  %-8s %.2f x %.2f x %.2f m' % (CLASSES[b.cid], *b.size))
    print('\nwrote %s (%d boxes + room shell)' % (out_path, len(boxes)))
    print('  python -m reconstruction_GT.voxelize_gt %s' % cap)


if __name__ == '__main__':
    main()
