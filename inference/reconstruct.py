"""
inference/reconstruct.py

The scene the CAMERA measured, with nothing inferred: the depth frame
triangulated into a surface and textured with the RGB frame, the way a
RealSense viewer shows it. It is the only geometry in this repo that is pure
measurement, so it is the reference a completion gets checked against, never
the other way round.

It opens a window on that surface alone, and deliberately knows nothing about
the prediction beyond where to find the frame: the readout reports the cell
under the cursor and how far away the camera measured it, nothing else. To see
the surface and the prediction TOGETHER use display_overlay.py, which imports
what is built here.

    python inference/reconstruct.py outputs/room01/ply/live_000000.ply
    python inference/reconstruct.py <ply> --surface label
    python inference/reconstruct.py <ply> --surface match --screenshot out.png

Buttons under the view cycle the surface colouring (photo -> label -> match)
and reframe. Hover to outline a cell and read its measured range; Ctrl+click
prints it. Buttons rather than key bindings -- see Viewer.

It takes the PREDICTION as its argument and finds that frame's depth, RGB and
pose the same way display_solid does -- the .bin header for NYU frames, the
capture's meta.json for a live one -- then prints how much of the measured
surface the prediction covers, broken down by the label it gave each cell.

*** THE SHARED FRAME ***

The surface is expressed in the prediction's own voxel-index coordinates
(inverting display_solid.voxel_centres_world) and reflected about the SAME
pivot (display_solid.view_pivot), so the two geometries land on top of each
other and any offset you can see is a real disagreement rather than a
bookkeeping one.

The depth PNG's encoding is not guessable and is not guessed: NYU frames carry
((d << 13) | (d >> 3)), capture.py writes plain uint16 millimetres, and
resolve_frame says which this frame is.
"""

import argparse
import os
import sys

import numpy as np
import open3d as o3d

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import display_solid as DS  # noqa: E402
from display_solid import (CLASS_NAMES, DEFAULT_COLORMAP, GRID,  # noqa: E402
                           VOX_UNIT, depth_to_grid_index, load_depth,
                           load_voxels, resolve_frame, to_view_index,
                           view_pivot)

COLOUR_MODES = DS.COLOUR_MODES   # the voxels' palette, as in display_solid

def depth_to_grid(depth, image, cam_K, cam_pose, vox_origin, pivot,
                  z_min=0.2, z_max=8.0, fix_axes=True):
    """Every valid depth pixel -> a vertex in the prediction's coordinates.

    Returns (H, W, 3) vertex positions, (H, W, 3) colours in 0..1, (H, W) valid
    mask and (H, W) range in metres. Laid out as an image so the triangulation
    below can just look at neighbouring pixels.
    """
    H, W = depth.shape
    z = depth.astype(np.float64)
    g = depth_to_grid_index(depth, cam_K, cam_pose, vox_origin)
    if fix_axes:
        g[..., 2] = pivot - g[..., 2]

    if image.shape[:2] != depth.shape:
        import cv2
        image = cv2.resize(image, (W, H), interpolation=cv2.INTER_AREA)
    valid = (z >= z_min) & (z <= z_max)
    return g, image / 255.0, valid, z


def triangulate(grid, colour, valid, rng, edge=0.12, stride=1, shaded=False,
                smooth=3):
    """Connect neighbouring pixels into a surface, the way a depth viewer does.

    A quad is emitted only where all four of its pixels are valid AND their
    ranges agree to within `edge` metres: without that test every depth
    discontinuity -- the gap between the bottle and the wall behind it -- gets
    bridged by a sheet of triangles that is pure fabrication.
    """
    g = grid[::stride, ::stride]
    c = colour[::stride, ::stride]
    ok = valid[::stride, ::stride]
    r = rng[::stride, ::stride]
    H, W = ok.shape

    a = np.s_[:-1, :-1]      # the four corners of each quad
    b = np.s_[:-1, 1:]
    d = np.s_[1:, :-1]
    e = np.s_[1:, 1:]
    keep = ok[a] & ok[b] & ok[d] & ok[e]
    span = (np.maximum(np.maximum(r[a], r[b]), np.maximum(r[d], r[e]))
            - np.minimum(np.minimum(r[a], r[b]), np.minimum(r[d], r[e])))
    keep &= span <= edge
    if not keep.any():
        raise SystemExit("no depth pixels survived -- is this frame empty?")

    flat = np.arange(H * W).reshape(H, W)
    ia, ib, id_, ie = (flat[a][keep], flat[b][keep], flat[d][keep], flat[e][keep])
    tris = np.concatenate([np.stack([ia, id_, ib], axis=1),
                           np.stack([ib, id_, ie], axis=1)])

    # keep only the pixels a triangle actually refers to, and renumber -- the
    # dropped ones include every invalid pixel, whose position is the camera
    # origin and would otherwise drag the framing towards it
    used = np.unique(tris)
    remap = np.full(H * W, -1, dtype=np.int64)
    remap[used] = np.arange(len(used))
    verts = g.reshape(-1, 3)[used]
    cols = c.reshape(-1, 3)[used]

    mesh = o3d.geometry.TriangleMesh(
        o3d.utility.Vector3dVector(verts),
        o3d.utility.Vector3iVector(remap[tris].astype(np.int32)))
    shade = np.ones((len(verts), 1))
    if shaded:
        # bake a Lambert term into the vertex colours and render unlit, so the
        # relief is readable without filament tone-mapping the photo colours
        #
        # The normals come from a SMOOTHED COPY. Vertex positions are never
        # touched -- they are the measurement -- but a depth sensor's per-pixel
        # noise makes raw normals swing wildly across a flat wall, and a
        # Lambert term then renders that noise as worm-like relief that looks
        # like real structure and isn't. Smoothing the shading normals leaves
        # the geometry exactly as measured and only stops the lighting from
        # lying about it. --smooth 0 turns it off.
        src = mesh
        if smooth > 0:
            src = mesh.filter_smooth_simple(number_of_iterations=int(smooth))
        src.compute_vertex_normals()
        n = np.asarray(src.vertex_normals)
        light = np.array([0.35, 0.80, 0.50])
        light /= np.linalg.norm(light)
        shade = (0.62 + 0.38 * np.abs(n @ light))[:, None]
    mesh.vertex_colors = o3d.utility.Vector3dVector(np.clip(cols * shade, 0, 1))
    # `used` indexes the strided colour image, so the surface can be repainted
    # (photo / label / match) without rebuilding any geometry
    return mesh, used, shade, (stride, g.shape[:2])


def paint_surface(mesh, colour_img, used, shade, layout):
    stride, shape = layout
    c = colour_img[::stride, ::stride].reshape(-1, 3)[used]
    mesh.vertex_colors = o3d.utility.Vector3dVector(np.clip(c * shade, 0, 1))
    return mesh


SURFACE_MODES = ['photo', 'label', 'match']


def cells_of(grid):
    """Grid cell each measured point falls in, plus whether it is in bounds."""
    cells = np.floor(grid + 0.5).astype(np.int64)
    inside = np.all((cells >= 0) & (cells < np.asarray(GRID)), axis=-1)
    return cells, inside


def surface_colours(grid, photo, view_idx, cls):
    """Three ways to paint the measured surface, all in 0..1.

    photo: what the camera saw -- the surface as it is.
    label: the class the model gave the cell each measured point falls in, so
        the prediction's labels are seen ON the real geometry.
    match: green where the model filled that cell, red where it left it empty.
        Measured points outside the 4.8 m volume are grey -- the model was
        never asked about them, and scoring them as misses would be unfair.
    """
    occ = np.full(GRID, -1, dtype=np.int32)
    occ[view_idx[:, 0], view_idx[:, 1], view_idx[:, 2]] = cls
    cells, inside = cells_of(grid)
    lab = np.full(grid.shape[:2], -2, dtype=np.int32)      # -2 = out of volume
    c = cells[inside]
    lab[inside] = occ[c[..., 0], c[..., 1], c[..., 2]]     # -1 = left empty

    label = np.zeros_like(photo)
    known = lab >= 0
    label[known] = DEFAULT_COLORMAP[lab[known]] / 255.0
    label[lab == -1] = (0.12, 0.12, 0.14)
    label[lab == -2] = (0.30, 0.30, 0.30)

    match = np.zeros_like(photo)
    match[known] = (0.20, 0.80, 0.35)
    match[lab == -1] = (0.90, 0.20, 0.20)
    match[lab == -2] = (0.35, 0.35, 0.35)
    return {'photo': photo, 'label': label, 'match': match}


# ------------------------------------------------------------- the comparison

def coverage(grid, valid, view_idx, cls, stride=4):
    """How much of the MEASURED surface the prediction actually fills.

    The measurement is the reference: each measured point is dropped into its
    grid cell and we ask whether the model put anything there. The reverse
    ratio is not reported because it is meaningless -- completing cells the
    camera never saw is the whole job.
    """
    g = grid[::stride, ::stride][valid[::stride, ::stride]]
    cells, inside = cells_of(g)
    cells = np.unique(cells[inside], axis=0)

    occ = np.full(GRID, -1, dtype=np.int32)
    occ[view_idx[:, 0], view_idx[:, 1], view_idx[:, 2]] = cls
    hit = occ[cells[:, 0], cells[:, 1], cells[:, 2]]
    lines = ["measured surface occupies %d grid cells (%d/%d points fell outside "
             "the volume)" % (len(cells), int((~inside).sum()), len(inside))]
    filled = hit >= 0
    lines.append("  the prediction fills %d of them (%.1f%%)"
                 % (filled.sum(), 100.0 * filled.mean()))
    ids, counts = np.unique(hit[filled], return_counts=True)
    for cid, n in sorted(zip(ids.tolist(), counts.tolist()), key=lambda kv: -kv[1]):
        lines.append("    %-8s %5d  (%4.1f%%)"
                     % (CLASS_NAMES[cid], n, 100.0 * n / filled.sum()))
    return "\n".join(lines)


# ------------------------------------------------- picking the surface

def raycast_scene(mesh):
    """A BVH over one mesh, for exact ray/triangle picking.

    Not a march over the voxel lattice: cell occupancy is an 8 cm proxy for a
    continuous surface, and a ray clipping the corner of a cell whose surface
    lies elsewhere in that cell stops early. Measured against a render that
    colours each vertex by its cell, the lattice march named the drawn cell for
    only 32.7% of pixels; hitting the actual triangles is exact.
    """
    scene = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(mesh))
    return scene


def first_hit(scenes, origin, direction):
    """Nearest hit across several meshes -> the point, or None.

    `direction` is normalised first so t_hit comes back in grid units and the
    scenes are comparable with each other.
    """
    d = np.asarray(direction, dtype=np.float64)
    n = np.linalg.norm(d)
    if n < 1e-12 or not scenes:
        return None
    d = d / n
    ray = o3d.core.Tensor([list(np.asarray(origin, np.float32))
                           + list(d.astype(np.float32))],
                          dtype=o3d.core.Dtype.Float32)
    best = np.inf
    for scene in scenes:
        t = float(scene.cast_rays(ray)['t_hit'].numpy()[0])
        if np.isfinite(t):
            best = min(best, t)
    if not np.isfinite(best):
        return None
    # nudge past the surface: a cube-face hit lands exactly on a cell boundary
    return np.asarray(origin, np.float64) + (best + 1e-4) * d


# ---------------------------------------------------- the comparison

def build_cell_state(grid, valid, ranges, view_idx, cls, stride=1):
    g = grid[::stride, ::stride]
    ok = valid[::stride, ::stride]
    r = ranges[::stride, ::stride]
    cells, inside = cells_of(g)
    take = ok & inside
    c = cells[take]
    flat = np.ravel_multi_index((c[:, 0], c[:, 1], c[:, 2]), GRID)

    n = int(np.prod(GRID))
    count = np.bincount(flat, minlength=n)
    total = np.bincount(flat, weights=r[take], minlength=n)
    measured = np.full(n, np.nan)
    seen = count > 0
    measured[seen] = total[seen] / count[seen]

    predicted = np.full(n, -1, dtype=np.int32)
    predicted[np.ravel_multi_index((view_idx[:, 0], view_idx[:, 1],
                                    view_idx[:, 2]), GRID)] = cls
    return measured.reshape(GRID), predicted.reshape(GRID)


# ------------------------------------------------------------------- viewer

class Viewer:
    """The measured surface on its own.

    Deliberately knows nothing about the prediction: this file is the
    measurement, and the readout reports only what the camera got -- the cell
    under the cursor and how far away its surface is. Put the two together with
    display_overlay.py, which reuses the surface built here.

    Buttons rather than key bindings: a Window's set_on_key callback is cast to
    a C++ type open3d 0.17 will not accept an EventCallbackResult for, and it
    raises `RuntimeError: Unable to cast Python instance to C++ type` out of
    Application.run() the first time a key is pressed. set_on_clicked takes no
    argument and returns nothing, so there is nothing to cast.
    """

    FOV = 60.0

    def __init__(self, surface, view_idx, measured, surf_palette, surf_paint,
                 title, surf_mode='photo', highlight=True):
        from open3d.visualization import gui, rendering
        self.gui, self.rendering = gui, rendering
        self.surface, self.view_idx, self.measured = surface, view_idx, measured
        self.surf_palette, self.surf_paint = surf_palette, surf_paint
        self.surf_mode = surf_mode
        self.highlight = highlight
        self._framed = False
        self._picked = None
        self._scene = raycast_scene(surface)

        self.window = gui.Application.instance.create_window(title, 1600, 1000)
        em = self.window.theme.font_size
        self.widget3d = gui.SceneWidget()
        self.widget3d.scene = rendering.Open3DScene(self.window.renderer)
        self.widget3d.scene.set_background([0.05, 0.05, 0.08, 1.0])
        self.widget3d.scene.view.set_post_processing(False)
        self.mat = rendering.MaterialRecord()
        self.mat.shader = "defaultUnlit"
        self.widget3d.scene.add_geometry("surface", surface, self.mat)
        if self.highlight:
            self.hi_mat = rendering.MaterialRecord()
            self.hi_mat.shader = "unlitLine"
            self.hi_mat.line_width = 3.0
            self.widget3d.scene.add_geometry("pick", DS.highlight_box(),
                                             self.hi_mat)
            self.widget3d.scene.show_geometry("pick", False)
        self.window.add_child(self.widget3d)

        self.panel = gui.Vert(0.25 * em, gui.Margins(0.5 * em, 0.35 * em,
                                                     0.5 * em, 0.35 * em))
        self.panel.background_color = gui.Color(0.05, 0.05, 0.08, 1.0)
        self.info = gui.Label("hover the surface")
        self.idle_text = self.info.text
        self.info.text_color = gui.Color(1.0, 1.0, 1.0)
        self.panel.add_child(self.info)

        row = gui.Horiz(0.4 * em)
        self.surf_button = gui.Button("surface: %s" % self.surf_mode)
        self.surf_button.horizontal_padding_em = 0.6
        self.surf_button.vertical_padding_em = 0.2
        self.surf_button.set_on_clicked(self.cycle_surface_colour)
        row.add_child(self.surf_button)
        reframe = gui.Button("reframe")
        reframe.horizontal_padding_em = 0.6
        reframe.vertical_padding_em = 0.2
        reframe.set_on_clicked(self.reframe)
        row.add_child(reframe)
        row.add_stretch()
        self.panel.add_child(row)

        self.state_label = gui.Label("")
        self.state_label.text_color = gui.Color(0.6, 0.75, 1.0)
        self.panel.add_child(self.state_label)
        self.window.add_child(self.panel)
        self._update_state()

        self.window.set_on_layout(self._on_layout)
        self.widget3d.set_on_mouse(self._on_mouse)

    # -- state / actions -------------------------------------------------
    def _update_state(self):
        n = int(np.isfinite(self.measured).sum())
        self.state_label.text = ("measured surface in %d cells   [%s]"
                                 % (n, self.surf_mode))
        self.surf_button.text = "surface: %s" % self.surf_mode

    def cycle_surface_colour(self):
        self.surf_mode = SURFACE_MODES[
            (SURFACE_MODES.index(self.surf_mode) + 1) % len(SURFACE_MODES)]
        used, shade, layout = self.surf_paint
        paint_surface(self.surface, self.surf_palette[self.surf_mode], used,
                      shade, layout)
        self.widget3d.scene.remove_geometry("surface")
        self.widget3d.scene.add_geometry("surface", self.surface, self.mat)
        self._update_state()
        self.window.post_redraw()

    def reframe(self):
        self.reset_view()
        self.window.post_redraw()

    # -- camera / layout -------------------------------------------------
    def reset_view(self):
        frame = self.widget3d.frame
        pts = self.view_idx.astype(np.float64)   # the prediction's extent, so
        lo, hi = pts.min(axis=0) - 0.5, pts.max(axis=0) + 0.5   # both viewers
        ext = hi - lo                                           # frame alike
        aspect = (frame.width / frame.height) if frame.height else 1.6
        tan = float(np.tan(np.radians(self.FOV / 2.0)))
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
        if not self._framed:
            self._framed = True
            self.reset_view()

    # -- picking ---------------------------------------------------------
    def _on_mouse(self, event):
        gui = self.gui
        moved = event.type == gui.MouseEvent.Type.MOVE
        clicked = (event.type == gui.MouseEvent.Type.BUTTON_DOWN
                   and event.is_modifier_down(gui.KeyModifier.CTRL))
        if not (moved or clicked) or DS._PICK_FAILED:
            return gui.Widget.EventCallbackResult.IGNORED
        frame = self.widget3d.frame
        x, y = int(event.x) - frame.x, int(event.y) - frame.y
        cell = None
        if 0 <= x < frame.width and 0 <= y < frame.height:
            ray = DS.guard_pick(DS.pick_ray, self.widget3d.scene.camera,
                                x, y, frame.width, frame.height)
            if ray is not None and ray[0] is not None:
                hit = DS.guard_pick(first_hit, [self._scene], ray[0], ray[1])
                if hit is not None:
                    c = np.floor(hit + 0.5).astype(np.int64)
                    cell = (tuple(int(v) for v in c)
                            if np.all((c >= 0) & (c < np.asarray(GRID)))
                            else 'outside')
        if cell != self._picked:
            self._picked = cell
            if cell is None or cell == 'outside':
                if self.highlight:
                    self.widget3d.scene.show_geometry("pick", False)
                self.info.text = (self.idle_text if cell is None else
                                  "measured surface outside the 4.8 m volume")
            else:
                if self.highlight:
                    t = np.eye(4)
                    t[:3, 3] = cell
                    self.widget3d.scene.set_geometry_transform("pick", t)
                    self.widget3d.scene.show_geometry("pick", True)
                rng = float(self.measured[cell])
                self.info.text = ("cell [%2d, %2d, %2d]   %s"
                                  % (cell + (("measured at %.2f m" % rng)
                                             if np.isfinite(rng)
                                             else "no range for this cell",)))
            self.window.post_redraw()
        if clicked and isinstance(cell, tuple):
            print(self.info.text, flush=True)
        return (gui.Widget.EventCallbackResult.HANDLED if clicked
                else gui.Widget.EventCallbackResult.IGNORED)


def surface_screenshot(surface, view_idx, path, width=1600, height=1000):
    """Offscreen render of the measured surface alone."""
    from open3d.visualization import rendering
    ren = rendering.OffscreenRenderer(width, height)
    ren.scene.set_background([0.05, 0.05, 0.08, 1.0])
    ren.scene.view.set_post_processing(False)
    mat = rendering.MaterialRecord()
    mat.shader = "defaultUnlit"
    ren.scene.add_geometry("surface", surface, mat)
    pts = view_idx.astype(np.float64)      # framed on the prediction's extent
    lo, hi = pts.min(axis=0) - 0.5, pts.max(axis=0) + 0.5
    ext, tan = hi - lo, float(np.tan(np.radians(30.0)))
    dist = 1.05 * max(ext[1] / 2.0 / tan, ext[0] / 2.0 / (tan * width / height))
    c = 0.5 * (lo + hi)
    ren.setup_camera(60.0, c, c + np.array([0.0, 0.0, dist]), [0.0, 1.0, 0.0])
    o3d.io.write_image(path, ren.render_to_image())
    print("wrote %s" % path)


def build(ply_path, args):
    """Everything display_overlay needs: the prediction, the frame, and the
    measured surface expressed in the prediction's coordinates."""
    idx, class_rgb, cls = load_voxels(ply_path)
    pivot = view_pivot(idx)
    view_idx = to_view_index(idx) if args.fix_axes else idx

    frame = resolve_frame(ply_path, args)
    if frame is None or frame.get('depth_path') is None:
        raise SystemExit(
            "Could not find the depth frame and camera pose behind %s.\n"
            "A reconstruction is measurement, so there is nothing to fall back "
            "on -- pass --depth and --rgb with --meta (a capture) or --bin "
            "(an NYU frame)." % ply_path)

    print("Prediction  %s  (%d voxels)" % (ply_path, len(idx)))
    print("Camera      %s" % frame['source'])
    print("Depth       %s  (%s)"
          % (frame['depth_path'],
             "NYU bit-rotated" if frame['nyu'] else "plain uint16 mm"))
    print("Colour      %s" % frame['rgb_path'])

    depth = load_depth(frame['depth_path'], frame['nyu'])
    grid, colour, valid, rng = depth_to_grid(
        depth, frame['image'], frame['cam_K'], frame['cam_pose'],
        frame['vox_origin'], pivot, z_min=args.z_range[0],
        z_max=args.z_range[1], fix_axes=args.fix_axes)
    print("%d/%d depth pixels valid in %.1f-%.1f m"
          % (valid.sum(), valid.size, args.z_range[0], args.z_range[1]))

    surf_palette = surface_colours(grid, colour, view_idx, cls)
    surface, used, shade, layout = triangulate(
        grid, surf_palette[args.surf_mode], valid, rng, edge=args.edge,
        stride=args.stride, shaded=args.shade, smooth=args.smooth)
    print("surface: %d vertices, %d triangles"
          % (len(np.asarray(surface.vertices)),
             len(np.asarray(surface.triangles))))
    print(coverage(grid, valid, view_idx, cls))
    measured, predicted = build_cell_state(grid, valid, rng, view_idx, cls)
    return dict(idx=idx, class_rgb=class_rgb, cls=cls, view_idx=view_idx,
                pivot=pivot, frame=frame, depth=depth, grid=grid,
                surface=surface, surf_palette=surf_palette,
                surf_paint=(used, shade, layout),
                measured=measured, predicted=predicted)


def add_common_args(ap):
    """The flags both this file and display_overlay.py take."""
    ap.add_argument('ply_path',
                    help="the prediction to check. Its capture (depth, rgb, "
                         "pose) is found the same way display_solid finds it")
    ap.add_argument('--surface', dest='surf_mode', default='photo',
                    choices=SURFACE_MODES,
                    help="photo: the measured surface as the camera saw it; "
                         "label: painted with the class the model gave each "
                         "cell; match: green where the model filled the cell, "
                         "red where it left it empty, grey outside the volume")
    ap.add_argument('--stride', type=int, default=1,
                    help="use every Nth depth pixel (default 1 -- full frame)")
    ap.add_argument('--edge', type=float, default=0.12,
                    help="metres of range disagreement across a quad before it "
                         "is treated as a depth discontinuity and left open")
    ap.add_argument('--z-range', nargs=2, type=float, default=(0.2, 8.0),
                    metavar=('MIN', 'MAX'),
                    help="depth values outside this (metres) are dropped")
    ap.add_argument('--shade', action='store_true',
                    help="add a Lambert term for relief. Off by default: the "
                         "photo texture already reads as a surface, and on raw "
                         "sensor normals the shading renders the depth "
                         "camera's own low-frequency waviness as structure")
    ap.add_argument('--smooth', type=int, default=3,
                    help="with --shade, smoothing iterations for the SHADING "
                         "NORMALS only; vertex positions are always the raw "
                         "measurement")
    ap.add_argument('--raw-axes', dest='fix_axes', action='store_false',
                    help="the file's left-handed axes, uncorrected")
    ap.add_argument('--depth', help="the frame's depth PNG")
    ap.add_argument('--rgb', help="the frame's colour image")
    ap.add_argument('--bin', help="NYU .bin holding vox_origin + cam_pose")
    ap.add_argument('--meta', help="capture.py meta.json for a live frame")
    ap.add_argument('--camera-height', type=float, default=None)
    ap.add_argument('--yaw', type=float, default=None)
    ap.add_argument('--screenshot', help="render offscreen to this PNG and exit")
    ap.add_argument('--no-highlight', dest='highlight', action='store_false',
                    help="don't outline the cell under the cursor")
    return ap


def main():
    ap = argparse.ArgumentParser(
        description=__doc__.split('***')[0],
        formatter_class=argparse.RawDescriptionHelpFormatter)
    add_common_args(ap)
    args = ap.parse_args()
    b = build(args.ply_path, args)
    if args.screenshot:
        surface_screenshot(b['surface'], b['view_idx'], args.screenshot)
        return
    from open3d.visualization import gui
    app = gui.Application.instance
    app.initialize()
    Viewer(b['surface'], b['view_idx'], b['measured'], b['surf_palette'],
           b['surf_paint'],
           os.path.basename(args.ply_path) + "  [reconstruction]",
           surf_mode=args.surf_mode, highlight=args.highlight)
    app.run()


if __name__ == '__main__':
    main()
