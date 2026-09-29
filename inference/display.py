"""
inference/display.py

Standalone viewer for the coloured voxel .ply files produced by
inference/visualize.py (voxel grid indices as x/y/z, per-class RGB colour).

    python inference/display.py ./outputs/room01/ply/live_000000.ply

Hover the cursor over the cloud and the readout bar below the 3D view reports
the voxel under the cursor: its grid index, its RGB triple, and the class that
colour belongs to. Ctrl+click prints the same line to stdout.

*** WHY THE VIEW NEEDED FIXING (and what --raw-axes turns off) ***

The .ply axes are voxel-grid indices. Probing frame_loader with single-patch
depth images pins the grid axis order down as:

    ply x  = grid axis 0 -> world X, increasing to the CAMERA'S RIGHT
    ply y  = grid axis 1 -> world Z, increasing UP
    ply z  = grid axis 2 -> world Y, increasing AWAY from the camera

(right, up, forward) is LEFT-handed -- world (X, Y, Z) = (right, forward, up)
had its last two axes swapped, with no sign flip, when frame_loader wrote
gz/gy/gx. Every viewer, open3d included, expects the OpenGL convention
(right, up, TOWARD the viewer), so the file's forward axis points the wrong
way and the scene is rendered inside-out: the eye sits behind the far wall
looking back at the camera, the wall is drawn over the furniture in front of
it, and orbiting turns the room the wrong way. Reflecting the depth axis --
z -> zmax + zmin - z, what this viewer does by default -- makes the basis
right-handed and puts the eye where the capture camera stood.

Verified rather than assumed: the ply was reprojected through the real camera
model (cam_pose + the capture's cam_K) into a 640x480 label image and compared
with captures/room01/rgb. The reflected view reproduces it -- desk objects
left, bed right, floor below, wall behind -- and the near surfaces occlude the
far ones. Note that left/right was ALREADY correct before the fix and is
unchanged by it: reflecting x instead (the obvious guess) breaks it.

The .ply files are left alone. Their layout is what test_NYU.py's
labeled_voxel2ply has always written and other tooling reads them, so the
reflection is display-only -- and the readout always reports the ORIGINAL grid
index, which still indexes the prediction arrays directly.
"""

import argparse
import os
import sys

import numpy as np
import open3d as o3d

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from visualize import DEFAULT_COLORMAP  # noqa: E402

try:
    from utils import CLASSES  # noqa: E402
except Exception:  # utils drags in torch; a viewer has no use for it
    CLASSES = ['empty', 'ceiling', 'floor', 'wall', 'window', 'chair', 'bed',
               'sofa', 'table', 'tvs', 'furn', 'objs']

# DEFAULT_COLORMAP has one row per class plus a 13th for ignore/label==255
CLASS_NAMES = list(CLASSES) + ['ignore'] * (len(DEFAULT_COLORMAP) - len(CLASSES))
COLOR_TO_CLASS = {tuple(int(c) for c in rgb): (i, CLASS_NAMES[i])
                  for i, rgb in enumerate(DEFAULT_COLORMAP)}


def load_cloud(ply_path):
    pcd = o3d.io.read_point_cloud(ply_path)
    if len(pcd.points) == 0:
        raise SystemExit(f"Warning: {ply_path} loaded but contains 0 points.")
    # .copy(): np.asarray on a Vector3dVector is a VIEW into open3d's buffer,
    # and assigning pcd.points later writes through it -- without the copy the
    # view-frame coordinates would overwrite the indices the readout reports.
    pts = np.asarray(pcd.points).copy()
    rgb = np.rint(np.asarray(pcd.colors) * 255.0).astype(np.int32)
    return pcd, pts, rgb


def to_view_frame(pts):
    """Left-handed (right, up, forward) -> right-handed (right, up, backward).

    Reflects the depth axis about its own extent, so the coordinates stay in
    the grid's 0..59 range and only the handedness changes.
    """
    out = pts.copy()
    out[:, 2] = pts[:, 2].max() + pts[:, 2].min() - pts[:, 2]
    return out


def describe(pts, rgb):
    lines = [f"{len(pts)} points",
             "bounds  x %.0f..%.0f   y %.0f..%.0f   z %.0f..%.0f"
             % (pts[:, 0].min(), pts[:, 0].max(), pts[:, 1].min(),
                pts[:, 1].max(), pts[:, 2].min(), pts[:, 2].max())]
    keys, counts = np.unique(rgb, axis=0, return_counts=True)
    for k, n in sorted(zip(keys.tolist(), counts.tolist()), key=lambda kv: -kv[1]):
        cid, name = COLOR_TO_CLASS.get(tuple(k), (-1, 'unknown colour'))
        lines.append(f"  {name:<8s} {n:6d}  ({100.0 * n / len(pts):4.1f}%)  rgb{tuple(k)}")
    return "\n".join(lines)


def readout(i, orig_pts, rgb):
    x, y, z = (int(round(v)) for v in orig_pts[i])
    r, g, b = (int(v) for v in rgb[i])
    cid, name = COLOR_TO_CLASS.get((r, g, b), (-1, 'unknown colour'))
    cls = f"{name} (class {cid})" if cid >= 0 else name
    return (f"voxel [{x:2d}, {y:2d}, {z:2d}]   rgb ({r:3d}, {g:3d}, {b:3d})   {cls}",
            (r, g, b))


# ---------------------------------------------------------------- gui viewer

class Viewer:
    """SceneWidget + a side panel, so the class under the cursor can be shown
    live. The picking is the depth-buffer round trip from open3d's own
    examples/python/visualization/mouse_and_point_coord.py: render the scene's
    depth, unproject the pixel under the mouse, snap to the nearest point.
    """

    FOV = 60.0   # vertical, degrees

    def __init__(self, pcd, view_pts, orig_pts, rgb, title, point_size):
        from open3d.visualization import gui, rendering
        self.gui = gui
        self.orig_pts, self.rgb = orig_pts, rgb
        self.kdtree = o3d.geometry.KDTreeFlann(pcd)
        self._busy = False

        self.window = gui.Application.instance.create_window(title, 1280, 800)
        em = self.window.theme.font_size
        self._view_pts = view_pts
        self._auto_point_size = point_size is None
        self._framed = False           # the fit needs the laid-out scene size
        point_size = point_size or 12.0

        self.widget3d = gui.SceneWidget()
        self.widget3d.scene = rendering.Open3DScene(self.window.renderer)
        self.widget3d.scene.set_background([0.05, 0.05, 0.08, 1.0])
        # filament tone-maps by default, which washes the class colours out and
        # makes them disagree with the RGB the panel reports; unlit + no
        # post-processing renders the colormap values exactly
        self.widget3d.scene.view.set_post_processing(False)
        self.mat = rendering.MaterialRecord()
        self.mat.shader = "defaultUnlit"
        self.mat.point_size = float(point_size)
        self.widget3d.scene.add_geometry("cloud", pcd, self.mat)
        self.window.add_child(self.widget3d)

        bounds = self.widget3d.scene.bounding_box
        self.widget3d.setup_camera(self.FOV, bounds, bounds.get_center())

        # the readout is a bar UNDER the scene rather than an overlay on it,
        # so it never sits on top of the geometry being inspected
        self.panel = gui.Vert(0.25 * em, gui.Margins(0.5 * em, 0.35 * em,
                                                     0.5 * em, 0.35 * em))
        self.panel.background_color = gui.Color(0.05, 0.05, 0.08, 1.0)
        self.info = gui.Label("hover the cloud")
        self.info.text_color = gui.Color(1.0, 1.0, 1.0)
        self.panel.add_child(self.info)
        legend = gui.Horiz(0.75 * em)
        for cid, name in enumerate(CLASS_NAMES):
            if name in ('empty', 'ignore'):
                continue
            r, g, b = (int(v) for v in DEFAULT_COLORMAP[cid])
            swatch = gui.Label(name)
            swatch.text_color = gui.Color(r / 255.0, g / 255.0, b / 255.0)
            legend.add_child(swatch)
        legend.add_stretch()
        self.panel.add_child(legend)
        self.window.add_child(self.panel)
        self.window.set_on_layout(self._on_layout)
        self.widget3d.set_on_mouse(self._on_mouse)

    def _tan(self):
        return float(np.tan(np.radians(self.FOV / 2.0)))

    def reset_view(self):
        """Stand where the capture camera stood, looking into the scene.

        setup_camera's own framing fits the volume's bounding SPHERE, which for
        a 60x36x60 grid backs off about twice as far as it needs to; this fits
        the occupied extent to the scene rect instead, at the volume's
        mid-depth -- fitting the front face would leave the far wall, which is
        most of what is actually occupied, at half height.
        """
        frame = self.widget3d.frame
        lo, hi = self._view_pts.min(axis=0), self._view_pts.max(axis=0)
        ext = hi - lo
        aspect = (frame.width / frame.height) if frame.height else 1.6
        tan = self._tan()
        dist = 1.05 * max(ext[1] / 2.0 / tan, ext[0] / 2.0 / (tan * aspect))
        centre = 0.5 * (lo + hi)
        self.widget3d.look_at(centre, centre + np.array([0.0, 0.0, dist]),
                              [0.0, 1.0, 0.0])
        if self._auto_point_size:
            # one voxel is 1.0 apart, so match the on-screen spacing at the
            # volume's mid-depth: any smaller and the surfaces come out as a
            # grid of dots with the background showing through
            self.mat.point_size = max(4.0, round(frame.height
                                                 / (2.0 * dist * tan)))
            self.widget3d.scene.modify_geometry_material("cloud", self.mat)

    def _on_layout(self, ctx):
        r = self.window.content_rect
        c = self.gui.Widget.Constraints()
        c.width, c.height = r.width, r.height
        bar = min(self.panel.calc_preferred_size(ctx, c).height, r.height // 3)
        self.widget3d.frame = self.gui.Rect(r.x, r.y, r.width, r.height - bar)
        self.panel.frame = self.gui.Rect(r.x, r.y + r.height - bar, r.width, bar)
        if not self._framed:      # first layout: now the scene rect is real
            self._framed = True
            self.reset_view()

    def _on_mouse(self, event):
        gui = self.gui
        moved = event.type == gui.MouseEvent.Type.MOVE
        clicked = (event.type == gui.MouseEvent.Type.BUTTON_DOWN
                   and event.is_modifier_down(gui.KeyModifier.CTRL))
        if not (moved or clicked) or self._busy:
            return gui.Widget.EventCallbackResult.IGNORED

        x = int(event.x - self.widget3d.frame.x)
        y = int(event.y - self.widget3d.frame.y)
        self._busy = True

        def on_depth(depth_image):
            depth = np.asarray(depth_image)
            text, colour = "no voxel under the cursor", (1.0, 1.0, 1.0)
            if 0 <= y < depth.shape[0] and 0 <= x < depth.shape[1]:
                d = float(depth[y, x])
                if d < 1.0:                      # 1.0 is the empty background
                    world = self.widget3d.scene.camera.unproject(
                        x, y, d, self.widget3d.frame.width,
                        self.widget3d.frame.height)
                    _, idx, _ = self.kdtree.search_knn_vector_3d(
                        np.asarray(world, dtype=np.float64), 1)
                    text, rgb = readout(idx[0], self.orig_pts, self.rgb)
                    colour = tuple(c / 255.0 for c in rgb)
                    if clicked:
                        print(text, flush=True)

            def apply():
                self.info.text = text
                self.info.text_color = self.gui.Color(*colour)
                self.window.post_redraw()
                self._busy = False

            gui.Application.instance.post_to_main_thread(self.window, apply)

        self.widget3d.scene.scene.render_to_depth_image(on_depth)
        # MOVE must stay IGNORED or the camera controls stop seeing drags
        return (gui.Widget.EventCallbackResult.HANDLED if clicked
                else gui.Widget.EventCallbackResult.IGNORED)


def run_gui(pcd, view_pts, orig_pts, rgb, title, point_size):
    from open3d.visualization import gui
    app = gui.Application.instance
    app.initialize()
    Viewer(pcd, view_pts, orig_pts, rgb, title, point_size)
    app.run()


def run_legacy(pcd, view_pts, title, point_size):
    """draw_geometries fallback: same corrected view, no hover readout."""
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name=title, width=1280, height=800)
    vis.add_geometry(pcd)
    vis.get_render_option().point_size = float(point_size or 12.0)
    vc = vis.get_view_control()
    vc.set_lookat(view_pts.mean(axis=0))
    vc.set_front([0.0, 0.0, 1.0])    # eye on the camera's side of the volume
    vc.set_up([0.0, 1.0, 0.0])
    vc.set_zoom(0.7)
    vis.run()
    vis.destroy_window()


def main():
    ap = argparse.ArgumentParser(
        description=__doc__.split('***')[0],
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('ply_path')
    ap.add_argument('--raw-axes', dest='fix_axes', action='store_false',
                    help="show the file's left-handed axes unchanged -- the "
                         "scene is then rendered inside-out, from behind")
    ap.add_argument('--point-size', type=float, default=None,
                    help="screen pixels per point; default sizes them so the "
                         "voxels just meet at the volume's mid-depth")
    ap.add_argument('--legacy', action='store_true',
                    help="draw_geometries instead of the gui app; no hover readout")
    args = ap.parse_args()

    pcd, orig_pts, rgb = load_cloud(args.ply_path)
    print(f"Loaded {args.ply_path}")
    print(describe(orig_pts, rgb))

    view_pts = to_view_frame(orig_pts) if args.fix_axes else orig_pts
    if args.fix_axes:
        pcd.points = o3d.utility.Vector3dVector(view_pts)

    title = os.path.basename(args.ply_path) + ('' if args.fix_axes else '  [raw axes]')
    if args.legacy:
        run_legacy(pcd, view_pts, title, args.point_size)
    else:
        run_gui(pcd, view_pts, orig_pts, rgb, title, args.point_size)


if __name__ == '__main__':
    main()
