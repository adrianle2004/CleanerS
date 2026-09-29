"""
inference/display_overlay.py

The prediction and the measurement in one window, so they can be compared.
Same family as display.py (points) and display_solid.py (cubes); this one adds
the measured surface underneath, built by reconstruct.py.

    python inference/display_overlay.py outputs/room01/ply/live_000000.ply
    python inference/display_overlay.py visual_pred/CleanerS/NYU0015.ply
    python inference/display_overlay.py <ply> --show surface

Buttons under the view:

    layers      solid -> transparent -> transparent over a dimmed surface.
                Transparency is the point of this file: a solid cube is opaque
                and ENCLOSES the measured surface running through it, so the
                two cannot be seen at once until the cubes let light through.
                At 0.55 alpha the class colours still read -- cyan wall, orange
                bed -- with the photo texture visible underneath.
    visibility  the rule the predicted voxels are coloured by, as in
                display_solid: points -> surface -> zbuffer, recomputed live.
    surface     the surface colouring: photo -> label -> match
    voxels      the voxel colouring: photo -> class
    hide/show   blank either layer outright
    reframe     back to the starting view

Hover to outline the cell under the cursor and read out what the camera
measured there and what the model said. Ctrl+click prints that line to stdout.

They are BUTTONS, not key bindings, and deliberately: a Window's set_on_key
callback is cast to a C++ type that open3d 0.17 will not accept an
EventCallbackResult for, and it raises

    RuntimeError: Unable to cast Python instance to C++ type

out of Application.run() the first time any key is pressed. Button's
set_on_clicked takes no argument and returns nothing, so there is nothing to
cast. Calling the handler directly from python does NOT reproduce the crash --
the cast only happens on the way back into C++ -- which is why this survived
several rounds of testing.

*** WHAT IS MEASURED AND WHAT IS INFERRED ***

Nothing here is estimated. The surface is depth pixels unprojected through the
frame's own cam_K/cam_pose; the voxels are the prediction, coloured by
--visibility (default `points`: a voxel takes colour only from measurements
INSIDE it, never from a neighbour's). Cells with no measurement of their own
keep their flat class colour, and the readout says which is which.
"""

import argparse
import os
import sys

import numpy as np
import open3d as o3d

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import display_solid as DS  # noqa: E402
import reconstruct as RC  # noqa: E402
from display_solid import (CLASS_NAMES, COLOUR_MODES, DEFAULT_COLORMAP,  # noqa: E402
                           GRID, VOX_UNIT, guard_pick, highlight_box, pick_ray)
from reconstruct import (SURFACE_MODES, first_hit,  # noqa: E402
                         paint_surface, raycast_scene)

# ------------------------------------------------------------------- gui

class Viewer:
    """Two geometries in one frame: the measured surface and the predicted
    voxels, each toggleable, with a readout that names the cell under the
    cursor and what the model put there.

    Picking casts the cursor's ray at the actual triangles (raycast_scene),
    one BVH per geometry and only the visible ones, then reports the cell the
    hit lands in: what the camera measured there and what the model said about
    it, together. Synchronous and about 0.04 ms, so the readout keeps up with
    the mouse, and the range it gives is the measured one rather than a
    distance to a rendered pixel.
    """

    FOV = 60.0

    def __init__(self, surface, cubes, view_idx, idx, cls, palette,
                 vox_of_vert, shade, title, mode,
                 vis_mode='points', recolour=None,
                 surf_palette=None, surf_mode='photo', surf_paint=None,
                 measured=None, predicted=None, pivot=0, fix_axes=True,
                 show_surface=True, show_vox=True, highlight=True, frame_box=None):
        from open3d.visualization import gui, rendering
        self.gui, self.rendering = gui, rendering
        self.surface, self.cubes = surface, cubes
        self.view_idx, self.idx, self.cls = view_idx, idx, cls
        self.palette, self.vox_of_vert, self.shade = palette, vox_of_vert, shade
        self.mode = mode
        self.vis_mode, self.recolour = vis_mode, recolour
        self.surf_palette = surf_palette or {}
        self.surf_mode = surf_mode
        self.surf_paint = surf_paint      # (used, shade, layout) for repainting
        self.measured, self.predicted = measured, predicted
        self.pivot, self.fix_axes = pivot, fix_axes
        self.frame_box = frame_box
        self.highlight = highlight
        self._framed = False
        self._picked = None
        self.show = {'surface': show_surface and surface is not None,
                     'vox': show_vox and cubes is not None}
        # how the two layers are stacked; see LAYERS / _apply_dim
        self.dim = 'solid'
        # one BVH per geometry; only the visible ones are cast against, so the
        # readout always describes something on screen
        self._scene = {'surface': raycast_scene(surface) if surface is not None
                       else None,
                       'vox': raycast_scene(cubes) if cubes is not None else None}

        self.window = gui.Application.instance.create_window(title, 1600, 1000)
        em = self.window.theme.font_size
        self.widget3d = gui.SceneWidget()
        self.widget3d.scene = rendering.Open3DScene(self.window.renderer)
        self.widget3d.scene.set_background([0.05, 0.05, 0.08, 1.0])
        self.widget3d.scene.view.set_post_processing(False)
        self.mat = rendering.MaterialRecord()
        self.mat.shader = "defaultUnlit"
        if surface is not None:
            self.widget3d.scene.add_geometry("surface", surface, self.mat)
            self.widget3d.scene.show_geometry("surface", self.show['surface'])
        if cubes is not None:
            cubes.compute_vertex_normals()   # the transparent shader is lit
            self.widget3d.scene.add_geometry("vox", cubes, self.mat)
            self.widget3d.scene.show_geometry("vox", self.show['vox'])
        self.rendering = rendering
        if self.highlight:
            self.hi_mat = rendering.MaterialRecord()
            self.hi_mat.shader = "unlitLine"
            self.hi_mat.line_width = 3.0
            self.widget3d.scene.add_geometry("pick", highlight_box(), self.hi_mat)
            self.widget3d.scene.show_geometry("pick", False)
        self.window.add_child(self.widget3d)

        self.panel = gui.Vert(0.25 * em, gui.Margins(0.5 * em, 0.35 * em,
                                                     0.5 * em, 0.35 * em))
        self.panel.background_color = gui.Color(0.05, 0.05, 0.08, 1.0)
        self.info = gui.Label("hover the geometry")
        self.idle_text = self.info.text
        self.info.text_color = gui.Color(1.0, 1.0, 1.0)
        self.panel.add_child(self.info)

        # Buttons rather than key bindings: a Window's set_on_key callback is
        # cast to a C++ type open3d 0.17 would not accept an
        # EventCallbackResult for, and it raises out of Application.run() the
        # first time a key is pressed. set_on_clicked takes no argument and
        # returns nothing, so there is nothing to cast.
        self.buttons = {}
        row = gui.Horiz(0.4 * em)
        def button(key, label, fn):
            b = gui.Button(label)
            b.horizontal_padding_em, b.vertical_padding_em = 0.6, 0.2
            b.set_on_clicked(fn)
            self.buttons[key] = b
            row.add_child(b)
        button('layers', "layers: solid", self.cycle_layers)
        if self.recolour is not None:
            button('vis', "visibility: %s" % self.vis_mode, self.cycle_visibility)
        if self.surf_paint is not None:
            button('surf', "surface: %s" % self.surf_mode, self.cycle_surface_colour)
        if cubes is not None:
            button('vox', "voxels: %s" % self.mode, self.cycle_voxel_colour)
        if surface is not None:
            button('showsurf', "hide surface", lambda: self._toggle('surface', 'surface'))
        if cubes is not None:
            button('showvox', "hide voxels", lambda: self._toggle('vox', 'vox'))
        button('reframe', "reframe", self.reframe)
        row.add_stretch()
        self.panel.add_child(row)

        self.legend = gui.Horiz(0.75 * em)
        self.state_label = gui.Label("")
        self.state_label.text_color = gui.Color(0.6, 0.75, 1.0)
        self.legend.add_child(self.state_label)
        for cid in sorted(set(int(c) for c in cls if c >= 0)) if cls is not None else []:
            r, g, b = (int(x) for x in DEFAULT_COLORMAP[cid])
            sw = gui.Label(CLASS_NAMES[cid])
            sw.text_color = gui.Color(r / 255.0, g / 255.0, b / 255.0)
            self.legend.add_child(sw)
        self.legend.add_stretch()
        self.panel.add_child(self.legend)
        self.window.add_child(self.panel)
        self._update_state()

        self.window.set_on_layout(self._on_layout)
        self.widget3d.set_on_mouse(self._on_mouse)

    # -- state -----------------------------------------------------------
    # (name, voxel alpha, surface brightness)
    LAYERS = [('solid',   1.00, 1.00),   # prediction only -- it hides the rest
              ('measure', 0.55, 1.00),   # see-through cubes over the surface
              ('predict', 0.55, 0.35)]   # same, surface pushed back
    OPAQUE, SEE_THROUGH = "defaultUnlit", "defaultLitTransparency"

    def _apply_dim(self):
        """Restack the two layers.

        Merely darkening the prediction achieves nothing: a cube is opaque and
        encloses the measured surface running through it, so a dark cube hides
        it exactly as well as a bright one. The cubes have to let light
        through, which needs a transparent shader -- and note that
        defaultUnlitTransparency is NOT usable in open3d 0.17: it aborts the
        process with `uniform named "srgbColor" not found`. defaultLitTransparency
        works. It is a lit shader, so the colours shift a little; at 0.55 alpha
        they still read clearly, which is what matters for a layer you are
        looking THROUGH.
        """
        alpha, bright = next((a, b) for n, a, b in self.LAYERS if n == self.dim)
        sc = self.widget3d.scene
        if self.cubes is not None:
            if sc.has_geometry("vox"):
                sc.remove_geometry("vox")
            if alpha >= 1.0:
                mat = self.mat
            else:
                mat = self.rendering.MaterialRecord()
                mat.shader = self.SEE_THROUGH
                mat.base_color = [1.0, 1.0, 1.0, alpha]
            sc.add_geometry("vox", self.cubes, mat)
            sc.show_geometry("vox", self.show['vox'])
        if self.surface is not None and self.surf_paint is not None:
            used, shade, layout = self.surf_paint
            paint_surface(self.surface, self.surf_palette[self.surf_mode], used,
                          shade * bright, layout)
            if sc.has_geometry("surface"):
                sc.remove_geometry("surface")
            sc.add_geometry("surface", self.surface, self.mat)
            sc.show_geometry("surface", self.show['surface'])
        self._update_state()
        self.window.post_redraw()

    def _update_state(self):
        bits = []
        if self.surface is not None:
            bits.append("surface %s [%s]%s"
                        % ("on" if self.show['surface'] else "off",
                           self.surf_mode,
                           " DIM" if self.dim == 'predict' else ""))
        if self.cubes is not None:
            bits.append("voxels %s [%s / %s]%s"
                        % ("on" if self.show['vox'] else "off", self.mode,
                           self.vis_mode,
                           " SEE-THROUGH" if self.dim != 'solid' else ""))
        self.state_label.text = "   ".join(bits)
        b = self.buttons
        if 'layers' in b:
            b['layers'].text = "layers: %s" % self.dim
        if 'vis' in b:
            b['vis'].text = "visibility: %s" % self.vis_mode
        if 'surf' in b:
            b['surf'].text = "surface: %s" % self.surf_mode
        if 'vox' in b:
            b['vox'].text = "voxels: %s" % self.mode
        if 'showsurf' in b:
            b['showsurf'].text = ("hide surface" if self.show['surface']
                                  else "show surface")
        if 'showvox' in b:
            b['showvox'].text = ("hide voxels" if self.show['vox']
                                 else "show voxels")

    def _toggle(self, name, geom):
        self.show[name] = not self.show[name]
        self.widget3d.scene.show_geometry(geom, self.show[name])
        self._update_state()
        self.window.post_redraw()

    def _repaint_surface(self):
        self._apply_dim()               # it repaints from surf_mode + self.dim

    def _repaint_voxels(self):
        DS.paint_mesh(self.cubes, self.palette[self.mode], self.vox_of_vert,
                      self.shade)
        self._apply_dim()

    # -- camera / layout -------------------------------------------------
    def _extent(self):
        # framed on the PREDICTION's extent, exactly as display_solid frames
        # it, so the two viewers show the scene at the same scale and size --
        # the surface is being compared against those voxels, and a framing
        # that moved with whichever geometry was visible would make the
        # comparison harder, not easier. --frame-grid frames the whole grid
        # instead, so any two files of one frame open at the same view.
        if self.frame_box is not None:
            return self.frame_box
        pts = self.view_idx.astype(np.float64)
        return pts.min(axis=0) - 0.5, pts.max(axis=0) + 0.5

    def reset_view(self):
        frame = self.widget3d.frame
        lo, hi = self._extent()
        ext = hi - lo
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

    # -- input -----------------------------------------------------------
    # -- actions (wired to the buttons above) ----------------------------
    def cycle_layers(self):
        names = [n for n, _, _ in self.LAYERS]
        self.dim = names[(names.index(self.dim) + 1) % len(names)]
        self._apply_dim()

    def cycle_visibility(self):
        if self.recolour is None:
            return
        m = DS.VISIBILITY_MODES
        self.vis_mode = m[(m.index(self.vis_mode) + 1) % len(m)]
        # only the voxel COLOURS depend on the rule. self.measured is the
        # per-cell measured-range grid from build_cell_state, which the readout
        # uses and which the visibility rule has nothing to do with -- assigning
        # photo_colours' per-voxel mask over it was an IndexError waiting in
        # _describe.
        self.palette, _, _ = self.recolour(self.vis_mode)
        self._repaint_voxels()

    def cycle_surface_colour(self):
        if self.surf_paint is None:
            return
        self.surf_mode = SURFACE_MODES[
            (SURFACE_MODES.index(self.surf_mode) + 1) % len(SURFACE_MODES)]
        self._repaint_surface()

    def cycle_voxel_colour(self):
        if self.cubes is None:
            return
        self.mode = COLOUR_MODES[(COLOUR_MODES.index(self.mode) + 1)
                                 % len(COLOUR_MODES)]
        self._repaint_voxels()

    def reframe(self):
        self.reset_view()
        self.window.post_redraw()

    # -- picking ---------------------------------------------------------
    def _pick_at(self, event):
        if DS._PICK_FAILED:
            return None
        frame = self.widget3d.frame
        x, y = int(event.x) - frame.x, int(event.y) - frame.y
        if not (0 <= x < frame.width and 0 <= y < frame.height):
            return None
        ray = guard_pick(pick_ray, self.widget3d.scene.camera,
                         x, y, frame.width, frame.height)
        if ray is None:                  # guard_pick returns None, not a pair
            return None
        origin, direction = ray
        if origin is None:
            return None
        # only what is visible is cast against, so the readout always
        # describes something on screen
        scenes = [self._scene[k] for k in ('surface', 'vox')
                  if self.show[k] and self._scene[k] is not None]
        hit = guard_pick(first_hit, scenes, origin, direction)
        if hit is None:
            return None
        cell = np.floor(hit + 0.5).astype(np.int64)
        if not np.all((cell >= 0) & (cell < np.asarray(GRID))):
            return 'outside'             # measured, but beyond the 4.8 m volume
        return cell

    def _show_pick(self, cell):
        key = (cell if isinstance(cell, str) or cell is None
               else tuple(int(v) for v in cell))
        if key == self._picked:
            return
        self._picked = key
        if key is None or key == 'outside':
            if self.highlight:
                self.widget3d.scene.show_geometry("pick", False)
            self.info.text = (self.idle_text if key is None else
                              "measured surface outside the 4.8 m volume -- "
                              "the model was never asked about this point")
        else:
            if self.highlight:
                t = np.eye(4)
                t[:3, 3] = key
                self.widget3d.scene.set_geometry_transform("pick", t)
                self.widget3d.scene.show_geometry("pick", True)
            self.info.text = self._describe(key)
        self.window.post_redraw()

    def _on_mouse(self, event):
        gui = self.gui
        moved = event.type == gui.MouseEvent.Type.MOVE
        clicked = (event.type == gui.MouseEvent.Type.BUTTON_DOWN
                   and event.is_modifier_down(gui.KeyModifier.CTRL))
        if not (moved or clicked):
            return gui.Widget.EventCallbackResult.IGNORED
        cell = self._pick_at(event)
        self._show_pick(cell)
        if clicked and cell is not None and not isinstance(cell, str):
            print(self.info.text, flush=True)
        # MOVE must stay IGNORED or the camera controls stop seeing drags
        return (gui.Widget.EventCallbackResult.HANDLED if clicked
                else gui.Widget.EventCallbackResult.IGNORED)

    def _describe(self, cell):
        """The measurement and the prediction for one cell, side by side.

        `cell` is in the DISPLAY's (reflected) coordinates; the index reported
        is the file's own, which is what indexes the prediction arrays.
        """
        rng = float(self.measured[cell])
        cid = int(self.predicted[cell])
        # the reflection is its own inverse, so undoing it is arithmetic, not
        # a search: z_file = pivot - z_view
        idx = (cell[0], cell[1],
               self.pivot - cell[2] if self.fix_axes else cell[2])
        left = ("measured at %.2f m" % rng if np.isfinite(rng)
                else "no measured surface here")
        right = ("model says %s (class %d)" % (CLASS_NAMES[cid], cid)
                 if cid >= 0 else "model left this cell EMPTY")
        return "cell [%2d, %2d, %2d]   %s   %s" % (idx[0], idx[1], idx[2],
                                                   left, right)


# ------------------------------------------------------------------- driver


# ------------------------------------------------------------------- driver

def grid_box():
    """The whole 60x36x60 grid in view coordinates (the depth reflection about
    FIXED_PIVOT = 59 maps 0..59 onto itself)."""
    return np.array([-0.5, -0.5, -0.5]), np.array(GRID, dtype=np.float64) - 0.5


def screenshot(geoms, view_idx, path, width=1600, height=1000, box=None):
    from open3d.visualization import rendering
    ren = rendering.OffscreenRenderer(width, height)
    ren.scene.set_background([0.05, 0.05, 0.08, 1.0])
    ren.scene.view.set_post_processing(False)
    mat = rendering.MaterialRecord()
    mat.shader = "defaultUnlit"
    for name, g in geoms:
        ren.scene.add_geometry(name, g, mat)
    pts = view_idx.astype(np.float64)      # same framing as the window
    lo, hi = box if box is not None else (pts.min(axis=0) - 0.5, pts.max(axis=0) + 0.5)
    ext, tan = hi - lo, float(np.tan(np.radians(30.0)))
    dist = 1.05 * max(ext[1] / 2.0 / tan, ext[0] / 2.0 / (tan * width / height))
    c = 0.5 * (lo + hi)
    ren.setup_camera(60.0, c, c + np.array([0.0, 0.0, dist]), [0.0, 1.0, 0.0])
    o3d.io.write_image(path, ren.render_to_image())
    print("wrote %s" % path)


def main():
    ap = argparse.ArgumentParser(
        description=__doc__.split('***')[0],
        formatter_class=argparse.RawDescriptionHelpFormatter)
    RC.add_common_args(ap)
    ap.add_argument('--show', choices=['surface', 'voxels', 'both'],
                    default='both',
                    help="what is visible at startup (default both -- the "
                         "layers button makes the cubes see-through so the "
                         "surface underneath shows)")
    ap.add_argument('--colour', '--color', dest='colour', default='class',
                    choices=COLOUR_MODES,
                    help="how to colour the predicted voxels next to the "
                         "measurement (default: class -- the flat label "
                         "colours, which is what you want against a "
                         "photo-textured surface)")
    ap.add_argument('--visibility', default='points',
                    choices=DS.VISIBILITY_MODES,
                    help="how the predicted voxels get their photo colour, as "
                         "in display_solid (default points -- no voxel is ever "
                         "coloured from another's measurements)")
    ap.add_argument('--reach', type=float, default=0.75, metavar='VOXELS',
                    help="--visibility surface only; see display_solid")
    ap.add_argument('--hidden', default='class', choices=DS.HIDDEN_MODES)
    ap.add_argument('--frame-grid', action='store_true',
                    help="frame the whole 60x36x60 grid and reflect about the grid, "
                         "not this file's voxels, so a prediction and its GT open "
                         "at exactly the same view (show_prediction_gt.py uses this)")
    args = ap.parse_args()
    if args.frame_grid:
        DS.FIXED_PIVOT = GRID[2] - 1

    b = RC.build(args.ply_path, args)
    idx, cls, view_idx = b['idx'], b['cls'], b['view_idx']
    class_rgb, frame, depth = b['class_rgb'], b['frame'], b['depth']

    def recolour(vis_mode):
        """Voxel colours for one --visibility rule. Handed to the viewer so the
        button can switch rule without restarting (~0.2 s a click)."""
        ph, seen, meas = DS.photo_colours(
            idx, frame['vox_origin'], frame['cam_pose'], frame['cam_K'],
            frame['image'], depth, mode=vis_mode, reach=args.reach)
        ph = DS.fill_hidden(ph, seen, class_rgb, view_idx, args.hidden)
        return {'photo': ph, 'class': class_rgb}, seen, meas

    palette, _, _ = recolour(args.visibility)
    cubes, vov, shade = DS.build_cube_mesh(view_idx, palette[args.colour])

    show_surface = args.show in ('both', 'surface')
    show_vox = args.show in ('both', 'voxels')
    if args.screenshot:
        geoms = ([("surface", b['surface'])] if show_surface else []) \
            + ([("vox", cubes)] if show_vox else [])
        screenshot(geoms, view_idx, args.screenshot, box=grid_box() if args.frame_grid else None)
        return

    from open3d.visualization import gui
    app = gui.Application.instance
    app.initialize()
    Viewer(b['surface'], cubes, view_idx, idx, cls, palette, vov, shade,
           os.path.basename(args.ply_path) + "  [overlay]", args.colour,
           vis_mode=args.visibility, recolour=recolour,
           surf_palette=b['surf_palette'], surf_mode=args.surf_mode,
           surf_paint=b['surf_paint'],
           measured=b['measured'], predicted=b['predicted'], pivot=b['pivot'],
           fix_axes=args.fix_axes,
           show_surface=show_surface, show_vox=show_vox,
           highlight=args.highlight, frame_box=grid_box() if args.frame_grid else None)
    app.run()


if __name__ == '__main__':
    main()
