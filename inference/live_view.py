"""
inference/live_view.py

The live window for `run_live --display`: one open3d window, the voxels
replaced in place every frame while the viewpoint you set stays put.

This is deliberately NOT display_solid.py. That file is the inspector -- it
owns the main thread through open3d's `gui` toolkit (`gui.Application.run()`),
which buys per-voxel picking and the legend panel and costs a worker thread
plus a file-shaped entry point. Here the renderer is the legacy
`o3d.visualization.Visualizer`, which is a plain window driven from whatever
loop you already have: `update_geometry` / `poll_events` / `update_renderer`
called straight from run_live's frame loop, no threads, no marshalling.

What it gives up against display_solid is picking (the legacy window has no
mouse-position callback, so this is not a matter of effort) and the widget
panels. What it keeps is everything that makes the picture: the geometry and
the colouring are display_solid's own functions, imported, not reimplemented
-- `build_cube_mesh`, `paint_mesh`, `photo_colours`, `fill_hidden`. Same
culled solid cubes, same baked face shading, same colour rules.

*** COLOUR MODES ***
`class` paints each voxel its semantic colour: free, and the mode that
answers "is the model labelling the room correctly", which is usually the
question a live run is asking.

The other two paint the voxels with what the camera actually saw, and they
differ in how hard they work at it (per frame, 9 k voxels, 265 k depth
points, measured on an RTX 3050 Ti box):

    zbuffer   17 ms   the voxel front-most over each pixel takes that pixel.
                      Needs no depth frame. Wrong for a surface seen edge-on
                      -- consecutive bed voxels sit far apart in depth, the
                      nearest wins every pixel, the rest fall back to class
                      colour.
    all       98 ms   the three rules stacked, each voxel taking the first
                      that reaches it: its own measured points, else a
                      neighbour's within `reach`, else the photo pixels it is
                      front-most on. Hidden voxels borrow the nearest visible
                      sibling OF THE SAME CLASS.

`all` was 424 ms before this window existed, which is what made it worth
optimising rather than excluding: see `display_solid.points_in_cubes`. The
budget it now fits in is a 317 ms frame, so `all` costs about a third of the
frame rate and `zbuffer` about a twentieth.

*** THE OPENING VIEWPOINT ***
Where the window first points is open3d's own fit to the geometry, which
frames the bounding SPHERE and so backs off about twice as far as it needs
to -- the room lands at roughly 40% of the window and you scroll once.

That is not a choice. In open3d 0.17 `Visualizer.get_view_control()` returns a
COPY, so `set_lookat` / `set_front` / `set_zoom` change an object the window
does not own: they report success, the readback through the same handle
agrees, and the picture never moves. Verified by screenshot -- the rendered
pixels are byte-identical at zoom 0.3 and 0.9. display_solid aims its camera
with `widget3d.look_at`, which works, but that is the `gui` toolkit and
belongs to the thread-owning architecture this file exists to avoid.

Mouse control is unaffected: orbit, pan and zoom go through the real view
control inside the window, and the viewpoint you set survives every update
because the geometry is replaced in place rather than re-added.

*** THE CAMERA PANELS ***
Beside the 3D window, a second one holds what the model was actually given:
the colour frame on top, the depth below it, colourised with
`colorize_depth.colorize` -- the RealSense Viewer's own rs-jet ramp with
per-frame histogram equalisation, the same function capture.py's `--preview`
uses, so the three agree.

They show the frame AFTER match_nyu_fov, not the sensor's native one: 640x480
with NYU's intrinsics, which is what the network sees and therefore what you
want to judge. A footer carries the frame name, the throughput, and the
fraction of depth pixels the sensor actually measured -- the number that
explains a thin prediction faster than the prediction does.

It is a separate OpenCV window because the 3D one cannot hold widgets (see
VIEWERS.md); the two are positioned side by side at startup.

Press `c` in the 3D window to cycle the colour modes, `s` to drop a
screenshot.
"""

import logging
import os
import time

import numpy as np

log = logging.getLogger('live_view')

GRID = (60, 36, 60)
MODES = ('class', 'zbuffer', 'all')


class LiveView:
    """One window, re-fed per frame. `update()` returns False once it closes.

    Nothing is imported until the window is actually asked for, so `run_live`
    without --display neither needs open3d nor pays to import it.
    """

    def __init__(self, colour='zbuffer', reach=0.75, width=1280, height=800,
                 out_dir=None, title='CleanerS live', panels=True,
                 panel_width=640):
        if colour not in MODES:
            raise ValueError('colour must be one of %s, not %r'
                             % (', '.join(MODES), colour))
        import open3d as o3d
        try:
            from . import display_solid as ds
        except ImportError:
            import display_solid as ds
        self.o3d, self.ds = o3d, ds

        # to_view_index reflects the depth axis about a pivot taken from the
        # frame's own voxel bounds, which MOVES as the scene changes -- the
        # whole room would jump between frames. Pin it to the grid, exactly as
        # display_overlay.py does to put two files in one coordinate system.
        ds.FIXED_PIVOT = GRID[2] - 1

        self.colour = colour
        self.reach = reach
        self.out_dir = out_dir
        self.frames = 0
        self.colour_ms = 0.0
        self.draw_ms = 0.0
        self.closed = False
        self.geom = None
        self._last = None          # (idx, cls, loader, rgb), to recolour

        self.panels = panels
        self.panel_w = panel_width
        self.panel_window = 'CleanerS camera' if panels else None
        self._cv = None
        if panels:
            import cv2
            try:
                from .colorize_depth import INVALID_COLORS, colorize
            except ImportError:
                from colorize_depth import INVALID_COLORS, colorize
            self._cv = cv2
            self._colorize = colorize
            self._invalid = INVALID_COLORS['black']

        self.vis = o3d.visualization.VisualizerWithKeyCallback()
        self.vis.create_window(title, width, height, left=0, top=0)
        opt = self.vis.get_render_option()
        # The face shading is baked into the vertex colours (display_solid's
        # FACES table). Real lighting on top would tone-map the class colours
        # and make them disagree with the legend, so the renderer is unlit.
        opt.light_on = False
        opt.mesh_color_option = o3d.visualization.MeshColorOption.Color
        opt.mesh_show_back_face = True
        opt.background_color = np.array([0.08, 0.08, 0.09])
        self.vis.register_key_callback(ord('C'), self._cycle)
        self.vis.register_key_callback(ord('S'), self._screenshot)
        if panels:
            self._cv.namedWindow(self.panel_window, self._cv.WINDOW_AUTOSIZE)
            self._cv.moveWindow(self.panel_window, width + 20, 0)

    # ---- the frame loop calls this ----
    def update(self, pred_label, mask, loader, rgb=None, name='', fps=None):
        """-> False when the user has closed the window.

        pred_label/mask: what run_live already hands save_prediction_ply.
        loader: the FrameLoader, for cam_K / cam_pose / vox_origin / depth.
        rgb: the frame's colour image, BGR as the loader got it.
        name/fps: for the panel footer only.
        """
        if self.closed:
            return False
        if self.panels:
            self._draw_panels(loader.depth, rgb, name, fps)
        idx, cls = occupied(pred_label, mask)
        if len(idx) == 0:                     # nothing predicted: keep the
            return self._draw()               # last picture rather than crash
        self._last = (idx, cls, loader, rgb)
        t0 = time.time()
        colours = self._paint(idx, cls, loader, rgb)
        t1 = time.time()
        mesh, _, _ = self.ds.build_cube_mesh(self.ds.to_view_index(idx),
                                             colours)
        self._swap(mesh)
        alive = self._draw()
        self.frames += 1
        self.colour_ms += 1000 * (t1 - t0)
        self.draw_ms += 1000 * (time.time() - t1)
        return alive

    def close(self):
        if not self.closed:
            self.vis.destroy_window()
            if self.panels:
                try:
                    self._cv.destroyWindow(self.panel_window)
                except Exception:        # already gone, or no GUI backend
                    pass
            self.closed = True

    # ---- the camera panels ----
    def _draw_panels(self, depth, rgb, name, fps):
        """Colour over depth, as the model received them.

        The depth ramp is colorize_depth's rs-jet with per-frame equalisation,
        which is what capture.py's preview and the RealSense Viewer both show.
        Equalising per frame means the colours are not comparable BETWEEN
        frames -- it is a shape readout, not a measurement -- and the footer
        carries the numbers for that.
        """
        cv2 = self._cv
        w = self.panel_w
        panes = []
        if rgb is not None:
            panes.append(('camera', np.asarray(rgb, np.uint8)))
        mm = np.clip(np.nan_to_num(depth) * 1000.0, 0, 65535).astype(np.uint16)
        panes.append(('depth, rs-jet (equalised per frame)',
                      self._colorize(mm, 'rs-jet', self._invalid, 'equalize')))

        out = []
        for label, img in panes:
            h = int(round(img.shape[0] * w / float(img.shape[1])))
            pane = cv2.resize(img, (w, h), interpolation=cv2.INTER_AREA)
            bar = np.zeros((22, w, 3), np.uint8)
            cv2.putText(bar, label, (6, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                        (200, 200, 200), 1, cv2.LINE_AA)
            out.extend([bar, pane])

        valid = float((depth > 0).mean())
        med = float(np.median(depth[depth > 0])) if valid else 0.0
        foot = np.zeros((24, w, 3), np.uint8)
        cv2.putText(foot, '%s  depth %.0f%% valid, median %.2f m%s'
                    % (name or '-', 100 * valid, med,
                       '   %.1f fps' % fps if fps else ''),
                    (6, 17), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                    (170, 220, 170) if valid > 0.5 else (140, 140, 240),
                    1, cv2.LINE_AA)
        out.append(foot)
        cv2.imshow(self.panel_window, np.vstack(out))
        cv2.waitKey(1)                   # pumps the window; it draws nothing

    def report(self):
        if not self.frames:
            return 'display: no frames drawn'
        return ('display: %d frames | colour %.0f ms | mesh+draw %.0f ms '
                '| mode %s' % (self.frames, self.colour_ms / self.frames,
                               self.draw_ms / self.frames, self.colour))

    # ---- colour ----
    def _paint(self, idx, cls, loader, rgb):
        class_rgb = self.ds.DEFAULT_COLORMAP[cls]
        if self.colour == 'class' or rgb is None:
            return class_rgb
        # display_solid's photo rules want RGB; the pipeline carries BGR
        image = np.ascontiguousarray(
            np.asarray(rgb)[:, :, ::-1]).astype(np.uint8)
        depth = None if self.colour == 'zbuffer' else loader.depth
        photo, visible, _, _ = self.ds.photo_colours(
            idx, loader.vox_origin, loader.cam_pose, loader.cam_K, image,
            depth, mode=self.colour, reach=self.reach)
        # 'all' reaches enough voxels that borrowing a same-class sibling's
        # colour reads better than dimming; 'zbuffer' misses too many for that
        # to be anything but a smear, so those stay dimmed class colour.
        hidden = 'nearest' if self.colour == 'all' else 'class'
        out, _ = self.ds.fill_hidden(photo, visible, class_rgb,
                                     self.ds.to_view_index(idx), hidden)
        return out

    # ---- window plumbing ----
    def _swap(self, mesh):
        if self.geom is None:
            self.geom = mesh
            # open3d fits the view to the geometry here, and that is the only
            # chance to aim the camera -- see the docstring on the opening
            # viewpoint
            self.vis.add_geometry(self.geom)
            return
        # one geometry, its arrays replaced: the vertex count changes every
        # frame, and add/remove would refit the camera and throw away the
        # viewpoint the user set
        self.geom.vertices = mesh.vertices
        self.geom.triangles = mesh.triangles
        self.geom.vertex_colors = mesh.vertex_colors
        self.vis.update_geometry(self.geom)

    def _draw(self):
        alive = self.vis.poll_events()
        self.vis.update_renderer()
        if not alive:
            self.close()
        return alive

    def _cycle(self, _vis):
        self.colour = MODES[(MODES.index(self.colour) + 1) % len(MODES)]
        log.info('display colour=%s', self.colour)
        if self._last is not None:                    # repaint the frame on
            idx, cls, loader, rgb = self._last        # screen, do not wait
            mesh, _, _ = self.ds.build_cube_mesh(
                self.ds.to_view_index(idx), self._paint(idx, cls, loader, rgb))
            self._swap(mesh)
            self._draw()
        return False

    def _screenshot(self, _vis):
        d = self.out_dir or '.'
        os.makedirs(d, exist_ok=True)
        path = os.path.join(d, 'live_%s.png' % time.strftime('%H%M%S'))
        self.vis.capture_screen_image(path, do_render=True)
        log.info('screenshot -> %s', path)
        return False


def occupied(pred_label, mask):
    """The voxels the .ply would hold, and their classes.

    Same selection as visualize.save_prediction_ply: zero the voxels the mask
    excludes, then keep what is left occupied. The (x, y, z) order of
    np.nonzero matches get_xyz's meshgrid, so these indices are the ones
    display_solid's ply reader would have produced.
    """
    lab = np.asarray(pred_label).reshape(GRID).copy()
    lab[~np.asarray(mask).reshape(GRID).astype(bool)] = 0
    lab[lab == 255] = 0
    occ = lab > 0
    return np.stack(np.nonzero(occ), axis=1).astype(np.int32), lab[occ]
