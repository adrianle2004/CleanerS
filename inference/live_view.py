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

*** FRAMING, AND WHY IT IS DONE TO THE GEOMETRY ***
In open3d 0.17 `Visualizer.get_view_control()` returns a COPY, so `set_zoom`,
`set_lookat` and `set_front` change an object the window does not own. They
report success, the readback through the same handle agrees, and the picture
never moves -- verified by screenshot, the rendered pixels being identical at
zoom 0.3 and 0.9. The camera cannot be aimed from Python here.

What CAN be controlled is the geometry, which this file rebuilds every frame
anyway. So the framing is done there, in two steps:

1. The camera is fitted once, to a proxy box covering the whole 60x36x60
   GRID, which is then removed. Fitting the first frame's voxels instead
   would make the opening view depend on what the model happened to predict
   in that frame, and the orbit pivot land somewhere arbitrary. The pivot is
   now the grid's centre, always.

2. Every frame's mesh is scaled about that same centre by `zoom`. Because the
   camera is fixed, scaling the content is exactly equivalent to moving the
   camera in or out, and it is immune to the bug above.

open3d's own fit frames the bounding SPHERE, so it sits about twice as far
back as it needs to. Measured, as a fraction of the window's height:

    zoom          1.0    1.7    2.0
    a live frame   31%    58%    65%
    an NYU frame   44%    86%    99%, touching the edges

2.0 is the default because --display is for watching a camera, where the
room occupies much less of the 4.8 x 4.8 x 2.88 m volume than NYU's
room-canonical placement does. Replaying NYU frames, --zoom 1.7 fits better.

The direction needs no correction -- the default view looks down -Z with +Y
up, which after `to_view_index` is exactly where the capture camera stood, the
same viewpoint display_solid opens with.

`+` and `-` in the window step the zoom and `0` resets it, but a key press has
to reach the window, and the CLI's `--zoom` is the path that cannot fail.

*** THE CAMERA PANELS ***
Beside the 3D window, a resizable second one holds what the model was given:
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
VIEWERS.md). The two are placed side by side at startup, and `panel_width`
(`--panel_width`) sets how wide this one starts: 420 plus the 3D window's
1280 fits a 1728-wide desktop, which is what a 3456x2160 panel at 2x gives
you. It is a WINDOW_NORMAL, so drag it to whatever size suits.

Press `c` in the 3D window to cycle the colour modes, `s` to drop a
screenshot.

*** ONE FRAME BEHIND ***

Painting is numpy, which releases the GIL, so `update()` hands the frame it
is given to a worker thread and draws the PREVIOUS one. The paint then runs
against the caller's next encode and forward pass and costs almost nothing:
measured 96 of 96 ms hidden. With no window in the way that is 377 -> 295 ms
a frame on a live camera, and a process painter measures the same as a
thread, so the GIL is not the limit. With the window open the honest claim
is that the paint stopped costing anything -- `all` now runs at `class`'s
frame rate.

What that buys is paid for in latency: the window is one frame behind the
camera. The RGB/depth panels are deferred with the mesh, from the same frame,
because a window whose two halves disagree is a worse instrument than a slow
one. `overlap=False` (`run_live --sync_display`) draws each frame in its own
iteration instead.

What governs the rate after that is the window itself (~0.4 fps, open3d's GL
wanting the same card as the inference) and how hot the card is: `infer`
drifts 204 -> 300 ms under sustained load on this laptop, which moves the
number by more than any of this does.
"""

import logging
import os
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np

log = logging.getLogger('live_view')

GRID = (60, 36, 60)
MODES = ('class', 'zbuffer', 'all')


def view_points(p_world, origin, vox, pivot):
    """World METRES -> the viewer's index frame, for the accumulating room.

    The one piece of arithmetic the room's two layers have to share. The
    cubes are placed by grid index and the sensor's reconstruction is a
    mesh in metres, and if these disagree by so much as half a voxel the
    comparison the window exists for is quietly wrong.

    A voxel CENTRE is at origin + (i + 0.5) * vox and its cube is drawn
    at i, hence the half. Then display_solid's axis order -- index axis 1
    is UP and the room's up axis is Z -- and the depth-axis reflection
    about a FIXED pivot, as display_overlay does, so nothing flips when
    the extent changes.

    Pure, and tested in tests/test_live_view.py: it imports no open3d and
    needs no window.
    """
    g = (np.asarray(p_world, np.float64).reshape(-1, 3)
         - np.asarray(origin, np.float64)) / float(vox) - 0.5
    return np.stack([g[:, 0], g[:, 2], float(pivot) - g[:, 1]], axis=1)


class LiveView:
    """One window, re-fed per frame. `update()` returns False once it closes.

    Nothing is imported until the window is actually asked for, so `run_live`
    without --display neither needs open3d nor pays to import it.
    """

    def __init__(self, colour='zbuffer', reach=0.75, width=1280, height=800,
                 out_dir=None, title='CleanerS live', panels=True,
                 panel_width=420, zoom=2.0, overlap=True):
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
        self.zoom = float(zoom)
        # the grid's centre in view-index coordinates: the camera's pivot
        # after the proxy fit, and what every frame is scaled about
        self.centre = np.array([(g - 1) / 2.0 for g in GRID], dtype=np.float64)
        self.out_dir = out_dir
        self.frames = 0
        self.colour_ms = 0.0
        self.draw_ms = 0.0
        self.wait_ms = 0.0
        self.closed = False
        self.geom = None
        self._last = None          # (idx, cls, loader, rgb), to recolour

        # --accumulate state. The room is a different grid from the frame,
        # so the proxy box the camera is fitted to, the orbit pivot and the
        # depth-axis reflection all have to come from it instead of GRID.
        self.proxy = GRID
        # None = decide per mode, which is the default and the only
        # honest one. In `photo` the room is a PLACE and everything is
        # drawn whole, so it looks like one. In `verdict` it is a
        # COMPARISON, and whole cubes hide it: face culling means you
        # only ever see the outermost shell, and that shell is the
        # model's unverified claims wrapped around the real room -- on
        # room07, all 3604 proved voxels are measured surfaces inside it
        # and not one is visible. So there the model's unmeasured claims
        # shrink and you see the room through the gaps. A number forces
        # one answer for every mode.
        self.shrink = None
        # 'photo' is the default and the one to show people: every voxel
        # wears the camera's own colour, filled in from the nearest
        # coloured voxel of the same class where the camera never saw it
        # -- display_solid's `all` rule, applied to a room instead of a
        # frame. 'verdict' is the comparison, and 'class' the semantics.
        self.world_colour = 'photo'
        self.frozen = False
        # metres; what the opening view is framed on. Independent of how
        # big the map is -- the map has to be large enough for the camera
        # to walk around in, the framing only has to show the room.
        self.view_extent = (8.0, 8.0, 3.04)
        self.mesh_every = 3
        self.record_dir = None
        # The sensor's own reconstruction, in world metres -- what the
        # tracker's TSDF already holds. `m` flips the window between it
        # and the room, at the same viewpoint, which is the comparison:
        # the mesh stops where the depth stopped, and the holes in it are
        # what the model is claiming to fill.
        self.recon = None
        self.show_mesh = False
        self._wpivot = None
        self._wframe = 0
        self._wmesh = None
        self._wstatus = []
        self._wforce = False

        # Painting a frame is pure numpy and releases the GIL, so it can run
        # while the GPU encodes and infers the NEXT frame -- measured 103 of
        # 103 ms hidden behind a 195 ms forward pass. The cost is that what
        # is on screen is one frame behind the camera; the panels are drawn
        # from the same frame as the mesh, so the two halves always agree.
        self.overlap = bool(overlap)
        self._pool = ThreadPoolExecutor(max_workers=1) if overlap else None
        self._job = None

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
        self.vis.register_key_callback(ord('M'), self._flip_mesh)
        self.vis.register_key_callback(ord('S'), self._screenshot)
        for key in (' ', 'F'):
            self.vis.register_key_callback(ord(key), self._freeze)
        for key in ('=', '+'):
            self.vis.register_key_callback(ord(key), self._zoom_in)
        for key in ('-', '_'):
            self.vis.register_key_callback(ord(key), self._zoom_out)
        self.vis.register_key_callback(ord('0'), self._zoom_reset)
        if panels:
            # NORMAL, not AUTOSIZE: AUTOSIZE pins the window to the image and
            # refuses to be resized, so a panel taller than the screen cannot
            # be shrunk or fully seen. NORMAL lets it be dragged to any size
            # and scales the image into it.
            self._cv.namedWindow(self.panel_window, self._cv.WINDOW_NORMAL)
            self._cv.resizeWindow(self.panel_window, panel_width,
                                  2 * int(round(panel_width * 3 / 4.0)) + 68)
            self._cv.moveWindow(self.panel_window, width + 20, 0)

    # ---- the frame loop calls this ----
    def update(self, pred_label, mask, loader, rgb=None, name='', fps=None):
        """-> False when the user has closed the window.

        pred_label/mask: what run_live already hands save_prediction_ply.
        loader: the FrameLoader, for cam_K / cam_pose / vox_origin / depth.
        rgb: the frame's colour image, BGR as the loader got it.
        name/fps: for the panel footer only.

        With `overlap` on (the default, except in `class` mode where there is
        nothing to hide), this DRAWS the previous frame and then starts
        painting this one, so the painting runs against the caller's next
        encode and forward pass. The window is therefore one frame behind.
        """
        if self.closed:
            return False
        job = self._job_for(pred_label, mask, loader, rgb, name, fps)

        if not self._overlapping():
            self._drain()                     # mode may have just changed
            if job is None:
                return self._draw()
            job['colours'] = self._paint_timed(job)
            return self._present(job)

        alive = True
        if self._job is not None:             # the frame painted last round
            done, self._job = self._job, None
            t0 = time.time()
            done['colours'] = done['future'].result()
            self.wait_ms += 1000 * (time.time() - t0)
            alive = self._present(done)
        if not alive:
            return False
        if job is None:
            return self._draw()
        job['future'] = self._pool.submit(self._paint_timed, job)
        self._job = job
        return True

    # ---- --accumulate: draw the ROOM, not the frame ----

    def update_world(self, world, depth=None, rgb=None, name='', fps=None,
                     note=None, force=False):
        """The accumulating path. -> False once the user closes the window.

        What separates the layers here is GEOMETRY, not transparency. The
        legacy Visualizer has no material API -- display_overlay's
        see-through cubes need the gui/rendering stack, which owns the main
        thread and so cannot be driven from run_live's loop. So:

            full 8 cm cube   the sensor has MEASURED something here
            half-size cube   the model COMPLETED it and nothing has
                             measured it yet -- you see between them, and
                             the real surface reads through the gaps
            green            a completion the sweep went on to PROVE
            red              a completion the sweep DISPROVED: it measured
                             free space where the model put something
            blue             the sensor found it and the model never said

        Which is readable without a legend: big solid blocks are things
        the camera has really seen, little blocks are guesses, green means
        a guess that turned out right.

        That legend is `verdict`'s. In `photo` and `class` the room is
        answering what is THERE, so a claim the sensor has measured
        through is dropped rather than drawn -- see _build_world.
        """
        if self.closed:
            return False
        if self._wpivot is None:
            # The orbit pivot is the MAP's centre, so it never moves as the
            # room fills. The proxy the camera is fitted to is the smaller
            # view_extent, so the opening view is framed on a room.
            dims = tuple(int(world.shape[i]) for i in (0, 2, 1))
            self.centre = np.array([(g - 1) / 2.0 for g in dims],
                                   dtype=np.float64)
            ex, ey, ez = self.view_extent
            self.proxy = tuple(
                min(d, max(1, int(round(e / world.vox))))
                for d, e in zip(dims, (ex, ez, ey)))
            self._wpivot = int(world.shape[1]) - 1

        # The room changes by a handful of voxels a frame, and rebuilding a
        # whole-room mesh every frame would spend the budget redrawing what
        # did not move. poll_events still runs every iteration below, so
        # the window stays alive while the mesh is a few frames stale.
        self._wframe += 1
        due = (force or self._wforce
               or self._wframe % max(1, self.mesh_every) == 1)
        self._wforce = False
        if due:
            t0 = time.time()
            # verdicts() walks every voxel in the room, so it is computed
            # ONCE here and shared by the mesh and the readout
            v = world.verdicts()
            self._wmesh = (self._build_recon(world)
                           if self.show_mesh and self.recon is not None
                           else self._build_world(world, v))
            self._wstatus = self._status_lines(v)
            self.colour_ms += 1000 * (time.time() - t0)

        t0 = time.time()
        status = list(self._wstatus)
        if note:
            status.append((note, (90, 90, 250)))
        if self.panels and depth is not None:
            self._draw_panels(depth, rgb, name, fps, status=status)
        if self._wmesh is not None:
            self._swap(self._wmesh)
        alive = self._draw()
        self.frames += 1
        self.draw_ms += 1000 * (time.time() - t0)
        if self.record_dir is not None:
            os.makedirs(self.record_dir, exist_ok=True)
            self.vis.capture_screen_image(
                os.path.join(self.record_dir, '%06d.png' % self.frames),
                do_render=True)
        return alive

    def hold_step(self, world, depth=None, rgb=None, name='', note=None):
        """One turn of a HELD window: answer the compositor, and rebuild
        only if a key asked for it. -> False once it is closed.

        Nothing is pulled from the source and nothing is recorded, so a
        run can sit here for as long as someone wants to look -- while
        frozen half way, or after the last frame.
        """
        if not self._wforce:
            return self.pump()
        rec, self.record_dir = self.record_dir, None
        try:
            alive = self.update_world(world, depth, rgb, name=name,
                                      note=note, force=True)
        finally:
            self.record_dir = rec
        return alive

    def _to_view(self, idx):
        """World grid index -> the viewer's left-to-right-handed frame.

        display_solid draws index axis 1 as UP (its FACES table shades +y
        brightest), and the world grid's up axis is Z, so the axes are
        reordered (X, Z, Y) -- the same order voxel_centres_world uses for
        a .ply. The depth axis is then reflected about a FIXED pivot, as
        display_overlay does, so the room does not flip when its extent
        changes.
        """
        return np.stack([idx[:, 0], idx[:, 2], self._wpivot - idx[:, 1]],
                        axis=1)

    WORLD_MODES = ('photo', 'verdict', 'class')
    # how big the model's unmeasured claims are drawn, per mode
    MODE_SHRINK = {'photo': 1.0, 'verdict': 0.45, 'class': 1.0}

    def _status_lines(self, v):
        """The counts, in the words that make them mean something.

        'untested' is reported as its own number and never folded into
        the confirmed count: most of what a completion model fills in is
        permanently unreachable by the sensor, and that is not a win.
        """
        n = dict((k, int(a.sum())) for k, a in v.items())
        settled = n['confirmed'] + n['contradicted']
        pct = (' -- %.0f%% right' % (100.0 * n['confirmed'] / settled)
               if settled else '')
        if self.show_mesh:
            return [
                ('%sthe SENSOR only -- its own reconstruction, %d triangles'
                 % ('FROZEN  ' if self.frozen else '',
                    len(self.recon.triangles) if self.recon is not None else 0),
                 (210, 210, 210)),
                ('every hole in it is somewhere the depth could not reach',
                 (170, 170, 170)),
                ('m: back to the room, which fills %d of them in'
                 % (n['confirmed'] + n['contradicted'] + n['untested']),
                 (170, 170, 170)),
            ]
        return [
            ('%s%s   measured %d   model filled in %d'
             % ('FROZEN  ' if self.frozen else '', self.world_colour,
                n['measured'], n['confirmed'] + n['contradicted']
                + n['untested']), (210, 210, 210)),
            ('completions swept so far: %d proved, %d disproved%s'
             % (n['confirmed'], n['contradicted'], pct),
             (120, 230, 140) if n['confirmed'] >= n['contradicted']
             else (140, 140, 240)),
            ('still untested %d   sensor-only %d   in free space %d'
             % (n['untested'], n['missed'], n['disproved']),
             (170, 170, 170)),
        ]

    def _world_colours(self, world, v, i, view):
        """One colour per occupied voxel, in the current mode.

        `photo` is display_solid's `all` rule applied to a room: a voxel
        wears the mean colour of the pixels that measured it, and one the
        camera never saw borrows from the nearest coloured voxel OF THE
        SAME CLASS -- nearest overall would smear the wall onto the bed
        standing in front of it. A class with no coloured member anywhere
        keeps a dimmed class colour, which reads as unseen. That is
        `fill_hidden`, imported, not reimplemented.
        """
        try:
            from .visualize import DEFAULT_COLORMAP
        except ImportError:
            from visualize import DEFAULT_COLORMAP
        k = world.predicted()[1][i[:, 0], i[:, 1], i[:, 2]]
        class_rgb = DEFAULT_COLORMAP[
            np.clip(k, 0, len(DEFAULT_COLORMAP) - 1)][:, :3].astype(np.int32)
        if self.world_colour == 'class':
            return class_rgb
        photo = world.colours()[i[:, 0], i[:, 1], i[:, 2]].astype(np.int32)
        seen = world.colour_n.reshape(world.shape)[
            i[:, 0], i[:, 1], i[:, 2]] > 0
        out = self.ds.fill_hidden(photo, seen, class_rgb, view,
                                  'nearest')[0].astype(np.int32)
        if self.world_colour != 'verdict':
            return out
        # the comparison, laid over the photo: anything the model claims
        # and the sensor has not measured goes neutral FIRST, then the
        # three verdicts paint over it
        g = v['predicted'] & ~v['measured']
        out[g[i[:, 0], i[:, 1], i[:, 2]]] = (168, 168, 174)
        for name, colour in (('confirmed', (60, 220, 90)),
                             ('disproved', (230, 60, 60)),
                             ('missed', (70, 130, 255))):
            m = v[name][i[:, 0], i[:, 1], i[:, 2]]
            if m.any():
                out[m] = colour
        return out

    # view units, pointing up and a little towards the viewer. The room's
    # cubes carry display_solid's baked face shading and the renderer is
    # unlit (see __init__), so the reconstruction has to be shaded the
    # same way or it arrives as a flat silhouette.
    _LIGHT = np.array([0.30, 0.90, 0.25])

    def _build_recon(self, world):
        """The reconstruction, in the room's own view coordinates.

        World metres -> continuous grid index -> _to_view's axis order and
        reflection -> the zoom every room mesh is scaled by. A voxel
        CENTRE sits at world origin + (i + 0.5) * vox and is drawn at
        index i, hence the half-voxel: without it the mesh floats half a
        voxel off the room it is meant to be compared with.
        """
        o3d = self.o3d
        src = self.recon
        pts = view_points(np.asarray(src.vertices), world.origin, world.vox,
                          self._wpivot)
        pts = (pts - self.centre) * self.zoom + self.centre
        m = o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(pts),
                                      src.triangles)
        # the axis swap is a permutation AND a reflection, so orientation
        # survives it and the winding does not have to be flipped
        m.compute_vertex_normals()
        n = np.asarray(m.vertex_normals)
        L = self._LIGHT / np.linalg.norm(self._LIGHT)
        shade = 0.42 + 0.58 * (0.5 + 0.5 * (n @ L))
        if len(src.vertex_colors):
            c = np.asarray(src.vertex_colors)
        else:
            c = np.full((len(pts), 3), 0.72)
        m.vertex_colors = o3d.utility.Vector3dVector(
            np.clip(c * shade[:, None], 0, 1))
        return m

    def _build_world(self, world, v):
        """One mesh of the whole room.

        Outside `verdict`, a claim the sensor has REFUTED is not drawn at
        all. The room is then answering what is there, and about space it
        has measured as empty the sensor outranks the model -- leaving
        the claim in puts matter in mid-air and thickens every wall by
        the voxel in front of it. `verdict` keeps them, in red: that mode
        is asking how the model did, and the mistakes are the answer.
        """
        ds, o3d = self.ds, self.o3d
        claim = v['predicted']
        if self.world_colour != 'verdict':
            claim = claim & ~v['in_free']
        i = np.argwhere(v['measured'] | claim)
        if not len(i):
            return None
        view = self._to_view(i)
        colours = self._world_colours(world, v, i, view)

        shrink = (self.MODE_SHRINK[self.world_colour]
                  if self.shrink is None else self.shrink)
        if shrink >= 1.0:
            # faces between two occupied voxels are dropped -- most of the
            # geometry in a filled room
            mesh = ds.build_cube_mesh(view, colours)[0]
        else:
            # only what the MODEL claims shrinks; measured voxels stay
            # whole, so the two are told apart by size again
            small = (claim & ~v['measured'])[i[:, 0], i[:, 1], i[:, 2]]
            mesh, vov, _ = ds.build_cube_mesh_all(view, colours)
            pts = np.asarray(mesh.vertices)
            mid = view[vov].astype(np.float64)
            scale = np.where(small[vov], shrink, 1.0)[:, None]
            mesh.vertices = o3d.utility.Vector3dVector(
                mid + (pts - mid) * scale)
        if self.zoom != 1.0:
            pts = np.asarray(mesh.vertices)
            mesh.vertices = o3d.utility.Vector3dVector(
                (pts - self.centre) * self.zoom + self.centre)
        return mesh

    def pump(self):
        """Answer the window system during a long step. -> False once closed.

        A window only breathes when `poll_events` runs, and this one is
        driven from `run_live`'s loop, so between frames that is every
        ~400 ms -- fine. The dangerous gaps are the ones around the loop: a
        first frame doing cudnn's autotune can take seconds, and a
        compositor that pings a window which does not answer puts up
        "not responding -- force quit or wait". Costs ~1 ms.
        """
        if self.closed:
            return False
        if self.panels:
            self._cv.waitKey(1)
        return self._draw()

    def _overlapping(self):
        # `class` paint is a colormap lookup -- overlapping it would buy
        # nothing and still cost a frame of lag
        return self.overlap and self.colour != 'class'

    def _job_for(self, pred_label, mask, loader, rgb, name, fps):
        """Everything one frame needs, or None when there is nothing to draw."""
        idx, cls = occupied(pred_label, mask)
        if len(idx) == 0:
            return None
        # the painter reads these one iteration later, so they must not be
        # views into a buffer the frame source is about to recycle. Every
        # run_live path already hands over owned arrays (astype and
        # cv2.remap both copy); this is the guard, not the assumption.
        if rgb is not None and rgb.base is not None:
            rgb = rgb.copy()
        return {'idx': idx, 'cls': cls, 'loader': loader, 'rgb': rgb,
                'depth': loader.depth, 'name': name, 'fps': fps,
                'colours': None, 'future': None}

    def _paint_timed(self, job):
        """Runs on the worker thread when overlapping, inline when not."""
        t0 = time.time()
        out = self._paint(job['idx'], job['cls'], job['loader'], job['rgb'])
        self.colour_ms += 1000 * (time.time() - t0)
        return out

    def _present(self, job):
        """Put one painted frame on screen: panels and mesh from the SAME
        frame, which is the point of deferring both together."""
        t0 = time.time()
        if self.panels:
            self._draw_panels(job['depth'], job['rgb'], job['name'],
                              job['fps'])
        self._swap(self._build(job['idx'], job['colours']))
        alive = self._draw()
        self._last = (job['idx'], job['cls'], job['loader'], job['rgb'])
        self.frames += 1
        self.draw_ms += 1000 * (time.time() - t0)
        return alive

    def _drain(self):
        """Finish any in-flight paint without drawing it. Called before
        anything that touches the geometry from this thread."""
        if self._job is not None:
            job, self._job = self._job, None
            try:
                job['future'].result()
            except Exception as e:            # a paint bug must not be lost
                log.warning('discarding a failed paint: %s', e)

    def close(self):
        if not self.closed:
            self.closed = True               # before draining: no re-entry
            self._drain()
            if self._pool is not None:
                self._pool.shutdown(wait=True)
            self.vis.destroy_window()
            if self.panels:
                try:
                    self._cv.destroyWindow(self.panel_window)
                except Exception:        # already gone, or no GUI backend
                    pass

    # ---- the camera panels ----
    def _draw_panels(self, depth, rgb, name, fps, status=None):
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
        # The 3D window is open3d's LEGACY Visualizer: no material API and
        # no text. The panels are cv2, so the accumulate readout lives here
        # -- which is also why --no_panels turns the readout off with them.
        for line, colour in (status or []):
            bar = np.zeros((20, w, 3), np.uint8)
            cv2.putText(bar, line, (6, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.42,
                        colour, 1, cv2.LINE_AA)
            out.append(bar)
        cv2.imshow(self.panel_window, np.vstack(out))
        cv2.waitKey(1)                   # pumps the window; it draws nothing

    def report(self):
        if not self.frames:
            return 'display: no frames drawn'
        bits = ['%d frames' % self.frames,
                'colour %.0f ms' % (self.colour_ms / self.frames),
                'mesh+draw %.0f ms' % (self.draw_ms / self.frames)]
        if self._overlapping():
            hidden = max(0.0, self.colour_ms - self.wait_ms) / self.frames
            bits.append('%.0f of it hidden behind inference' % hidden)
        bits.append('mode %s' % self.colour)
        return 'display: ' + ' | '.join(bits)

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
    def _build(self, idx, colours):
        """The solid cubes, scaled about the grid centre by `zoom`.

        Scaling the content rather than moving the camera, because the camera
        cannot be moved from Python in this open3d -- see the module
        docstring. The centre is the GRID's, not this frame's, so the scene
        does not shift when the prediction changes.
        """
        mesh, _, _ = self.ds.build_cube_mesh(self.ds.to_view_index(idx),
                                             colours)
        if self.zoom != 1.0:
            v = np.asarray(mesh.vertices)
            mesh.vertices = self.o3d.utility.Vector3dVector(
                (v - self.centre) * self.zoom + self.centre)
        return mesh

    def _swap(self, mesh):
        if self.geom is None:
            # Fit the camera to the whole grid, then drop the box that did it.
            # This is the only moment open3d will aim the camera, and fitting
            # a constant box rather than the first frame's voxels is what
            # makes --zoom mean the same thing from run to run.
            # centred on the pivot, not at the origin: in --accumulate the
            # proxy is deliberately SMALLER than the map, so that the
            # opening view frames a room rather than the whole 16 m box the
            # camera is free to wander inside
            box = self.o3d.geometry.TriangleMesh.create_box(
                *[float(g) for g in self.proxy]).translate(
                    self.centre - np.asarray(self.proxy, np.float64) / 2.0)
            self.vis.add_geometry(box)
            self.geom = mesh
            self.vis.add_geometry(self.geom, reset_bounding_box=False)
            self.vis.remove_geometry(box, reset_bounding_box=False)
            return
        # one geometry, its arrays replaced: the vertex count changes every
        # frame, and add/remove would refit the camera and throw away the
        # viewpoint the user set
        self.geom.vertices = mesh.vertices
        self.geom.triangles = mesh.triangles
        self.geom.vertex_colors = mesh.vertex_colors
        self.vis.update_geometry(self.geom)

    def _flip_mesh(self, _vis):
        """Flip between the room and the sensor's reconstruction.

        Same window, same viewpoint, so the two are read against each
        other rather than side by side.
        """
        if self.recon is None:
            log.info('no reconstruction yet -- it is built when you freeze '
                     '(space) and again when the run ends')
            return False
        self.show_mesh = not self.show_mesh
        log.info('showing %s', 'the SENSOR\'s reconstruction (the holes in '
                 'it are what the model is filling in)' if self.show_mesh
                 else 'the room')
        self._repaint()
        return False

    def _draw(self):
        alive = self.vis.poll_events()
        self.vis.update_renderer()
        if not alive:
            self.close()
        return alive

    def _set_zoom(self, value):
        self.zoom = float(np.clip(value, 0.2, 6.0))
        log.info('display zoom=%.2f', self.zoom)
        self._repaint()
        return False

    def _zoom_in(self, _vis):
        return self._set_zoom(self.zoom * 1.15)

    def _zoom_out(self, _vis):
        return self._set_zoom(self.zoom / 1.15)

    def _zoom_reset(self, _vis):
        return self._set_zoom(2.0)

    def _repaint(self):
        """Redraw with the current settings, rather than making the user
        wait for the next frame."""
        self._drain()            # never build geometry alongside the painter
        if self._wpivot is not None:
            # --accumulate: the geometry is the room, not a frame, and it
            # is rebuilt from the map on the next update. Without this the
            # zoom and colour keys would log a change and do nothing.
            self._wforce = True
            return
        if self._last is None:
            return
        idx, cls, loader, rgb = self._last
        self._swap(self._build(idx, self._paint(idx, cls, loader, rgb)))
        self._draw()

    def _cycle(self, _vis):
        if self._wpivot is not None:
            # in a room the switch is which QUESTION the colours answer:
            # what does it look like, did the model get it right, or what
            # does the model think each thing is
            m = self.WORLD_MODES
            self.world_colour = m[(m.index(self.world_colour) + 1) % len(m)]
            log.info('room colour=%s', self.world_colour)
        else:
            self.colour = MODES[(MODES.index(self.colour) + 1) % len(MODES)]
            log.info('display colour=%s', self.colour)
        self._repaint()
        return False

    def _freeze(self, _vis):
        """Stop taking frames, and hold the room exactly as it is.

        The loop driving this window stops pulling from the source while
        it holds -- a camera's frames are dropped on the floor, and a
        replay resumes on the frame it stopped at rather than skipping
        ahead. The window stays live throughout, so the state at that
        moment can be orbited and looked at. Press again to carry on.
        """
        self.frozen = not self.frozen
        log.info('%s', 'FROZEN -- no more frames until you press space '
                 'again' if self.frozen else 'running again')
        self._wforce = True
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
