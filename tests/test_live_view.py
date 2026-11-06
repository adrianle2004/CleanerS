"""Unit tests for the one piece of arithmetic the room's two layers share.

inference/live_view.py draws the accumulating room as cubes placed by GRID
INDEX, and the sensor's own reconstruction as a mesh in WORLD METRES. The
window exists so the two can be read against each other at one viewpoint,
which is worth nothing if they do not land in the same place -- and half a
voxel is 4 cm, enough to float a mesh off the floor it was measured from.

view_points is pure numpy and imports no open3d, so that placement is
testable without a window, a GPU or a bag. The rest of live_view needs all
three.
"""

import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from inference.live_view import (GRID, SHRINK_STEPS, VOX_UNIT_LOW,
                                 LiveView, view_points)
from inference.world_map import FREE, OCCUPIED, UNKNOWN, WorldMap


class TestViewPoints(unittest.TestCase):
    def room(self):
        return WorldMap(origin=[-1.0, -2.0, -0.13], shape=(20, 20, 10))

    def to_view_index(self, w, idx):
        """What _build_world does with a voxel index, repeated here so a
        change to either side has to be made deliberately."""
        idx = np.asarray(idx)
        pivot = w.shape[1] - 1
        return np.stack([idx[:, 0], idx[:, 2], pivot - idx[:, 1]], axis=1)

    def test_a_voxel_centre_lands_exactly_on_its_cube(self):
        """The property the comparison rests on: a point measured at the
        middle of a voxel draws at the index that voxel's cube is at."""
        w = self.room()
        idx = np.array([[0, 0, 0], [3, 7, 2], [19, 19, 9], [11, 4, 6]])
        got = view_points(w.centres(idx), w.origin, w.vox, w.shape[1] - 1)
        np.testing.assert_allclose(got, self.to_view_index(w, idx),
                                   atol=1e-9)

    def test_the_half_voxel_is_not_optional(self):
        """Without it every vertex is half a voxel out -- 4 cm, which is
        a mesh floating off its own floor."""
        w = self.room()
        corner = w.origin + np.array([0.0, 0.0, 0.0])
        got = view_points([corner], w.origin, w.vox, w.shape[1] - 1)[0]
        np.testing.assert_allclose(got, [-0.5, -0.5, w.shape[1] - 1 + 0.5])

    def test_the_depth_axis_is_reflected_and_the_others_are_not(self):
        w = self.room()
        a = view_points([w.centres(np.array([[5, 5, 5]]))[0]], w.origin,
                        w.vox, w.shape[1] - 1)[0]
        b = view_points([w.centres(np.array([[5, 6, 5]]))[0]], w.origin,
                        w.vox, w.shape[1] - 1)[0]
        self.assertEqual(a[0], b[0])
        self.assertEqual(a[1], b[1])
        self.assertAlmostEqual(b[2] - a[2], -1.0)   # world +Y draws nearer

    def test_it_preserves_orientation(self):
        """A permutation AND a reflection, so a mesh's winding survives
        it and does not have to be flipped -- _build_recon relies on
        this to keep its normals pointing out."""
        w = self.room()
        o = view_points([[0.0, 0.0, 0.0]], w.origin, w.vox, 1.0)[0]
        axes = np.array([
            view_points([[w.vox, 0.0, 0.0]], w.origin, w.vox, 1.0)[0] - o,
            view_points([[0.0, w.vox, 0.0]], w.origin, w.vox, 1.0)[0] - o,
            view_points([[0.0, 0.0, w.vox]], w.origin, w.vox, 1.0)[0] - o])
        self.assertGreater(np.linalg.det(axes), 0.0)


class _Mesh(object):
    """Enough of an open3d mesh for the colour code, which reads the two
    vertex arrays and nothing else."""

    def __init__(self, verts, cols):
        self.vertices = np.asarray(verts, np.float64)
        self.vertex_colors = np.asarray(cols, np.float64)


class TestSurfaceColourSource(unittest.TestCase):
    """Where a surface vertex's colour comes from.

    The room only photographs what the camera reached. Everything else
    falls back to display_solid's dimmed class colour -- and in room07
    that is the whole FLOOR, which no frame ever coloured, so the middle
    layer laid a dimmed palette green over it. A floor does not look
    like that, and the layer's whole claim is that the completion looks
    like a surface.

    So a voxel NO photograph coloured asks the sensor's own mesh, at any
    range: the nearest measured colour is a photograph of this room, and
    a palette tint never is. It is the same borrow fill_hidden already
    makes across a class, from a better source.

    A voxel that has a colour of its own keeps the short reach: out
    there its own colour beats a wall's from across the room.
    """

    def room(self):
        return WorldMap(origin=[0.0, 0.0, 0.0], shape=(10, 10, 10))

    def view(self, mode='photo'):
        v = LiveView.__new__(LiveView)
        v.sensor_colour = True
        v.world_colour = mode
        # one red reference, on the centre of voxel (0, 0, 0)
        v.recon = _Mesh([[0.04, 0.04, 0.04]], [[1.0, 0.0, 0.0]])
        return v

    def test_a_vertex_the_mesh_can_reach_takes_its_colour(self):
        f = self.view()._sensor_recolour(self.room())
        out = f(np.zeros((1, 3)), np.zeros((1, 3)), np.array([False]))
        np.testing.assert_allclose(out[0], [255.0, 0.0, 0.0])

    def test_a_photographed_vertex_out_of_reach_keeps_its_colour(self):
        f = self.view()._sensor_recolour(self.room())
        base = np.array([[9.0, 9.0, 9.0]])
        out = f(np.array([[9.0, 9.0, 9.0]]), base, np.array([False]))
        np.testing.assert_allclose(out[0], base[0])

    def test_a_vertex_no_photograph_coloured_reaches_as_far_as_it_must(self):
        f = self.view()._sensor_recolour(self.room())
        out = f(np.array([[9.0, 9.0, 9.0]]), np.array([[9.0, 9.0, 9.0]]),
                np.array([True]))
        np.testing.assert_allclose(out[0], [255.0, 0.0, 0.0])

    def test_without_the_mask_nothing_reaches_further_than_before(self):
        """The frame's surface passes no mask, and must keep the cap."""
        f = self.view()._sensor_recolour(self.room())
        base = np.array([[9.0, 9.0, 9.0]])
        out = f(np.array([[9.0, 9.0, 9.0]]), base)
        np.testing.assert_allclose(out[0], base[0])

    def test_class_mode_keeps_its_palette(self):
        """There the colour IS the answer -- which class the model
        called each voxel -- and a photograph painted over it answers
        nothing."""
        self.assertIsNone(self.view('class')._sensor_recolour(self.room()))

    def test_verdict_mode_keeps_its_palette(self):
        self.assertIsNone(self.view('verdict')._sensor_recolour(self.room()))


class TestTheMaskReachesTheRecolour(unittest.TestCase):
    """_surface_mesh holds one colour per VOXEL and hands out one per
    VERTEX, so the mask has to make the same trip or it is read against
    the wrong array."""

    def test_the_recolour_is_told_which_vertices_were_photographed(self):
        v = LiveView.__new__(LiveView)
        v.surface_smooth = 0
        v._lit = lambda pts, tris, cols: cols
        occ = np.zeros((5, 5, 5), bool)
        occ[1:4, 1:4, 1:4] = True
        rows = np.argwhere(occ)
        unlit = np.all(rows == [1, 1, 1], axis=1)
        got = {}

        def recolour(verts, base, vunlit=None):
            got['mask'] = np.asarray(vunlit)
            out = np.array(base, copy=True)
            out[got['mask']] = [255.0, 0.0, 0.0]
            return out

        cols = v._surface_mesh(occ, rows, np.zeros((len(rows), 3), np.int32),
                               lambda p: p, recolour=recolour, unlit=unlit)
        self.assertEqual(len(got['mask']), len(cols))
        # the one unphotographed voxel owns some vertices, and only its
        self.assertGreater(got['mask'].sum(), 0)
        self.assertLess(got['mask'].sum(), len(cols))
        np.testing.assert_allclose(cols[got['mask']][0], [1.0, 0.0, 0.0])


class TestWhatWasPhotographed(unittest.TestCase):
    """_world_colours knows which voxels a photograph reached -- it is
    the one place that runs fill_hidden -- so it is the one place that
    can say, and the surface needs to be told."""

    def room(self):
        w = WorldMap(origin=[0.0, 0.0, 0.0], shape=(4, 4, 4))
        w.votes[:, 3] = 10            # one class everywhere, so predicted()
        w.n_seen[:] = 10              # calls every voxel occupied
        return w

    def view(self, mode='photo'):
        from inference import display_solid
        v = LiveView.__new__(LiveView)
        v.ds = display_solid
        v.world_colour = mode
        return v

    def test_a_voxel_the_camera_coloured_is_reported_as_photographed(self):
        w = self.room()
        w.colour_n[0] = 4
        w.colour_ema[0] = (200.0, 100.0, 50.0)
        i = np.array([[0, 0, 0]])
        _, lit = self.view()._world_colours(w, None, i, i.astype(float))
        self.assertTrue(lit[0])

    def test_a_voxel_that_borrowed_a_siblings_colour_counts_too(self):
        """It is wearing a photograph of this room, taken a voxel
        away."""
        w = self.room()
        w.colour_n[0] = 4
        w.colour_ema[0] = (200.0, 100.0, 50.0)
        i = np.array([[0, 0, 0], [0, 0, 1]])
        _, lit = self.view()._world_colours(w, None, i, i.astype(float))
        self.assertTrue(lit.all())

    def test_a_voxel_no_photograph_ever_reached_is_not(self):
        w = self.room()
        i = np.array([[0, 0, 0], [0, 0, 1]])
        _, lit = self.view()._world_colours(w, None, i, i.astype(float))
        self.assertFalse(lit.any())

    def test_class_mode_photographs_nothing(self):
        w = self.room()
        w.colour_n[0] = 4
        i = np.array([[0, 0, 0]])
        _, lit = self.view('class')._world_colours(w, None, i,
                                                   i.astype(float))
        self.assertFalse(lit.any())



class TestFrameGeometry(unittest.TestCase):
    """Which of the THREE layers the single-frame window draws.

    `m` used to need --accumulate, because the mesh was the tracker's
    volume and there was no tracker without the flag. A plain --display
    run now builds the frame's OWN surface instead, so the key has to
    choose between that and the prediction -- and has to fall back to the
    prediction when the surface for this frame does not exist yet, which
    is the state the window is in for one frame after the key is pressed.

    The middle layer is the completion MESHED: the same voxels the first
    layer draws as cubes, as one surface. It needs no sensor mesh and no
    tracker, so it is the one layer that is always available.

    LiveView.__init__ imports open3d and opens a window; the chooser
    touches neither, so it is built without them.
    """

    def view(self, layer):
        v = LiveView.__new__(LiveView)
        v.layer = layer
        v._build = lambda idx, colours: 'the prediction'
        v._build_frame_surface = \
            lambda idx, colours, tsdf, lit, loader, recon: \
            'the completion, meshed'
        v._build_frame_recon = lambda loader, recon: recon
        return v

    def test_the_prediction_is_what_it_draws_by_default(self):
        v = self.view('voxels')
        self.assertEqual(v._frame_geometry(None, None, None, 'a mesh'),
                         'the prediction')

    def test_the_middle_layer_meshes_the_completion(self):
        v = self.view('surface')
        self.assertEqual(v._frame_geometry(None, None, None, 'a mesh'),
                         'the completion, meshed')

    def test_the_middle_layer_needs_no_sensor_mesh(self):
        """It is built from the voxels, so a run with no reconstruction
        at all still has something to show between the two."""
        v = self.view('surface')
        self.assertEqual(v._frame_geometry(None, None, None, None),
                         'the completion, meshed')

    def test_a_completion_too_thin_to_mesh_falls_back_to_cubes(self):
        v = self.view('surface')
        v._build_frame_surface = \
            lambda idx, colours, tsdf, lit, loader, recon: None
        self.assertEqual(v._frame_geometry(None, None, None, None),
                         'the prediction')

    def test_m_draws_the_frames_own_surface(self):
        v = self.view('sensor')
        self.assertEqual(v._frame_geometry(None, None, None, 'a mesh'),
                         'a mesh')

    def test_m_before_a_surface_exists_still_draws_something(self):
        """The key may be pressed before run_live has built one -- it is
        built on the NEXT frame. Drawing nothing would read as a crash."""
        v = self.view('sensor')
        self.assertEqual(v._frame_geometry(None, None, None, None),
                         'the prediction')

    def test_the_surface_drawn_is_the_one_that_came_with_the_frame(self):
        """Not whatever self.recon happens to hold. The window is a
        frame behind by design, so reading the live attribute at draw
        time pairs this frame's cubes with the NEXT frame's surface."""
        v = self.view('sensor')
        v.recon = 'the next frame already built this one'
        self.assertEqual(v._frame_geometry(None, None, None, 'this frame'),
                         'this frame')


class TestLayerCycle(unittest.TestCase):
    """`m` advances one layer, and never dead-ends.

    Three layers, and only two of them are always available: the sensor's
    own surface needs a reconstruction, which a plain --display run can
    build per frame and an --accumulate run only has once the sweep is
    frozen or finished. When it cannot be had, the key must carry on
    round rather than refusing -- refusing is what sent someone to press
    space again.
    """

    def view(self, layer='voxels', make_surface=None, recon=None, live=True):
        v = LiveView.__new__(LiveView)
        v.layer = layer
        v.recon, v.live_recon, v.can_reconstruct = recon, live, True
        v._last = ('idx', 'cls', 'loader', 'rgb')
        v.make_surface = make_surface
        v.repainted = []
        v._drain = lambda: None
        v._repaint = lambda: v.repainted.append(1)
        return v

    def test_it_goes_voxels_then_surface(self):
        v = self.view('voxels')
        v._cycle_layer(None)
        self.assertEqual(v.layer, 'surface')

    def test_the_middle_layer_needs_nothing_to_be_available(self):
        """No recon, no make_surface, no tracker -- still reachable."""
        v = self.view('voxels', make_surface=None, live=False)
        v._cycle_layer(None)
        self.assertEqual(v.layer, 'surface')
        self.assertEqual(len(v.repainted), 1)

    def test_surface_then_sensor_when_one_can_be_built(self):
        v = self.view('surface', make_surface=lambda loader, rgb: 'a surface')
        v._cycle_layer(None)
        self.assertEqual(v.layer, 'sensor')
        self.assertEqual(v.recon, 'a surface')

    def test_sensor_wraps_back_to_voxels(self):
        v = self.view('sensor', recon='the sweep')
        v._cycle_layer(None)
        self.assertEqual(v.layer, 'voxels')

    def test_an_unavailable_sensor_layer_is_skipped_not_stuck(self):
        """--accumulate before the first freeze: there is no mesh and no
        make_surface, so the cycle goes straight back to the cubes."""
        v = self.view('surface', make_surface=None, live=False)
        v._cycle_layer(None)
        self.assertEqual(v.layer, 'voxels')

    def test_a_frame_that_measured_nothing_skips_the_sensor_layer(self):
        v = self.view('surface', make_surface=lambda loader, rgb: None)
        v._cycle_layer(None)
        self.assertEqual(v.layer, 'voxels')

    def test_a_failure_to_build_never_escapes(self):
        def boom(loader, rgb):
            raise RuntimeError('no room on the card')
        v = self.view('surface', make_surface=boom)
        with self.assertLogs('live_view', 'WARNING') as caught:
            v._cycle_layer(None)        # must not raise
        self.assertIn('no room on the card', caught.output[0])
        self.assertEqual(v.layer, 'voxels')


class TestFrameSurfacePlacement(unittest.TestCase):
    """The frame's surface arrives in the same metres as its voxel grid,
    and has to land on the same cubes."""

    def test_a_voxel_centre_lands_on_its_own_cube(self):
        origin = np.array([-2.4, 0.0, -0.05])
        pivot = GRID[2] - 1
        for idx in ((0, 0, 0), (13, 7, 41), (59, 35, 59)):
            # world metres of that voxel's CENTRE: the grid is indexed
            # (X, Z, Y) -- see README's coordinate conventions
            centre = origin + (np.array([idx[0], idx[2], idx[1]]) + 0.5) \
                * VOX_UNIT_LOW
            got = view_points(centre[None, :], origin, VOX_UNIT_LOW, pivot)[0]
            want = np.array([idx[0], idx[1], pivot - idx[2]], np.float64)
            np.testing.assert_allclose(got, want, atol=1e-9)


class TestMBuildsOnDemand(unittest.TestCase):
    """Pressing m has to produce a surface for the frame ON SCREEN.

    The first cut built one only while m was already showing, which
    sounds equivalent and is not: the moment anyone presses the key is
    while the window is FROZEN, and then no frame is being processed, so
    nothing ever built one and the key toggled a flag that changed
    nothing. The window therefore asks for one itself.
    """

    def view(self, make_surface, recon=None, live=True):
        v = LiveView.__new__(LiveView)
        # one step short of the sensor layer, so a single press enters it
        v.layer, v.recon, v.live_recon = 'surface', recon, live
        v.can_reconstruct = True
        v._last = ('idx', 'cls', 'loader', 'rgb')
        v.make_surface = make_surface
        v.repainted = []
        v._drain = lambda: None
        v._repaint = lambda: v.repainted.append(1)
        return v

    def test_pressing_m_with_no_surface_yet_builds_one(self):
        v = self.view(lambda loader, rgb: 'a surface')
        v._cycle_layer(None)
        self.assertEqual(v.layer, 'sensor')
        self.assertEqual(v.recon, 'a surface')
        self.assertEqual(len(v.repainted), 1)

    def test_it_is_built_for_the_frame_on_screen(self):
        seen = []
        v = self.view(lambda loader, rgb: seen.append((loader, rgb)) or 'm')
        v._cycle_layer(None)
        self.assertEqual(seen, [('loader', 'rgb')])

    def test_m_always_rebuilds_rather_than_reusing(self):
        """Reported: freeze, m on, m off, play a hundred frames to
        somewhere else, freeze, m on -- and the surface was still the
        one built before the camera moved. A per-frame surface belongs
        to one frame, so a held one is stale, not spare."""
        calls = []
        v = self.view(lambda loader, rgb: calls.append(1) or 'rebuilt',
                      recon='built a hundred frames ago')
        v._cycle_layer(None)
        self.assertEqual(len(calls), 1)
        self.assertEqual(v.recon, 'rebuilt')

    def test_the_sweeps_own_mesh_is_never_rebuilt_by_a_keypress(self):
        """--accumulate: recon is the whole sweep's mesh and there is no
        make_surface, so entering the layer must use what is there."""
        v = self.view(None, recon='the sweep', live=False)
        v._cycle_layer(None)
        self.assertEqual(v.layer, 'sensor')
        self.assertEqual(v.recon, 'the sweep')


class TestHeldKeys(unittest.TestCase):
    """A held key must not keep firing a toggle.

    Reported as the window "jumping between 2 modes": X11 repeats a held
    key and open3d delivers every repeat, so holding `m` flipped the
    display dozens of times a second, each flip a full repaint -- which
    is where the lag came from too. A quick press was also long enough
    to catch one repeat and toggle twice, so the mode appeared not to
    change at all.
    """

    RELEASE, PRESS, REPEAT = 0, 1, 2

    def test_only_the_press_fires(self):
        fired = []
        cb = LiveView._press_only(lambda vis: fired.append(vis) or False)
        cb('vis', self.PRESS, 0)
        self.assertEqual(fired, ['vis'])

    def test_holding_the_key_fires_nothing_more(self):
        fired = []
        cb = LiveView._press_only(lambda vis: fired.append(1) or False)
        cb(None, self.PRESS, 0)
        for _ in range(40):              # ~a second of X11 auto-repeat
            cb(None, self.REPEAT, 0)
        cb(None, self.RELEASE, 0)
        self.assertEqual(len(fired), 1)

    def test_pressing_twice_fires_twice(self):
        """Suppressing the repeat must not suppress a real second press."""
        fired = []
        cb = LiveView._press_only(lambda vis: fired.append(1) or False)
        cb(None, self.PRESS, 0)
        cb(None, self.RELEASE, 0)
        cb(None, self.PRESS, 0)
        self.assertEqual(len(fired), 2)

    def test_it_passes_the_callbacks_answer_through(self):
        """open3d reads the return value as 'redraw needed'."""
        cb = LiveView._press_only(lambda vis: True)
        self.assertTrue(cb(None, self.PRESS, 0))
        self.assertFalse(cb(None, self.REPEAT, 0))

class TestOverlapReport(unittest.TestCase):
    """The run report may only claim the colour was hidden behind
    inference if a frame was actually painted on the pool.

    `_overlapping()` answers a question about the CONFIGURATION -- is
    overlap on, and is the mode one worth overlapping. With
    --accumulate nothing is ever submitted: update_world builds the
    room on the loop and update() is never called, so wait_ms stays 0
    and the line reported the whole room build as hidden. It is the
    line anyone reads to find out what the room costs, and it said
    nothing.
    """

    def view(self, overlapped):
        v = LiveView.__new__(LiveView)
        v.frames, v.colour_ms, v.draw_ms, v.wait_ms = 10, 500.0, 100.0, 0.0
        v.overlap, v.colour, v.overlapped = True, 'all', overlapped
        return v

    def test_a_run_that_overlapped_nothing_claims_nothing(self):
        self.assertNotIn('hidden', self.view(0).report())

    def test_a_run_that_did_says_how_much(self):
        self.assertIn('50 of it hidden behind inference',
                      self.view(10).report())

    def test_what_it_waited_for_was_not_hidden(self):
        v = self.view(10)
        v.wait_ms = 300.0
        self.assertIn('20 of it hidden', v.report())


class _Loader(object):
    """Enough of a FrameLoader for the colour code: it reads where the
    frame's grid starts and nothing else."""

    def __init__(self, vox_origin):
        self.vox_origin = np.asarray(vox_origin, np.float64)


class TestFrameSurfaceColour(unittest.TestCase):
    """The single-frame surface takes its colour from the frame's own
    measured surface, exactly as the room takes its from the sweep's.

    The arithmetic is the part that can go quietly wrong. The room's
    grid is indexed in world order and the FRAME's is not -- index axis
    1 is UP and the world's up axis is Z -- so the two axes have to be
    swapped on the way out or every lookup lands somewhere else in the
    room and the surface wears the colour of whatever was there.
    """

    def frame(self):
        """A grid origin, a voxel index, and that voxel's centre in the
        metres the frame's mesh is built in."""
        origin = np.array([-2.4, 0.0, -0.05])
        idx = np.array([13, 7, 41])
        centre = origin + (idx[[0, 2, 1]] + 0.5) * VOX_UNIT_LOW
        return _Loader(origin), idx, centre

    def view(self, mode='all'):
        v = LiveView.__new__(LiveView)
        v.sensor_colour = True
        v.colour = mode
        return v

    def test_it_asks_about_the_metres_that_voxel_is_really_in(self):
        loader, idx, centre = self.frame()
        f = self.view()._frame_recolour(loader,
                                        _Mesh([centre], [[1.0, 0.0, 0.0]]))
        out = f(np.array([idx], np.float64), np.zeros((1, 3)),
                np.array([False]))
        np.testing.assert_allclose(out[0], [255.0, 0.0, 0.0])

    def test_the_axis_swap_is_not_optional(self):
        """Without it the lookup lands at the index read as metres in
        the wrong order, which is a different place in the room."""
        loader, idx, centre = self.frame()
        wrong = loader.vox_origin + (idx + 0.5) * VOX_UNIT_LOW
        self.assertGreater(np.linalg.norm(wrong - centre), 2 * VOX_UNIT_LOW)
        f = self.view()._frame_recolour(loader,
                                        _Mesh([wrong], [[1.0, 0.0, 0.0]]))
        base = np.array([[7.0, 8.0, 9.0]])
        out = f(np.array([idx], np.float64), base, np.array([False]))
        np.testing.assert_allclose(out[0], base[0])

    def test_a_voxel_no_photograph_coloured_reaches_further(self):
        loader, idx, centre = self.frame()
        far = centre + np.array([1.0, 0.0, 0.0])
        f = self.view()._frame_recolour(loader,
                                        _Mesh([far], [[1.0, 0.0, 0.0]]))
        out = f(np.array([idx], np.float64), np.zeros((1, 3)),
                np.array([True]))
        np.testing.assert_allclose(out[0], [255.0, 0.0, 0.0])

    def test_class_mode_keeps_its_palette(self):
        loader, _, centre = self.frame()
        self.assertIsNone(self.view('class')._frame_recolour(
            loader, _Mesh([centre], [[1.0, 0.0, 0.0]])))

    def test_no_surface_for_this_frame_is_not_an_error(self):
        """`m` can be pressed before one has been built, and the layer
        falls back to the model's own colour rather than nothing."""
        loader, _, _ = self.frame()
        self.assertIsNone(self.view()._frame_recolour(loader, None))


class TestFrameGeometryPassesTheColourSource(unittest.TestCase):
    """The mask and the frame's own surface have to reach the builder,
    or the layer silently keeps the colour it had."""

    def test_the_surface_is_given_the_mask_the_loader_and_the_mesh(self):
        got = {}
        v = LiveView.__new__(LiveView)
        v.layer = 'surface'
        v._build_frame_surface = lambda *a: got.setdefault('args', a)
        v._frame_geometry('idx', 'colours', 'loader', 'mesh', 'tsdf', 'lit')
        self.assertEqual(got['args'],
                         ('idx', 'colours', 'tsdf', 'lit', 'loader', 'mesh'))


class TestEnteringTheSurfaceLayer(unittest.TestCase):
    """`m` into the surface layer now wants a surface for this frame as
    well -- not to draw, but to take the colour off.

    It must not behave like the sensor layer about it. That one is
    SKIPPED when no surface can be built, because without one it has
    nothing to show. This one is built from the model's own voxels and
    always has something, so a failure costs the colour and never the
    layer.
    """

    def view(self, make_surface):
        v = LiveView.__new__(LiveView)
        v.layer, v.recon, v.live_recon = 'voxels', None, True
        v.can_reconstruct = True
        v._last = ('idx', 'cls', 'loader', 'rgb')
        v.make_surface = make_surface
        v._drain = lambda: None
        v._repaint = lambda: None
        return v

    def test_it_builds_one_to_take_the_colour_from(self):
        v = self.view(lambda loader, rgb: 'this frame')
        v._cycle_layer(None)
        self.assertEqual(v.layer, 'surface')
        self.assertEqual(v.recon, 'this frame')

    def test_a_frame_that_measured_nothing_still_shows_the_layer(self):
        v = self.view(lambda loader, rgb: None)
        v._cycle_layer(None)
        self.assertEqual(v.layer, 'surface')
        self.assertIsNone(v.recon)

    def test_a_failure_to_build_never_escapes(self):
        def boom(loader, rgb):
            raise RuntimeError('no room on the card')

        v = self.view(boom)
        v._cycle_layer(None)
        self.assertEqual(v.layer, 'surface')

    def test_accumulate_keeps_the_sweeps_mesh(self):
        """There recon is the whole sweep and a keypress must not
        rebuild it -- make_surface is None, which says so."""
        v = self.view(None)
        v.recon, v.live_recon = 'the sweep', False
        v._cycle_layer(None)
        self.assertEqual(v.layer, 'surface')
        self.assertEqual(v.recon, 'the sweep')


class TestSurfaceMeshOrder(unittest.TestCase):
    """The colour is sampled where the surface ENDS UP.

    The single-frame layer shrinks the completion onto the measurement
    it is standing in front of, and then asks the sensor's mesh what
    colour that place is. Asking first and moving afterwards samples a
    position the surface does not occupy -- about a voxel proud of it,
    which at a wall is the wrong side of the corner.

    Splitting after the shrink is also four times less of it to do.
    """

    def test_the_recolour_sees_the_shrunk_vertices(self):
        v = LiveView.__new__(LiveView)
        v.surface_smooth = 0
        v._lit = lambda pts, tris, cols: pts
        occ = np.zeros((5, 5, 5), bool)
        occ[1:4, 1:4, 1:4] = True
        rows = np.argwhere(occ)
        seen = {}

        def recolour(verts, base, unlit=None):
            seen['z'] = np.asarray(verts)[:, 2].min()
            return base

        def pull(verts):
            out = np.array(verts, copy=True)
            out[:, 2] += 10.0
            return out

        v._surface_mesh(occ, rows, np.zeros((len(rows), 3), np.int32),
                        lambda p: p, pull=pull, recolour=recolour)
        self.assertGreater(seen['z'], 5.0)


class TestTheShrinkKeys(unittest.TestCase):
    """`[` and `]` move the shrink along SHRINK_STEPS.

    The shrink is the one correction this layer makes to the model's
    geometry, and the two ends of the ladder are its before and after.
    On a key they can be read against each other; as a flag they are two
    runs nobody holds in their head.
    """

    def view(self, pull=1.5, layer='surface', room=False):
        v = LiveView.__new__(LiveView)
        v.surface_pull = pull
        v.layer = layer
        v._wpivot = 7 if room else None
        v.repainted = []
        v._repaint = lambda: v.repainted.append(1)
        return v

    def test_it_steps_down_the_ladder(self):
        v = self.view(1.5)
        v._shrink_less(None)
        self.assertEqual(v.surface_pull, 1.0)

    def test_it_steps_up_the_ladder(self):
        v = self.view(1.5)
        v._shrink_more(None)
        self.assertEqual(v.surface_pull, 2.0)

    def test_down_all_the_way_turns_it_off(self):
        """0 is a value on the ladder, not a floor it stops short of:
        the shrink off is the model's own geometry, which is half of
        what the key is for."""
        v = self.view(SHRINK_STEPS[-1])
        for _ in range(len(SHRINK_STEPS) * 2):
            v._shrink_less(None)
        self.assertEqual(v.surface_pull, 0.0)

    def test_it_stops_at_the_top(self):
        """3.0 is TSDF_VOXELS -- the whole distance the field can
        express. Past it the step is clipped anyway, so there is nothing
        above to offer."""
        v = self.view(0.0)
        for _ in range(len(SHRINK_STEPS) * 2):
            v._shrink_more(None)
        self.assertEqual(v.surface_pull, SHRINK_STEPS[-1])

    def test_an_off_ladder_value_snaps_onto_it(self):
        """--surface_pull 0.8 is legal and the key must not be stuck
        beside the ladder, nor drift to 0.8 +/- a step."""
        v = self.view(0.8)
        v._shrink_more(None)
        self.assertEqual(v.surface_pull, 1.0)
        v = self.view(0.8)
        v._shrink_less(None)
        self.assertEqual(v.surface_pull, 0.5)

    def test_the_surface_layer_is_redrawn(self):
        v = self.view(1.5, layer='surface')
        v._shrink_more(None)
        self.assertEqual(len(v.repainted), 1)

    def test_the_other_layers_change_the_value_without_redrawing(self):
        """Only the surface is shrunk. The value still has to move, so
        it can be set before m arrives there."""
        v = self.view(1.5, layer='voxels')
        v._shrink_more(None)
        self.assertEqual(v.surface_pull, 2.0)
        self.assertEqual(v.repainted, [])

    def test_a_press_at_the_end_redraws_nothing(self):
        """Nothing moved, so there is nothing to re-mesh -- and a
        re-mesh is ~150 ms."""
        v = self.view(SHRINK_STEPS[-1])
        v._shrink_more(None)
        self.assertEqual(v.surface_pull, SHRINK_STEPS[-1])
        self.assertEqual(v.repainted, [])

    def test_the_room_steps_too(self):
        """One value, both modes. The room shrinks onto the sweep's own
        hit/pass vote where a frame shrinks onto its TSDF -- different
        evidence, same step, so the key means one thing."""
        v = self.view(1.5, room=True)
        v._shrink_more(None)
        self.assertEqual(v.surface_pull, 2.0)
        v._shrink_less(None)
        self.assertEqual(v.surface_pull, 1.5)
        self.assertEqual(len(v.repainted), 2)

    def test_every_rung_is_reachable_in_order(self):
        v = self.view(SHRINK_STEPS[0])
        seen = [v.surface_pull]
        for _ in range(len(SHRINK_STEPS) - 1):
            v._shrink_more(None)
            seen.append(v.surface_pull)
        self.assertEqual(tuple(seen), SHRINK_STEPS)

    def test_the_handlers_answer_false(self):
        """open3d reads the return value as 'redraw needed', and the
        repaint is done here rather than by the renderer."""
        v = self.view(1.5)
        self.assertFalse(v._shrink_more(None))
        self.assertFalse(v._shrink_less(None))

    def test_it_says_the_distance_in_millimetres(self):
        """A voxel is 8 cm and nobody reads a room in voxels."""
        said = LiveView._shrink_said(1.5)
        self.assertIn('120 mm', said)
        self.assertIn('OFF', LiveView._shrink_said(0.0))

    def test_the_panel_bar_shows_it_on_the_surface_layer_only(self):
        v = self.view(1.5, layer='surface')
        self.assertEqual(len(v._panel_status(None)), 1)
        self.assertIn('120 mm', v._panel_status(None)[0][0])
        self.assertEqual(len(v._panel_status(None)[0][1]), 3)
        v = self.view(1.5, layer='voxels')
        self.assertEqual(v._panel_status(None), [])
        v = self.view(1.5, layer='surface', room=True)
        self.assertEqual(len(v._panel_status(None)), 1)

    def test_the_panel_bar_keeps_the_rooms_own_readout(self):
        """--accumulate's status lines go through the same function, and
        the shrink is appended rather than replacing them."""
        v = self.view(1.5, layer='surface', room=True)
        rows = [('a', (1, 2, 3)), ('b', (4, 5, 6))]
        got = v._panel_status(rows)
        self.assertEqual(got[:2], rows)
        self.assertIn('shrink', got[2][0])

    def test_the_short_form_fits_the_panel(self):
        """420 px at 0.42 scale is about 46 characters."""
        for pull in SHRINK_STEPS:
            v = self.view(pull, layer='surface')
            self.assertLessEqual(len(v._panel_status(None)[0][0]), 46)


class _VoteWorld(object):
    """Only the parts _world_pull reads off a WorldMap."""

    def __init__(self, st):
        self._st = np.asarray(st, np.uint8)
        self.shape = self._st.shape
        self.n_hit = np.zeros(self._st.size, np.uint16)
        self.n_pass = np.zeros(self._st.size, np.uint16)

    def measured_state(self):
        return self._st


class TestTheRoomsShrink(unittest.TestCase):
    """The room shrinks onto the sweep's own vote, not onto a TSDF.

    OCCUPIED is the measurement, FREE is where the camera measured
    THROUGH -- the room's "in front of something" -- and UNKNOWN is no
    evidence at all, which is the room's version of the TSDF saturating.
    A vertex may only move out of FREE, which is the whole correctness
    argument: everything else would be inventing a measurement.
    """

    def world(self):
        """A measured wall at x=2, free space in front of it, and
        unobserved space behind."""
        st = np.full((6, 4, 4), UNKNOWN, np.uint8)
        st[2] = OCCUPIED
        st[3] = FREE
        st[4] = FREE
        return _VoteWorld(st)

    def view(self, pull=1.5):
        v = LiveView.__new__(LiveView)
        v.surface_pull = pull
        return v

    def test_a_vertex_in_measured_free_space_is_pulled_back(self):
        v, w = self.view(), self.world()
        pull = v._world_pull(w)
        before = np.array([[3.0, 2.0, 2.0]])
        after = pull(before)
        self.assertLess(after[0][0], before[0][0])

    def test_a_vertex_in_unobserved_space_is_left_alone(self):
        """Nothing was measured there, so there is nothing to shrink
        onto and moving it would be inventing a measurement."""
        v, w = self.view(), self.world()
        before = np.array([[0.0, 2.0, 2.0]])
        np.testing.assert_allclose(v._world_pull(w)(before), before)

    def test_the_step_is_capped_by_the_ladder(self):
        v, w = self.view(0.5), self.world()
        before = np.array([[4.0, 2.0, 2.0]])
        step = np.linalg.norm(v._world_pull(w)(before) - before)
        self.assertLessEqual(step, 0.5 + 1e-9)

    def test_the_shrink_off_builds_no_field_at_all(self):
        """0 is off, and off must not pay 87 ms for a field nobody
        asked for."""
        v = self.view(0.0)
        self.assertIsNone(v._world_pull(self.world()))

    def test_the_field_is_cached_on_the_vote(self):
        """It costs 87 ms to build and 11 ms to differentiate over the
        room's 1.52 M voxels, and a shrink keypress integrates no
        frame. Proven by moving the state WITHOUT the counters: a
        rebuild would see it, the cache must not."""
        v, w = self.view(), self.world()
        v._world_pull(w)
        w._st[:] = OCCUPIED              # no counter moved
        before = np.array([[3.0, 2.0, 2.0]])
        self.assertLess(v._world_pull(w)(before)[0][0], before[0][0])

    def test_a_new_measurement_rebuilds_it(self):
        v, w = self.view(), self.world()
        v._world_pull(w)
        w._st[:] = UNKNOWN
        w.n_hit[0] += 1                  # a frame was integrated
        before = np.array([[3.0, 2.0, 2.0]])
        np.testing.assert_allclose(v._world_pull(w)(before), before)

    def test_a_vertex_outside_the_grid_does_not_wrap(self):
        """floor(v + 0.5) off the end would index backwards and read
        some other corner of the room's vote."""
        v, w = self.view(), self.world()
        far = np.array([[-9.0, 2.0, 2.0], [99.0, 2.0, 2.0]])
        out = v._world_pull(w)(far)
        self.assertEqual(out.shape, far.shape)


if __name__ == '__main__':
    unittest.main()
