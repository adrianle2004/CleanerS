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

from inference.live_view import GRID, VOX_UNIT_LOW, LiveView, view_points
from inference.world_map import WorldMap


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


if __name__ == '__main__':
    unittest.main()


class TestFrameGeometry(unittest.TestCase):
    """Which of the two layers the single-frame window draws.

    `m` used to need --accumulate, because the mesh was the tracker's
    volume and there was no tracker without the flag. A plain --display
    run now builds the frame's OWN surface instead, so the key has to
    choose between that and the prediction -- and has to fall back to the
    prediction when the surface for this frame does not exist yet, which
    is the state the window is in for one frame after the key is pressed.

    LiveView.__init__ imports open3d and opens a window; the chooser
    touches neither, so it is built without them.
    """

    def view(self, show_mesh):
        v = LiveView.__new__(LiveView)
        v.show_mesh = show_mesh
        v._build = lambda idx, colours: 'the prediction'
        v._build_frame_recon = lambda loader, recon: recon
        return v

    def test_the_prediction_is_what_it_draws_by_default(self):
        v = self.view(False)
        self.assertEqual(v._frame_geometry(None, None, None, 'a mesh'),
                         'the prediction')

    def test_m_draws_the_frames_own_surface(self):
        v = self.view(True)
        self.assertEqual(v._frame_geometry(None, None, None, 'a mesh'),
                         'a mesh')

    def test_m_before_a_surface_exists_still_draws_something(self):
        """The key may be pressed before run_live has built one -- it is
        built on the NEXT frame. Drawing nothing would read as a crash."""
        v = self.view(True)
        self.assertEqual(v._frame_geometry(None, None, None, None),
                         'the prediction')

    def test_the_surface_drawn_is_the_one_that_came_with_the_frame(self):
        """Not whatever self.recon happens to hold. The window is a
        frame behind by design, so reading the live attribute at draw
        time pairs this frame's cubes with the NEXT frame's surface."""
        v = self.view(True)
        v.recon = 'the next frame already built this one'
        self.assertEqual(v._frame_geometry(None, None, None, 'this frame'),
                         'this frame')


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
        v.show_mesh, v.recon, v.live_recon = False, recon, live
        v.can_reconstruct = True
        v._last = ('idx', 'cls', 'loader', 'rgb')
        v.make_surface = make_surface
        v.repainted = []
        v._drain = lambda: None
        v._repaint = lambda: v.repainted.append(1)
        return v

    def test_pressing_m_with_no_surface_yet_builds_one(self):
        v = self.view(lambda loader, rgb: 'a surface')
        v._flip_mesh(None)
        self.assertTrue(v.show_mesh)
        self.assertEqual(v.recon, 'a surface')
        self.assertEqual(len(v.repainted), 1)

    def test_it_is_built_for_the_frame_on_screen(self):
        seen = []
        v = self.view(lambda loader, rgb: seen.append((loader, rgb)) or 'm')
        v._flip_mesh(None)
        self.assertEqual(seen, [('loader', 'rgb')])

    def test_m_always_rebuilds_rather_than_reusing(self):
        """Reported: freeze, m on, m off, play a hundred frames to
        somewhere else, freeze, m on -- and the surface was still the
        one built before the camera moved. A per-frame surface belongs
        to one frame, so a held one is stale, not spare."""
        calls = []
        v = self.view(lambda loader, rgb: calls.append(1) or 'rebuilt',
                      recon='built a hundred frames ago')
        v._flip_mesh(None)
        self.assertEqual(len(calls), 1)
        self.assertEqual(v.recon, 'rebuilt')

    def test_a_frame_that_measured_nothing_does_not_leave_m_on(self):
        """Showing nothing at all reads as a crash, so it says so and
        stays on the prediction."""
        v = self.view(lambda loader, rgb: None)
        v._flip_mesh(None)
        self.assertFalse(v.show_mesh)
        self.assertEqual(v.repainted, [])

    def test_a_failure_to_build_never_escapes(self):
        def boom(loader, rgb):
            raise RuntimeError('no room on the card')
        v = self.view(boom)
        v._flip_mesh(None)          # must not raise
        self.assertFalse(v.show_mesh)

    def test_accumulate_still_says_to_freeze_first(self):
        """There the mesh is the sweep's, built on freeze and at the end,
        and there is no make_surface -- the old message is still right."""
        v = self.view(None, live=False)
        v._flip_mesh(None)
        self.assertFalse(v.show_mesh)


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
