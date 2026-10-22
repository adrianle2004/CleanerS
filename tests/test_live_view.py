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

    def view(self, show_mesh, recon):
        v = LiveView.__new__(LiveView)
        v.show_mesh, v.recon = show_mesh, recon
        v._build = lambda idx, colours: 'the prediction'
        v._build_frame_recon = lambda loader: 'the sensor surface'
        return v

    def test_the_prediction_is_what_it_draws_by_default(self):
        v = self.view(False, 'a mesh')
        self.assertEqual(v._frame_geometry(None, None, None),
                         'the prediction')

    def test_m_draws_the_frames_own_surface(self):
        v = self.view(True, 'a mesh')
        self.assertEqual(v._frame_geometry(None, None, None),
                         'the sensor surface')

    def test_m_before_a_surface_exists_still_draws_something(self):
        """The key may be pressed before run_live has built one -- it is
        built on the NEXT frame. Drawing nothing would read as a crash."""
        v = self.view(True, None)
        self.assertEqual(v._frame_geometry(None, None, None),
                         'the prediction')


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
