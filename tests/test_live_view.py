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

from inference.live_view import view_points
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
