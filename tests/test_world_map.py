"""Unit tests for inference/world_map.py -- the fusion rules as assertions.

Pure numpy: no bag, no checkpoint, no GPU, no open3d. Run with

    python -m unittest discover -s tests -v
"""

import os
import sys
import tempfile
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from inference.world_map import (VOX, UNKNOWN, FREE, OCCUPIED, WorldMap,
                                 observe, gather_prediction)


class TestGrid(unittest.TestCase):
    def map(self):
        return WorldMap(origin=[-1.0, -1.0, 0.0], shape=(25, 25, 50))

    def test_centre_round_trips_to_its_own_index(self):
        m = self.map()
        idx = np.array([[0, 0, 0], [12, 7, 49], [24, 24, 0]])
        back, inside = m.index_of(m.centres(idx))
        self.assertTrue(inside.all())
        np.testing.assert_array_equal(back, idx)

    def test_points_outside_the_box_are_not_inside(self):
        m = self.map()
        _, inside = m.index_of([[-1.01, 0.0, 0.5], [0.0, 0.0, 4.01]])
        self.assertFalse(inside.any())

    def test_around_aligns_z_with_the_frame_grids(self):
        """Every frame grid starts at vox_origin[2] = -0.05, and gravity
        never rotates z, so the world's z layers must coincide with
        theirs: the offset is a whole number of voxels."""
        m = WorldMap.around((0.0, 0.0))
        self.assertEqual(m.shape, (200, 200, 38))
        self.assertAlmostEqual(m.origin[2], -0.13)
        self.assertAlmostEqual(
            ((-0.05 - m.origin[2]) / VOX) % 1.0, 0.0, places=9)


class TestMeasurement(unittest.TestCase):
    def map(self):
        return WorldMap(origin=[0.0, 0.0, 0.0], shape=(4, 4, 4))

    def test_one_frame_is_not_enough_and_two_are(self):
        m = self.map()
        m.add_measurement([[1, 1, 1]], [], frame=0)
        self.assertEqual(m.measured_state()[1, 1, 1], UNKNOWN)
        m.add_measurement([[1, 1, 1]], [], frame=1)
        self.assertEqual(m.measured_state()[1, 1, 1], OCCUPIED)

    def test_two_frames_of_free_space_make_it_free(self):
        m = self.map()
        for f in (0, 1):
            m.add_measurement([], [[2, 2, 2]], frame=f)
        self.assertEqual(m.measured_state()[2, 2, 2], FREE)

    def test_hits_beat_passes(self):
        m = self.map()
        for f in range(5):
            m.add_measurement([[0, 0, 0]], [[0, 0, 0]], frame=f)
        self.assertEqual(m.measured_state()[0, 0, 0], OCCUPIED)

    def test_counts_are_per_frame_not_per_sample(self):
        m = self.map()
        m.add_measurement([[1, 1, 1]] * 50, [], frame=0)
        self.assertEqual(m.measured_state()[1, 1, 1], UNKNOWN)

    def test_first_meas_is_when_the_state_settled(self):
        m = self.map()
        m.add_measurement([[1, 1, 1]], [], frame=7)
        m.add_measurement([[1, 1, 1]], [], frame=9)
        self.assertEqual(m.first_meas[m._flat([[1, 1, 1]])[0][0]], 9)

    def test_colour_is_the_mean_over_the_frames_that_saw_it(self):
        m = self.map()
        m.add_measurement([[1, 1, 1]], [], 0, rgb=[[10, 20, 30]])
        m.add_measurement([[1, 1, 1]], [], 1, rgb=[[30, 40, 50]])
        np.testing.assert_array_equal(m.colours()[1, 1, 1], [20, 30, 40])


class TestObserve(unittest.TestCase):
    def setUp(self):
        self.m = WorldMap(origin=[-1.0, -1.0, 0.0], shape=(25, 25, 50))
        self.K = np.array([[100.0, 0, 50.0], [0, 100.0, 50.0], [0, 0, 1.0]])
        self.T = np.eye(4)

    def wall(self, d):
        return np.full((100, 100), d, np.float32)

    def test_a_wall_is_hit_and_the_space_before_it_is_free(self):
        hit, _, free = observe(self.m, self.wall(2.02), self.K, self.T)
        hz = set(int(k) for k in hit[:, 2])
        fz = set(int(k) for k in free[:, 2])
        self.assertEqual(hz, {25})            # centre 2.04, |d - z| = 0.02
        self.assertNotIn(25, fz)
        self.assertIn(12, fz)                 # centre 1.00, well in front
        self.assertNotIn(30, fz)              # behind the wall: unknown

    def test_zero_depth_measures_nothing(self):
        hit, _, free = observe(self.m, np.zeros((100, 100), np.float32),
                               self.K, self.T)
        self.assertEqual(len(hit), 0)
        self.assertEqual(len(free), 0)

    def test_beyond_max_range_measures_nothing(self):
        hit, _, _ = observe(self.m, self.wall(2.02), self.K, self.T,
                            max_range=1.0)
        self.assertEqual(len(hit), 0)

    def test_colour_comes_from_the_pixel_a_hit_voxel_projects_to(self):
        bgr = np.zeros((100, 100, 3), np.uint8)
        bgr[:, :] = (10, 20, 30)              # BGR in, RGB out
        hit, rgb, _ = observe(self.m, self.wall(2.02), self.K, self.T,
                              rgb=bgr)
        self.assertEqual(len(rgb), len(hit))
        np.testing.assert_array_equal(rgb[0], [30, 20, 10])


class TestPrediction(unittest.TestCase):
    def map(self):
        return WorldMap(origin=[0.0, 0.0, 0.0], shape=(4, 4, 4))

    def test_a_majority_of_the_frames_that_could_see_it(self):
        m = self.map()
        v = [[1, 1, 1]]
        m.add_prediction(v, [7], 0)
        m.add_prediction(v, [7], 1)
        m.add_prediction(v, [0], 2)
        occ, cls = m.predicted()
        self.assertTrue(occ[1, 1, 1])
        self.assertEqual(cls[1, 1, 1], 7)

    def test_one_frame_in_three_is_not_a_majority(self):
        m = self.map()
        v = [[1, 1, 1]]
        m.add_prediction(v, [7], 0)
        m.add_prediction(v, [0], 1)
        m.add_prediction(v, [0], 2)
        occ, cls = m.predicted()
        self.assertFalse(occ[1, 1, 1])
        self.assertEqual(cls[1, 1, 1], 0)

    def test_blind_needs_the_sensor_to_know_nothing_at_all(self):
        m = self.map()
        f = m._flat([[1, 1, 1]])[0][0]
        m.add_prediction([[1, 1, 1]], [7], 3)
        self.assertTrue(m.blind[f])
        self.assertEqual(m.first_pred[f], 3)

    def test_a_voxel_the_sensor_already_measured_is_not_a_completion(self):
        m = self.map()
        g = m._flat([[2, 2, 2]])[0][0]
        for fr in (0, 1):
            m.add_measurement([[2, 2, 2]], [], fr)
        m.add_prediction([[2, 2, 2]], [7], 2)
        self.assertFalse(m.blind[g])

    def test_one_ray_is_enough_to_stop_it_being_a_completion(self):
        """The state needs two frames to settle, but a completion is
        about what the sensor KNEW, not about what has settled. A surface
        the camera is looking straight at on frame 0 is not a completion,
        or the model takes credit for predicting what it can see."""
        for meas in ([[1, 1, 1]], []), ([], [[1, 1, 1]]):
            m = self.map()
            f = m._flat([[1, 1, 1]])[0][0]
            m.add_measurement(meas[0], meas[1], 0)
            self.assertEqual(m.measured_state()[1, 1, 1], UNKNOWN)
            m.add_prediction([[1, 1, 1]], [7], 0)
            self.assertFalse(m.blind[f])

    def test_first_pred_is_the_first_frame_only(self):
        m = self.map()
        f = m._flat([[1, 1, 1]])[0][0]
        m.add_prediction([[1, 1, 1]], [7], 5)
        m.add_prediction([[1, 1, 1]], [7], 6)
        self.assertEqual(m.first_pred[f], 5)


class TestGather(unittest.TestCase):
    """world_T_cam == cam_pose is the FIRST frame, which fuse_scan anchors
    the world on, so the composition cam_pose @ inv(world_T_cam) must
    collapse to the identity. A transposed rotation or an inverted pose
    shows up here as a translation or a tipped-over room.

    These also pin the AXIS CONVENTION. The network's grid is indexed
    [z_g, y_g, x_g]: axis 0 walks world X, axis 1 walks world Z (height,
    the 36-cell one) and axis 2 walks world Y. Assuming (X, Y, Z) puts the
    room on its side, and the only symptom is that the prediction stops
    agreeing with the measurement.
    """

    VOX_ORIGIN = np.array([-2.4, 0.0, -0.05])
    GRID = (6, 4, 7)            # deliberately unequal: (X, Z, Y) extents

    def frame(self):
        cam_pose = np.eye(4)
        cam_pose[:3, 3] = [0.3, -0.2, 1.1]
        rng = np.random.RandomState(0)
        pred = rng.randint(0, 12, self.GRID).astype(np.int64)
        return cam_pose, pred, np.ones(self.GRID, bool)

    def world_index(self, f, off):
        """array index [iX, iZ, iY] -> world-order index + offset."""
        return (f[0] + off[0], f[2] + off[1], f[1] + off[2])

    def test_on_an_aligned_grid_every_frame_voxel_lands_where_it_should(self):
        """Offset the world by a whole number of voxels and the mapping is
        a bijection, so the frame can be checked voxel for voxel."""
        cam_pose, pred, keep = self.frame()
        off = np.array([7, 10, 2])
        m = WorldMap(origin=self.VOX_ORIGIN - off * VOX, shape=(40, 40, 20))
        idx, cls = gather_prediction(m, pred, keep, self.VOX_ORIGIN,
                                     cam_pose, cam_pose, grid=self.GRID)
        got = dict((tuple(a), int(b)) for a, b in zip(idx, cls))
        self.assertEqual(len(got), int(np.prod(self.GRID)))
        for f in np.argwhere(keep):
            self.assertEqual(got[self.world_index(f, off)],
                             int(pred[tuple(f)]))

    def test_the_height_axis_is_the_36_cell_one(self):
        """The grid is 4.8 x 4.8 m wide and 2.88 m TALL. Read as
        (X, Y, Z) it would be 4.8 x 2.88 x 4.8 and the room would lie on
        its side, so check the world extent each axis actually covers."""
        cam_pose = np.eye(4)
        cam_pose[:3, 3] = [0.3, -0.2, 1.1]
        grid = (60, 36, 60)
        pred = np.ones(grid, np.int64)
        m = WorldMap(origin=[-6.0, -6.0, -0.13], shape=(150, 150, 38))
        idx, _ = gather_prediction(m, pred, np.ones(grid, bool),
                                   self.VOX_ORIGIN, cam_pose, cam_pose,
                                   grid=grid)
        span = (idx.max(axis=0) - idx.min(axis=0) + 1)
        np.testing.assert_array_equal(span, [60, 60, 36])

    def test_on_an_offset_grid_each_voxel_takes_the_class_at_its_centre(self):
        """Grids that do not share a lattice are still correct: a world
        voxel takes the class of the frame voxel its CENTRE falls in."""
        cam_pose, pred, keep = self.frame()
        m = WorldMap(origin=[-3.0, -1.0, -0.2], shape=(80, 80, 45))
        idx, cls = gather_prediction(m, pred, keep, self.VOX_ORIGIN,
                                     cam_pose, cam_pose, grid=self.GRID)
        self.assertEqual(len(set(map(tuple, idx))), len(idx))
        fw = np.floor((m.centres(idx) - self.VOX_ORIGIN) / VOX).astype(int)
        f = fw[:, [0, 2, 1]]
        np.testing.assert_array_equal(cls, pred[f[:, 0], f[:, 1], f[:, 2]])

    def test_a_moved_camera_moves_the_prediction_in_the_world(self):
        """The one case the identity test cannot catch: once the camera
        has moved, the same frame voxel must land somewhere else."""
        cam_pose, pred, keep = self.frame()
        m = WorldMap(origin=[-6.0, -6.0, -0.13], shape=(150, 150, 38))
        a, _ = gather_prediction(m, pred, keep, self.VOX_ORIGIN, cam_pose,
                                 cam_pose, grid=self.GRID)
        moved = cam_pose.copy()
        moved[:3, 3] += [1.6, 0.0, 0.0]        # 20 voxels along world x
        b, _ = gather_prediction(m, pred, keep, self.VOX_ORIGIN, cam_pose,
                                 moved, grid=self.GRID)
        np.testing.assert_array_equal(np.sort(b[:, 0]),
                                      np.sort(a[:, 0]) + 20)
        np.testing.assert_array_equal(np.sort(b[:, 2]), np.sort(a[:, 2]))


class TestVerdicts(unittest.TestCase):
    def map(self):
        return WorldMap(origin=[0.0, 0.0, 0.0], shape=(4, 4, 4))

    def test_confirmed(self):
        m = self.map()
        m.add_prediction([[1, 1, 1]], [7], 0)
        for f in (1, 2):
            m.add_measurement([[1, 1, 1]], [], f)
        self.assertTrue(m.verdicts()['confirmed'][1, 1, 1])

    def test_contradicted(self):
        m = self.map()
        m.add_prediction([[1, 1, 1]], [7], 0)
        for f in (1, 2):
            m.add_measurement([], [[1, 1, 1]], f)
        self.assertTrue(m.verdicts()['contradicted'][1, 1, 1])

    def test_untested(self):
        m = self.map()
        m.add_prediction([[1, 1, 1]], [7], 0)
        self.assertTrue(m.verdicts()['untested'][1, 1, 1])

    def test_missed(self):
        m = self.map()
        for f in (0, 1):
            m.add_measurement([[2, 2, 2]], [], f)
        self.assertTrue(m.verdicts()['missed'][2, 2, 2])

    def test_disproved_is_wider_than_contradicted(self):
        """Scoring a COMPLETION needs the blind test; showing an ERROR
        does not. A voxel the model fills in plain view, that the sensor
        then measures as free space, is still wrong -- and without this it
        drew the same neutral grey as "nothing has measured this yet"."""
        m = self.map()
        v = [[1, 1, 1]]
        m.add_measurement([], v, 0)          # one pass: seen, not settled
        m.add_prediction(v, [7], 0)          # so this is NOT a completion
        m.add_measurement([], v, 1)          # now it settles as free
        d = m.verdicts()
        self.assertFalse(m.blind[m._flat(v)[0][0]])
        self.assertFalse(d['contradicted'][1, 1, 1])
        self.assertTrue(d['disproved'][1, 1, 1])

    def test_every_contradicted_completion_is_also_disproved(self):
        m = self.map()
        m.add_prediction([[1, 1, 1]], [7], 0)
        for f in (1, 2):
            m.add_measurement([], [[1, 1, 1]], f)
        d = m.verdicts()
        self.assertTrue(d['contradicted'][1, 1, 1])
        self.assertTrue(d['disproved'][1, 1, 1])

    def test_a_prediction_that_lost_the_vote_has_no_verdict(self):
        m = self.map()
        v = [[1, 1, 1]]
        m.add_prediction(v, [7], 0)
        m.add_prediction(v, [0], 1)
        m.add_prediction(v, [0], 2)
        d = m.verdicts()
        self.assertFalse(d['untested'][1, 1, 1])
        self.assertFalse(d['predicted'][1, 1, 1])

    def test_round_trips_through_a_file(self):
        m = self.map()
        m.add_measurement([[1, 1, 1]], [[2, 2, 2]], 0, rgb=[[1, 2, 3]])
        m.add_prediction([[1, 1, 1]], [7], 0)
        d = tempfile.mkdtemp()
        p = os.path.join(d, 'w.npz')
        m.save(p)
        b = WorldMap.load(p)
        self.assertEqual(b.shape, m.shape)
        np.testing.assert_array_equal(b.votes, m.votes)
        np.testing.assert_array_equal(b.first_pred, m.first_pred)
        self.assertEqual(b.counts(), m.counts())


if __name__ == '__main__':
    unittest.main()
