"""Unit tests for inference/tracker.py's gate.

The gate is a pure function and the rule around it a pure class, on
purpose: what decides whether a frame may touch the map, and what a
tracker that has lost the pose must show before it is believed again, are
both testable without open3d, a GPU or a bag. Tracker itself needs all
three and is checked against the committed trajectory.txt by
inference/check_world_map.py instead.
"""

import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from inference.tracker import Belief, Tracker, reject


def turn_about_z(deg):
    a = np.radians(deg)
    T = np.eye(4)
    T[:3, :3] = [[np.cos(a), -np.sin(a), 0.0],
                 [np.sin(a), np.cos(a), 0.0],
                 [0.0, 0.0, 1.0]]
    return T


class TestGate(unittest.TestCase):
    def test_a_still_camera_is_accepted(self):
        self.assertIsNone(reject(0.9, np.eye(4), 30.0))

    def test_a_plausible_handheld_step_is_accepted(self):
        T = np.eye(4)
        T[:3, 3] = [0.01, 0.0, 0.005]          # 1 cm at 30 fps = 0.3 m/s
        self.assertIsNone(reject(0.9, T, 30.0))

    def test_low_fitness_is_rejected(self):
        self.assertIn('fitness', reject(0.05, np.eye(4), 30.0))

    def test_an_impossible_jump_is_rejected(self):
        T = np.eye(4)
        T[:3, 3] = [0.5, 0.0, 0.0]             # 0.5 m in one frame
        self.assertIn('jump', reject(0.9, T, 30.0))

    def test_an_impossible_turn_is_rejected(self):
        self.assertIn('jump', reject(0.9, turn_about_z(30.0), 30.0))

    def test_a_plausible_turn_is_accepted(self):
        self.assertIsNone(reject(0.9, turn_about_z(2.0), 30.0))

    def test_the_limits_scale_with_the_frame_rate(self):
        """1 m/s is 1 cm at 100 fps and 10 cm at 10 fps."""
        T = np.eye(4)
        T[:3, 3] = [0.05, 0.0, 0.0]
        self.assertIsNone(reject(0.9, T, 10.0))
        self.assertIn('jump', reject(0.9, T, 100.0))

    def test_fitness_alone_decides_when_there_is_no_step_to_gate(self):
        """How far the camera went while nothing was believed is exactly
        what is not knowable, so re-acquiring passes None."""
        self.assertIsNone(reject(0.9, None))
        self.assertIn('fitness', reject(0.05, None))

    def test_without_a_known_rate_the_allowance_is_generous(self):
        """A guessed frame rate is worse than none. room07's bag is 15 fps
        and assuming 30 rejected ordinary handheld steps of 3.6-15.7 cm,
        which lost the tracker half way through the room."""
        for step in (0.036, 0.157, 0.4):
            T = np.eye(4)
            T[:3, 3] = [step, 0.0, 0.0]
            self.assertIsNone(reject(0.9, T))
        T = np.eye(4)
        T[:3, 3] = [0.8, 0.0, 0.0]        # implausible at any frame rate
        self.assertIn('jump', reject(0.9, T))


def step(d=0.0, deg=0.0):
    T = turn_about_z(deg)
    T[:3, 3] = [d, 0.0, 0.0]
    return T


class TestBelief(unittest.TestCase):
    """The rule that decides what gets into the map, and what gets a
    tracker that has lost the pose back into it."""

    def run_of(self, b, n, fitness=0.9, d=0.01):
        out = []
        for _ in range(n):
            out.append(b.see(fitness, step(d))[0])
        return out

    def test_a_good_run_goes_straight_in(self):
        b = Belief()
        self.assertEqual(self.run_of(b, 10), ['integrate'] * 10)
        self.assertEqual(b.dropped, 0)
        self.assertFalse(b.lost)

    def test_the_floor_is_fuse_scans_until_there_is_a_run_to_compare_with(self):
        b = Belief()
        self.assertEqual(b.floor(), 0.1)
        self.run_of(b, 5, fitness=0.9)
        self.assertAlmostEqual(b.floor(), 0.54)

    def test_the_floor_follows_what_this_run_achieves(self):
        """A sensor or a scene that matches worse is not a jump. The same
        60% of a run holding 0.4 asks for 0.24, not 0.54."""
        b = Belief()
        self.run_of(b, 10, fitness=0.4)
        self.assertAlmostEqual(b.floor(), 0.24)

    def test_the_floor_is_capped_so_it_cannot_outrun_the_evidence(self):
        """The worst frame of 1871 measured on two full sweeps was 0.647;
        the floor is never allowed above 0.55."""
        b = Belief()
        self.run_of(b, 10, fitness=0.99)
        self.assertAlmostEqual(b.floor(), 0.55)

    def test_the_jump_that_used_to_be_welded_in_is_dropped(self):
        """Resuming room07 at frame 200 after frame 59 read 0.15-0.23.
        fuse_scan's 0.1 floor passes it; this one must not."""
        b = Belief()
        self.run_of(b, 10, fitness=0.78)
        verb, why = b.see(0.18, step(0.25, 7.3))
        self.assertEqual(verb, 'drop')
        self.assertIn('fitness', why)

    def test_a_dropped_frame_does_not_move_the_reference(self):
        """Or a slide of marginal frames would talk the floor down to
        meet itself."""
        b = Belief()
        self.run_of(b, 10, fitness=0.9)
        for _ in range(5):
            b.see(0.2, step())
        self.assertAlmostEqual(b.floor(), 0.54)

    def test_lost_is_only_said_after_a_run_of_them(self):
        b = Belief(lost_after=15)
        self.run_of(b, 10)
        for i in range(14):
            b.see(0.01, step())
            self.assertFalse(b.lost)
        b.see(0.01, step())
        self.assertTrue(b.lost)

    def test_one_good_frame_is_not_enough_to_be_believed_again(self):
        """Three in a row that agree, because one lucky match welds the
        room into the wrong place for good."""
        b = Belief(confirm=3)
        self.run_of(b, 10)
        for _ in range(20):
            b.see(0.01, step())
        self.assertEqual(b.see(0.9, step())[0], 'confirm')
        self.assertEqual(b.see(0.9, step(0.01))[0], 'confirm')
        self.assertEqual(b.see(0.9, step(0.01))[0], 'integrate')
        self.assertFalse(b.lost)
        self.assertEqual(b.reacquired, 1)

    def test_confirming_starts_over_if_the_frames_disagree(self):
        b = Belief(confirm=3)
        self.run_of(b, 10)
        for _ in range(20):
            b.see(0.01, step())
        self.assertEqual(b.see(0.9, step())[0], 'confirm')
        self.assertEqual(b.see(0.9, step(2.0))[0], 'drop')    # not one place
        self.assertEqual(b.see(0.9, step())[0], 'confirm')    # and start over

    def test_the_jump_gate_is_dropped_for_the_first_frame_back(self):
        """The camera may be anywhere by then; agreement is what is asked
        for instead."""
        b = Belief(confirm=2)
        self.run_of(b, 10)
        for _ in range(20):
            b.see(0.01, step())
        self.assertEqual(b.see(0.9, step(3.0))[0], 'confirm')
        self.assertEqual(b.see(0.9, step(0.01))[0], 'integrate')

    def test_nothing_is_integrated_while_it_is_confirming(self):
        b = Belief(confirm=3, lost_after=2)
        self.run_of(b, 10)
        for _ in range(2):
            b.see(0.01, step())
        self.assertTrue(b.lost)
        held = b.held
        self.assertEqual(b.see(0.9, step())[0], 'confirm')
        self.assertEqual(b.held, held + 1)

    def test_a_hiccup_is_not_a_relocalisation(self):
        """One dropped frame out of a good run costs one frame, not four.
        Asking for three agreeing frames there fed itself on a live
        camera: the confirming frames are not integrated, so the model
        stopped keeping up, so the next frame matched worse."""
        b = Belief(confirm=3, lost_after=15)
        self.run_of(b, 10)
        self.assertEqual(b.see(0.01, step())[0], 'drop')
        self.assertFalse(b.lost)
        self.assertEqual(b.see(0.9, step(0.01))[0], 'integrate')
        self.assertEqual(b.run_lost, 0)
        self.assertEqual(b.reacquired, 0)

    def test_a_hiccup_still_has_to_pass_the_jump_gate(self):
        """The gate is only dropped for a frame coming back from LOST,
        where the gap really is unknowable."""
        b = Belief(confirm=3, lost_after=15)
        self.run_of(b, 10)
        b.see(0.01, step())
        self.assertIn('jump', b.see(0.9, step(3.0))[1])


VRAM_ERROR = (
    '[Open3D Error] (void open3d::t::geometry::kernel::voxel_grid::'
    'ExtractTriangleMeshCUDA(...)) /root/Open3D/cpp/open3d/t/geometry/'
    'kernel/VoxelBlockGridImpl.h:1334: Unable to allocate assistance mesh '
    'structure for Marching Cubes with 463 active voxel blocks. Please '
    'consider using a larger voxel size (currently 0.02) for TSDF '
    'integration, or using tsdf_volume.cpu() to perform mesh extraction '
    'on CPU.')


class StubGrid:
    """Stands in for open3d's VoxelBlockGrid. The host path is one call."""

    def __init__(self, mesh):
        self.mesh = mesh
        self.copied = 0

    def hashmap(self):
        return self

    def size(self):
        return 463

    def cpu(self):
        self.copied += 1
        return self

    def extract_triangle_mesh(self, weight):
        return self.mesh


class StubModel:
    def __init__(self, raises=None, mesh='gpu mesh'):
        self.raises, self.mesh = raises, mesh
        self.voxel_grid = StubGrid('host mesh')

    def extract_trianglemesh(self, weight):
        if self.raises:
            raise RuntimeError(self.raises)
        return self.mesh


class TestExtractFallback(unittest.TestCase):
    """A 4 GB card holding the network, the encoder, the tracking volume
    and the window has nothing left for the marching cubes' scratch
    structure, and open3d says so rather than returning a worse mesh."""

    def tracker(self, model):
        t = Tracker.__new__(Tracker)      # no open3d, no GPU, no bag
        t.model, t.on_host = model, 0
        return t

    def test_the_card_is_used_when_it_will_have_it(self):
        t = self.tracker(StubModel())
        self.assertEqual(t._extract(3.0), 'gpu mesh')
        self.assertEqual(t.on_host, 0)
        self.assertEqual(t.model.voxel_grid.copied, 0)

    def test_no_room_on_the_card_means_the_host(self):
        t = self.tracker(StubModel(raises=VRAM_ERROR))
        self.assertEqual(t._extract(3.0), 'host mesh')
        self.assertEqual(t.on_host, 1)
        self.assertEqual(t.model.voxel_grid.copied, 1)

    def test_plain_cuda_oom_is_also_the_host(self):
        t = self.tracker(StubModel(raises='CUDA error: out of memory'))
        self.assertEqual(t._extract(3.0), 'host mesh')

    def test_anything_else_is_not_swallowed(self):
        t = self.tracker(StubModel(raises='the volume is on fire'))
        with self.assertRaises(RuntimeError):
            t._extract(3.0)


if __name__ == '__main__':
    unittest.main()
