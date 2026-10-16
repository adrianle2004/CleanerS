"""Unit tests for inference/tracker.py's gate.

The gate is a pure function on purpose, so the rule that decides whether a
frame is allowed to touch the map can be tested without open3d, a GPU or a
bag. Tracker itself needs all three and is checked against the committed
trajectory.txt by inference/check_world_map.py instead.
"""

import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from inference.tracker import reject


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


if __name__ == '__main__':
    unittest.main()
