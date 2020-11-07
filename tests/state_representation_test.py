import unittest
import numpy as np
from numpy.testing import assert_array_equal, assert_array_almost_equal
from environments.spe_ed import Player, directions_by_name
from state_representation import occupancy_map


class TestOccupancyMap(unittest.TestCase):
    def test_no_opponent(self):
        """Having no active opponent results in no occupancy"""
        occ = occupancy_map(cells=np.zeros((5, 5), dtype=np.int32), opponents=[], rounds=1)

        assert_array_equal(occ, np.zeros((5, 5), dtype=np.float32))  #
        self.assertEqual(occ.dtype, np.float32)

    def test_no_active_opponent(self):
        """Having no active opponent results in no occupancy"""
        occ = occupancy_map(
            cells=np.zeros((5, 5), dtype=np.int32),
            opponents=[Player(1, 2, 2, directions_by_name["right"], 1, False)],
            rounds=1,
        )

        assert_array_equal(occ, np.zeros((5, 5), dtype=np.float32))
        self.assertEqual(occ.dtype, np.float32)

    def test_depth_1(self):
        """Compute occupancy for two steps."""
        cells = np.zeros((5, 5), dtype=np.int32)
        cells[2, 2] = 1

        occ = occupancy_map(
            cells=cells,
            opponents=[Player(1, 2, 2, directions_by_name["right"], 1, True)],
            rounds=1,
            depth=1,
        )

        assert_array_almost_equal(
            occ,
            np.array([
                [0, 0, 0, 0, 0],
                [0, 0, 1, 0, 0],
                [0, 0, 5, 2, 1],
                [0, 0, 1, 0, 0],
                [0, 0, 0, 0, 0],
            ]) / 5
        )

    def test_depth_2(self):
        """Compute occupancy for two steps."""
        cells = np.zeros((5, 5), dtype=np.int32)
        cells[2, 2] = 1

        occ = occupancy_map(
            cells=cells,
            opponents=[Player(1, 2, 2, directions_by_name["right"], 1, True)],
            rounds=1,
            depth=2,
        )

        assert_array_almost_equal(
            occ,
            np.array([
                [0, 0, 2, 0, 1],
                [0, 1, 5, 2, 1],
                [0, 0, 25, 10, 7],
                [0, 1, 5, 2, 1],
                [0, 0, 2, 0, 1],
            ]) / 25
        )

    def test_jump(self):
        """Compute occupancy for one step with jump."""
        cells = np.zeros((5, 5), dtype=np.int32)
        cells[2, 0] = 1

        occ = occupancy_map(
            cells=cells,
            opponents=[Player(1, 0, 2, directions_by_name["right"], 3, True)],
            rounds=6,
            depth=1,
        )

        assert_array_almost_equal(
            occ,
            np.array([
                [0, 0, 0, 0, 0],
                [1, 0, 0, 0, 0],
                [5, 3, 1, 1, 1],
                [1, 0, 0, 0, 0],
                [0, 0, 0, 0, 0],
            ]) / 5
        )
