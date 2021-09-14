import unittest

import numpy as np
from numpy.testing import assert_array_almost_equal, assert_array_equal

from environments.spe_ed import Player, SavedGame, directions_by_name
from state_representation import occupancy_map, padded_window


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
        """Compute occupancy for one step."""
        cells = np.zeros((5, 5), dtype=np.int32)
        cells[2, 2] = 1

        occ = occupancy_map(
            cells=cells,
            opponents=[Player(1, 2, 2, directions_by_name["right"], 1, True)],
            rounds=1,
            depth=1,
        )

        assert_array_almost_equal(
            occ, [
                [0, 0, 0, 0, 0],
                [0, 0, 1 / 5, 0, 0],
                [0, 0, 1, 2 / 5, 1 / 5],
                [0, 0, 1 / 5, 0, 0],
                [0, 0, 0, 0, 0],
            ]
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
            [
                [0, 0, 2 / 25, 0, 1 / 25],
                [0, 1 / 25, 1 / 5, 49 / 625, 1 / 25],
                [0, 0, 1, 2 / 5, 33 / 125],
                [0, 1 / 25, 1 / 5, 49 / 625, 1 / 25],
                [0, 0, 2 / 25, 0, 1 / 25],
            ],
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
            occ, [
                [0, 0, 0, 0, 0],
                [1 / 5, 0, 0, 0, 0],
                [1, 3 / 5, 1 / 5, 1 / 5, 1 / 5],
                [1 / 5, 0, 0, 0, 0],
                [0, 0, 0, 0, 0],
            ]
        )


class TestPaddedWindow(unittest.TestCase):
    def test_window(self):
        game = SavedGame.load(r"tests/logs/20201019-182018.json")

        t = 0
        window = padded_window(game.cell_states[t], game.player_states[t][0].x, game.player_states[t][0].y, 2, -1)
        assert_array_equal(
            window, [
                [0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0],
                [0, 0, 1, 0, 0],
                [0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0],
            ]
        )

        t = 4
        window = padded_window(game.cell_states[t], game.player_states[t][0].x, game.player_states[t][0].y, 2, -1)
        assert_array_equal(
            window, [
                [0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0],
                [0, 0, 1, 0, 0],
                [0, 0, 1, 0, 0],
                [0, 0, 1, 1, 1],
            ]
        )
        t = 23
        window = padded_window(game.cell_states[t], game.player_states[t][0].x, game.player_states[t][0].y, 2, -1)
        assert_array_equal(
            window, [
                [0, 0, 1, 0, 0],
                [0, 0, 1, 0, 0],
                [0, 0, 1, 0, 0],
                [0, 0, 0, 0, 0],
                [-1, -1, -1, -1, -1],
            ]
        )
