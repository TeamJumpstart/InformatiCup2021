import unittest
from environments.spe_ed import Cells, SavedGame
from tests.heuristic_test import default_round1_board


class TestCells(unittest.TestCase):
    def test_dimension(self):
        """Check proper width and height."""
        cells = Cells(default_round1_board()[0])

        self.assertEqual(cells.width, 5)
        self.assertEqual(cells.height, 5)

        game = SavedGame.load(r"tests/logs/20201019-182018.json")
        cells = Cells(game.cell_states[0] != 0)

        self.assertEqual(cells.width, game.width)
        self.assertEqual(cells.height, game.height)

    def test_is_free(self):
        """Check test_is_free."""
        cells = Cells(default_round1_board()[0])

        self.assertEqual(cells.is_free([0, 0]), False)
        self.assertEqual(cells.is_free([0, 1]), False)
        self.assertEqual(cells.is_free([1, 0]), True)

        self.assertEqual(cells.is_free([2, 2]), False)
        self.assertEqual(cells.is_free([2, 3]), True)
        self.assertEqual(cells.is_free([3, 2]), False)

        self.assertEqual(cells.is_free([-1, 0]), False)
        self.assertEqual(cells.is_free([0, -1]), False)
        self.assertEqual(cells.is_free([-1, -1]), False)
        self.assertEqual(cells.is_free([4, 5]), False)
        self.assertEqual(cells.is_free([5, 4]), False)
        self.assertEqual(cells.is_free([5, -5]), False)
