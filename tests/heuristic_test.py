import unittest
import heuristics
from environments import spe_ed
import numpy as np


def empty_board():
    """ board state visualised: board size = 5x5
    - - - - -
    - - - - -
    - - - - -
    - - - - -
    - - - - -
    """
    cells = np.zeros((5, 5), dtype=bool)
    player = spe_ed.Player(player_id=0, x=2, y=2, direction=spe_ed.directions[0], speed=1, active=True)
    opponents = []
    rounds = 0
    return (cells, player, opponents, rounds)


def default_round1_board():
    """ board state visualised: board size = 5x5
    # - - - -
    # - - - -
    - - # # -
    - - - - -
    - - - - -
    """
    cells = np.zeros((5, 5), dtype=bool)
    cells[2, 2:3] = True  # path of player_id 0
    cells[0:1, 0] = True  # path of player_id 1
    player = spe_ed.Player(
        player_id=0, x=3, y=2, direction=spe_ed.directions[0], speed=1, active=True
    )  # pos: 3/2, direction: right
    opponents = [
        spe_ed.Player(player_id=1, x=0, y=1, direction=spe_ed.directions[1], speed=1, active=True)
    ]  # pos: 0/1 direction: down
    rounds = 1
    return (cells, player, opponents, rounds)


def default_almost_full_board():
    """ board state visualised: board size = 5x5
    - # # # #
    - # # # #
    # # - - #
    # # # # #
    # # # # #
    """
    cells = np.ones((5, 5), dtype=bool)
    cells[2, 2:3] = False  # free path for player_id 0
    cells[0:1, 0] = False  # free path for player_id 1
    player = spe_ed.Player(
        player_id=0, x=4, y=2, direction=spe_ed.directions[2], speed=1, active=True
    )  # pos: 4/2, direction: left
    opponents = [
        spe_ed.Player(player_id=1, x=0, y=2, direction=spe_ed.directions[3], speed=1, active=True)
    ]  # pos:0/2, direction: up
    rounds = 30
    return (cells, player, opponents, rounds)


class TestRandomHeuristic(unittest.TestCase):
    def test_random_output(self):
        """The heuristic should return a value between 0 and 1, independently of the input."""
        score = heuristics.RandomHeuristic().score(None, None, None, None, None)
        self.assertGreaterEqual(score, 0)
        self.assertLessEqual(score, 1)

        score = heuristics.RandomHeuristic().score(*default_round1_board())
        self.assertGreaterEqual(score, 0)
        self.assertLessEqual(score, 1)

        score = heuristics.RandomHeuristic().score(*default_round1_board())
        self.assertGreaterEqual(score, 0)
        self.assertLessEqual(score, 1)

        score = heuristics.RandomHeuristic().score(*default_almost_full_board())
        self.assertGreaterEqual(score, 0)
        self.assertLessEqual(score, 1)


class TestRegionHeuristic(unittest.TestCase):
    def test_empty_board(self):
        score = heuristics.RegionHeuristic().score(*empty_board())
        self.assertEqual(score, 1)

    def test_default_round1_board(self):
        score = heuristics.RegionHeuristic().score(*default_round1_board())
        self.assertEqual(score, 21.0 / 25.0)

    def test_default_almost_full_board(self):
        score = heuristics.RegionHeuristic().score(*default_almost_full_board())
        self.assertEqual(score, 2.0 / 25.0)

    def test_immutable_input(self):
        """Check if the heuristic modifies the input data itself."""
        board_state = default_round1_board()
        heuristics.RegionHeuristic().score(*board_state)
        self.assertEqual(board_state, default_round1_board())


class TestOpponentDistanceHeuristic(unittest.TestCase):
    def test_empty_board(self):
        score = heuristics.OpponentDistanceHeuristic(dist_threshold=16).score(*empty_board())
        self.assertEqual(score, 0)

    def test_default_round1_board(self):
        score = heuristics.OpponentDistanceHeuristic(dist_threshold=16).score(*default_round1_board())
        self.assertEqual(score, 4.0 / 10.0)

    def test_default_almost_full_board(self):
        score = heuristics.OpponentDistanceHeuristic(dist_threshold=16).score(*default_almost_full_board())
        self.assertEqual(score, 4.0 / 10.0)

    def test_immutable_input(self):
        """Check if the heuristic modifies the input data itself."""
        board_state = default_round1_board()
        heuristics.OpponentDistanceHeuristic().score(*board_state)
        self.assertEqual(board_state, default_round1_board())


class TestGeodesicVoronoiHeuristic(unittest.TestCase):
    def test_empty_board(self):
        score = heuristics.GeodesicVoronoiHeuristic(dist_threshold=16).score(*empty_board())
        self.assertEqual(score, 1.0)

    def test_default_round1_board(self):
        score = heuristics.GeodesicVoronoiHeuristic(dist_threshold=16).score(*default_round1_board())
        self.assertEqual(score, 11.0 / 25.0)

    def test_default_almost_full_board(self):
        score = heuristics.GeodesicVoronoiHeuristic(dist_threshold=16).score(*default_almost_full_board())
        self.assertEqual(score, 2.0 / 25.0)

    def test_immutable_input(self):
        """Check if the heuristic modifies the input data itself."""
        board_state = default_round1_board()
        heuristics.GeodesicVoronoiHeuristic().score(*board_state)
        self.assertEqual(board_state, default_round1_board())


class TestRandomProbingHeuristic(unittest.TestCase):
    def test_empty_board(self):
        """Evaluating the policy should not throw any error."""
        heuristics.RandomProbingHeuristic(heuristics.RegionHeuristic(), n_steps=5, n_probes=100).score(*empty_board())

    def test_default_round1_board(self):
        """Evaluating the policy should not throw any error."""
        heuristics.RandomProbingHeuristic(heuristics.RegionHeuristic(), n_steps=5,
                                          n_probes=100).score(*default_round1_board())

    def test_default_almost_full_board(self):
        """Evaluating the policy should not throw any error."""
        heuristics.RandomProbingHeuristic(heuristics.RegionHeuristic(), n_steps=5,
                                          n_probes=100).score(*default_almost_full_board())

    def test_immutable_input(self):
        """Check if the heuristic modifies the input data itself."""
        board_state = default_round1_board()
        heuristics.RandomProbingHeuristic(heuristic=heuristics.RandomHeuristic()).score(*board_state)
        self.assertEqual(board_state, default_round1_board())


class TestPathLengthHeuristic(unittest.TestCase):
    def test_empty_board(self):
        """Evaluating the policy should not throw any error."""
        score = heuristics.PathLengthHeuristic(n_steps=5, n_probes=10).score(*empty_board())
        self.assertGreater(score, 2.0 / 5.0)
        self.assertLessEqual(score, 1.0)

    def test_default_round1_board(self):
        """Evaluating the policy should not throw any error."""
        score = heuristics.PathLengthHeuristic(n_steps=5, n_probes=10).score(*default_round1_board())
        self.assertGreater(score, 2.0 / 5.0)
        self.assertLessEqual(score, 1.0)

    def test_default_almost_full_board(self):
        """Evaluating the policy should not throw any error."""
        score = heuristics.PathLengthHeuristic(n_steps=5, n_probes=10).score(*default_almost_full_board())
        self.assertEqual(score, 2.0 / 5.0)

    def test_immutable_input(self):
        """Check if the heuristic modifies the input data itself."""
        board_state = default_round1_board()
        heuristics.PathLengthHeuristic().score(*board_state)
        self.assertEqual(board_state, default_round1_board())


class TestCompositeHeuristic(unittest.TestCase):
    def test_normalized_output_value(self):
        """Should normalize weight values and return a value between 0 and 1."""
        score = heuristics.CompositeHeuristic(
            heuristics=[
                heuristics.RandomHeuristic(),
                heuristics.RandomHeuristic(),
                heuristics.RandomHeuristic(),
            ],
            weights=[1, 20000, 1000]
        ).score(*empty_board())
        self.assertGreaterEqual(score, 0.0)
        self.assertLessEqual(score, 1.0)

    def test_double_stacked_composites(self):
        """ Composite heuristic should be callable within other composite heuristic.
        Should return always a normalized result.
        """
        score = heuristics.CompositeHeuristic(
            heuristics=[
                heuristics.OpponentDistanceHeuristic(),
                heuristics.PathLengthHeuristic(n_steps=2, n_probes=2),
                heuristics.CompositeHeuristic(
                    heuristics=[
                        heuristics.RandomHeuristic(),
                        heuristics.PathLengthHeuristic(n_steps=2, n_probes=2),
                    ],
                    weights=[1, 2000]
                ),
            ]
        ).score(*default_round1_board())
        self.assertGreaterEqual(score, 0.0)
        self.assertLessEqual(score, 1.0)
