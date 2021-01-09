import unittest
from numpy.testing import assert_array_equal
import heuristics
import numpy as np
from environments import spe_ed


def empty_board_1player():
    """ board state visualised: board size = 5x5
    - - - - -
    - - - - -
    - - 1 - -
    - - - - -
    - - - - -
    """
    cells = np.zeros((5, 5), dtype=bool)
    player = spe_ed.Player(player_id=1, x=2, y=2, direction=spe_ed.directions[0], speed=1, active=True)
    opponents = []
    rounds = 0
    return (cells, player, opponents, rounds)


def empty_board_2players():
    """ board state visualised: board size = 5x5
    2 - - - -
    - - - - -
    - - - - -
    - - - - -
    - - - - 1
    """
    cells = np.zeros((5, 5), dtype=bool)
    player = spe_ed.Player(player_id=1, x=4, y=4, direction=spe_ed.directions[3], speed=1, active=True)
    opponents = [spe_ed.Player(player_id=2, x=0, y=0, direction=spe_ed.directions[1], speed=1, active=True)]
    rounds = 0
    return (cells, player, opponents, rounds)


def empty_board_3players():
    """ board state visualised: board size = 5x5
    2 - - - 3
    - - - - -
    - - - - -
    - - - - -
    - - - - 1
    """
    cells = np.zeros((5, 5), dtype=bool)
    player = spe_ed.Player(player_id=1, x=4, y=4, direction=spe_ed.directions[3], speed=1, active=True)
    opponents = [
        spe_ed.Player(player_id=2, x=0, y=0, direction=spe_ed.directions[1], speed=1, active=True),
        spe_ed.Player(player_id=3, x=0, y=4, direction=spe_ed.directions[1], speed=1, active=True)
    ]
    rounds = 0
    return (cells, player, opponents, rounds)


def default_round1_board():
    """ board state visualised: board size = 5x5
    # - - - -
    2 - - - -
    - - # 1 -
    - - - - -
    - - - - -
    """
    cells = np.zeros((5, 5), dtype=bool)
    cells[2, 2:4] = True  # path of player_id 1
    cells[0:2, 0] = True  # path of player_id 2
    player = spe_ed.Player(
        player_id=1, x=3, y=2, direction=spe_ed.directions[0], speed=1, active=True
    )  # pos: 3/2, direction: right
    opponents = [
        spe_ed.Player(player_id=2, x=0, y=1, direction=spe_ed.directions[1], speed=1, active=True)
    ]  # pos: 0/1 direction: down
    rounds = 1
    return (cells, player, opponents, rounds)


def default_almost_full_board():
    """ board state visualised: board size = 5x5
    - # # # #
    - # # # #
    2 # - - 1
    # # # # #
    # # # # #
    """
    cells = np.ones((5, 5), dtype=bool)
    cells[2, 2:4] = False  # free path for player_id 1
    cells[0:2, 0] = False  # free path for player_id 2
    player = spe_ed.Player(
        player_id=1, x=4, y=2, direction=spe_ed.directions[2], speed=1, active=True
    )  # pos: 4/2, direction: left
    opponents = [
        spe_ed.Player(player_id=2, x=0, y=2, direction=spe_ed.directions[3], speed=1, active=True)
    ]  # pos:0/2, direction: up
    rounds = 30
    return (cells, player, opponents, rounds)


class TestRandomHeuristic(unittest.TestCase):
    def test_random_output(self):
        """The heuristic should return a value between 0 and 1, independently of the input."""
        score = heuristics.RandomHeuristic().score(None, None, None, None)
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
        score = heuristics.RegionHeuristic().score(*empty_board_1player())
        self.assertEqual(score, 1)

    def test_default_round1_board(self):
        score = heuristics.RegionHeuristic().score(*default_round1_board())
        self.assertEqual(score, (23 / 2 / 25) * (1 - 23 / 2 / 25))

        score = heuristics.RegionHeuristic(include_opponent_regions=False).score(*default_round1_board())
        self.assertEqual(score, (23 / 2 / 25))

    def test_default_almost_full_board(self):
        score = heuristics.RegionHeuristic().score(*default_almost_full_board())
        self.assertEqual(score, (3 / 1 / 25) * (1 - 3 / 1 / 25))

        score = heuristics.RegionHeuristic(include_opponent_regions=False).score(*default_almost_full_board())
        self.assertEqual(score, (3 / 1 / 25))

    def test_immutable_input(self):
        """Check if the heuristic modifies the input data itself."""
        board_state = default_round1_board()
        heuristics.RegionHeuristic().score(*board_state)

        assert_array_equal(board_state[0], default_round1_board()[0])


class TestOpponentDistanceHeuristic(unittest.TestCase):
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

        assert_array_equal(board_state[0], default_round1_board()[0])


class TestVoronoiHeuristic(unittest.TestCase):
    def test_empty_board(self):
        score = heuristics.VoronoiHeuristic(max_steps=16, opening_iterations=0).score(*empty_board_1player())
        self.assertEqual(score, 1.0)

    def test_default_round1_board(self):
        score = heuristics.VoronoiHeuristic(max_steps=16, opening_iterations=0).score(*default_round1_board())
        self.assertEqual(score, 12.0 / 25.0)

    def test_default_almost_full_board(self):
        score = heuristics.VoronoiHeuristic(max_steps=16, opening_iterations=0).score(*default_almost_full_board())
        self.assertEqual(score, 3.0 / 25.0)

    def test_immutable_input(self):
        """Check if the heuristic modifies the input data itself."""
        board_state = default_round1_board()
        heuristics.VoronoiHeuristic(max_steps=16, opening_iterations=0).score(*board_state)

        assert_array_equal(board_state[0], default_round1_board()[0])


class TestRandomProbingHeuristic(unittest.TestCase):
    def test_empty_board(self):
        """Evaluating the policy should not throw any error."""
        heuristics.RandomProbingHeuristic(heuristics.RegionHeuristic(), n_steps=5,
                                          n_probes=100).score(*empty_board_1player())

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
        heuristics.RandomProbingHeuristic(heuristic=heuristics.RandomHeuristic(), n_steps=5,
                                          n_probes=10).score(*board_state)

        assert_array_equal(board_state[0], default_round1_board()[0])


class TestPathLengthHeuristic(unittest.TestCase):
    def test_small_board(self):
        """ board state visualised: board size = 1x3
        1 - - 
        """
        cells = np.zeros((1, 3), dtype=bool)
        player = spe_ed.Player(player_id=1, x=0, y=0, direction=spe_ed.directions[0], speed=1, active=True)
        opponents = []
        rounds = 0

        score = heuristics.PathLengthHeuristic(n_steps=1).score(cells, player, opponents, rounds)
        self.assertEqual(score, 1.0)

        score = heuristics.PathLengthHeuristic(n_steps=2).score(cells, player, opponents, rounds)
        self.assertEqual(score, 1.0)

        score = heuristics.PathLengthHeuristic(n_steps=3).score(cells, player, opponents, rounds)
        self.assertEqual(score, 2 / 3)

    def test_small_board2(self):
        """ board state visualised: board size = 1x3
        1 - - 
        - - -
        """
        cells = np.zeros((2, 3), dtype=bool)
        player = spe_ed.Player(player_id=1, x=0, y=0, direction=spe_ed.directions[0], speed=1, active=True)
        opponents = []
        rounds = 0

        score = heuristics.PathLengthHeuristic(n_steps=5).score(cells, player, opponents, rounds)
        self.assertEqual(score, 1.0)

        score = heuristics.PathLengthHeuristic(n_steps=6).score(cells, player, opponents, rounds)
        self.assertEqual(score, 1.0)

        score = heuristics.PathLengthHeuristic(n_steps=7).score(cells, player, opponents, rounds)
        self.assertEqual(score, 6 / 7)

    def test_empty_board(self):
        board_state = empty_board_1player()

        score = heuristics.PathLengthHeuristic(n_steps=5).score(*board_state)
        self.assertEqual(score, 1.0)

        score = heuristics.PathLengthHeuristic(n_steps=10).score(*board_state)
        self.assertEqual(score, 1.0)

        # score = heuristics.PathLengthHeuristic(n_steps=25).score(*board_state)
        # self.assertEqual(score, 1.0)

    def test_default_round1_board(self):
        score = heuristics.PathLengthHeuristic(n_steps=5).score(*default_round1_board())
        self.assertGreater(score, 2.0 / 5.0)
        self.assertLessEqual(score, 1.0)

    def test_default_almost_full_board(self):
        score = heuristics.PathLengthHeuristic(n_steps=5).score(*default_almost_full_board())
        self.assertEqual(score, 2.0 / 5.0)

    def test_immutable_input(self):
        """Check if the heuristic modifies the input data itself."""
        board_state = default_round1_board()
        heuristics.PathLengthHeuristic(n_steps=5).score(*board_state)

        assert_array_equal(board_state[0], default_round1_board()[0])


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
        ).score(*empty_board_1player())
        self.assertGreaterEqual(score, 0.0)
        self.assertLessEqual(score, 1.0)

    def test_default_weights(self):
        """ Composite heuristic should be callable within other composite heuristic.
        Should return always a normalized result.
        """
        score = heuristics.CompositeHeuristic(
            heuristics=[
                heuristics.OpponentDistanceHeuristic(),
                heuristics.PathLengthHeuristic(n_steps=2),
            ]
        ).score(*default_round1_board())
        self.assertGreaterEqual(score, 0.0)
        self.assertLessEqual(score, 1.0)

    def test_double_stacked_composites(self):
        """ Composite heuristic should be callable within other composite heuristic.
        Should return always a normalized result.
        """
        score = heuristics.CompositeHeuristic(
            heuristics=[
                heuristics.OpponentDistanceHeuristic(),
                heuristics.PathLengthHeuristic(n_steps=2),
                heuristics.CompositeHeuristic(
                    heuristics=[
                        heuristics.RandomHeuristic(),
                        heuristics.PathLengthHeuristic(n_steps=2),
                    ],
                    weights=[1, 2000]
                ),
            ]
        ).score(*default_round1_board())
        self.assertGreaterEqual(score, 0.0)
        self.assertLessEqual(score, 1.0)
