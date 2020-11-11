import unittest
import heuristics
from environments import spe_ed
import numpy as np


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


def empty_board15x15_1player():
    """ board state visualised: board size = 15x15 """
    cells = np.zeros((15, 15), dtype=bool)
    player = spe_ed.Player(player_id=1, x=7, y=7, direction=spe_ed.directions[0], speed=1, active=True)
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


def empty_board15x15_2players():
    """ board state visualised: board size = 15x15 """
    cells = np.zeros((15, 15), dtype=bool)
    player = spe_ed.Player(player_id=1, x=14, y=14, direction=spe_ed.directions[3], speed=1, active=True)
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


def empty_board15x15_3players():
    """ board state visualised: board size = 15x15 """
    cells = np.zeros((15, 15), dtype=bool)
    player = spe_ed.Player(player_id=1, x=14, y=14, direction=spe_ed.directions[3], speed=1, active=True)
    opponents = [
        spe_ed.Player(player_id=2, x=0, y=0, direction=spe_ed.directions[1], speed=1, active=True),
        spe_ed.Player(player_id=3, x=0, y=14, direction=spe_ed.directions[1], speed=1, active=True)
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
        self.assertEqual(score, 22.0 / 25.0)

    def test_default_almost_full_board(self):
        score = heuristics.RegionHeuristic().score(*default_almost_full_board())
        self.assertEqual(score, 3.0 / 25.0)

    def test_immutable_input(self):
        """Check if the heuristic modifies the input data itself."""
        board_state = default_round1_board()
        heuristics.RegionHeuristic().score(*board_state)
        default_board_state = default_round1_board()
        self.assertTrue(np.array_equal(board_state[0], default_board_state[0]))
        self.assertEqual(board_state[1], default_board_state[1])
        self.assertEqual(board_state[2], default_board_state[2])
        self.assertEqual(board_state[3], default_board_state[3])


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
        default_board_state = default_round1_board()
        self.assertTrue(np.array_equal(board_state[0], default_board_state[0]))
        self.assertEqual(board_state[1], default_board_state[1])
        self.assertEqual(board_state[2], default_board_state[2])
        self.assertEqual(board_state[3], default_board_state[3])


class TestGeodesicVoronoiHeuristic(unittest.TestCase):
    def test_empty_board(self):
        score = heuristics.GeodesicVoronoiHeuristic(max_distance=16).score(*empty_board_1player())
        self.assertEqual(score, 1.0)

    def test_default_round1_board(self):
        score = heuristics.GeodesicVoronoiHeuristic(max_distance=16).score(*default_round1_board())
        self.assertEqual(score, 12.0 / 25.0)

    def test_default_almost_full_board(self):
        score = heuristics.GeodesicVoronoiHeuristic(max_distance=16).score(*default_almost_full_board())
        self.assertEqual(score, 3.0 / 25.0)

    def test_immutable_input(self):
        """Check if the heuristic modifies the input data itself."""
        board_state = default_round1_board()
        heuristics.GeodesicVoronoiHeuristic().score(*board_state)
        default_board_state = default_round1_board()
        self.assertTrue(np.array_equal(board_state[0], default_board_state[0]))
        self.assertEqual(board_state[1], default_board_state[1])
        self.assertEqual(board_state[2], default_board_state[2])
        self.assertEqual(board_state[3], default_board_state[3])


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
        default_board_state = default_round1_board()
        self.assertTrue(np.array_equal(board_state[0], default_board_state[0]))
        self.assertEqual(board_state[1], default_board_state[1])
        self.assertEqual(board_state[2], default_board_state[2])
        self.assertEqual(board_state[3], default_board_state[3])


class TestPathLengthHeuristic(unittest.TestCase):
    def test_empty_board(self):
        """Evaluating the policy should not throw any error."""
        score = heuristics.PathLengthHeuristic(n_steps=5, n_probes=10).score(*empty_board_1player())
        self.assertGreater(score, 2.0 / 5.0)
        self.assertLessEqual(score, 1.0)

    def test_default_round1_board(self):
        """Evaluating the policy should not throw any error."""
        score = heuristics.PathLengthHeuristic(n_steps=5, n_probes=10).score(*default_round1_board())
        self.assertGreater(score, 2.0 / 5.0)
        self.assertLessEqual(score, 1.0)

    def test_default_almost_full_board(self):
        """Evaluating the policy should not throw any error."""
        score = heuristics.PathLengthHeuristic(n_steps=5, n_probes=2).score(*default_almost_full_board())
        self.assertEqual(score, 2.0 / 5.0)

    def test_immutable_input(self):
        """Check if the heuristic modifies the input data itself."""
        board_state = default_round1_board()
        heuristics.PathLengthHeuristic(n_steps=5, n_probes=10).score(*board_state)
        default_board_state = default_round1_board()
        self.assertTrue(np.array_equal(board_state[0], default_board_state[0]))
        self.assertEqual(board_state[1], default_board_state[1])
        self.assertEqual(board_state[2], default_board_state[2])
        self.assertEqual(board_state[3], default_board_state[3])


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
                heuristics.PathLengthHeuristic(n_steps=2, n_probes=2),
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


class TestCloseRegionHeuristic(unittest.TestCase):
    def test_immutable_input(self):
        """Check if the heuristic modifies the input data itself."""
        board_state = default_round1_board()
        heuristics.CloseRegionHeuristic().score(*board_state)
        default_board_state = default_round1_board()
        self.assertTrue(np.array_equal(board_state[0], default_board_state[0]))
        self.assertEqual(board_state[1], default_board_state[1])
        self.assertEqual(board_state[2], default_board_state[2])
        self.assertEqual(board_state[3], default_board_state[3])

    def test_one_opening(self):
        """ board state schematicaly visualised: board size = 15x15
        2 - # - 3
        - - # - -
        - - - - -
        - - # - -
        - - # - 1
        """
        board_state_2players = empty_board15x15_2players()
        cells_2players = board_state_2players[0]
        cells_2players[7, 0:] = True
        cells_2players[7, 7] = False
        score = heuristics.CloseRegionHeuristic().score(*board_state_2players)
        self.assertEqual(score, 105.0 / 225.0)  # 105 = region size we are in, 225 = board size

        board_state_3players = empty_board15x15_3players()
        cells_3players = board_state_3players[0]
        cells_3players[7, 0:] = True
        cells_3players[7, 7] = False
        score = heuristics.CloseRegionHeuristic().score(*board_state_3players)
        self.assertEqual(score, 105.0 / 450.0)  # 105 = region size we are in, 450 = board size * 2 opponents

    def test_two_opening(self):
        """ board state schematicaly visualised: board size = 15x15
        2 - # - 3
        - - # - -
        - - - - -
        - - - - -
        - - # - 1
        """
        board_state_2players = empty_board15x15_2players()
        cells_2players = board_state_2players[0]
        cells_2players[7, 0:] = True
        cells_2players[7, 7:9] = False
        score = heuristics.CloseRegionHeuristic().score(*board_state_2players)
        self.assertEqual(score, 105.0 / 225.0)  # 105 = region size we are in, 225 = board size

        board_state_3players = empty_board15x15_3players()
        cells_3players = board_state_3players[0]
        cells_3players[7, 0:] = True
        cells_3players[7, 7:9] = False
        score = heuristics.CloseRegionHeuristic().score(*board_state_3players)
        self.assertEqual(score, 105.0 / 450.0)  # 105 = region size we are in, 450 = board size * 2 opponents

    def test_three_opening(self):
        """ board state schematicaly visualised: board size = 15x15
        2 - # - 3
        - - - - -
        - - - - -
        - - - - -
        - - # - 1
        """
        board_state_2players = empty_board15x15_2players()
        cells_2players = board_state_2players[0]
        cells_2players[7, 0:] = True
        cells_2players[7, 6:9] = False
        score = heuristics.CloseRegionHeuristic().score(*board_state_2players)
        self.assertEqual(score, 213.0 / 450.0)  # 213 = region size we are in, 450 = board size * 2 opponents

        board_state_3players = empty_board15x15_3players()
        cells_3players = board_state_3players[0]
        cells_3players[7, 0:] = True
        cells_3players[7, 6:9] = False
        score = heuristics.CloseRegionHeuristic().score(*board_state_3players)
        self.assertEqual(score, 213.0 / 675.0)  # 213 = region size we are in, 675 = board size * 3 opponents

    def test_diagonal_opening1(self):
        """ board state schematicaly visualised: board size = 15x15
        2 - # - 3
        - - # - -
        # - - - -
        - - - - -
        - - - - 1
        """
        board_state_2players = empty_board15x15_2players()
        cells_2players = board_state_2players[0]
        cells_2players[7, 0:7] = True
        cells_2players[0:6, 8] = True
        score = heuristics.CloseRegionHeuristic().score(*board_state_2players)
        self.assertEqual(score, 156.0 / 225.0)  # 156 = region size we are in, 225 = board size

        board_state_3players = empty_board15x15_3players()
        cells_3players = board_state_3players[0]
        cells_3players[7, 0:7] = True
        cells_3players[0:6, 8] = True
        score = heuristics.CloseRegionHeuristic().score(*board_state_3players)
        self.assertEqual(score, 156.0 / 450.0)  # 156 = region size we are in, 450 = board size * 2 opponents

    def test_diagonal_opening2(self):
        """ board state schematicaly visualised: board size = 15x15
        2 - # - 3
        - - # - -
        - - - - -
        # - - - -
        - - - - 1
        """
        board_state_2players = empty_board15x15_2players()
        cells_2players = board_state_2players[0]
        cells_2players[7, 0:7] = True
        cells_2players[0:7, 8] = True
        score = heuristics.CloseRegionHeuristic().score(*board_state_2players)
        self.assertEqual(score, 154.0 / 225.0)  # 154 = region size we are in, 225 = board size

        board_state_3players = empty_board15x15_3players()
        cells_3players = board_state_3players[0]
        cells_3players[7, 0:7] = True
        cells_3players[0:7, 8] = True
        score = heuristics.CloseRegionHeuristic().score(*board_state_3players)
        self.assertEqual(score, 154.0 / 450.0)  # 154 = region size we are in, 450 = board size * 2 opponents

    def test_one_opening_special(self):
        """ board state schematicaly visualised: board size = 15x15
        2 - # - 3
        - - # - -
        # - - - -
        - - # - -
        - - # - 1
        """
        board_state_2players = empty_board15x15_2players()
        cells_2players = board_state_2players[0]
        cells_2players[7, 0:] = True
        cells_2players[7, 7] = False
        cells_2players[0:6, 7] = True
        score = heuristics.CloseRegionHeuristic().score(*board_state_2players)
        self.assertEqual(score, 105.0 / 225.0)  # 105 = region size we are in, 225 = board size

        board_state_3players = empty_board15x15_3players()
        cells_3players = board_state_3players[0]
        cells_3players[7, 0:] = True
        cells_3players[7, 7] = False
        cells_3players[0:6, 7] = True
        score = heuristics.CloseRegionHeuristic().score(*board_state_3players)
        self.assertEqual(score, 105.0 / 450.0)  # 105 = region size we are in, 450 = board size * 2 opponents

    def test_three_opening_special(self):
        """ board state schematicaly visualised: board size = 15x15
        2 - # - 3
        - - - - -
        # - - - -
        - - - - -
        - - # - 1
        """
        board_state_2players = empty_board15x15_2players()
        cells_2players = board_state_2players[0]
        cells_2players[7, 0:] = True
        cells_2players[7, 6:9] = False
        cells_2players[0:6, 7] = True
        score = heuristics.CloseRegionHeuristic().score(*board_state_2players)
        self.assertEqual(score, 109.0 / 225.0)  # 109 = region size we are in, 225 = board size

        board_state_3players = empty_board15x15_3players()
        cells_3players = board_state_3players[0]
        cells_3players[7, 0:] = True
        cells_3players[7, 6:9] = False
        cells_3players[0:6, 7] = True
        score = heuristics.CloseRegionHeuristic().score(*board_state_3players)
        self.assertEqual(score, 109.0 / 450.0)  # 109 = region size we are in, 450 = board size * 2 opponents

    def test_wall_size1(self):
        """ board state schematicaly visualised: board size = 15x15
        2 - # - 3
        - - # - -
        - - # - -
        - - # - -
        - - # - 1
        """
        board_state_2players = empty_board15x15_2players()
        cells_2players = board_state_2players[0]
        cells_2players[7, 0:] = True
        score = heuristics.CloseRegionHeuristic().score(*board_state_2players)
        self.assertEqual(score, 105.0 / 225.0)  # 105 = region size we are in, 225 = board size

        board_state_3players = empty_board15x15_3players()
        cells_3players = board_state_3players[0]
        cells_3players[7, 0:] = True
        score = heuristics.CloseRegionHeuristic().score(*board_state_3players)
        self.assertEqual(score, 105.0 / 450.0)  # 105 = region size we are in, 225 = board size * 2 opponents

    def test_wall_size2(self):
        """ board state schematicaly visualised: board size = 15x15
        2 # # - 3
        - # # - -
        - # # - -
        - # # - -
        - # # - 1
        """
        board_state_2players = empty_board15x15_2players()
        cells_2players = board_state_2players[0]
        cells_2players[6:8, 0:] = True
        score = heuristics.CloseRegionHeuristic().score(*board_state_2players)
        self.assertEqual(score, 105.0 / 225.0)  # 105 = region size we are in, 225 = board size

        board_state_3players = empty_board15x15_3players()
        cells_3players = board_state_3players[0]
        cells_3players[6:8, 0:] = True
        score = heuristics.CloseRegionHeuristic().score(*board_state_3players)
        self.assertEqual(score, 105.0 / 450.0)  # 105 = region size we are in, 225 = board size * 2 opponents

    def test_wall_size3(self):
        """ board state schematicaly visualised: board size = 15x15
        2 # - # 3
        - # - # -
        - # - # -
        - # - # -
        - # - # 1
        """
        board_state_2players = empty_board15x15_2players()
        cells_2players = board_state_2players[0]
        cells_2players[6, 0:] = True
        cells_2players[8, 0:] = True
        score = heuristics.CloseRegionHeuristic().score(*board_state_2players)
        self.assertEqual(score, 90.0 / 225.0)  # 90 = region size we are in, 225 = board size

        board_state_3players = empty_board15x15_3players()
        cells_3players = board_state_3players[0]
        cells_3players[6, 0:] = True
        cells_3players[8, 0:] = True
        score = heuristics.CloseRegionHeuristic().score(*board_state_3players)
        self.assertEqual(score, 90.0 / 450.0)  # 90 = region size we are in, 225 = board size * 2 opponents

    def test_wall_size4(self):
        """ board state schematicaly visualised: board size = 15x15
        2 # - # 3
        - # - # -
        - # - # -
        - # - # -
        - # - # 1
        """
        board_state_2players = empty_board15x15_2players()
        cells_2players = board_state_2players[0]
        cells_2players[6, 0:] = True
        cells_2players[9, 0:] = True
        score = heuristics.CloseRegionHeuristic().score(*board_state_2players)
        self.assertEqual(score, 75.0 / 225.0)  # 75 = region size we are in, 225 = board size

        board_state_3players = empty_board15x15_3players()
        cells_3players = board_state_3players[0]
        cells_3players[6, 0:] = True
        cells_3players[9, 0:] = True
        score = heuristics.CloseRegionHeuristic().score(*board_state_3players)
        self.assertEqual(score, 75.0 / 450.0)  # 75 = region size we are in, 225 = board size * 2 opponents

    def test_wall_size5(self):
        """ board state schematicaly visualised: board size = 15x15
        2 # - # 3
        - # - # -
        - # - # -
        - # - # -
        - # - # 1
        """
        board_state_2players = empty_board15x15_2players()
        cells_2players = board_state_2players[0]
        cells_2players[6, 0:] = True
        cells_2players[10, 0:] = True
        score = heuristics.CloseRegionHeuristic().score(*board_state_2players)
        self.assertEqual(score, 60.0 / 225.0)  # 60 = region size we are in, 225 = board size

        board_state_3players = empty_board15x15_3players()
        cells_3players = board_state_3players[0]
        cells_3players[6, 0:] = True
        cells_3players[10, 0:] = True
        score = heuristics.CloseRegionHeuristic().score(*board_state_3players)
        self.assertEqual(score, 60.0 / 450.0)  # 60 = region size we are in, 225 = board size * 2 opponents
