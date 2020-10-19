import unittest
from numpy.testing import assert_array_equal
from environments import SimulatedSpe_edEnv
from environments.spe_ed import directions


class TestSimulator(unittest.TestCase):
    def test_step_change_nothing(self):
        env = SimulatedSpe_edEnv(5, 5, [], seed=1)  # Seed places the player in the middle
        env.reset()
        env.players[0].direction = directions[0]  # Make player facing right

        env.step("change_nothing")

        assert_array_equal(
            env.cells, [
                [0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0],
                [0, 0, 1, 1, 0],
                [0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0],
            ]
        )
        self.assertTrue(env.players[0].active)  # Player lives
        self.assertEqual(env.players[0].speed, 1)  # Speed does not change
        self.assertEqual(env.players[0].direction.name, "right")  # Direction does not change

    def test_step_speed_up(self):
        env = SimulatedSpe_edEnv(5, 5, [], seed=1)  # Seed places the player in the middle
        env.reset()
        env.players[0].direction = directions[0]  # Make player facing right

        env.step("speed_up")

        assert_array_equal(
            env.cells, [
                [0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0],
                [0, 0, 1, 1, 1],
                [0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0],
            ]
        )
        self.assertTrue(env.players[0].active)  # Player lives
        self.assertEqual(env.players[0].speed, 2)  # Speed increases
        self.assertEqual(env.players[0].direction.name, "right")  # Direction does not change

    def test_step_slow_down(self):
        env = SimulatedSpe_edEnv(5, 5, [], seed=1)  # Seed places the player in the middle
        env.reset()
        env.players[0].direction = directions[0]  # Make player facing right

        env.step("slow_down")

        assert_array_equal(
            env.cells, [
                [0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0],
                [0, 0, 1, 0, 0],
                [0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0],
            ]
        )
        self.assertFalse(env.players[0].active)  # Player dies
        self.assertEqual(env.players[0].speed, 1)  # Speed does not change
        self.assertEqual(env.players[0].direction.name, "right")  # Direction does not change

    def test_step_turn_left(self):
        env = SimulatedSpe_edEnv(5, 5, [], seed=1)  # Seed places the player in the middle
        env.reset()
        env.players[0].direction = directions[0]  # Make player facing right

        env.step("turn_left")

        assert_array_equal(
            env.cells, [
                [0, 0, 0, 0, 0],
                [0, 0, 1, 0, 0],
                [0, 0, 1, 0, 0],
                [0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0],
            ]
        )
        self.assertTrue(env.players[0].active)  # Player lives
        self.assertEqual(env.players[0].speed, 1)  # Speed does not change
        self.assertEqual(env.players[0].direction.name, "up")

    def test_step_turn_right(self):
        env = SimulatedSpe_edEnv(5, 5, [], seed=1)  # Seed places the player in the middle
        env.reset()
        env.players[0].direction = directions[0]  # Make player facing right

        env.step("turn_right")

        assert_array_equal(
            env.cells, [
                [0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0],
                [0, 0, 1, 0, 0],
                [0, 0, 1, 0, 0],
                [0, 0, 0, 0, 0],
            ]
        )
        self.assertTrue(env.players[0].active)  # Player lives
        self.assertEqual(env.players[0].speed, 1)  # Speed does not change
        self.assertEqual(env.players[0].direction.name, "down")
