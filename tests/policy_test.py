import unittest

import policies
from environments import SimulatedSpe_edEnv
from heuristics import CompositeHeuristic, RandomHeuristic


def run_policy(env, pol):
    obs = env.reset()
    done = False
    while not done:
        action = pol.act(*obs)
        obs, _, done, _ = env.step(action)


class TestCirclePolicy(unittest.TestCase):
    def test_execution(self):
        """Executing the policy should not throw any error."""
        env = SimulatedSpe_edEnv(5, 5, [policies.CirclePolicy() for _ in range(5)])
        pol = policies.CirclePolicy()
        run_policy(env, pol)


class TestMazeWalkerPolicy(unittest.TestCase):
    def test_execution(self):
        """Executing the policy should not throw any error."""
        env = SimulatedSpe_edEnv(5, 5, [policies.MazeWalkerPolicy() for _ in range(5)])
        pol = policies.MazeWalkerPolicy()
        run_policy(env, pol)


class TestRandomPolicy(unittest.TestCase):
    def test_execution(self):
        """Executing the policy should not throw any error."""
        env = SimulatedSpe_edEnv(5, 5, [policies.RandomPolicy() for _ in range(5)])
        pol = policies.RandomPolicy()
        run_policy(env, pol)


class TestScriptedPolicy(unittest.TestCase):
    def test_execution(self):
        """Executing the policy should not throw any error."""
        env = SimulatedSpe_edEnv(5, 5, [policies.ScriptedPolicy([]) for _ in range(5)])
        pol = policies.ScriptedPolicy(["turn_left", "turn_right", "slow_down", "speed_up", "change_nothing"])

        run_policy(env, pol)

    def test_action_sequence(self):
        """Returns planned actions for the corresponding rounds."""
        pol = policies.ScriptedPolicy(["turn_left", "turn_right", "slow_down", "speed_up", "change_nothing"])

        self.assertEqual(pol.act(None, None, None, 1, None), "turn_left")
        self.assertEqual(pol.act(None, None, None, 2, None), "turn_right")
        self.assertEqual(pol.act(None, None, None, 3, None), "slow_down")
        self.assertEqual(pol.act(None, None, None, 4, None), "speed_up")
        self.assertEqual(pol.act(None, None, None, 5, None), "change_nothing")

    def test_repeatability(self):
        """Action only depends on the round and not on internal state."""
        pol = policies.ScriptedPolicy(["turn_left", "turn_right", "slow_down", "speed_up", "change_nothing"])

        self.assertEqual(pol.act(None, None, None, 1, None), "turn_left")
        self.assertEqual(pol.act(None, None, None, 1, None), "turn_left")
        self.assertEqual(pol.act(None, None, None, 5, None), "change_nothing")
        self.assertEqual(pol.act(None, None, None, 5, None), "change_nothing")

    def test_run_out(self):
        """Should return `"change_nothing"` after finishing the planned actions."""
        pol = policies.ScriptedPolicy(["slow_down"])

        self.assertEqual(pol.act(None, None, None, 1, None), "slow_down")
        self.assertEqual(pol.act(None, None, None, 2, None), "change_nothing")
        self.assertEqual(pol.act(None, None, None, 3, None), "change_nothing")
        self.assertEqual(pol.act(None, None, None, 100, None), "change_nothing")

    def test_empty_plan(self):
        """Can handle an empoty action sequence."""
        pol = policies.ScriptedPolicy([])

        self.assertEqual(pol.act(None, None, None, 1, None), "change_nothing")


class TestSpiralPolicy(unittest.TestCase):
    def test_execution(self):
        """Executing the policy should not throw any error."""
        env = SimulatedSpe_edEnv(5, 5, [policies.SpiralPolicy() for _ in range(5)])
        pol = policies.SpiralPolicy()
        run_policy(env, pol)


class TestHeuristicPolicy(unittest.TestCase):
    def test_execution(self):
        """Executing the policy should not throw any error."""
        env = SimulatedSpe_edEnv(5, 5, [policies.HeuristicPolicy(heuristic=RandomHeuristic()) for _ in range(5)])
        pol = policies.HeuristicPolicy(heuristic=RandomHeuristic())
        run_policy(env, pol)

    def test_composite_execution(self):
        env = SimulatedSpe_edEnv(5, 5, [policies.HeuristicPolicy(heuristic=RandomHeuristic()) for _ in range(5)])
        pol = policies.HeuristicPolicy(
            heuristic=CompositeHeuristic(
                [
                    RandomHeuristic(),
                    RandomHeuristic(),
                    CompositeHeuristic(
                        [
                            RandomHeuristic(),
                            RandomHeuristic(),
                        ]
                    ),
                ],
                weights=[1, 2, 3],
            )
        )
        run_policy(env, pol)


class TestNamedPolicies(unittest.TestCase):
    def test_loading(self):
        """Adam should be a heuristicPolicy."""
        pol = policies.load_named_policy("Adam")

        self.assertEqual(type(pol), policies.HeuristicPolicy)

    def test_new_instances(self):
        """Named polcies should be new instances."""
        pol1 = policies.load_named_policy("Adam")
        pol2 = policies.load_named_policy("Adam")

        self.assertFalse(pol1 is pol2)

    def test_name(self):
        """Named polcies have their proper names."""
        pol = policies.load_named_policy("Adam")

        self.assertEqual(str(pol), "Adam")
        self.assertEqual(repr(pol)[:20], "HeuristicPolicy(heur")  # ...
