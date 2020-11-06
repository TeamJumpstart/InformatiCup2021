import unittest
from environments import SimulatedSpe_edEnv
import policies


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


class TestFutureStepsPolicy(unittest.TestCase):
    def test_execution(self):
        """Executing the policy should not throw any error."""
        env = SimulatedSpe_edEnv(5, 5, [policies.FutureStepsPolicy() for _ in range(5)])
        pol = policies.FutureStepsPolicy()

        run_policy(env, pol)

    def test_execution_dynamic(self):
        """Executing the policy should not throw any error."""
        env = SimulatedSpe_edEnv(5, 5, [policies.FutureStepsPolicy(dynamic=True) for _ in range(5)])
        pol = policies.FutureStepsPolicy(dynamic=True)

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

        self.assertEqual(pol.act(None, None, None, 1), "turn_left")
        self.assertEqual(pol.act(None, None, None, 2), "turn_right")
        self.assertEqual(pol.act(None, None, None, 3), "slow_down")
        self.assertEqual(pol.act(None, None, None, 4), "speed_up")
        self.assertEqual(pol.act(None, None, None, 5), "change_nothing")

    def test_repeatability(self):
        """Action only depends on the round and not on internal state."""
        pol = policies.ScriptedPolicy(["turn_left", "turn_right", "slow_down", "speed_up", "change_nothing"])

        self.assertEqual(pol.act(None, None, None, 1), "turn_left")
        self.assertEqual(pol.act(None, None, None, 1), "turn_left")
        self.assertEqual(pol.act(None, None, None, 5), "change_nothing")
        self.assertEqual(pol.act(None, None, None, 5), "change_nothing")

    def test_run_out(self):
        """Should return `"change_nothing"` after finishing the planned actions."""
        pol = policies.ScriptedPolicy(["slow_down"])

        self.assertEqual(pol.act(None, None, None, 1), "slow_down")
        self.assertEqual(pol.act(None, None, None, 2), "change_nothing")
        self.assertEqual(pol.act(None, None, None, 3), "change_nothing")
        self.assertEqual(pol.act(None, None, None, 100), "change_nothing")

    def test_empty_plan(self):
        """Can handle an empoty action sequence."""
        pol = policies.ScriptedPolicy([])

        self.assertEqual(pol.act(None, None, None, 1), "change_nothing")


class TestSpiralPolicy(unittest.TestCase):
    def test_execution(self):
        """Executing the policy should not throw any error."""
        env = SimulatedSpe_edEnv(5, 5, [policies.SpiralPolicy() for _ in range(5)])
        pol = policies.SpiralPolicy()
        run_policy(env, pol)
