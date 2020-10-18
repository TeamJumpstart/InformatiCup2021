import unittest
from policies import ScriptedPolicy


class TestScriptedPolicy(unittest.TestCase):
    def test_action_sequence(self):
        """Returns planned actions for the corresponding rounds."""
        pol = ScriptedPolicy(["turn_left", "turn_right", "slow_down", "speed_up", "change_nothing"])

        self.assertEqual(pol.act(None, None, None, 1), "turn_left")
        self.assertEqual(pol.act(None, None, None, 2), "turn_right")
        self.assertEqual(pol.act(None, None, None, 3), "slow_down")
        self.assertEqual(pol.act(None, None, None, 4), "speed_up")
        self.assertEqual(pol.act(None, None, None, 5), "change_nothing")

    def test_repeatability(self):
        """Action only depends on the round and not on internal state."""
        pol = ScriptedPolicy(["turn_left", "turn_right", "slow_down", "speed_up", "change_nothing"])

        self.assertEqual(pol.act(None, None, None, 1), "turn_left")
        self.assertEqual(pol.act(None, None, None, 1), "turn_left")
        self.assertEqual(pol.act(None, None, None, 5), "change_nothing")
        self.assertEqual(pol.act(None, None, None, 5), "change_nothing")

    def test_run_out(self):
        """Should return `"change_nothing"` after finishing the planned actions."""
        pol = ScriptedPolicy(["slow_down"])

        self.assertEqual(pol.act(None, None, None, 1), "slow_down")
        self.assertEqual(pol.act(None, None, None, 2), "change_nothing")
        self.assertEqual(pol.act(None, None, None, 3), "change_nothing")
        self.assertEqual(pol.act(None, None, None, 100), "change_nothing")

    def test_empty_plan(self):
        """Can handle an empoty action sequence."""
        pol = ScriptedPolicy([])

        self.assertEqual(pol.act(None, None, None, 1), "change_nothing")
