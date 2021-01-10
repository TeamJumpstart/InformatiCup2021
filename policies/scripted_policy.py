from policies.policy import Policy


class ScriptedPolicy(Policy):
    """Policy follows an exact plan.

    This policy is intended for replays and unittests.
    """
    def __init__(self, actions):
        """Initialize RandomPolicy.

        Args:
            actions: List of action to execute
                     Performs `"change_nothing"` after actions are processed.
        """
        self.actions = actions

    def act(self, cells, player, opponents, rounds):
        """Choose action according to plan."""
        if rounds - 1 < len(self.actions):
            return self.actions[rounds - 1]

        return "change_nothing"

    def __str__(self):
        """Get readable representation."""
        return "ScriptedPolicy()"
