from heuristics.conditions.condition import Condition


class RoundsCondition(Condition):
    """Returns number of rounds."""
    def __init__(self):
        """Initialize RoundsCondition. """

    def score(self, cells, player, opponents, rounds, deadline):
        """Return number of rounds."""
        return rounds

    def __str__(self):
        """Get readable representation."""
        return "RoundsCondition()"
