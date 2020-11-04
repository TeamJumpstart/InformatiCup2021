from heuristics.heuristic import Heuristic


class RoundsHeuristic(Heuristic):
    """Heuristic that return the number of rounds.
    """
    def __init__(self, n_steps=None):
        """Initialize RoundsHeuristic."""
        self.threshold = n_steps

    def score(self, cells, player, opponents, rounds):
        """Return current number of rounds as score value."""
        if player.active:
            return rounds
        else:
            return 0

    def normalizedScore(self, cells, player, opponents, rounds):
        """Return current number of rounds as normalized score value."""
        if self.threshold is None:
            return 0
        else:
            if player.active:
                return rounds / self.threshold
            else:
                return 0

    def normalizedScoreAvailable(self):
        """Returns `True` iff `n_steps` was set to a value."""
        if self.threshold is None:
            return False
        else:
            return True

    def earlyOutThreshold(self):
        """Returns `n_steps` iff `n_steps` was initialized to a value."""
        if self.threshold is None:
            return float('inf')
        else:
            return self.threshold

    def normalizedEarlyOutThreshold(self):
        """Returns 1.0."""
        if self.threshold is None:
            return 1.0
        else:
            return 1.0
