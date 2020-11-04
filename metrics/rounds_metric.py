from metrics.metric import Metric


class RoundsMetric(Metric):
    """Metric that return the number of rounds.
    """
    def __init__(self, n_steps=None):
        """Initialize RoundsMetric."""
        self.threshold = n_steps

    def score(self, cells, player, opponents, rounds):
        """Return current number of rounds as score value."""
        if player.active:
            return rounds
        else:
            return rounds - 1

    def normalizedScore(self, cells, player, opponents, rounds):
        """Return current number of rounds as normalized score value."""
        if self.threshold is None:
            return 0
        else:
            if player.active:
                return rounds / self.threshold
            else:
                return (rounds - 1) / self.threshold

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
