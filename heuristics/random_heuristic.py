from heuristics.heuristic import Heuristic
import numpy as np


class RandomHeuristic(Heuristic):
    """Returns a random number."""
    def __init__(self):
        """Initialize RandomMetric.
        """

    def score(self, cells, player, opponents, rounds):
        """Returns a random number in range [0, 1] if the player is active, otherwise 0."""
        return np.random.uniform() if player.active else 0

    def normalizedScore(self, cells, player, opponents, rounds):
        """Return current number of rounds as normalized score value."""
        return np.random.uniform() if player.active else 0

    def normalizedScoreAvailable(self):
        """Returns `True` iff `n_steps` was set to a value."""
        return True

    def earlyOutThreshold(self):
        """Returns 1.0."""
        return 1.0

    def normalizedEarlyOutThreshold(self):
        """Returns 1.0."""
        return 1.0
