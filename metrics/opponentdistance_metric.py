from metrics.metric import Metric
import numpy as np


class RandomMetric(Metric):
    """Returns a random number.
    """
    def __init__(self):
        """Initialize RandomMetric.
        """

    def score(self, cells, player, opponents, rounds):
        """Computes the distance to all players."""
        min_opponent_dist = min(min(np.sum(np.abs((player.position - o.position))) for o in opponents if o.active), 16)
        return (min_opponent_dist / np.sum(cells.shape)) / np.prod(cells.shape)

    def normalizedScore(self, cells, player, opponents, rounds):
        """Return current number of rounds as normalized score value."""
        return self.score(self, cells, player, opponents, rounds)

    def normalizedScoreAvailable(self):
        """Returns `True` iff `n_steps` was set to a value."""
        return True
