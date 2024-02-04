import numpy as np

from heuristics.heuristic import Heuristic


class RandomHeuristic(Heuristic):
    """Returns a random number."""

    def score(self, cells, player, opponents, rounds, deadline):
        """Return a random number in range [0, 1]."""
        return np.random.uniform()

    def __str__(self):
        """Get readable representation."""
        return "RandomHeuristic()"
