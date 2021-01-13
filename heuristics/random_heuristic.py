from heuristics.heuristic import Heuristic
import numpy as np


class RandomHeuristic(Heuristic):
    """Returns a random number."""
    def score(self, cells, player, opponents, rounds):
        """Returns a random number in range [0, 1]"""
        return np.random.uniform()

    def __str__(self):
        """Get readable representation."""
        return "RandomHeuristic()"
