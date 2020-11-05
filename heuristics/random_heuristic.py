from heuristics.heuristic import Heuristic
import numpy as np


class RandomHeuristic(Heuristic):
    """Returns a random number."""
    def score(self, cells, player, opponents, rounds):
        """Returns a random number in range [0, 1] if the player is active, otherwise 0."""
        rnd = np.random.uniform()
        return (rnd, rnd)
