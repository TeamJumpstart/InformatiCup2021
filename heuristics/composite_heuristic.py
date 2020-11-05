from heuristics.heuristic import Heuristic
import numpy as np


class CompositeHeuristic(Heuristic):
    """Computes the distance sum to all players up to a threshold."""
    def __init__(self, heuristics, weights=None):
        """Initialize OpponentDistanceHeuristic.

        weights: Propability weights of each action. Pass `None` for uniform distribution.
        """
        self.heuristic = heuristics
        if weights is not None and len(weights) != len(heuristics):
            raise ValueError(f"Number of weights {weights} does mot match number of heuristics {heuristics}")
        if weights is None:
            self.weights = np.ones(len(heuristics)) / len(heuristics)
        else:
            self.weights = weights / np.sum(weights)

    def score(self, cells, player, opponents, rounds):
        """Computes the distance to all players."""
        return sum(
            weight * heuristic.score(cells, player, opponents, rounds)
            for heuristic, weight in zip(self.weights, self.heuristics)
        )
