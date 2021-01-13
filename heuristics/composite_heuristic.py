from heuristics.heuristic import Heuristic
import numpy as np


class CompositeHeuristic(Heuristic):
    """Allows to combine multiple heuristics into a single score evaluating the same board state."""
    def __init__(self, heuristics, weights=None):
        """Initialize OpponentDistanceHeuristic.

        Args:
            heuristics: An array containing different `Heuristics` which should be evaluated in combination.
            weights: Weighting of each `Heuristic`. Pass `None` for uniform weighting.
        """
        self.heuristics = heuristics
        if weights is not None and len(weights) != len(heuristics):
            raise ValueError(f"Number of weights {weights} does mot match number of heuristics {heuristics}")
        if weights is None:
            self.weights = np.ones(len(heuristics)) / len(heuristics)
        else:
            self.weights = weights / np.sum(weights)

    def score(self, cells, player, opponents, rounds):
        """Compute the combined heuristic score."""
        return sum(
            weight * heuristic.score(cells, player, opponents, rounds)
            for weight, heuristic in zip(self.weights, self.heuristics)
        )

    def __str__(self):
        """Get readable representation."""
        return "CompositeHeuristic(" + \
            f"[{','.join([str(heuristic) for heuristic in self.heuristics])}], " + \
            f"weights={self.weights}, " + \
            ")"
