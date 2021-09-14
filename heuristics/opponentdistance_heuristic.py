import numpy as np

from heuristics.heuristic import Heuristic


class OpponentDistanceHeuristic(Heuristic):
    """Computes the distance sum to all players up to a threshold."""
    def __init__(self, dist_threshold=16):
        """Initialize OpponentDistanceHeuristic."""
        self.dist_threshold = dist_threshold

    def score(self, cells, player, opponents, rounds, deadline):
        """Computes the distance to all players."""
        min_opponent_dist = min(
            min(np.sum(np.abs((player.position - o.position))) for o in opponents if o.active), self.dist_threshold
        )
        return min_opponent_dist / np.sum(cells.shape)

    def __str__(self):
        """Get readable representation."""
        return "OpponentDistanceHeuristic(" + \
            f"dist_threshold={self.dist_threshold}" + \
            ")"
