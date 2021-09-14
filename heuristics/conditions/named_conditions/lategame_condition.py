import numpy as np

from heuristics.conditions import (
    CompositeCondition, Condition, NearestOpponentDistanceCondition, OccupiedCellsCondition,
    PlayerInBiggestRegionCondition
)


class LategameCondition(Condition):
    """ Check if we are in the LategameCondition."""
    def __init__(self):
        """Initialize LategameConditionCondition. """
        in_biggest_region = PlayerInBiggestRegionCondition()
        nearest_distance_to_opponent = NearestOpponentDistanceCondition(opening_iterations=1)
        occupied_cells_percent = OccupiedCellsCondition()

        self.lategame_cond = CompositeCondition(
            [in_biggest_region, nearest_distance_to_opponent, occupied_cells_percent],
            thresholds=[True, 10, 0.2],
            compare_op=[np.greater_equal, np.greater_equal, np.greater_equal]
        )

    def score(self, cells, player, opponents, rounds, deadline):
        """Return if the player is in the Lategame phase."""
        return self.lategame_cond.score(cells, player, opponents, rounds, deadline)

    def __str__(self):
        """Get readable representation."""
        return "LategameCondition()"
