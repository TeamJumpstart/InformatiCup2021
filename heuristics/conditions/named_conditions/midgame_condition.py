import numpy as np

from heuristics.conditions import CompositeCondition, Condition, OccupiedCellsCondition, PlayerInBiggestRegionCondition


class MidgameCondition(Condition):
    """ Check if we are in the Midgame."""
    def __init__(self):
        """Initialize MidgameCondition. """
        in_biggest_region_opening = PlayerInBiggestRegionCondition(opening_iterations=1)
        occupied_cells_percent = OccupiedCellsCondition()

        self.in_biggest_region_and_some_cells_occupied = CompositeCondition(
            [in_biggest_region_opening, occupied_cells_percent],
            thresholds=[True, 0.20],
            compare_op=[np.greater_equal, np.greater_equal],
        )

    def score(self, cells, player, opponents, rounds, deadline):
        """Return if the player is in the endgame phase."""
        return self.in_biggest_region_and_some_cells_occupied.score(cells, player, opponents, rounds, deadline)

    def __str__(self):
        """Get readable representation."""
        return "MidgameCondition()"
