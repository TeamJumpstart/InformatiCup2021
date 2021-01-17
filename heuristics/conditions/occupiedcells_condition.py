from heuristics.conditions.condition import Condition
import numpy as np


class OccupiedCellsCondition(Condition):
    """Returns number of rounds."""
    def __init__(self):
        """Initialize OccupiedCellsCondition. """

    def score(self, cells, player, opponents, rounds, deadline):
        """Return number of rounds."""
        return np.sum(cells) / np.prod(cells.shape)

    def __str__(self):
        """Get readable representation."""
        return "OccupiedCellsCondition()"
