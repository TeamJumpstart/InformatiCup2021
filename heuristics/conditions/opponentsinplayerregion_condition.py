from heuristics.conditions.condition import Condition
import numpy as np
from scipy import ndimage
from scipy.ndimage import morphology


class OpponentsInPlayerRegionCondition(Condition):
    """Returns number of opponents in the players region."""
    def __init__(self, closing=0):
        """Initialize OpponentsInPlayerRegionCondition. """
        self.closing_iterations = closing

    def score(self, cells, player, opponents, rounds):
        """Return number of opponents in own region."""

        if self.closing_iterations:
            cells = morphology.binary_closing(cells, iterations=self.closing_iterations)

        players = [player] + opponents

        # inverse map (mask occupied cells)
        empty = cells == 0
        # Clear cell we're in and for all active opponents
        for p in players:
            empty[p.y, p.x] = True

        # compute distinct regions
        labelled, _ = ndimage.label(empty)

        # Get the region each player is in
        regions = np.array([labelled[p.y, p.x] for p in players])

        # return number of opponents
        return np.sum(regions == regions[0]) - 1

    def __str__(self):
        """Get readable representation."""
        return f"OpponentsInPlayerRegionCondition(iterations={self.closing_iterations})"
