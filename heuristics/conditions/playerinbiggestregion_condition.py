from heuristics.conditions import Condition
from scipy.ndimage import morphology
from scipy import ndimage
import numpy as np


class PlayerInBiggestRegionCondition(Condition):
    """ Checks if the controlled player is in the biggest contiguous region of all players."""
    def __init__(self, opening_iterations=0):
        """Initialize PlayerInBiggestRegionCondition.

        Args:
            opening_iterations: number of performed closing operations on the cell state before the computation
                of the regions to ommit smaller regions. default: 0
        """
        self.opening_iterations = opening_iterations

    def score(self, cells, player, opponents, rounds, deadline):
        """Compute the size of all players regions and check if we are in the biggest one. """
        # close all 1 cell wide openings aka "articulating points"
        if self.opening_iterations:
            cells = morphology.binary_opening(cells, iterations=self.opening_iterations)

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
        # Compute the sizes and divide by numbers of players in each region
        region_sizes = np.array([np.sum(labelled == region) for region in regions])

        # Check if player region is the biggest one
        return region_sizes[0] == max(region_sizes)

    def __str__(self):
        """Get readable representation."""
        return "PlayerInBiggestRegionCondition(" + \
            f"closing_iterations={self.opening_iterations}, " + \
            ")"
