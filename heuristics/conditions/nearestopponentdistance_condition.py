from heuristics.conditions import Condition
from scipy.ndimage import morphology
from scipy import ndimage
import numpy as np


class NearestOpponentDistanceCondition(Condition):
    """ Computes the player region size."""
    def __init__(self, opening_iterations=0):
        """Initialize NearestOpponentDistanceCondition.

        Args:
            closing_iterations: number of performed closing operations on the cell state before the computation
                of the regions to ommit smaller regions. default: 0
        """
        self.opening_iterations = opening_iterations

    def score(self, cells, player, opponents, rounds, deadline):
        """Compute the relative size of the region we're in."""
        # close all 1 cell wide openings aka "articulating points"
        if self.opening_iterations:
            cells = morphology.binary_opening(cells, iterations=self.opening_iterations)

        players = [player] + opponents

        # inverse map (mask occupied cells)
        empty = cells == 0
        # Clear cell for all active players
        for p in players:
            empty[p.y, p.x] = True

        # compute distinct regions
        labelled, _ = ndimage.label(empty)

        # Get the region each player is in
        player_regions = np.array([labelled[p.y, p.x] for p in players])

        def dist(pos1, pos2):
            return sum(np.abs(pos1 - pos2))

        min_distance = min(
            dist(player.position, p.position) for p in opponents if labelled[p.y, p.x] == player_regions[0]
        )

        # sum player region size divided by the board size, score in [0..1]
        return min_distance

    def __str__(self):
        """Get readable representation."""
        return "RegionCondition(" + \
            f"closing_iterations={self.closing_iterations}, " + \
            ")"
