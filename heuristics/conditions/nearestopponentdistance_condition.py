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
        if not player.active:
            return 0

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

        # check if we are the only one in region
        if sum(player_regions == player_regions[0]) < 2:
            return float("Inf")

        def dist(pos1, pos2):
            return sum(np.abs(pos1 - pos2))

        # compute the distance to the nearest opponent in our region
        min_distance = min(
            dist(player.position, o.position) for o in opponents if labelled[o.y, o.x] == player_regions[0]
        )

        # return minimal distance
        return min_distance

    def __str__(self):
        """Get readable representation."""
        return "RegionCondition(" + \
            f"closing_iterations={self.closing_iterations}, " + \
            ")"
