from numpy import np
from scipy import ndimage
from scipy.ndimage import morphology

from heuristics.conditions import Condition


class RegionCondition(Condition):
    """ Computes the player region size."""
    def __init__(self, closing_iterations=0):
        """Initialize RegionCondition.

        Args:
            closing_iterations: number of performed closing operations on the cell state before the computation
                of the regions to ommit smaller regions. default: 0
        """
        self.closing_iterations = closing_iterations

    def score(self, cells, player, opponents, rounds, deadline):
        """Compute the relative size of the region we're in."""
        # close all 1 cell wide openings aka "articulating points"
        if self.closing_iterations:
            cells = morphology.binary_closing(cells, iterations=self.closing_iterations)

        players = [player] + opponents

        # inverse map (mask occupied cells)
        empty = cells == 0
        # Clear cell for all active players
        for p in players:
            empty[p.y, p.x] = True

        # compute distinct regions
        labelled, _ = ndimage.label(empty)

        # get player region label
        player_region = labelled[player.y, player.x]

        # sum player region size divided by the board size, score in [0..1]
        return sum(labelled == player_region) / np.prod(labelled.shape)

    def __str__(self):
        """Get readable representation."""
        return "RegionCondition(" + \
            f"closing_iterations={self.closing_iterations}, " + \
            ")"
