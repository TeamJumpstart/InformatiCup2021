from heuristics.heuristic import Heuristic
from scipy.ndimage import morphology
from scipy import ndimage
import numpy as np


class RegionHeuristic(Heuristic):
    def __init__(self, closing_iterations=0):
        """Initialize RegionHeuristic.

        Args:
            closing_iterations: number of performed closing operations on the cell state before the computation
                of the regions to ommit smaller regions. default: 0
        """
        self.iterations = closing_iterations

    def score(self, cells, player, opponents, rounds):
        """Compute the relative size of the region we're in."""
        empty = cells.copy()

        # close all 1 cell wide openings aka "articulating points"
        if self.iterations:
            empty = np.pad(empty, (1, ))
            empty = morphology.binary_closing(empty, iterations=self.iterations)
            empty = empty[1:-1, 1:-1]

        # inverse map (mask occupied cells)
        empty = empty == 0
        # Clear cell we're in and for all active opponents
        empty[player.y, player.x] = True
        for o in opponents:
            if o.active:
                empty[o.y, o.x] = True

        # compute distinct regions
        labelled, _ = ndimage.label(empty)

        # Get the region we're in and compute its size
        region = labelled[player.y, player.x]
        region_size = np.sum(labelled == region)

        # Compute the number of opponents in our region with which we fight in the region
        opponents_in_region = sum([labelled[o.y, o.x] == region for o in opponents if o.active])
        score = region_size / (1 + opponents_in_region)

        # return a normalized score
        return score / np.prod(cells.shape)
