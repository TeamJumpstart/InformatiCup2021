from heuristics.heuristic import Heuristic
from scipy import ndimage
import numpy as np


class CloseRegionHeuristic(Heuristic):
    """Returns the size of the region we're in."""
    def score(self, cells, player, opponents, rounds):
        """Compute the size of the region we're in after taking an action."""
        empty = cells.copy()

        # apply padding, to reconstruct boarders without introducing noice
        empty = np.pad(empty, (1, ))

        # use different kernels to catch diagonal edge cases,
        # where the end points of a wall have a Manhattan distance of 4
        struct1 = ndimage.generate_binary_structure(2, 1)
        struct2 = ndimage.generate_binary_structure(2, 2)

        # close all 1 cell wide openings aka "articulating points"
        empty = ndimage.morphology.binary_dilation(empty, border_value=1, structure=struct2)
        empty = ndimage.morphology.binary_erosion(empty, border_value=0, structure=struct1)

        # remove padding
        empty = empty[1:-1, 1:-1]

        # inverse map (mask occupied cells)
        empty = empty == 0
        # Clear cell we're in and for all active opponents
        empty[player.y, player.x] = True
        for o in opponents:
            empty[o.y, o.x] = o.active

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
