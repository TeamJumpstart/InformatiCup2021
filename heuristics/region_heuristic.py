from heuristics.heuristic import Heuristic
from scipy import ndimage
import numpy as np


class RegionHeuristic(Heuristic):
    """Returns the size of the region we're in."""
    def score(self, cells, player, opponents, rounds):
        """Compute the size of the region we're in after taking action."""
        empty = cells == 0
        empty[player.y, player.x] = True  # Clear cell we're in
        labelled, _ = ndimage.label(empty)
        region = labelled[player.y, player.x]  # Get the region we're in
        region_size = np.sum(labelled == region)
        return (region_size, region_size / np.prod(cells.shape))
