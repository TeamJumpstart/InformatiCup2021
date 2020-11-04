from metrics.metric import Metric
from scipy import ndimage
import numpy as np


class RegionMetric(Metric):
    """Returns a random number.
    """
    def __init__(self, cell_shape=None):
        """Initialize RegionMetric."""

        if cell_shape is None:
            self.threshold = None
        else:
            self.threshold = 0.5 * np.prod(cell_shape)
        self.normalizedThreshold = 0.5

    def score(self, cells, player, opponents, rounds):
        """Compute the of the region we're in after taking action."""
        if not player.active:
            return 0
        empty = cells == 0
        empty[player.y, player.x] = True  # Clear cell we're in
        labelled, _ = ndimage.label(empty)
        region = labelled[player.y, player.x]  # Get the region we're in
        region_size = np.sum(labelled == region)
        return region_size

    def normalizedScore(self, cells, player, opponents, rounds):
        """Compute the of the region we're in after taking action normalized by the board size."""
        return self.score(cells, player, opponents, rounds) / np.prod(cells.shape)

    def normalizedScoreAvailable(self):
        return True

    def earlyOutThreshold(self):
        if self.threshold is None:
            return float('inf')
        else:
            return self.threshold

    def normalizedEarlyOutThreshold(self):
        return self.normalizedThreshold
