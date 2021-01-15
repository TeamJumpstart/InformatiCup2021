from heuristics.heuristic import Heuristic
from scipy.ndimage import morphology, gaussian_filter
from scipy import ndimage
import numpy as np


class RegionHeuristic(Heuristic):
    def __init__(self, closing_iterations=0, include_opponent_regions=True):
        """Initialize RegionHeuristic.

        Args:
            closing_iterations: number of performed closing operations on the cell state before the computation
                of the regions to ommit smaller regions. default: 0
        """
        self.closing_iterations = closing_iterations
        self.include_opponent_regions = include_opponent_regions

    def score(self, cells, player, opponents, rounds):
        """Compute the relative size of the region we're in."""
        # close all 1 cell wide openings aka "articulating points"
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
        # Compute the sizes and divide by numbers of players in each region
        region_sizes = np.array([np.sum(labelled == region) / np.sum(regions == region) for region in regions])

        # Normalize by grid size
        region_sizes /= np.prod(cells.shape)

        score = region_sizes[0]

        if self.include_opponent_regions and len(opponents) > 0:
            # Use product of own region size times the inverse of the average of opponent regions sizes
            # Note: Using the average prevents reckless speeding up, as otherwise this would be strongly
            # preferred as it may reduce the regions of multiple players at once.
            score *= (1 - np.mean(region_sizes[1:]))

        return score

    def __str__(self):
        """Get readable representation."""
        return "RegionHeuristic"


class RegionHeuristic2(Heuristic):
    def __init__(self, closing_iterations=0, include_opponent_regions=True):
        """Initialize RegionHeuristic.

        Args:
            closing_iterations: number of performed closing operations on the cell state before the computation
                of the regions to ommit smaller regions. default: 0
        """
        self.closing_iterations = closing_iterations
        self.include_opponent_regions = include_opponent_regions

    def score(self, cells, player, opponents, rounds):
        """Compute the relative size of the region we're in."""
        # close all 1 cell wide openings aka "articulating points"
        if self.closing_iterations:
            cells = morphology.binary_closing(cells, iterations=self.closing_iterations)

        players = [player] + opponents

        # inverse map (mask occupied cells)
        empty = cells == 0
        # Clear cell we're in and for all active opponents
        for p in players:
            empty[p.y, p.x] = True

        # compute distinct regions
        labelled, n = ndimage.label(empty)

        # Get the region each player is in
        regions = np.array([labelled[p.y, p.x] for p in players])
        # Compute the sizes and divide by numbers of players in each region
        region_sizes = np.array([np.sum(labelled == region) / np.sum(regions == region) for region in regions])

        regions2 = np.zeros_like(cells, dtype=np.float32)
        for region in range(1, n + 1):
            mask = labelled == region
            size = np.sum(mask)

            regions2[mask] = size / max(np.sum(regions[1:] == region), 1)
        regions2 = gaussian_filter(regions2, 10, mode="constant", cval=np.max(regions2))
        region_sizes2 = np.array([regions2[p.y, p.x] for p in players])

        region_sizes = region_sizes2

        # Normalize by grid size
        region_sizes /= np.prod(cells.shape)

        score = region_sizes[0]

        if self.include_opponent_regions and len(opponents) > 0:
            # Use product of own region size times the inverse of the average of opponent regions sizes
            # Note: Using the average prevents reckless speeding up, as otherwise this would be strongly
            # preferred as it may reduce the regions of multiple players at once.
            score *= (1 - np.mean(region_sizes[1:]))

        return score

    def __str__(self):
        """Get readable representation."""
        return "RegionHeuristic"
