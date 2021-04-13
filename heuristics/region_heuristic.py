from heuristics.heuristic import Heuristic
from scipy.ndimage import morphology
from scipy import ndimage
import numpy as np


class RegionHeuristic(Heuristic):
    """Heuristic evaluation the the of the region the player is in.

    Larger reagion -> higher score

    More opponents in same region -> lower score

    Morphological operations can be applied.
    """
    def __init__(self, closing_iterations=0, opening_iterations=0, include_opponent_regions=True):
        """Initialize RegionHeuristic.

        Args:
            closing_iterations: number of performed closing operations on the cell state before the computation
                of the regions to ommit smaller regions. default: 0
            include_opponent_regions: Multiply the score with the inverse region size of opponents
        """
        self.closing_iterations = closing_iterations
        self.include_opponent_regions = include_opponent_regions
        self.opening_iterations = opening_iterations

    def score(self, cells, player, opponents, rounds, deadline):
        """Compute the relative size of the region we're in."""
        # close all 1 cell wide openings aka "articulating points"
        if self.closing_iterations or self.opening_iterations:
            cells = morphology.binary_closing(cells, iterations=self.closing_iterations)
            cells = morphology.binary_closing(cells, iterations=self.opening_iterations)

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
        return "RegionHeuristic(" + \
            f"closing_iterations={self.closing_iterations}, " + \
            f"include_opponent_regions={self.include_opponent_regions}, " + \
            ")"
