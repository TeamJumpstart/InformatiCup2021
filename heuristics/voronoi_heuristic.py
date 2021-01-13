from heuristics.heuristic import Heuristic
import numpy as np
from scipy.ndimage import morphology


class VoronoiHeuristic(Heuristic):
    """Tries to maximize the area that can be reached by the agent before the opponents."""
    def __init__(self, max_steps=40, opening_iterations=0, minimize_opponents=False):
        """Initialize VoronoiHeuristic.

        Args:
            max_steps: The maximal number of steps from the cell center to cell border
            opening_iterations: number of performed opening operations on the cell state before the computation
                of the voronoi diagram to account for jumps. default: 0
            minimize_opponents: TODO
        """
        self.max_steps = max_steps
        self.opening_iterations = opening_iterations
        self.minimize_opponents = minimize_opponents

    def score(self, cells, player, opponents, rounds):
        """Computes an approximation of the geodesic voronoi diagram using binary dilation."""
        unmodified_cells = cells
        # open all {self.opening_iterations} wide walls aka "articulating points" to account for jumps
        if self.opening_iterations:
            cells = np.pad(cells, (self.opening_iterations, ))
            cells = morphology.binary_opening(cells, iterations=self.opening_iterations)
            cells = cells[self.opening_iterations:-self.opening_iterations,
                          self.opening_iterations:-self.opening_iterations]

        # initialize arrays, one binary map for each player
        mask = cells == 0
        voronoi = np.zeros((len(opponents) + 1, *cells.shape), dtype=np.bool)
        for idx, p in enumerate((player, *opponents)):
            mask[p.y, p.x] = True  # reset player position to allow dilation
            voronoi[idx, p.y, p.x] = True

        # geodesic voronoi cell computation
        for _ in range(self.max_steps):
            for i in range(len(voronoi)):
                voronoi[i] = morphology.binary_dilation(voronoi[i], mask=mask)  # expand cell by one step
            xor = np.sum(voronoi, axis=0) > 1  # compute overlaps for every cell
            mask[xor] = 0  # mask out all overlaps
            voronoi[:, xor] = 0  # reset all overlapping cell borders

        if self.opening_iterations:
            voronoi[:, unmodified_cells > 0] = 0  # reset all occupied cells
            for idx, p in enumerate((player, *opponents)):
                voronoi[idx, p.y, p.x] = True  # reset player positions

        if self.minimize_opponents:
            return 1 - (np.sum(voronoi[1:]) / np.prod(cells.shape))
        else:
            # return the relative size of the voronoi cell for the controlled player
            return np.sum(voronoi[0]) / np.prod(cells.shape)

    def __str__(self):
        """Get readable representation."""
        return "VoronoiHeuristic(" + \
            f"max_steps={self.max_steps}, " + \
            f"opening_iterations={self.opening_iterations}, " + \
            f"minimize_opponents={self.minimize_opponents}, " + \
            ")"
