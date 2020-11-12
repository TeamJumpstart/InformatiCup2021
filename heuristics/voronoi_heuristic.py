from heuristics.heuristic import Heuristic
import numpy as np
from scipy.ndimage import morphology


class VoronoiHeuristic(Heuristic):
    """Tries to maximize the area that can be reached by the agent before the opponents."""
    def __init__(self, max_steps=40, opening_iterations=0):
        """Initialize VoronoiHeuristic.

        Args:
            max_steps: The maximal number of steps from the cell center to cell border
            opening_iterations: number of performed opening operations on the cell state before the computation
                of the voronoi diagram to account for jumps. default: 0
        """
        self.max_steps = max_steps
        self.iterations = opening_iterations

    def score(self, cells, player, opponents, rounds):
        """ Computes an approximation of the geodesic voronoi diagram using binary dilation."""
        unmodified_cells = cells
        # open all {self.iterations} wide walls aka "articulating points" to account for jumps
        if self.iterations:
            cells = np.pad(cells, (self.iterations, ))
            cells = morphology.binary_opening(cells, iterations=self.iterations)
            cells = cells[self.iterations:-self.iterations, self.iterations:-self.iterations]

        # initialize arrays
        map_voronoi = {}
        bin_voronoi = []
        mask = cells == 0
        for p in [player, *opponents]:
            if p.active:
                # create arrays only for active players
                map_voronoi[p.player_id] = len(bin_voronoi)
                bin_voronoi.append(np.zeros_like(cells, dtype=np.bool))
                # reset player position to allow dilation
                mask[p.y, p.x] = True
                bin_voronoi[map_voronoi[p.player_id]][p.y, p.x] = True

        bin_voronoi = np.array(bin_voronoi)

        # geodesic voronoi cell computation
        for _ in range(self.max_steps):
            for i, _ in enumerate(bin_voronoi):
                bin_voronoi[i] = morphology.binary_dilation(bin_voronoi[i], mask=mask)  # expand cell by one step
            xor = np.sum(bin_voronoi, axis=0) > 1  # compute overlaps for every cell
            mask[xor] = 0  # mask out all overlaps
            bin_voronoi[:, xor] = 0  # reset all overlapping cell borders

        if self.iterations:
            bin_voronoi[:, unmodified_cells > 0] = 0  # reset all occupied cells
            for p in [player, *opponents]:
                if p.active:
                    bin_voronoi[map_voronoi[p.player_id]][p.y, p.x] = True  # reset player position

        # return the relative size of the voronoi cell for the controlled player
        return np.sum(bin_voronoi[map_voronoi[player.player_id]]) / np.prod(cells.shape)
