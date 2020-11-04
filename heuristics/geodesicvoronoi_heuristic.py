from heuristics.heuristic import Heuristic
import numpy as np


class GeodesicVoronoiHeuristic(Heuristic):
    """Tries to maximize the area that can be reached by the agent before the opponents."""
    def __init__(self, max_distance=100):
        """Initialize GeodesicVoronoiHeuristic.

        Args:
            max_distance: The maximal size for a voronoi cell taken into account given as geodesic distance from center.
            seed: Seed for the random number generator. Use a fixed seed for reproducibility,
                  or pass `None` for a random seed.
        """
        self.max_distance = max_distance
        self.threshold = 0.5 * max_distance * max_distance
        self.normalizedThreshold = 0.5

    def score(self, cells, player, opponents, rounds):
        """ Returns a score based on the computed geodesic voronoi diagram. """

        if not player.active:
            return 0  # score is 0 for dead players - no need to compute anything

        cells = np.transpose(cells)  # transpose array, as cells are encoded with (y,x)
        kernel = np.array([[1, 0], [-1, 0], [0, -1], [0, 1]], dtype=int)  # defines the connectivity of a cell
        voronoi = np.zeros_like(cells, dtype=np.int8)  # voronoi diagram

        # populate map with positions of active player as cell centers
        voronoi[player.x, player.y] = player.player_id
        for opponent in opponents:
            if opponent.active:
                voronoi[opponent.x, opponent.y] = opponent.player_id

        # initialize first set of open indizes - cells that need to be checked
        open_indizes = set()

        def getFreeIndizes(cells, voronoi, pos, kernel):
            """ Check if neighbourhood of the cell at pos contains free cells.
            Returns their indices.
            """
            free_indizes = set()
            shape = cells.shape
            for offset in kernel:
                o_pos = (pos[0] + offset[0], pos[1] + offset[1])
                if 0 <= o_pos[0] < shape[0] and 0 <= o_pos[1] < shape[1]:
                    if cells[o_pos[0], o_pos[1]] == 0 and voronoi[o_pos[0], o_pos[1]] == 0:
                        free_indizes.add(o_pos)
            return free_indizes

        open_indizes.update(getFreeIndizes(cells, voronoi, player.position, kernel))
        for opponent in opponents:
            if opponent.active:
                open_indizes.update(getFreeIndizes(cells, voronoi, opponent.position, kernel))

        # iteratively evaluate all open indizes
        def checkCell(cells, voronoi, pos, kernel):
            """ Checks the neighbourhood of the cell.
            Returns a label, if the neighbourhood contains only one unique label, otherwise 0.
            """
            labels = []
            shape = cells.shape
            for offset in kernel:
                o_pos = (pos[0] + offset[0], pos[1] + offset[1])
                if 0 <= o_pos[0] < shape[0] and 0 <= o_pos[1] < shape[1]:
                    if voronoi[o_pos[0], o_pos[1]] != 0:
                        labels += voronoi[o_pos[0], o_pos[1]]
            if len(np.unique(labels)) == 1:
                return labels[0]
            return 0

        for _ in range(self.max_distance):
            new_indizes = set()
            voronoi_step = np.zeros_like(voronoi)

            # iterate over open indizes
            for index in open_indizes:
                label = checkCell(cells, voronoi, index, kernel)
                if label != 0:
                    # propagate the cell label from the neighbourhood
                    voronoi_step[index] = label
                    # and check neighbourhood for free cells
                    new_indizes.update(getFreeIndizes(cells, voronoi, index, kernel))

            if np.sum(voronoi_step) == 0:
                break  # early out - no cells were updated in this step

            # update the voronoi map
            voronoi = voronoi + voronoi_step
            # update open indizes, discard already checked indizes
            open_indizes = new_indizes - open_indizes

        # compute the voronoi cell size
        unique, counts = np.unique(voronoi, return_counts=True)
        scores = dict(zip(unique, counts))

        # return the relative size of the voronoi cell for the controlled player
        return scores[player.player_id]

    def normalizedScore(self, cells, player, opponents, rounds):
        """ Returns a normalized score based on the computed geodesic voronoi diagram.
        The score is normalized by the grid size, which is the upper bound of the cell size.
        """
        return self.score(cells, player, opponents, rounds) / np.prod(cells.shape)

    def normalizedScoreAvailable(self):
        return True

    def earlyOutThreshold(self):
        return self.threshold

    def normalizedEarlyOutThreshold(self):
        return self.normalizedThreshold
