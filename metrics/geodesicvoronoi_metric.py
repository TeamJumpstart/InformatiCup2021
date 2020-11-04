import numpy as np
from metrics.metric import Metric


class GeodesicVoronoiMetric(Metric):
    """Tries to maximize the area that can be reached by the agent before the opponents.
    """
    def __init__(self, max_distance=100, seed=None):
        """Initialize GeodesicVoronoiBoardState.

        Args:
            max_distance: The maximal size for a voronoi cell taken into account given as geodesic distance from center.
            seed: Seed for the random number generator. Use a fixed seed for reproducibility,
                  or pass `None` for a random seed.
        """
        self.rng = np.random.default_rng(seed)
        self.max_distance = max_distance

    def score(self, cells, player, opponents, rounds):
        """ Returns a score based on the computed geodesic voronoi diagram.
        """

        if not player.active:
            return 0  # score is 0 for dead players - no need to compute anything

        cells = np.transpose(cells)  # transpose array, as cells are encoded with (y,x)
        kernel = np.array([[1, 0], [-1, 0], [0, -1], [0, 1]], dtype=int)  # defines the connectivity of a cell
        voronoi = np.zeros_like(cells).astype(int)  # voronoi diagram

        # populate map with positions of active player as cell centers
        if player.active:
            voronoi[player.x, player.y] = player.player_id
        for opponent in opponents:
            if opponent.active:
                voronoi[opponent.x, opponent.y] = opponent.player_id

        # initialize first set of open indizes - cells that need to be checked
        open_indizes = set([])

        def getFreeIndizes(cells, voronoi, pos, kernel):
            """ Check if neighbourhood of the cell at pos contains free cells.
            Returns their indices.
            """
            free_indizes = set([])
            shape = cells.shape
            for offset in kernel:
                o_pos = (pos[0] + offset[0], pos[1] + offset[1])
                if 0 <= o_pos[0] and o_pos[0] < shape[0] and 0 <= o_pos[1] and o_pos[1] < shape[1]:
                    if cells[o_pos[0], o_pos[1]] == 0 and voronoi[o_pos[0], o_pos[1]] == 0:
                        free_indizes.add((o_pos[0], o_pos[1]))
            return free_indizes

        if player.active:
            open_indizes.update(getFreeIndizes(cells, voronoi, player.position, kernel))
        for opponent in opponents:
            if opponent.active:
                open_indizes.update(getFreeIndizes(cells, voronoi, opponent.position, kernel))

        # iteratively evaluate all open indizes
        def checkCell(cells, voronoi, pos, kernel):
            """ Checks the neighbourhood of the cell.
            Returns a label, if the neighbourhood contains only one unique label, otherwise 0.
            """
            labels = np.array([])
            shape = cells.shape
            for offset in kernel:
                o_pos = (pos[0] + offset[0], pos[1] + offset[1])
                if 0 <= o_pos[0] and o_pos[0] < shape[0] and 0 <= o_pos[1] and o_pos[1] < shape[1]:
                    if voronoi[o_pos[0], o_pos[1]] != 0:
                        labels = np.append(labels, voronoi[o_pos[0], o_pos[1]])
            if len(np.unique(labels)) == 1:
                return np.unique(labels)[0]
            return 0

        for _ in range(self.max_distance):
            new_indizes = set([])
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

        # debug output
        def debug_output(cells, voronoi):
            import matplotlib.pyplot as plt
            from matplotlib.colors import ListedColormap
            _cmap = ListedColormap(
                [
                    (1.0, 1.0, 1.0, 1.0),  # Background - white
                    (0.0, 0.0, 0.0, 1.0),  # Collision - black
                    (1.0, 0.0, 0.0, 0.2),  # Player 1 - red
                    (0.0, 1.0, 0.0, 0.2),  # Player 2 - green
                    (0.0, 0.0, 1.0, 0.2),  # Player 3 - blue
                    (1.0, 1.0, 0.0, 0.2),  # Player 4 - yellow
                    (0.0, 1.0, 1.0, 0.2),  # Player 5 - light blue
                    (1.0, 0.0, 1.0, 0.2),  # Player 6 - pink
                    (1.0, 0.0, 0.0, 1.0),  # Player 1 - red
                    (0.0, 1.0, 0.0, 1.0),  # Player 2 - green
                    (0.0, 0.0, 1.0, 1.0),  # Player 3 - blue
                    (1.0, 1.0, 0.0, 1.0),  # Player 4 - yellow
                    (0.0, 1.0, 1.0, 1.0),  # Player 5 - light blue
                    (1.0, 0.0, 1.0, 1.0),  # Player 6 - pink
                ]
            )

            # append voronoi diagram to the cells and convert the values to fit the color map:
            # player positions - solid color, player voronoi cell - light color
            cells = cells.astype(int) + np.where(voronoi, voronoi + 1, voronoi)
            if player.active:
                cells[player.x, player.y] = 7 + player.player_id
            for opponent in opponents:
                if opponent.active:
                    cells[opponent.x, opponent.y] = 7 + opponent.player_id

            # create a window and show the cell - transpose cells back to (y, x) indexing
            plt.logging.getLogger('matplotlib.font_manager').disabled = True
            plt.imshow(np.transpose(cells), cmap=_cmap)
            plt.show()

        #debug_output(cells, voronoi)

        # return the relative size of the voronoi cell for the controlled player
        output = scores[player.player_id]
        return output
