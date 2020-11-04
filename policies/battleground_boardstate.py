import numpy as np
from policies.boardstate import BoardState


class BattlegroundBoardState(BoardState):
    """Tries to maximize the area that can be reached by the agent before the opponents.
    """
    def __init__(self, seed=None, debug=False):
        """Initialize BattlegroundPolicy.

        Args:
            seed: Seed for the random number generator. Use a fixed seed for reproducibility,
                  or pass `None` for a random seed.
        """
        self.rng = np.random.default_rng(seed)
        self.debug = debug

    def score(self, cells, player, opponents, rounds):

        if not player.active:
            return 0  # score is 0 for dead players - no need to compute anything

        def check_cell(pos, cells, new_indizes):
            """Check all neighbours of given cell. Return a color and update `new_indizes`
                iff only one unique color is within the neighbourhood.
            """
            if cells[pos[1], pos[0]] == 0:
                neighbours = []

                free_cells_indizes = set([])
                # check for array boundry, walls, etc.
                for offset in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                    if 0 <= pos[0] + offset[0] \
                            and pos[0] + offset[0] < cells.shape[1] \
                            and 0 <= pos[1] + offset[1] \
                            and pos[1] + offset[1] < cells.shape[0]:
                        if cells[pos[1] + offset[1], pos[0] + offset[0]] == 0:
                            free_cells_indizes.add((pos[0] + offset[0], pos[1] + offset[1]))
                        elif cells[pos[1] + offset[1], pos[0] + offset[0]] > 1:
                            neighbours = np.append(neighbours, cells[pos[1] + offset[1], pos[0] + offset[0]])

                # check for distinct values
                if len(np.unique(neighbours)) != 1:
                    # cell stays unchanged, iff not it has not exactly one neighbour
                    return 0
                else:
                    # cell updated, add free cell neighbours to check in next iteration step
                    new_indizes.update(free_cells_indizes)
                    return np.unique(neighbours)[0]
            else:
                return 0

        def geodesic_voronoi_2d(cells, cell_centers, n_steps):
            """computes geodesic voronoi diagram with a maximal distance of `n_steps`
                distance metric: manhattan-distance

                cells: defines the grid and initial cell state
                cell_centers: defines the initial centers to start the computation with
                n_steps: maximal distance from a cell center

                cell values:
                    - 0: free cell - can be assigned a label
                    - 1: occupied cell - cannot be assigned a lebel
                    - > 1: labels - cell is assigned with the corresponding label
            """
            open_indizes = set([])
            for pos in cell_centers:
                for offset in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                    if 0 <= pos[0] + offset[0] \
                            and pos[0] + offset[0] < cells.shape[1] \
                            and 0 <= pos[1] + offset[1] \
                            and pos[1] + offset[1] < cells.shape[0]:
                        open_indizes.add((pos[0] + offset[0], pos[1] + offset[1]))

            for i in range(n_steps):

                cells_step = np.zeros_like(cells)
                new_open_indizes = set([])

                for index in open_indizes:
                    cells_step[index[1], index[0]] = check_cell(index, cells, new_open_indizes)

                if np.sum(cells_step) == 0:
                    break  # no change in last step - early stop

                # update the global cell state
                cells = cells + cells_step
                # update cell indizes to check in next iteration
                open_indizes = new_open_indizes.copy()

            return cells

        cells = cells.astype(int)

        cell_centers = set([])
        if player.active:
            cell_centers.add((player.x, player.y))
            cells[(player.y, player.x)] = 1 + player.player_id
        cell_centers.update([(opponent.x, opponent.y) for opponent in opponents if opponent.active])
        for opponent in opponents:
            if opponent.active:
                cells[opponent.y, opponent.x] = 1 + opponent.player_id

        cells = geodesic_voronoi_2d(cells, cell_centers, 100)

        unique, counts = np.unique(cells, return_counts=True)
        score = dict(zip(unique, counts))
        """_cmap = ListedColormap(
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

        if player.active:
            cells[(player.y, player.x)] = 7 + player.player_id
        for opponent in opponents:
            if opponent.active:
                cells[opponent.y, opponent.x] = 7 + opponent.player_id

        plt.logging.getLogger('matplotlib.font_manager').disabled = True
        plt.imshow(cells, cmap=_cmap)
        plt.show()"""

        return score[1 + player.player_id] / sum(score)
