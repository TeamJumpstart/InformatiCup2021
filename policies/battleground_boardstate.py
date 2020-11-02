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
        def check_cell(x, y, cells):
            """Check all neighbours. Return a color iff only one unique color is within the neighbourhood."""
            if cells[y, x] == 0:
                neighbours = []

                # check for array boundry, walls, etc. add only colored spaces
                if x - 1 >= 0 and cells[y, x - 1] > 1:
                    neighbours = np.append(neighbours, cells[y, x - 1])
                if x + 1 < cells.shape[1] and cells[y, x + 1] > 1:
                    neighbours = np.append(neighbours, cells[y, x + 1])
                if y - 1 >= 0 and cells[y - 1, x] > 1:
                    neighbours = np.append(neighbours, cells[y - 1, x])
                if y + 1 < cells.shape[0] and cells[y + 1, x] > 1:
                    neighbours = np.append(neighbours, cells[y + 1, x])

                # check for distinct values
                if len(np.unique(neighbours)) != 1:
                    return 0
                else:
                    return np.unique(neighbours)[0]
            else:
                return 0

        def battleground_floodfill(cells, n_steps):
            """perform battleground flood fill for max 100 steps"""
            for _ in range(n_steps):
                cells_step = np.zeros_like(cells)
                for (y, x), _ in np.ndenumerate(cells):
                    cells_step[y, x] = check_cell(x, y, cells)

                if np.sum(cells_step) == 0:
                    break  # no change in last step - return early
                cells = cells + cells_step

            return cells

        if not player.active:
            return 0  # score is 0 for dead players - no need to compute anything

        # init battleground flood fill
        battleground = cells.copy().astype(int)
        if player.active:
            battleground[player.y, player.x] = 1 + player.player_id
        for opponent in opponents:
            if opponent.active:
                battleground[opponent.y, opponent.x] = 1 + opponent.player_id

        battleground = battleground_floodfill(battleground, 100)

        unique, counts = np.unique(battleground, return_counts=True)
        score = dict(zip(unique, counts))

        return score[1 + player.player_id]
