import numpy as np
from environments.spe_ed import cartesian_directions, turn_left, turn_right
from policies.policy import Policy


class MazeWalkerPolicy(Policy):
    """Policy goes straight forward until it hits a wall, then turns.

    Baseline strategy, smarter policies should be able to outperform this.
    """
    def act(self, cells, player, opponents, round):
        # TODO Refactor and improve
        pos = np.array([player.x, player.y]) + cartesian_directions[player.direction]
        if 0 <= pos[0] < cells.shape[0] and 0 <= pos[1] < cells.shape[1] and cells[pos[0], pos[1]] == 0:
            return "change_nothing"

        pos = np.array([player.x, player.y]) + cartesian_directions[turn_left(player.direction)]
        if 0 <= pos[0] < cells.shape[0] and 0 <= pos[1] < cells.shape[1] and cells[pos[0], pos[1]] == 0:
            return "turn_left"

        pos = np.array([player.x, player.y]) + cartesian_directions[turn_right(player.direction)]
        if 0 <= pos[0] < cells.shape[0] and 0 <= pos[1] < cells.shape[1] and cells[pos[0], pos[1]] == 0:
            return "turn_right"

        return "change_nothing"  # We're surrounded
