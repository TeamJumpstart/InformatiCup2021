from environments.spe_ed import cartesian_directions, turn_left, turn_right
from policies.policy import Policy


class MazeWalkerPolicy(Policy):
    """Policy goes straight forward until it hits a wall, then turns.

    Baseline strategy, smarter policies should be able to outperform this.
    """
    def act(self, cells, player, opponents, round):
        # TODO Refactor and improve
        if cells.is_free(player.position + cartesian_directions[player.direction]):
            return "change_nothing"

        if cells.is_free(player.position + cartesian_directions[turn_left(player.direction)]):
            return "turn_left"

        if cells.is_free(player.position + cartesian_directions[turn_right(player.direction)]):
            return "turn_right"

        return "change_nothing"  # We're surrounded
