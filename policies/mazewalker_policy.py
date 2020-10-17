from policies.policy import Policy


class MazeWalkerPolicy(Policy):
    """Policy goes straight forward until it hits a wall, then turns.

    Baseline strategy, smarter policies should be able to outperform this.
    """
    def act(self, cells, player, opponents, round):
        # TODO Refactor and improve
        if cells.is_free(player.position + player.direction.cartesian):
            return "change_nothing"

        if cells.is_free(player.position + player.direction.turn_left().cartesian):
            return "turn_left"

        if cells.is_free(player.position + player.direction.turn_right().cartesian):
            return "turn_right"

        return "change_nothing"  # We're surrounded
