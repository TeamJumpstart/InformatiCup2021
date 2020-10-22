from policies.policy import Policy


class SpiralPolicy(Policy):
    """Policy goes straight forward until it hits a wall, then turns.

    Baseline strategy, smarter policies should be able to outperform this.
    """
    def __init__(self):
        """Initialize MazeWalkerPolicy."""
        self.clockwise = True

    def act(self, cells, player, opponents, round):

        # directions - relative to player direction
        forward = player.direction.cartesian
        left = player.direction.turn_left().cartesian
        right = player.direction.turn_right().cartesian

        def is_free(pos):
            """is_free relative to player position."""
            return cells.is_free(player.position + pos)

        # check if we can create a spiral loop
        if is_free(forward) & is_free(2 * forward):
            return "change_nothing"

        if self.clockwise:
            if is_free(right) & is_free(left) & (not is_free(forward)):
                self.clockwise = False
                return "turn_left"
            if is_free(right):
                return "turn_right"
            elif is_free(left):
                return "turn_left"
        else:
            if is_free(right) & is_free(left) & (not is_free(forward)):
                self.clockwise = True
                return "turn_right"
            if is_free(left):
                return "turn_left"
            elif is_free(right):
                return "turn_right"

        return "change_nothing"  # We're surrounded