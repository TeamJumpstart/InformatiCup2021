from policies.policy import Policy


class MazeWalkerPolicy(Policy):
    """Policy goes straight forward until it hits a wall, then follows the wall.

    Baseline strategy, smarter policies should be able to outperform this.
    """

    def __init__(self):
        """Initialize MazeWalkerPolicy."""
        self.hit_wall = False

    def act(self, cells, player, opponents, rounds, deadline):
        """Choose action."""

        def is_free(pos):
            """is_free, but relative to player position."""
            return cells.is_free(player.position + pos)

        # directions - relative to player direction
        forward = player.direction
        left = player.direction.turn_left()
        right = player.direction.turn_right()

        if self.hit_wall:  # follow the wall
            if is_free(left):
                return "turn_left"
            elif is_free(forward):
                return "change_nothing"
            elif is_free(right):
                return "turn_right"
        else:  # search for the wall
            if is_free(forward):
                return "change_nothing"
            else:
                self.hit_wall = True
                return "turn_right"

        return "change_nothing"  # We're surrounded

    def __repr__(self):
        """Get exact representation."""
        return "MazeWalkerPolicy()"
