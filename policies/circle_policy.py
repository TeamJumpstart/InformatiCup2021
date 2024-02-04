from policies.policy import Policy


class CirclePolicy(Policy):
    """Policy goes in circles starting at the spawn positions with naive collision avoidance.

    Baseline strategy, smarter policies should be able to outperform this.
    """

    def act(self, cells, player, opponents, rounds, deadline):
        """Choose action."""
        # directions - relative to player direction
        forward = player.direction
        left = player.direction.turn_left()
        right = player.direction.turn_right()

        # if_free - relative to player position
        def is_free(pos):
            return cells.is_free(player.position + pos)

        if is_free(right):
            return "turn_right"
        elif is_free(forward):
            return "change_nothing"
        elif is_free(left):
            return "turn_left"

        return "change_nothing"  # We're surrounded

    def __repr__(self):
        """Get exact representation."""
        return "CirclePolicy()"
