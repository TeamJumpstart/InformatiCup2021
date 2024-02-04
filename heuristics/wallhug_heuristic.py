from heuristics.heuristic import Heuristic


class WallhugHeuristic(Heuristic):
    """Returns a constant number."""

    def score(self, cells, player, opponents, rounds, deadline):
        """Return the constant number."""
        # directions - relative to player direction
        forward = player.direction
        left = player.direction.turn_left()
        right = player.direction.turn_right()

        def is_occ(pos):
            """is_free relative to player position."""
            return not cells.is_free(player.position + pos)

        return (is_occ(forward) + is_occ(left) + is_occ(right)) / 3

    def __str__(self):
        """Get readable representation."""
        return "WallhugHeuristic()"
