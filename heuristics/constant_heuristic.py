from heuristics.heuristic import Heuristic


class ConstantHeuristic(Heuristic):
    """Returns a constant number."""
    def __init__(self, value=1):
        """Initialize ConstantHeuristic.

        Args:
            value: The constant value.
        """
        if value < 0 or value > 1:
            raise ValueError(f"{value} if out of bounds [0, 1]")
        self.value = value

    def score(self, cells, player, opponents, rounds):
        """Return the constant number."""
        return self.value

    def __str__(self):
        """Get readable representation."""
        return f"ConstantHeuristic({self.value})"
