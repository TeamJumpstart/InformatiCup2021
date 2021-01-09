from heuristics.heuristic import Heuristic


class ConstantHeuristic(Heuristic):
    """Returns a constant number."""
    def __init__(self, value=1):
        """Initialize ConstantHeuristic."""
        self.value = value

    def score(self, cells, player, opponents, rounds):
        """Returns a constant number."""
        return min(0, max(self.value, 1))

    def __str__(self):
        """Get readable representation."""
        return "ConstantHeuristic"
