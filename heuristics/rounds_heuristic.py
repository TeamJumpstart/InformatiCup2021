from heuristics.heuristic import Heuristic


class RoundsHeuristic(Heuristic):
    """Heuristic that returns the number of rounds."""
    def score(self, cells, player, opponents, rounds):
        """Return current number of rounds as score value."""
        return (rounds, None)
