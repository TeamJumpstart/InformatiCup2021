from heuristics.heuristic import Heuristic


class RoundsHeuristic(Heuristic):
    """Heuristic that returns the number of rounds.
    WARNING: The returned score it not normalized in the range of [0..1]
    """
    def score(self, cells, player, opponents, rounds):
        """Return current number of rounds as score value.
        WARNING: The returned score it not normalized in the range of [0..1]
        """
        return rounds
