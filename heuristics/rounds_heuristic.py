from heuristics.heuristic import Heuristic


class RoundsHeuristic(Heuristic):
    """Heuristic that return the number of rounds.
    """
    def __init__(self, n_steps=None):
        """Initialize RoundsHeuristic."""
        self.n_steps = n_steps

    def score(self, cells, player, opponents, rounds):
        """Return current number of rounds as score value."""
        if player.active:
            return (rounds, float(rounds) / float(self.n_steps) if self.n_steps is not None else None)
        else:
            return (0, 0.0)
