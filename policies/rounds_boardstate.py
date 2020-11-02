from policies.boardstate import BoardState


class RoundsBoardState(BoardState):
    """Metric that return the number of rounds.
    """
    def __init__(self):
        """Initialize RoundsBoardState.
        """

    def score(self, cells, player, opponents, rounds):
        """Return current number of rounds as score value."""
        if player.active:
            return rounds
        else:
            return rounds - 1
