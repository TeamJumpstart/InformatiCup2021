from abc import ABC, abstractmethod


class Heuristic(ABC):
    """Abstract class to represent a board state heuristic."""

    @abstractmethod
    def score(self, cells, player, opponents, rounds, deadline):
        """Compute a score value for a given game state.

        Args:
            cells: binary ndarray of cell occupancies.
            player: Controlled player
            opponents: List of other active players
            rounds: Number of this round. Starts with 1, thus `rounds % 6 == 0` indicates a jump.
            deadline: A deadline after which the heuristic must return immediately.

        Returns:
            Returns a scalar values which describe the `goodness` of the current board state for `player`.
            The value is normalized in the range [0, 1].
        """
        pass
