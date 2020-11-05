from abc import ABC, abstractmethod


class Heuristic(ABC):
    """ Defines an abstract class to represent a board state heuristic."""
    @abstractmethod
    def score(self, cells, player, opponents, rounds):
        """Compute a score value for a given game state

        Args:
            cells: binary ndarray of cell occupancies.
            player: Controlled player
            opponents: List of other active players
            rounds: Number of this round. Starts with 1, thus `rounds % 6 == 0` indicates a jump.

        Returns:
            Returns two scalar values which describe the `goodness` of the current board state for `player`.
            The second value is normalized in the range [0, 1] if possible, otherwise it is set to `None`.
        """
        pass
