from abc import ABC, abstractmethod


class Condition(ABC):
    """Abstract class to represent a board state condition."""
    @abstractmethod
    def score(self, cells, player, opponents, rounds):
        """Compute a score value for a given game state.

        Args:
            cells: binary ndarray of cell occupancies.
            player: Controlled player
            opponents: List of other active players
            rounds: Number of this round. Starts with 1, thus `rounds % 6 == 0` indicates a jump.

        Returns:
            Returns a scalar values which describes a certain condition of the board state.
        """
        pass
