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
            Returns a scalar value which describes the `goodness` of the current board state for `player`.
        """
        pass

    def normalizedScore(self, cells, player, opponents, rounds):
        """Compute a normalized score value in the range of [0..1] for a given game state,
        where 0 represents to possibly worst and 1 the possibly best board state.

        Args:
            cells: binary ndarray of cell occupancies.
            player: Controlled player
            opponents: List of other active players
            rounds: Number of this round. Starts with 1, thus `rounds % 6 == 0` indicates a jump.

        Returns:
            Returns a scalar value which describes the `goodness` of the current board state for `player`.
        """
        return 0

    def normalizedScoreAvailable(self):
        """Returns if a implementation of a normalized score is available.

        Iff False than normalizedScore(...) should always return 0, the possibly worst value.
        """
        return False

    def earlyOutThreshold(self):
        """Return a threshold for a score, which might be considered optimal or near optimal."""
        return float('inf')

    def normalizedEarlyOutThreshold(self):
        """Return a threshold for a normalized score, which might be considered optimal or near optimal."""
        return 1
