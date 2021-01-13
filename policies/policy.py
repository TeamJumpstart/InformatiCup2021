from abc import ABC, abstractmethod


class Policy(ABC):
    """Abstract base class for all policies."""
    @abstractmethod
    def act(self, cells, you, opponents, rounds):
        """Produce an action for a given game state.

        Args:
            cells: binary ndarray of cell occupancies.
            you: Controlled player
            opponents: List of other active players
            rounds: Number of this round. Starts with 1, thus `rounds % 6 == 0` indicates a jump.

        Returns:
            Selected action, one of `environments.spe_ed.directions`
        """
        pass
