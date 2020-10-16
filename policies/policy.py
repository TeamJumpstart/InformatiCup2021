from abc import ABC, abstractmethod


class Policy(ABC):
    @abstractmethod
    def act(self, cells, you, opponents, round):
        """Produce an action for a given game state

        Args:
            cells: binary ndarray of cell occupancies.
            you: Controlled player
            opponents: List of other active players
            round: Number of this round. Starts with 1, thus `round % 6 == 0` indicates a jump.

        Returns:
            Selected action, one of `environments.spe_ed.directions`
        """
        pass
