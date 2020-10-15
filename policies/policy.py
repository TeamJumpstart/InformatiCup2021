from abc import ABC, abstractmethod


class Policy(ABC):
    @abstractmethod
    def act(self, cells, you, opponents):
        """Produce an action for a given game state

        Args:
            cells: two dimensional ndarray
            you: Player object of controlled player
            opponents: List of other players

        Returns:
            Selected action, one of `environments.spe_ed.directions`
        """
        pass
