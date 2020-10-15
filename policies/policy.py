from abc import ABC, abstractmethod


class Policy(ABC):
    @abstractmethod
    def act(self, cells, you, opponents):
        """Produce an action for a given game state

        Args:
            cells: binary ndarray of cell occupancies. 
            you: Controlled player
            opponents: List of other active players

        Returns:
            Selected action, one of `environments.spe_ed.directions`
        """
        pass
