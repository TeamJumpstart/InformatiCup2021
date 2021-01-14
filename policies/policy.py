from abc import ABC, abstractmethod
from pathlib import Path
from importlib.machinery import SourceFileLoader


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


def load_named_policy(name):
    """Load a named policy by it's given name.

    A named policy is contained in a python source file in `policies/named_policies` as `pol` variable.

    Args:
        name: Name of the policy (base name of it's source file)

    Returns:
        pol: A new instance of the corresponding policy
    """
    policy_file = Path(__file__).parent / "named_policies" / (name + ".py")
    if not policy_file.is_file():
        raise ValueError("There is no named policy '{name}'")

    pol = SourceFileLoader('tournament_config', str(policy_file)).load_module().pol
    return pol
