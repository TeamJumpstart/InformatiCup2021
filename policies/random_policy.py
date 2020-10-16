import numpy as np
from environments.spe_ed import actions
from policies.policy import Policy


class RandomPolicy(Policy):
    """Policy that chooses action randomly.

    May have different probabilities of different actions.

    Baseline strategy, each smarter policy should be able to outperform this.
    """
    def __init__(self, seed=None, p=None):
        """Initialize RandomPolicy.

        Args:
            seed: Seed for the random number generator. Use a fixed seed for reproducibility,
                  or pass `None` for a random seed.
            p: Propability weights of each action. Pass `None` for uniform distribution.
        """
        self.rng = np.random.default_rng(seed)
        if p is not None and len(p) != len(actions):
            raise ValueError(f"Number of probabilities {p} does mot match number of actions {actions}")
        self.p = p

    def act(self, cells, you, opponents, jumping):
        """Choose action randomly."""
        return self.rng.choice(actions, p=self.p)
