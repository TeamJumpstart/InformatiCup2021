import numpy as np
from policies.policy import Policy
from environments.simulator import Spe_edSimulator
from environments import spe_ed


class HeuristicPolicy(Policy):
    """Policy that performs a single action in every valid direction and
    evaluates the score of the given metric applied on the future board state to choose next actions.
    """
    def __init__(self, heuristic):
        """Initialize HeuristicPolicy.

        Args:
            heuristic: `Heuristic` that will be evaluated after one action of the player was performed.
        """

        self.heuristic = heuristic

    def act(self, cells, player, opponents, rounds):
        """Chooses action based on weighted heuristic scores."""

        scores = np.zeros(len(spe_ed.actions), dtype=np.float32)

        for a, action in enumerate(spe_ed.actions):
            # perform a single action
            env = Spe_edSimulator(cells.cells, [player], rounds).step([action])
            # evaluate the heuristic, if the player is active
            if env.players[0].active:
                scores[a] = self.heuristic.score(env.cells, env.players[0], opponents, env.rounds)

        # select action with the highest score
        return spe_ed.actions[np.argmax(scores)]

    def __str__(self):
        """Get readable representation."""
        return "HeuristicPolicy[]"  # TODO Encode heuristic
