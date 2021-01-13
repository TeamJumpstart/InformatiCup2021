import numpy as np
from policies.policy import Policy
from environments.simulator import Spe_edSimulator
from environments import spe_ed
from state_representation import occupancy_map


class HeuristicPolicy(Policy):
    """Policy that performs a single action in every valid direction and
    evaluates the score of the given metric applied on the future board state to choose next actions.
    """
    def __init__(self, heuristic, occupancy_map_depth=0, actions=None):
        """Initialize HeuristicPolicy.

        Args:
            heuristic: `Heuristic` that will be evaluated after one action of the player was performed.
            occupancy_map_depth: defines the depth of the occupoancy map. If > 0, uses it to weight scores.
            actions: considers only given actions, if `None` uses all given action.
        """

        self.heuristic = heuristic
        self.occupancy_map_depth = occupancy_map_depth
        self.actions = spe_ed.actions if actions is None else actions

    def act(self, cells, player, opponents, rounds):
        """Chooses action based on weighted heuristic scores."""

        scores = np.zeros(len(self.actions), dtype=np.float32)
        occ_map = occupancy_map(cells, opponents, rounds, self.occupancy_map_depth)
        init_env = Spe_edSimulator(cells, [player], rounds)

        for a, action in enumerate(self.actions):
            # perform a single action
            env = init_env.step([action])
            # evaluate the heuristic, if the player is active
            if env.players[0].active:
                scores[a] = self.heuristic.score(env.cells, env.players[0], opponents, env.rounds)
                scores[a] *= np.prod(1 - occ_map[np.logical_xor(env.cells, init_env.cells)])

        # select action with the highest score
        return self.actions[np.argmax(scores)]

    def __str__(self):
        """Get readable representation."""
        return f"HeuristicPolicy(heuristic={str(self.heuristic)}, \
            occupancy_map_depth={str(self.occupancy_map_depth)}, \
            actions={str(self.actions)})"
