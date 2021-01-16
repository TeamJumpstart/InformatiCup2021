from math import prod
import numpy as np
from policies.policy import Policy
from environments.simulator import Spe_edSimulator
from environments import spe_ed
from state_representation import occupancy_map


class HeuristicPolicy(Policy):
    """Policy that moved into that direction with the most promising Heuristic.

    A single action is performed in every valid direction and evaluated by the given metric.
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
        if self.occupancy_map_depth > 0:  # Only compute occupancy if required
            occ_map = occupancy_map(cells, opponents, rounds, self.occupancy_map_depth)
        cur_state = Spe_edSimulator(cells, [player], rounds)

        for a, action in enumerate(self.actions):
            # perform a single action
            next_state = cur_state.step([action])
            # evaluate the heuristic, if the player is active
            if next_state.player.active:
                scores[a] = self.heuristic.score(next_state.cells, next_state.player, opponents, next_state.rounds)
                if self.occupancy_map_depth > 0:  # Factor in occupancy of newly occupied cells
                    scores[a] *= prod(1 - occ_map[y, x] for x, y in next_state.changed)

        # select action with the highest score
        return self.actions[np.argmax(scores)]

    def __repr__(self):
        """Get exact representation."""
        return "HeuristicPolicy(" + \
            f"heuristic={str(self.heuristic)}, " + \
            f"occupancy_map_depth={self.occupancy_map_depth}, " + \
            f"actions={self.actions}, " + \
            ")"
