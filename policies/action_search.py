import time
from math import prod
from queue import PriorityQueue

from environments import spe_ed
from environments.simulator import Spe_edSimulator
from policies.policy import Policy
from state_representation import occupancy_map


class ActionSearchPolicy(Policy):
    """Policy that performs a greedy search for that action that will maximize the heuristic."""

    def __init__(self, heuristic, depth_limit=6, expanded_node_limit=100, occupancy_map_depth=0):
        """Initialize ActionSearchPolicy.

        Args:
            heuristic: `Heuristic` that will be evaluated after every step.
        """
        self.heuristic = heuristic
        self.depth_limit = depth_limit
        self.expanded_node_limit = expanded_node_limit
        self.occupancy_map_depth = occupancy_map_depth

    def act(self, cells, player, opponents, rounds, deadline):
        """Search action sequence based on heuristic scores."""
        if self.occupancy_map_depth > 0:
            occ_maps = [occupancy_map(cells, opponents, rounds, depth=d + 1) for d in range(self.occupancy_map_depth)]

        states = PriorityQueue()
        states.put((0, [], Spe_edSimulator(cells, [player], rounds), 1))  # Current state as inital

        actions_scores = []
        expanded = 0
        while not states.empty() and expanded < self.expanded_node_limit and time.time() < deadline:
            _, prev_actions, prev_state, prev_freeness = states.get()

            for action in spe_ed.actions:
                state = prev_state.step([action])
                if not state.player.active:
                    continue

                actions = prev_actions + [action]

                # Evaluate heuristic
                score = self.heuristic.score(state.cells, state.player, opponents, state.rounds, time.time() + 0.1)
                if self.occupancy_map_depth > 0:
                    occ_map = occ_maps[min(state.rounds - rounds, self.occupancy_map_depth) - 1]
                    freeness = prev_freeness * prod(1 - occ_map[cell[1], cell[0]] for cell in state.changed)
                    score *= freeness
                else:
                    freeness = 1

                actions_scores.append((actions, score))
                if len(actions) < self.depth_limit:  # Search depth
                    states.put((-score, actions, state, freeness))

            if len(prev_actions) == 0 and states.qsize() == 1:  # Only one possible root action
                break

            expanded += 1

        if len(actions_scores) == 0:
            return "change_nothing"

        # Select action which leads to maximal score
        best_action = max(actions_scores, key=lambda x: x[1])[0][0]

        return best_action

    def __repr__(self):
        """Get exact representation."""
        return (
            f"ActionSearchPolicy(heuristic={self.heuristic}, "
            + f"depth_limit={self.depth_limit}, "
            + f"expanded_node_limit={self.expanded_node_limit}, "
            + f"occupancy_map_depth={self.occupancy_map_depth})"
        )
