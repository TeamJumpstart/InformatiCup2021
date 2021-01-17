from queue import PriorityQueue
import numpy as np
from policies.policy import Policy
from environments.simulator import Spe_edSimulator
from environments import spe_ed
from heuristics import PathLengthHeuristic
from state_representation import occupancy_map


class ActionSearchPolicy(Policy):
    """Policy that performs a greedy search for that action that will maximize the heuristic."""
    def __init__(self, heuristic, depth_limit=6, expanded_node_limit=100):
        """Initialize ActionSearchPolicy.

        Args:
            heuristic: `Heuristic` that will be evaluated after every step.
        """
        self.heuristic = heuristic
        self.depth_limit = depth_limit
        self.expanded_node_limit = expanded_node_limit

    def act(self, cells, player, opponents, rounds, deadline):
        """Search action sequence based on heuristic scores."""
        states = PriorityQueue()
        states.put((0, [], Spe_edSimulator(cells, [player], rounds)))  # Current state as inital

        actions_scores = []
        expanded = 0
        data = []
        while not states.empty() and expanded < 100:
            prev_score, prev_actions, prev_state = states.get()

            for action in spe_ed.actions:
                state = prev_state.step([action])
                if not state.player.active:
                    continue

                actions = prev_actions + [action]

                # Evaluate heuristic
                score = self.heuristic.score(state.cells, state.player, opponents, state.rounds, deadline)
                data.append((actions, score))

                actions_scores.append((actions, score))
                if len(actions) < self.depth_limit:  # Search depth
                    states.put((-score, actions, state))

            if states.qsize() == 1:  # Only one possible action
                break

            expanded += 1

        if len(actions_scores) == 0:
            return "change_nothing"

        # Select action which leads to maximal score
        best_action = max(actions_scores, key=lambda x: x[1])[0][0]

        return best_action

    def __repr__(self):
        """Get exact representation."""
        return f"ActionSearchPolicy(heuristic={str(self.heuristic)}, " + \
            f"depth_limit={self.depth_limit}, " + \
            f"expanded_node_limit={self.expanded_node_limit})"
