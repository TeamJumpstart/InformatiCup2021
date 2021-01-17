import time
from queue import PriorityQueue
from policies.policy import Policy
from environments.simulator import Spe_edSimulator
from environments import spe_ed


class Maximin_SearchPolicy(Policy):
    """Policy that performs a greedy search for that action that will maximize the heuristic."""
    def __init__(self, heuristic, depth_limit=6, actions=None):
        """Initialize ActionSearchPolicy.

        Args:
            heuristic: `Heuristic` that will be evaluated after every step.
        """
        self.heuristic = heuristic
        self.depth_limit = depth_limit
        self.actions = actions

    def act(self, cells, player, opponents, rounds, deadline):
        """Search action sequence based on heuristic scores."""

        action_selection = spe_ed.actions if self.actions is None else self.actions

        states = PriorityQueue()
        states.put((0, -1, 0, [], Spe_edSimulator(cells, [player], rounds)))  # Current state as inital

        best_action = action_selection[0]
        best_score = 0
        expanded = 0
        lower_bound = 0
        count = 1
        while not states.empty() and time.time() < deadline:
            _, prev_score_neg, _, prev_actions, prev_state = states.get()

            if -prev_score_neg <= lower_bound:  # Check bound
                continue

            for action in action_selection:
                state = prev_state.step([action])
                if not state.player.active:
                    continue

                actions = prev_actions + [action]

                # Evaluate heuristic
                score = self.heuristic.score(state.cells, state.player, opponents, state.rounds, time.time() + 0.1)
                score = min(score, -prev_score_neg)

                if lower_bound == 0 and score > best_score:
                    best_action = action
                    best_score = score

                if score <= lower_bound:  # Bound
                    continue

                if len(actions) < self.depth_limit:  # Search depth limit not reached
                    states.put((-len(actions), -score, count, actions, state))  # Expand
                    count += 1
                else:  # Reched end
                    if score > lower_bound:  # Update lower bound
                        lower_bound = score
                        best_action = actions[0]

            if len(prev_actions) == 0 and states.qsize() == 1:  # Only one possible root action
                return states.get()[3][0]  # Return only remaining action

            expanded += 1

        return best_action

    def __repr__(self):
        """Get exact representation."""
        return f"Maximin_SearchPolicy(heuristic={str(self.heuristic)}, " + \
            f"depth_limit={self.depth_limit})"
