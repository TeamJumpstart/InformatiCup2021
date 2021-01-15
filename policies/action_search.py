from queue import PriorityQueue
import numpy as np
from policies.policy import Policy
from environments.simulator import Spe_edSimulator
from environments import spe_ed
from heuristics import PathLengthHeuristic
from state_representation import occupancy_map


class ActionSearchPolicy(Policy):
    """Policy that performs a single action in every valid direction and
    evaluates the score of the given metric applied on the future board state to choose next actions.
    """
    def __init__(self, heuristic):
        """Initialize HeuristicPolicy.

        Args:
            heuristic: `Heuristic` that will be evaluated after one action of the player was performed.
        """
        self.heuristic = heuristic
        self.initial_heurisitc = PathLengthHeuristic(20)

    def act(self, cells, player, opponents, rounds):
        """Chooses action based on weighted heuristic scores."""

        occ = occupancy_map(cells, opponents, rounds)

        states = PriorityQueue()
        states.put((0, [], Spe_edSimulator(cells, [player], rounds)))

        actions_scores = []
        expanded = 0
        initial_scores = {}
        data = []
        while not states.empty() and expanded < 100:
            prev_score, prev_actions, prev_state = states.get()

            for action in spe_ed.actions:
                state = prev_state.step([action])
                if not state.player.active:
                    continue

                actions = prev_actions + [action]

                # Evaluate heuristic
                if len(prev_actions) == 0:
                    initial_scores[action] = self.initial_heurisitc.score(
                        state.cells, state.player, opponents, state.rounds
                    )
                score = self.heuristic.score(state.cells, state.player, opponents, state.rounds)
                score *= initial_scores[actions[0]]
                score *= np.prod(1 - occ[np.logical_xor(state.cells, state.cells)])  # Factor in occupancy
                data.append((actions, score))

                actions_scores.append((actions, score))
                if len(actions) < 6:  # Search depth
                    states.put((-score, actions, state))

            if states.qsize() == 1:  # Only one possible action
                break

            expanded += 1

        if len(actions_scores) == 0:
            return "change_nothing"

        best_action = max(actions_scores, key=lambda x: x[1])[0][0]
        best_actions = max(actions_scores, key=lambda x: x[1])[0]

        if len(best_actions) > 2:
            with open("actionsearch", "w") as out_file:
                for actions, score in data:
                    # Colors
                    node_color = "green" if best_actions == actions else "black"
                    edge_color = "green" if best_actions[:len(actions)] == actions else "black"

                    # Node
                    node_name = "_".join(actions)
                    out_file.write(f"{node_name}[label={score:.06f},color={node_color}];\n")

                    # Edge
                    prev_node_name = "_".join(actions[:-1]) if len(actions) > 1 else "root"
                    out_file.write(f"{prev_node_name} -> {node_name}[label={actions[-1]},color={edge_color}];\n")

            sim = Spe_edSimulator(cells, [player], rounds)
            for action in best_actions:
                sim = sim.step([action])

            import matplotlib.pyplot as plt
            plt.subplot(1, 2, 1)
            plt.imshow(cells)

            plt.subplot(1, 2, 2)
            plt.imshow(sim.cells)

            plt.show()

            quit()

        return best_action

    def __str__(self):
        """Get readable representation."""
        return f"HeuristicPolicy[{str(self.heuristic)}]"
