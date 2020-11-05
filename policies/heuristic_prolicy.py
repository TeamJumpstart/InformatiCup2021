import numpy as np
from policies.policy import Policy
from environments.simulator import Spe_edSimulator
from heuristics import RandomHeuristic, RandomProbingHeuristic


class HeuristicPolicy(Policy):
    """Policy that performs a single action in every valid direction and
    evaluates the score of given metrics applied on the future board state to choose next actions.
    """
    def __init__(
        self, full_action_set=False, heuristics=[RandomHeuristic(), RandomProbingHeuristic()], weights=None, seed=None
    ):
        """Initialize HeuristicPolicy.

        Args:
            full_action_set: `True`, selects one of all available actions:
                ("change_nothing", "turn_left", "turn_right", "speed_up", "slow_down"),
                `False`, selects one of the following actions:
                ("change_nothing", "turn_left", "turn_right").
            heuristics: `Heuristic`s that will be evaluated after one action of the player was performed.
            weights: Weights for given 'Heuristic's. Use custom weights
                     or pass `None` for uniform weighting.
            seed: Seed for the random number generator. Use a fixed seed for reproducibility,
                  or pass `None` for a random seed.
        """

        self.rng = np.random.default_rng(seed)
        self.heuristics = heuristics

        if full_action_set:
            self.actions = np.array(["change_nothing", "turn_left", "turn_right", "speed_up", "slow_down"])
        else:
            self.actions = np.array(["change_nothing", "turn_left", "turn_right"])

        if weights is not None and len(weights) != len(heuristics):
            raise ValueError(f"Number of weights {weights} does mot match number of heuristics {heuristics}")
        if weights is None:
            self.weights = np.ones(len(heuristics))
        else:
            self.weights = weights

    def act(self, cells, player, opponents, rounds):
        """Chooses action based on weighted heuristic scores."""

        sum_actions = np.zeros((len(self.heuristics), len(self.actions)), dtype=np.float32)

        for h, heuristic in enumerate(self.heuristics):
            for a, action in enumerate(self.actions):
                # perform a single action
                env = Spe_edSimulator(cells.cells, [player], rounds).step([action])

                # evaluate the heuristic, if the player is active
                if env.players[0].active:
                    score, score_normalized = heuristic.score(env.cells, env.players[0], opponents, env.rounds)
                else:
                    score, score_normalized = (0, 0.0)
                sum_actions[h, a] = score if score_normalized is None else score_normalized

        # apply weights to each heuristic score
        sum_actions = np.reshape(self.weights, (len(self.weights), 1)) * sum_actions

        # select action with the highest score sum
        selected_action = self.actions[np.argmax(np.sum(sum_actions, axis=0))]

        return selected_action
