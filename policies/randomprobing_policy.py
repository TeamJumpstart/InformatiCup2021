import numpy as np
from policies.policy import Policy
from environments.simulator import Spe_edSimulator
from heuristics import RoundsHeuristic, RandomHeuristic


class RandomProbingPolicy(Policy):
    """Policy that performs `n_probe` random runs to check whether
    the agent will survive for `n_steps` within the current cell state.

    Baseline strategy, each smarter policy should be able to outperform this.
    """
    def __init__(
        self,
        n_steps=[3, 0],
        n_probes=[10, 1],
        full_action_set=False,
        heuristics=[RoundsHeuristic(), RandomHeuristic()],
        weights=None,
        early_out_thresholds=None,
        seed=None
    ):
        """Initialize RandomProbingPolicy.

        Args:
            n_steps: Defines the number of steps each probe run is performed at most.
            n_probes: Defines the number of probe runs per available `p_action`.
            full_action_set: `True`, selects one of all available actions:
                ("change_nothing", "turn_left", "turn_right", "speed_up", "slow_down"),
                `False`, selects one of the following actions:
                ("change_nothing", "turn_left", "turn_right").
            seed: Seed for the random number generator. Use a fixed seed for reproducibility,
                  or pass `None` for a random seed.
        """

        self.n_steps = n_steps
        self.n_probes = n_probes
        self.full_action_set = full_action_set
        self.rng = np.random.default_rng(seed)
        self.heuristics = heuristics

        if weights is not None and len(weights) != len(heuristics):
            raise ValueError(f"Number of weights {weights} does mot match number of heuristics {heuristics}")
        if weights is None:
            self.weights = np.ones(len(heuristics))
        else:
            self.weights = weights

        if early_out_thresholds is not None and len(early_out_thresholds) != len(heuristics):
            raise ValueError(
                f"Number of weights {early_out_thresholds} does mot match number of heuristics {heuristics}"
            )
        if early_out_thresholds is None:
            self.early_out_thresholds = np.ones(len(heuristics))
        else:
            self.early_out_thresholds = early_out_thresholds

    def act(self, cells, player, opponents, rounds):
        """Chooses action based on random probe runs."""

        if self.full_action_set:
            actions = np.array(["change_nothing", "turn_left", "turn_right", "speed_up", "slow_down"])
        else:
            actions = np.array(["change_nothing", "turn_left", "turn_right"])
        sum_actions = np.zeros((len(self.heuristics), ) + actions.shape, dtype=np.float32)

        def perform_probe_run(fixed_actions, random_steps, heuristic):
            """Performs one recursive probe run with random actions and returns the number of steps survived.

            Args:
                fixed_actions: Sequence of fixed actions taken at start.
                random_steps: Number of random actions taken afterwards
            """
            env = Spe_edSimulator(cells.cells, [player], rounds)

            for action in fixed_actions:
                env = env.step([action])
                if not env.players[0].active:
                    return (0.0, 0.0)  # fixed actions result in certain death

            for _ in range(random_steps):
                dead_end = True
                for action in self.rng.permutation(actions):
                    env = env.step([action])
                    if env.players[0].active:
                        # We survive, go to next step
                        dead_end = False
                        break

                    # We die, try alternative action
                    env = env.undo()
                if dead_end:  # No way out
                    break

            # return the board state score value
            return heuristic.score(env.cells, env.players[0], opponents, env.rounds)

        # perform 3 or 5 * `n_probes` runs each with maximum of `n_steps`
        for num_heuristic in range(len(self.heuristics)):
            for _ in range(self.n_probes[num_heuristic]):
                for a, action in enumerate(actions):
                    # perform a single probe run and remember the biggest resulting score
                    score, score_normalized = perform_probe_run(
                        [action], self.n_steps[num_heuristic], self.heuristics[num_heuristic]
                    )
                    if score_normalized is not None:
                        sum_actions[num_heuristic, a] = max(score_normalized, sum_actions[num_heuristic, a])
                        if score_normalized >= self.early_out_thresholds[num_heuristic]:
                            break  # early out for current action
                    else:
                        sum_actions[num_heuristic, a] = max(score, sum_actions[num_heuristic, a])

        # apply weights to each heuristic score
        sum_actions = np.reshape(self.weights, (len(self.weights), 1)) * sum_actions
        return actions[np.argmax(np.sum(sum_actions, axis=0))]
