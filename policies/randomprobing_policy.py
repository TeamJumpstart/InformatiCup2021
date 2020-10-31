import numpy as np
from policies.policy import Policy
from environments.simulator import Spe_edSimulator


class RandomProbingPolicy(Policy):
    """Policy that performs `n_probe` random runs to check whether
    the agent will survive for `n_steps` within the current cell state.

    Baseline strategy, each smarter policy should be able to outperform this.
    """
    def __init__(self, n_steps=3, n_probes=10, full_action_set=False, seed=None):
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

    def act(self, cells, player, opponents, rounds):
        """Chooses action based on random probe runs."""

        if self.full_action_set:
            actions = np.array(["change_nothing", "turn_left", "turn_right", "speed_up", "slow_down"])
        else:
            actions = np.array(["change_nothing", "turn_left", "turn_right"])
        sum_actions = np.zeros(actions.shape, dtype=np.float32)

        def perform_probe_run(fixed_actions, random_steps):
            """Performs one recursive probe run with random actions and returns the number of steps survived.

            Args:
                fixed_actions: Sequence of fixed actions taken at start.
                random_steps: Number of random actions taken afterwards
            """
            env = Spe_edSimulator(cells.cells, [player], rounds)

            for action in fixed_actions:
                env = env.step([action])
                if not env.players[0].active:
                    return env.undo().rounds  # Last step was the last survived

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

            return env.rounds  # Return number of survived steps

        # perform 3 or 5 * `n_probes` runs each with maximum of `n_steps`
        for _ in range(self.n_probes):
            for a, action in enumerate(actions):
                sum_actions[a] = max(perform_probe_run([action], self.n_steps), sum_actions[a])
        sum_actions += self.rng.uniform(0, 0.1, size=sum_actions.shape)  # Add tie-breaker

        return actions[np.argmax(sum_actions)]
