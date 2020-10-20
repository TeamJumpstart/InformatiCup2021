import numpy as np
from policies.policy import Policy


class RandomProbingPolicy(Policy):
<<<<<<< HEAD
    """Policy that performs {n_probe} random runs to check whether
    the agent will survive for {n_steps} within the current cell state.
=======
    """Policy that performs `n_probe` random runs to check whether
    the agent will survive for `n_steps` within the current cell state.
>>>>>>> bc49d37... RandomProbingPolicy: formating

    TODO: extend by `probing_policy` argument, which defines the behaviour for the probe runs.

    Baseline strategy, each smarter policy should be able to outperform this.
    """
    def __init__(self, n_steps=10, n_probes=3, seed=None):
        """Initialize RandomProbingPolicy.

        Args:
            n_steps: Defines the number of steps each probe run is performed at most.
            n_probes: Defines the number of probe runs per available `p_action`.
            seed: Seed for the random number generator. Use a fixed seed for reproducibility,
                  or pass `None` for a random seed.
        """

        self.n_steps = n_steps
        self.n_probes = n_probes
        self.rng = np.random.default_rng(seed)

    def act(self, cells, player, opponents, rounds):
<<<<<<< HEAD
        """Choose action randomly."""
=======
        """Chooses action based on random probe runs."""
>>>>>>> bc49d37... RandomProbingPolicy: formating

        actions = np.array(["change_nothing", "turn_left", "turn_right"])
        directions = np.array([player.direction, player.direction.turn_left(), player.direction.turn_right()])
        sum_actions = np.zeros(actions.shape)

<<<<<<< HEAD
        def perform_probe_run(pos, direction, n_steps, steps=None):
=======
        def perform_probe_run(pos, direction, n_steps, steps=[]):
>>>>>>> bc49d37... RandomProbingPolicy: formating
            """Performs one recursive probe run with random actions and returns the number of steps survived."""
            pos = pos + direction.cartesian
            if cells.is_free(pos) and (not np.any(steps == pos)) and (n_steps > 0):
                steps = steps + [pos]
                action = self.rng.choice(directions)
                return 1 + perform_probe_run(pos, action, n_steps - 1, steps)
            else:
                return 0

        # perform 3 * `n_probes` runs each with maximum of `n_steps`
        for _ in range(self.n_probes):
            for d, direction in enumerate(directions):
                sum_actions[d] += perform_probe_run(player.position, direction, self.n_steps)

        return actions[np.argmax(sum_actions)]
