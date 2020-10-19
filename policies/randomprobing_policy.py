import numpy as np
from policies.policy import Policy


class RandomProbingPolicy(Policy):
    """Policy that performs {n_probe} random runs to check whether
    the agent will survive for {n_steps} within the current cell state.    

    TODO: extend by `probing_policy` argument, which defines the behaviour for the probe runs.

    Baseline strategy, each smarter policy should be able to outperform this.
    """
    def __init__(self, n_steps=10, n_probes=3, seed=None):
        """Initialize RandomProbingPolicy.

        Args:
            n_steps: Defines the number of steps each probe run is performed at most.
            n_probes: Defines the number of probe runs per available {p_action}.
            TODO: p_actions: Defines a set of possible actions the agent can perform.
            seed: Seed for the random number generator. Use a fixed seed for reproducibility,
                  or pass `None` for a random seed.
            TODO: p: Propability weights of each action. Pass `None` for uniform distribution.
        """

        self.n_steps = n_steps
        self.n_probes = n_probes
        #self.p_actions = p_actions
        self.rng = np.random.default_rng(seed)
        #if p is not None and len(p) != len(p_actions):
        #    raise ValueError(f"Number of probabilities {p} does mot match number of actions {p_actions}")
        #self.p = p

    def act(self, cells, player, opponents, rounds):        
        """Choose action randomly."""

        actions = np.array(["change_nothing", "turn_left", "turn_right"])
        directions = np.array([player.direction, player.direction.turn_left(), player.direction.turn_right()])
        sum_actions = np.zeros(actions.shape)

        def perform_probe_run(pos, direction, n_steps, steps=None):
            """Performs one recursive probe run with random actions and returns the number of steps survived."""            
            pos = pos + direction.cartesian
            if cells.is_free(pos) & (not np.any(steps == pos)) & (n_steps > 0):
                np.append(steps, pos, axis=0)
                action = self.rng.choice(directions)
                return int(1 + perform_probe_run(pos, action, n_steps - 1, steps))
            else:
                return int(0)

        # perform 3 * {n_probes} runs each with maximum of {n_steps}
        for _ in range(self.n_probes):
            for d, direction in enumerate(directions):
                sum_actions[d] += perform_probe_run(player.position, direction, self.n_steps, [])

        return actions[np.argmax(sum_actions)]
