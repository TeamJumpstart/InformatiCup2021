from heuristics.heuristic import Heuristic
import numpy as np
from environments.simulator import Spe_edSimulator
from environments import spe_ed


class PathLengthHeuristic(Heuristic):
    """Performs a random probe run and evaluates length of the path."""
    def __init__(self, n_steps=20, n_probes=100, seed=None):
        """Initialize PathLengthHeuristic."""
        self.n_steps = n_steps
        self.n_probes = n_probes
        self.rng = np.random.default_rng(seed)

    def score(self, cells, player, opponents, rounds):
        """Performs probe runs with random actions and returns the number of steps survived."""
        def perform_probe_run(env):
            """Simulate the given environment for maximum of `n_steps` with valid random steps or
            until the player cannot make a valid move, return the environment.
            """
            for _ in range(self.n_steps):
                dead_end = True
                for action in self.rng.permutation(spe_ed.actions):
                    env = env.step([action])
                    if env.players[0].active:
                        # We survive, go to next step
                        dead_end = False
                        break
                    else:
                        env = env.undo()  # We die, try alternative action
                if dead_end:  # No way out
                    break
            return env

        probe_length = 0
        for _ in range(self.n_probes):
            # perform a single probe run
            env = perform_probe_run(Spe_edSimulator(cells, [player], rounds))
            # remember only the length of the best probe run
            probe_length = max(env.rounds - rounds, probe_length)
            # early out - we found one path with the maximal distance
            if probe_length >= self.n_steps:
                return 1.0

        # return the board state score value
        return probe_length / self.n_steps

    def __str__(self):
        """Get readable representation."""
        return "PathLenghtHeuristic"
