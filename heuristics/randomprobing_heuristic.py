import time

import numpy as np

from environments import spe_ed
from environments.simulator import Spe_edSimulator
from heuristics.heuristic import Heuristic


class RandomProbingHeuristic(Heuristic):
    """Performs a random probe run and evaluates the board state afterwards by the given heuristics."""
    def __init__(self, heuristic, n_steps, n_probes, seed=None):
        """Initialize RandomProbingHeuristic.

        Args:
            heuristic: Heuristic to evaluate for each probe
            n_steps: Number of random steps for each probe run
            n_probes: Number of probe runs
            seed: Random seed of the action selection
        """
        self.heuristic = heuristic
        self.n_steps = n_steps
        self.n_probes = n_probes
        self.rng = np.random.default_rng(seed)

    def score(self, cells, player, opponents, rounds, deadline):
        """Perform one recursive probe run with random actions and returns the number of steps survived."""
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

        score = 0
        for _ in range(self.n_probes):
            # perform a single probe run
            env = perform_probe_run(Spe_edSimulator(cells, [player], rounds))
            probe_score = self.heuristic.score(env.cells, env.players[0], opponents, env.rounds, deadline)
            # remember only the score of the best probe run
            score = max(probe_score, score)

            if time.time() >= deadline:  # Check deadline
                break

        # return the board state score value
        return score

    def __str__(self):
        """Get readable representation."""
        return "RandomProbingHeuristic(" + \
            f"heuristic={str(self.heuristic)}, " + \
            f"n_steps={self.n_steps}, " + \
            f"n_probes={self.n_probes}, " + \
            ")"
