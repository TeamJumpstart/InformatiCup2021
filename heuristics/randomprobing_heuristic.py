from heuristics.heuristic import Heuristic
import numpy as np
from environments.simulator import Spe_edSimulator
from environments import spe_ed


class RandomProbingHeuristic(Heuristic):
    """Performs a random probe run and evaluates the board state afterwards by the given heuristics."""
    def __init__(self, heuristic, n_steps, n_probes, seed=None):
        """Initialize RandomProbingHeuristic."""
        self.heuristic = heuristic
        self.n_steps = n_steps
        self.n_probes = n_probes
        self.rng = np.random.default_rng(seed)

    def score(self, cells, player, opponents, rounds):
        """Performs one recursive probe run with random actions and returns the number of steps survived.

        Args:
            fixed_actions: Sequence of fixed actions taken at start.
            random_steps: Number of random actions taken afterwards
        """
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
            probe_score = self.heuristic.score(env.cells, env.players[0], opponents, env.rounds)
            # remember only the score of the best probe run
            score = max(probe_score, score)

        # return the board state score value
        return score

    def __str__(self):
        """Get readable representation."""
        return f"RandomProbingHeuristic(heuristic={str(self.heuristic)}, \
            n_steps={str(self.n_steps)}, \
            n_probes={str(self.n_probes)})"
