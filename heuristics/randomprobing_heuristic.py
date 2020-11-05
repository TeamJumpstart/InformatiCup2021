from heuristics.heuristic import Heuristic
import numpy as np
from environments.simulator import Spe_edSimulator


class RandomProbingHeuristic(Heuristic):
    """Performs a random probe run and evaluates the board state afterwards by the given heuristics."""
    def __init__(self, n_steps=5, n_probes=10, full_action_set=False, heuristics=[None], weights=None, seed=None):
        """Initialize RandomProbingHeuristic."""
        self.n_steps = n_steps
        self.n_probes = n_probes
        self.heuristics = heuristics
        self.rng = np.random.default_rng(seed)

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
                for action in self.rng.permutation(self.actions):
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

        final_score = 0
        all_scores_normalized = True
        for _ in range(self.n_probes):
            probe_score = 0
            # perform a single probe run
            env = perform_probe_run(Spe_edSimulator(cells, [player], rounds))

            # evaluate the current board state at the end of the random probe for each metric
            for h, heuristic in enumerate(self.heuristics):
                if heuristic is None:  # default score - probe length
                    probe_length = env.rounds - rounds
                    absolute_score, normalized_score = (probe_length, probe_length / self.n_steps)
                else:  # use a custom heuristic provided
                    absolute_score, normalized_score = heuristic.score(env.cells, env.players[0], opponents, env.rounds)

                # sum up the scores and check if a normalized value was provided
                if normalized_score is None:
                    all_scores_normalized = False
                    probe_score += self.weights[h] * absolute_score
                else:
                    probe_score += self.weights[h] * normalized_score

            # remember only the score of the best probe run
            final_score = max(probe_score, final_score)

        # return the board state score value
        if all_scores_normalized:  # all scores were provided as a normalized output, normalize the score, too.
            return (final_score, final_score / len(self.heuristics))
        else:  # at least one heuristic did not provide a normalized score. No way to compute a normalized score.
            return (final_score, None)
