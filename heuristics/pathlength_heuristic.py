import time

from environments.simulator import Spe_edSimulator
from heuristics.heuristic import Heuristic

# Reorder actions to hit early out condition as fast as possible
# change_nothing first, as it's the most common action
# turn_* before speed_up, as this leads to longer paths
# slow_down before speed_up, as it terminates earlier.
ordered_actions = ("change_nothing", "turn_left", "turn_right", "slow_down", "speed_up")


class PathLengthHeuristic(Heuristic):
    """Performs a random probe run and evaluates length of the path."""

    def __init__(self, n_steps, time_limit=None):
        """Initialize PathLengthHeuristic.

        Args:
            n_steps: Number of steps to look into the future
            expanded_node_limit: Threshold to prevent long execution times
        """
        self.n_steps = n_steps
        self.time_limit = time_limit

    def score(self, cells, player, opponents, rounds, deadline):
        """Perform a DFS to seach the longest path reachable."""
        expanded = 0

        if self.time_limit is not None:
            deadline = min(time.time() + self.time_limit, deadline)

        def _dfs(sim):
            """Depth-first search"""
            nonlocal expanded

            path_length = sim.rounds - rounds
            if path_length >= self.n_steps or time.time() > deadline:  # Maximum search depth reached
                return path_length  # Early out

            for action in ordered_actions:
                sub_sim = sim.step([action])
                if not sub_sim.player.active:  # Backtrack
                    continue

                sub_path_length = _dfs(sub_sim)
                if sub_path_length >= self.n_steps:  # Maximum search depth reached
                    return sub_path_length  # Early out

                if sub_path_length > path_length:
                    path_length = sub_path_length

            expanded += 1  # Count expanded nodes

            return path_length

        path_length = _dfs(Spe_edSimulator(cells, [player], rounds))

        # return the board state score value
        return path_length / self.n_steps

    def __str__(self):
        """Get readable representation."""
        return f"PathLenghtHeuristic(n_steps={self.n_steps}, time_limit={self.time_limit})"
