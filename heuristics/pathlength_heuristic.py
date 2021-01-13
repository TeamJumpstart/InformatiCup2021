from heuristics.heuristic import Heuristic
from environments.simulator import Spe_edSimulator


class PathLengthHeuristic(Heuristic):
    """Performs a random probe run and evaluates length of the path."""
    def __init__(self, n_steps, expanded_node_limit=1000):
        """Initialize PathLengthHeuristic.

        Args:
            n_steps: Number of steps to look into the future
        """
        self.n_steps = n_steps
        self.expanded_node_limit = expanded_node_limit

    def score(self, cells, player, opponents, rounds):
        """Perform a DFS to seach the longest path reachable."""
        expanded = 0

        def _dfs(sim):
            """Depth-first search"""
            nonlocal expanded

            path_length = sim.rounds - rounds
            if path_length >= self.n_steps or expanded > self.expanded_node_limit:  # Maximum search depth reached
                return path_length  # Early out

            for action in ("change_nothing", "turn_left", "turn_right", "slow_down", "speed_up"):
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
        return f"PathLenghtHeuristic(n_steps={str(self.n_steps)}, \
            expanded_node_limit={str(self.expanded_node_limit)})"
