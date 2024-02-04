from heuristics import CompositeHeuristic, PathLengthHeuristic, RandomHeuristic, RandomProbingHeuristic, RegionHeuristic
from policies import HeuristicPolicy

# Variant of Elspeth with closing_iterations working
pol = HeuristicPolicy(
    CompositeHeuristic(
        [
            PathLengthHeuristic(20),
            RandomProbingHeuristic(
                CompositeHeuristic(
                    [
                        RegionHeuristic(),
                        RegionHeuristic(closing_iterations=1),
                    ]
                ),
                n_steps=6,
                n_probes=20,
            ),
            RegionHeuristic(),
            RandomHeuristic(),
        ],
        weights=[20, 10, 1, 1e-4],
    ),
    occupancy_map_depth=3,
)
