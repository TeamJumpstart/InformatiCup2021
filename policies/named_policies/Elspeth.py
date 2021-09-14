from heuristics import CompositeHeuristic, PathLengthHeuristic, RandomHeuristic, RandomProbingHeuristic, RegionHeuristic
from policies import HeuristicPolicy

pol = HeuristicPolicy(
    CompositeHeuristic(
        [
            PathLengthHeuristic(20),
            RandomProbingHeuristic(
                RegionHeuristic(),
                n_steps=6,
                n_probes=20,
            ),
            RegionHeuristic(),
            RandomHeuristic(),
        ],
        weights=[20, 10, 1, 1e-4]
    ),
    occupancy_map_depth=3
)
