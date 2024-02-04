from heuristics import CompositeHeuristic, PathLengthHeuristic, RandomHeuristic, RegionHeuristic
from policies import HeuristicPolicy

pol = HeuristicPolicy(
    CompositeHeuristic(
        [
            PathLengthHeuristic(20),
            RegionHeuristic(),
            RandomHeuristic(),
        ],
        weights=[20, 1, 1e-4],
    ),
    occupancy_map_depth=3,
)
