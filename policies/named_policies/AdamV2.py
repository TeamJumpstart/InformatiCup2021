from policies import HeuristicPolicy
from heuristics import CompositeHeuristic, PathLengthHeuristic, RegionHeuristic, RandomHeuristic

pol = HeuristicPolicy(
    CompositeHeuristic([
        PathLengthHeuristic(20),
        RegionHeuristic(),
        RandomHeuristic(),
    ], weights=[20, 1, 1e-4]),
    occupancy_map_depth=3
)
