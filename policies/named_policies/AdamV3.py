from heuristics import CompositeHeuristic, PathLengthHeuristic, RegionHeuristic
from policies import HeuristicPolicy

pol = HeuristicPolicy(
    CompositeHeuristic([
        PathLengthHeuristic(20),
        RegionHeuristic(),
    ], weights=[20, 1]), occupancy_map_depth=3
)
