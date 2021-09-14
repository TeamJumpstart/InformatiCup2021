from heuristics import CompositeHeuristic, PathLengthHeuristic, RandomHeuristic, VoronoiHeuristic
from policies import HeuristicPolicy

pol = HeuristicPolicy(
    CompositeHeuristic([
        PathLengthHeuristic(20),
        VoronoiHeuristic(),
        RandomHeuristic(),
    ], weights=[20, 1, 1e-4]),
    occupancy_map_depth=3
)
