from heuristics import (
    CompositeHeuristic, OpponentDistanceHeuristic, PathLengthHeuristic, RandomHeuristic, RegionHeuristic
)
from policies import HeuristicPolicy

pol = HeuristicPolicy(
    CompositeHeuristic(
        [
            PathLengthHeuristic(20),
            RegionHeuristic(include_opponent_regions=False),
            OpponentDistanceHeuristic(dist_threshold=16),
            RandomHeuristic(),
        ],
        weights=[20, 1, 1e-3, 1e-4]
    )
)
