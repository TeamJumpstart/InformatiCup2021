from policies import HeuristicPolicy
from heuristics import (
    CompositeHeuristic, PathLengthHeuristic, RegionHeuristic, OpponentDistanceHeuristic, RandomHeuristic
)

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
