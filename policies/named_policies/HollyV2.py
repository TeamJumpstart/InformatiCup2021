from heuristics import CompositeHeuristic, RegionHeuristic, WallhugHeuristic
from policies import Maximin_SearchPolicy

# Endgame policy, for composites
pol = Maximin_SearchPolicy(
    CompositeHeuristic(
        [
            RegionHeuristic(include_opponent_regions=False),
            WallhugHeuristic(),
        ],
        weights=[1, 1e-6],
    ),
    depth_limit=5,
    actions=("change_nothing", "turn_left", "turn_right", "slow_down"),
)
