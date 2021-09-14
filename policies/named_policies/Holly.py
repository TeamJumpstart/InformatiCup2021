from heuristics import RegionHeuristic
from policies import Maximin_SearchPolicy

# Endgame policy, for composites
pol = Maximin_SearchPolicy(
    RegionHeuristic(include_opponent_regions=False),
    depth_limit=5,
    actions=("change_nothing", "turn_left", "turn_right", "slow_down"),
)
