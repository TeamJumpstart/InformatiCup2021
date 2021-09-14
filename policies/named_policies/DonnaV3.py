from heuristics import CompositeHeuristic, PathLengthHeuristic, RegionHeuristic
from policies import ActionSearchPolicy

pol = ActionSearchPolicy(
    CompositeHeuristic(
        [
            PathLengthHeuristic(15),
            RegionHeuristic(),
        ],
        weights=[15, 1],
    ),
    depth_limit=6,
    expanded_node_limit=5000,
    occupancy_map_depth=3,
)
