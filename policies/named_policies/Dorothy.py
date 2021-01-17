from policies import ActionSearchPolicy
from heuristics import CompositeHeuristic, PathLengthHeuristic, RegionHeuristic

pol = ActionSearchPolicy(
    CompositeHeuristic(
        [
            PathLengthHeuristic(15),
            RegionHeuristic(opening_iterations=1),
            RegionHeuristic(),
        ],
        weights=[15, 1, 5],
    ),
    depth_limit=6,
    expanded_node_limit=100,
)
