from policies import RandomPolicy, HeuristicPolicy, SpiralPolicy
from heuristics import PathLengthHeuristic, RandomHeuristic, CompositeHeuristic, RegionHeuristic, OpponentDistanceHeuristic

policy_list = [
    {
        "name":
            "Composite",
        "pol":
            HeuristicPolicy(
                CompositeHeuristic(
                    [
                        PathLengthHeuristic(20, 100),
                        RegionHeuristic(),
                        OpponentDistanceHeuristic(dist_threshold=16),
                        RandomHeuristic(),
                    ],
                    weights=[20, 1, 1e-3, 1e-4]
                )
            )
    },
    {
        "name": "Random",
        "pol": RandomPolicy()
    },
    {
        "name": "Spiral",
        "pol": SpiralPolicy()
    },
    {
        "name": "PathLength",
        "pol": HeuristicPolicy(PathLengthHeuristic())
    },
    {
        "name": "Region",
        "pol": HeuristicPolicy(RegionHeuristic())
    },
    {
        "name": "OppDist",
        "pol": HeuristicPolicy(OpponentDistanceHeuristic())
    },
]
number_games = 1
width_height_pairs = [(30, 30), (50, 50)]
