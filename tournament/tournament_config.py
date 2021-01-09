from policies import RandomPolicy, HeuristicPolicy, SpiralPolicy
from heuristics import PathLengthHeuristic, RandomHeuristic, CompositeHeuristic, RegionHeuristic, OpponentDistanceHeuristic
import random

# participating policies with short nick names
policies = [
    {
        "name":
            "Composite",
        "pol":
            HeuristicPolicy(
                CompositeHeuristic(
                    [
                        PathLengthHeuristic(20),
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
        "pol": HeuristicPolicy(PathLengthHeuristic(10))
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
# different map sizes to be played
width_height_pairs = [
    (30, 30), (50, 50)
]  # or use randomly generated ranges: [(randint(20,51), randint(20,51)), (randint(50, 101), randint(50, 101))]
# number of games to be played for each constellation
number_games = 2
