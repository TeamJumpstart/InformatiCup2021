from policies import HeuristicPolicy
from heuristics import (
    CompositeHeuristic, PathLengthHeuristic, RegionHeuristic, RandomProbingHeuristic, VoronoiHeuristic
)

pol = HeuristicPolicy(
    CompositeHeuristic(
        [
            # longest path search - longer path is always better
            PathLengthHeuristic(20),
            RandomProbingHeuristic(
                RegionHeuristic(),
                n_steps=3,
                n_probes=10,
            ),
            # kill near opponents and minimize their regions
            RandomProbingHeuristic(
                CompositeHeuristic([
                    VoronoiHeuristic(max_steps=12, minimize_opponents=True),
                    RegionHeuristic(),
                ]),
                n_steps=2,
                n_probes=10,
            ),
            # supports the endgame
            VoronoiHeuristic(),
            RandomProbingHeuristic(
                CompositeHeuristic([
                    RegionHeuristic(),
                ]),
                n_steps=1,
                n_probes=1,
            ),
        ],
        weights=[20, 5, 4, 1, 1]
    ),
    # defines how aggresive our policy is (bigger value - avoids enemys more)
    occupancy_map_depth=2
)
