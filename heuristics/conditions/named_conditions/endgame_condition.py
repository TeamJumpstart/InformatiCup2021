import numpy as np

from heuristics.conditions import (
    CompositeCondition, Condition, OpponentsInPlayerRegionCondition, PlayerInBiggestRegionCondition
)


class EndgameCondition(Condition):
    """ Check if we are in the Endgame."""
    def __init__(self):
        """Initialize EndgameCondition. """
        in_biggest_region = PlayerInBiggestRegionCondition()
        opp_num_in_region = OpponentsInPlayerRegionCondition()

        self.only_two_players_in_region = CompositeCondition(
            [in_biggest_region, opp_num_in_region], thresholds=[True, 0.0], compare_op=[np.equal, np.equal]
        )

    def score(self, cells, player, opponents, rounds, deadline):
        """Return if the player is in the endgame phase."""
        return self.only_two_players_in_region.score(cells, player, opponents, rounds, deadline)

    def __str__(self):
        """Get readable representation."""
        return "EndgameCondition()"
