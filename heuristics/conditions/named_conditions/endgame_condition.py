from heuristics.conditions import Condition, OpponentsInPlayerRegionCondition, CompositeCondition
import numpy as np


class EndgameCondition(Condition):
    """ Check if we are in the Endgame."""
    def __init__(self):
        """Initialize EndgameCondition. """
        self.oppNum_closedregion = OpponentsInPlayerRegionCondition(closing=1)
        self.oppNum_region = OpponentsInPlayerRegionCondition()

        self.only_two_players_in_region = CompositeCondition(
            [self.oppNum_closedregion, self.oppNum_region], compare_op=np.less_equal
        )

    def score(self, cells, player, opponents, rounds):
        """Return if the player is in the endgame phase."""
        return self.only_two_players_in_region.score(cells, player, opponents, rounds)

    def __str__(self):
        """Get readable representation."""
        return "EndgameCondition()"
