import numpy as np
from scipy import ndimage
from scipy.ndimage import morphology
from heuristics.heuristic import Heuristic


def applyMorphology(cells, closing=0, opening=0, erosion=0, dilation=0):
    """ Applys morphological operations on the given cells and returns them.
        Multiple operations and multiple iterations of the operation can be specified at once.
        Operations are executed in the following order: [closing, opening, erosion, dilation].
    """
    # apply padding
    iterations = max(closing, opening, erosion, dilation)
    if iterations:
        cells = np.pad(cells, (iterations, ))
        # perform morphological operations/iterations
        if closing:
            cells = morphology.binary_closing(cells, iterations=closing)
        if opening:
            cells = morphology.binary_opening(cells, iterations=opening)
        if erosion:
            cells = morphology.binary_erosion(cells, iterations=erosion)
        if dilation:
            cells = morphology.binary_dilation(cells, iterations=dilation)
        # remove padding
        cells = cells[iterations:-iterations, iterations:-iterations]
    return cells


def labelCells(cells, players):
    """ Returns cells labeled on the region they belong to. Player positions are masked to belong to a region. """
    # inverse map (mask occupied cells)
    empty_cells = cells == 0
    # Clear cell for all players
    for p in players:
        empty_cells[p.y, p.x] = True
    # compute distinct regions
    labelled_cells, _ = ndimage.label(empty_cells)
    return labelled_cells


def numOppInRegion(cells, players):
    cells_lebelled = labelCells(cells, players)
    # Get the region each player is in
    regions = np.array([cells_lebelled[p.y, p.x] for p in players])
    # Compare each player label with the label of player one to get number of opponents
    opponents_in_our_region = np.sum([1 for region in regions[1:] if region == regions[0]])
    return opponents_in_our_region


class EndgameConditionHeuristic(Heuristic):
    """Returns a constant number."""
    def __init__(self, value=1):
        """Initialize EndgameConditionHeuristic."""
        self.value = value

    def score(self, cells, player, opponents, rounds):
        """Returns a constant number. """
        players = [player] + opponents

        num_opp_region = numOppInRegion(cells, players)
        num_opp_closing_opening = numOppInRegion(applyMorphology(cells, closing=1, opening=1), players)

        if (num_opp_region + num_opp_closing_opening) < 1:
            return 1
        else:
            return 0

    def __str__(self):
        """Get readable representation."""
        return f"ConstantHeuristic({str(self.value)})"
