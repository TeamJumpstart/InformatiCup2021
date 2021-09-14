import time

import numpy as np
from scipy import ndimage
from scipy.ndimage import morphology

from environments import spe_ed
from environments.simulator import Spe_edSimulator
from heuristics import PathLengthHeuristic
from policies.policy import Policy


def applyMorphology(cells, closing=0, opening=0, erosion=0, dilation=0):
    """Applys morphological operations on the given cells and returns them.

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
    """Returns cells labeled on the region they belong to.

    Player positions are masked to belong to a region.
    """
    # inverse map (mask occupied cells)
    empty_cells = cells == 0
    # Clear cell for all players
    for p in players:
        empty_cells[p.y, p.x] = True
    # compute distinct regions
    labelled_cells, _ = ndimage.label(empty_cells)
    return labelled_cells


def computeRegionSize(cells, players):
    """Computes the size of the region the controlled player is in."""
    labelled_cells = labelCells(cells, players)
    # Get the region we're in and compute its size
    region_label = labelled_cells[players[0].y, players[0].x]
    player_region_size = np.sum(labelled_cells == region_label)
    return player_region_size


def computeRegionNumber(cells, players):
    """Computes the number of unique regions."""
    # inverse map (mask occupied cells)
    cells = np.pad(cells, (1, ))
    empty_cells = cells == 0
    # compute distinct regions
    _, num_cells = ndimage.label(empty_cells)
    return num_cells


def computeOccupiedCells(cells, players):
    """Computes the number of occupied cells."""
    return np.sum(cells)


def computePathLength(cells, players):
    """Evaluates the 'PathLengthHeuristic' with constant parameters."""
    path_length_heuristic = PathLengthHeuristic(n_steps=200)
    return path_length_heuristic.score(cells, players[0], [], 0, deadline=time.time + 0.1)  # TODO Magic number


def tiebreakerFunc(env, remaining_actions, score_func=computeRegionSize, eval_func=max, morph_kwargs={}):
    """A general tiebreaker function to decide given an environment which actions are preferable and should be executed.

    Args:
        env: The current game state given in `Spe_edSimulator`
        remaining_actions: A list of actions to choose from.
        score_func: A function, which accepts 'cells' and 'players' and returns a scalar value.
        eval_func: accepts either `max` or `min` to decide, whether prefer a lower or higher score.
        morph_kwargs: keyword arguments, to define morphological operations on the cells beforehand.

    Return:
        remaining_actions: A possibly reduced list of actions which were choosen to process further.
        scores: A dictionary of action-score tuples for every input action.
    """
    if len(remaining_actions) <= 1:
        return remaining_actions, {a: 0 for a in remaining_actions}

    if eval_func is max:
        scores = {action: float('-Inf') for action in remaining_actions}
    elif eval_func is min:
        scores = {action: float('Inf') for action in remaining_actions}
    else:
        print("ERROR: function not handled")

    for action in scores:
        env = env.step([action])
        if env.players[0].active:
            cells = applyMorphology(env.cells, **morph_kwargs)
            scores[action] = score_func(cells, env.players)
        env = env.undo()

    score_list = list(scores.values())
    remaining_actions = [k for k, v in scores.items() if v == eval_func(score_list)]
    return remaining_actions, scores


class EndgamePolicy(Policy):
    """Provides a policy which can be used to master the endgame.

    In the case we are stuck in one region and cannot interact with other players,
    it tries to maximize the number of rounds that the policy survives until filling all available space.
    An optimal or even satisfiable behavior is not guaranteed for any other circumstances.
    """
    def __init__(self, actions=None):
        """Initialize endgame policy.

        Args:
            actions: specifies which actions are considered at all. Default: uses all actions except 'speed_up'.
        """
        self.actions = [a for a in spe_ed.actions if a != "speed_up"] if actions is None else actions

    def act(self, cells, player, opponents, rounds, deadline):
        """Choose action."""
        env = Spe_edSimulator(cells, [player], rounds)
        remaining_actions = self.actions

        # bigger region is always better
        remaining_actions, _ = tiebreakerFunc(env, remaining_actions, computeRegionSize, max)
        # less regions is preferable
        remaining_actions, _ = tiebreakerFunc(env, remaining_actions, computeRegionNumber, min)

        # tie breaker: morphological operations
        for i in range(2, 0, -1):
            remaining_actions, _ = tiebreakerFunc(env, remaining_actions, computeOccupiedCells, min, {'closing': i})
            remaining_actions, _ = tiebreakerFunc(env, remaining_actions, computeOccupiedCells, min, {'dilation': i})

        # tie breaker: random walk
        remaining_actions, _ = tiebreakerFunc(env, remaining_actions, computePathLength, max)

        # Choose last of remaining actions, as it's more likely change_nothing which is to prefer in the endgame
        return remaining_actions[-1]

    def __repr__(self):
        """Get exact representation."""
        return f"EndgamePolicy(actions={self.actions})"
