import numpy as np
from policies.policy import Policy
from environments.simulator import Spe_edSimulator
from environments import spe_ed
from scipy import ndimage
from scipy.ndimage import morphology
from heuristics import PathLengthHeuristic


def applyMorphology(cells, closing=0, opening=0, erosion=0, dilation=0):
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
    # inverse map (mask occupied cells)
    empty_cells = cells == 0
    # Clear cell for all players
    for p in players:
        empty_cells[p.y, p.x] = True
    # compute distinct regions
    labelled_cells, _ = ndimage.label(empty_cells)
    return labelled_cells


def computeRegionSize(cells, players):
    labelled_cells = labelCells(cells, players)
    # Get the region we're in and compute its size
    region_label = labelled_cells[players[0].y, players[0].x]
    player_region_size = np.sum(labelled_cells == region_label)
    return player_region_size


def computeVoronoiSize(cells, players):
    max_steps = 2
    # initialize arrays, one binary map for each player
    mask = cells == 0
    voronoi = np.zeros((len(players), *cells.shape), dtype=np.bool)
    for idx, p in enumerate(players):
        mask[p.y, p.x] = True  # reset player position to allow dilation
        voronoi[idx, p.y, p.x] = True

    # geodesic voronoi cell computation
    for _ in range(max_steps):
        for i in range(len(voronoi)):
            voronoi[i] = morphology.binary_dilation(voronoi[i], mask=mask)  # expand cell by one step
        xor = np.sum(voronoi, axis=0) > 1  # compute overlaps for every cell
        mask[xor] = 0  # mask out all overlaps
        voronoi[:, xor] = 0  # reset all overlapping cell borders

    return np.sum(voronoi[0])


def computeRegionNumber(cells, players):
    # inverse map (mask occupied cells)
    cells = np.pad(cells, (1, ))
    empty_cells = cells == 0
    # compute distinct regions
    _, num_cells = ndimage.label(empty_cells)
    return num_cells


def computeOccupiedCells(cells, players):
    return np.sum(cells)


def computePathLength(cells, players):
    path_length_heuristic = PathLengthHeuristic(n_steps=200, n_probes=100)
    return path_length_heuristic.score(cells, players[0], players[1:], 0)


def tiebreakerFunc(env, remaining_actions, score_func=computeRegionSize, eval_func=max, morph_kwargs={}):
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
    """EndgamePolicy."""
    def __init__(self, actions=None):
        self.actions = [a for a in spe_ed.actions if a != "speed_up"] if actions is None else actions

    def act(self, cells, player, opponents, rounds):
        env = Spe_edSimulator(cells, [player], rounds)
        remaining_actions = self.actions

        # bigger region is always better
        remaining_actions, scores = tiebreakerFunc(env, remaining_actions, computeRegionSize, max)
        # less regions is preferable
        remaining_actions, scores = tiebreakerFunc(env, remaining_actions, computeRegionNumber, min)

        # tie breaker: morphological operations
        for i in range(2, 0, -1):
            remaining_actions, scores = tiebreakerFunc(
                env, remaining_actions, computeOccupiedCells, min, {'closing': i}
            )
            remaining_actions, scores = tiebreakerFunc(
                env, remaining_actions, computeOccupiedCells, min, {'dilation': i}
            )

        # tie breaker: random walk
        remaining_actions, scores = tiebreakerFunc(env, remaining_actions, computePathLength, max)

        # return remaining action
        return remaining_actions[-1]

    def __str__(self):
        """Get readable representation."""
        return "EndgamePolicy()"
