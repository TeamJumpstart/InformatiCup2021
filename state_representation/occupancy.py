import numpy as np
from environments import spe_ed, Spe_edSimulator


def occupancy_map(cells, opponents, rounds, depth=3):
    """Compute occupancy probabilities in presence of opponens.

    Assumes actions of opponents to be uniformly distributed.

    Args:
        cells, opponents, roungs: Game state
        depth: How many steps to project opponent actions into the future

    Returns:
        occ: ndarray with occupancy probabilities
    """

    occ = (cells != 0).astype(np.float32)
    N_actions = len(spe_ed.actions)

    def _occupancy_recursion(sim, probability=1, level=1):
        for a in spe_ed.actions:
            sub_sim = sim.step([a])
            sub_prob = probability / N_actions
            diff = sim.cells != sub_sim.cells
            np.add(occ, diff * sub_prob, occ)

            if level < depth and sub_sim.players[0].active:
                _occupancy_recursion(sub_sim, sub_prob, level + 1)

    for opponent in opponents:
        if opponent.active:
            _occupancy_recursion(Spe_edSimulator(cells, [opponent], rounds))

    return np.minimum(occ, 1)  # Clip to [0, 1]
