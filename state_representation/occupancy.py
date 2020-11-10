import numpy as np
from environments import spe_ed, Spe_edSimulator


def occupancy_map(cells, opponents, rounds, depth=3):
    """Compute occupancy probabilities in presence of opponents for each cell.

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
        probs = np.zeros_like(cells, dtype=float)
        # Sum probs fro all actions
        for a in spe_ed.actions:
            sub_sim = sim.step([a])
            sub_probability = probability / N_actions  # Assume uniform distribution

            probs += (sim.cells != sub_sim.cells) * sub_probability

            if level < depth and sub_sim.players[0].active:
                _occupancy_recursion(sub_sim, sub_probability, level + 1)

        # Update occupancy
        occ[:] = 1 - (1 - occ) * (1 - probs)

    for opponent in opponents:
        if opponent.active:
            _occupancy_recursion(Spe_edSimulator(cells, [opponent], rounds))

    return occ
