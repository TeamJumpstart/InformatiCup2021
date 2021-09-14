import numpy as np

from state_representation.occupancy import occupancy_map
from state_representation.window import padded_window


def windowed_abstraction(game, player_id, radius):
    """Compute occupancy window abstract of a game from the perspective of one player.

    Abstracts away the direction by rotating the window accordingly.

    Places the speed at the center of the occupancy map, as the center otherwise contains no information.

    Args:
        player_id: Regarded player
        radius: Radius for the occupancy window

    Returns:
        windows: (n_steps, 2*radius+1, 2*radius+1) Occupancy windows of regarded player
        actions: (n_steps, )
    """
    windows = []
    for t in range(game.rounds):
        player = game.player_states[t][player_id - 1]
        if not player.active:
            break
        occ = occupancy_map(game.cell_states[t], [p for p in game.player_states[t] if p.player_id != player_id], t + 1)
        window = padded_window(occ, player.x, player.y, radius, 1)

        # Rotate window, so direction is always facing right
        window = np.rot90(window, k=player.direction.index)

        # Add speed to center of window
        window[radius, radius] = (player.speed - 1) / 9  # Normalize speed
        windows.append(window)
    return windows
