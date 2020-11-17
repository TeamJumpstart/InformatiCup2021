import numpy as np
from state_representation.occupancy import occupancy_map


class Spe_edAxOccupancy():
    """Matplotlib cells plot with update functionality."""
    def __init__(self, fig, ax, cells, players, depth=3):
        self.img = ax.imshow(cells, vmin=0, vmax=1, cmap='binary')
        self.heads = ax.scatter([p.x for p in players], [p.y for p in players], c='white', marker='.')

        # Disables ticks and tick labels
        ax.xaxis.set_visible(False)
        ax.yaxis.set_visible(False)

        def handle_close(evt):
            self.closed = True

        self.closed = False
        fig.canvas.mpl_connect('close_event', handle_close)

        self.depth = depth

    def update(self, cells, players, rounds=1):
        """Draw a new cells state."""
        occ_map = occupancy_map(cells, [p for p in players if p.player_id > 1], rounds, self.depth)

        # update cells
        self.img.set_data(occ_map)

        # Update player heads
        # Dead players don't get a head
        self.heads.set_offsets([(p.x, p.y) if p.active else (np.nan, np.nan) for p in players])
