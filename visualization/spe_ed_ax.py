import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

_cmap = ListedColormap(
    [
        (0.0, 0.0, 0.0, 1.0),  # Collision - black
        (1.0, 1.0, 1.0, 1.0),  # Background - white
        (1.0, 0.0, 0.0, 1.0),  # Player 1 - red
        (0.0, 1.0, 0.0, 1.0),  # Player 2 - green
        (0.0, 0.0, 1.0, 1.0),  # Player 3 - blue
        (1.0, 1.0, 0.0, 1.0),  # Player 4 - yellow
        (0.0, 1.0, 1.0, 1.0),  # Player 5 - light blue
        (1.0, 0.0, 1.0, 1.0),  # Player 6 - pink
    ]
)


class Spe_edAx():
    """Matplotlib cells plot with update functionality."""
    def __init__(self, fig, ax, cells, players):
        self.img = ax.imshow((cells + 1) / 7, cmap=_cmap, vmin=0, vmax=1)
        self.heads = ax.scatter([p.x for p in players], [p.y for p in players], c='white', marker='.')

        # Disables ticks and tick labels
        ax.xaxis.set_visible(False)
        ax.yaxis.set_visible(False)

        def handle_close(evt):
            self.closed = True

        self.closed = False
        fig.canvas.mpl_connect('close_event', handle_close)

    def update(self, cells, players):
        """Draw a new cells state."""
        # Update cells
        self.img.set_data((cells + 1) / 7)

        # Update player heads
        # Dead players don't get a head
        self.heads.set_offsets([(p.x, p.y) if p.active else (np.nan, np.nan) for p in players])
