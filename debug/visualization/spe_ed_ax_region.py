import numpy as np
from scipy import ndimage
from scipy.ndimage import morphology


class Spe_edAxRegion():
    """Matplotlib cells plot with update functionality."""
    def __init__(self, fig, ax, cells, players, cmap=None):
        self.img = ax.imshow(cells, vmin=0, vmax=1, cmap='gnuplot')
        self.heads = ax.scatter([p.x for p in players], [p.y for p in players], c='white', marker='.')

        # Disables ticks and tick labels
        ax.xaxis.set_visible(False)
        ax.yaxis.set_visible(False)

        def handle_close(evt):
            self.closed = True

        self.closed = False
        fig.canvas.mpl_connect('close_event', handle_close)

    def update(self, cells, players, iterations=0):
        """Draw a new cells state."""
        # filter inactive players
        players = [p for p in players if p.active]

        # close all 1 cell wide openings aka "articulating points"
        if iterations:
            cells = morphology.binary_closing(cells, iterations=iterations)

        # inverse map (mask occupied cells)
        empty = cells == 0
        # Clear cell for all players
        for p in players:
            empty[p.y, p.x] = True

        # compute distinct regions
        labelled, num_features = ndimage.label(empty)

        # Reset cells for all players
        for p in players:
            labelled[p.y, p.x] = 0

        # update cells
        self.img.set_data(labelled / num_features)

        # Update player heads
        # Dead players don't get a head
        self.heads.set_offsets([(p.x, p.y) if p.active else (np.nan, np.nan) for p in players])
