import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.dates import AutoDateLocator
import pandas as pd

player_colors = [
    (204.0 / 255, 7.0 / 255, 30.0 / 255, 1.0),  # Player 1 - red
    (87.0 / 255, 171 / 255.0, 39.0 / 255, 1.0),  # Player 2 - green
    (0.0 / 255, 84.0 / 255, 159.0 / 255, 1.0),  # Player 3 - blue
    (246.0 / 255, 168.0 / 255, 0.0 / 255, 1.0),  # Player 4 - orange
    (97.0 / 255, 33.0 / 255, 88.0 / 255, 1.0),  # Player 5 - violet
    (0.0 / 255, 177.0 / 255, 183.0 / 255, 1.0),  # Player 6 - turquoise
]

_cmap = ListedColormap(
    [
        (0.0, 0.0, 0.0, 1.0),  # Collision - black
        (1.0, 1.0, 1.0, 1.0),  # Background - white
    ] + player_colors
)


class Spe_edAx():
    """Matplotlib cells plot with update functionality."""
    def __init__(self, fig, ax, cells, players, cmap=_cmap):
        self.img = ax.imshow((cells + 1) / 7, cmap=cmap, vmin=0, vmax=1)
        marker_size = (0.5 * fig.get_size_inches()[1] * fig.dpi / cells.shape[0])**2
        self.heads = ax.scatter([p.x for p in players], [p.y for p in players], c='white', marker='.', s=marker_size)

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


class WinRateAx():
    def __init__(self, fig, ax, date, won, groupby="D"):
        won = pd.concat([date, won], axis=1).groupby(date.dt.floor(groupby)).agg(['mean', 'count', 'std']).loc[:, 0]

        date = won.index
        mean, count, std = won["mean"], won["count"], won["std"]

        # Compute confidence interval
        low = mean - 1.96 * std / np.sqrt(count)
        high = mean + 1.96 * std / np.sqrt(count)

        ax.axhline(0.5, color="black", linestyle="dashed")

        line, = ax.plot(date, mean, label="win rate", c=player_colors[0])
        c = line.get_color()
        ax.fill_between(date, low, high, facecolor=c, alpha=0.25, interpolate=True, label="confidence interval")
        #ax.xaxis.set_major_locator(AutoDateLocator(maxticks=6))
        ax.set_xlim(date[0], date[-1] + pd.offsets.Day(1))
        ax.set_ylim(0, 1)
