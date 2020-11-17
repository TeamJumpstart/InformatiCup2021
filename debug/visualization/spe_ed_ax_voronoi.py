import numpy as np
from scipy.ndimage import morphology
from matplotlib.colors import ListedColormap

_cmap = ListedColormap(
    [
        (0.0, 0.0, 0.0, 1.0),  # Collision - black
        (1.0, 1.0, 1.0, 1.0),  # Background - white
        (204.0 / 255, 7.0 / 255, 30.0 / 255, 1.0),  # Player 1 - red
        (87.0 / 255, 171 / 255.0, 39.0 / 255, 1.0),  # Player 2 - green
        (0.0 / 255, 84.0 / 255, 159.0 / 255, 1.0),  # Player 3 - blue
        (246.0 / 255, 168.0 / 255, 0.0 / 255, 1.0),  # Player 4 - orange
        (97.0 / 255, 33.0 / 255, 88.0 / 255, 1.0),  # Player 5 - violet
        (0.0 / 255, 177.0 / 255, 183.0 / 255, 1.0),  # Player 6 - turquoise
    ]
)


class Spe_edAxVoronoi():
    """Matplotlib cells plot with update functionality."""
    def __init__(self, fig, ax, cells, players):
        self.img = ax.imshow(cells, vmin=0, vmax=1, cmap=_cmap)
        self.heads = ax.scatter([p.x for p in players], [p.y for p in players], c='white', marker='.')

        # Disables ticks and tick labels
        ax.xaxis.set_visible(False)
        ax.yaxis.set_visible(False)

        def handle_close(evt):
            self.closed = True

        self.closed = False
        fig.canvas.mpl_connect('close_event', handle_close)

    def update(self, cells, players, iterations=0):
        """Draw a new cells state - geodesic voronoi diagram."""
        # filter inactive players
        players = [p for p in players if p.active]

        occupied_cells = cells != 0
        # open all {self.iterations} wide walls aka "articulating points" to account for jumps
        if iterations:
            cells = np.pad(cells, (iterations, ))
            cells = morphology.binary_opening(cells, iterations=iterations)
            cells = cells[iterations:-iterations, iterations:-iterations]

        # initialize arrays, one binary map for each player
        mask = cells == 0
        voronoi = np.zeros((len(players), *cells.shape), dtype=np.bool)
        for idx, p in enumerate(players):
            mask[p.y, p.x] = True  # reset player position to allow dilation
            voronoi[idx, p.y, p.x] = True

        # geodesic voronoi cell computation
        for _ in range(40):
            for i in range(len(voronoi)):
                voronoi[i] = morphology.binary_dilation(voronoi[i], mask=mask)  # expand cell by one step
            xor = np.sum(voronoi, axis=0) > 1  # compute overlaps for every cell
            mask[xor] = 0  # mask out all overlaps
            voronoi[:, xor] = 0  # reset all overlapping cell borders

        if iterations:
            voronoi[:, occupied_cells] = 0  # reset all occupied cells
            for idx, p in enumerate(players):
                voronoi[idx, p.y, p.x] = True  # reset player positions

        # map binarized cells to player id's
        voronoi_map = np.zeros_like(cells, dtype=np.int16)
        for idx, p in enumerate(players):
            voronoi_map[voronoi[idx]] = p.player_id

        voronoi_map = voronoi_map + 1  # update player color index and free cells
        voronoi_map[occupied_cells] = 0  # set all occupied cells to zero

        # update cell data
        self.img.set_data(voronoi_map / 7)

        # Update player heads
        # Dead players don't get a head
        self.heads.set_offsets([(p.x, p.y) for p in players])
