from pathlib import Path
from environments.spe_ed import SavedGame
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from tqdm import tqdm

cell_size = 16

cmap = ListedColormap(
    [
        (0.0, 0.0, 0.0, 1.0),  # Collision - black
        (1.0, 1.0, 1.0, 0.0),  # Background - white
        (0.7, 0.7, 0.7, 1.0),  # Player 1
        (0.6, 0.6, 0.6, 1.0),  # Player 2
        (0.5, 0.5, 0.5, 1.0),  # Player 3
        (0.4, 0.4, 0.4, 1.0),  # Player 4
        (0.3, 0.3, 0.3, 1.0),  # Player 5
        (0.2, 0.2, 0.2, 1.0),  # Player 6
    ]
)


def render_logfile(log_file, render_dir):
    """Render logfile to mp4.

    Resulting .mp4 is placed alongside the .json file.

    Args:
        log_file: Log file to render.
        fps: FPS of generated video.
        silent: Show no progress bar.
    """
    from visualization import Spe_edAx

    render_dir.mkdir(exist_ok=True)

    game = SavedGame.load(log_file)
    game.move_controlled_player_to_front()

    fig = plt.figure(figsize=(game.width * cell_size / 100, game.height * cell_size / 100), dpi=100)

    ax = plt.subplot(1, 1, 1)
    ax.axis('off')
    viewer = Spe_edAx(fig, ax, game.cell_states[0], game.player_states[0], cmap=cmap)
    plt.tight_layout(pad=0)

    for i in tqdm(range(len(game.cell_states)), desc=f"Rendering {log_file.name}"):
        viewer.update(game.cell_states[i], game.player_states[i])
        fig.canvas.draw()

        plt.savefig(render_dir / f"{i:04}.png", transparent=True)

    # Cleanup
    plt.close(fig)


render_logfile(
    log_file=Path(r"F:\spe_ed\logs\20210117-234331.json"),
    render_dir=Path(r"F:/spe_ed2/render/20210117-234331"),
)
