import argparse
import time
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from environments import SimulatedSpe_edEnv, WebsocketEnv
from environments.logging import Spe_edLogger, CloudUploader
from environments.spe_ed import SavedGame
from policies import RandomPolicy, HeuristicPolicy
from heuristics import PathLengthHeuristic, RandomHeuristic, CompositeHeuristic, RegionHeuristic, OpponentDistanceHeuristic
import os
import logging
from pathlib import Path

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
)


def play(env, pol, show=False, render_file=None, fps=10, logger=None, silent=False):
    obs = env.reset()

    if show and not env.render(screen_width=720, screen_height=720):
        return
    if render_file is not None:  # Initialize video writer
        from imageio_ffmpeg import write_frames

        writer = write_frames(render_file, (720, 720), fps=fps, codec="libx264", quality=8)
        writer.send(None)  # seed the generator
        writer.send(env.render(mode="rgb_array", screen_width=720, screen_height=720).copy(order='C'))
    if logger is not None:  # Log initial state
        states = [env.game_state()]

    done = False
    with tqdm(disable=not silent) as pbar:
        while not done:
            action = pol.act(*obs)
            obs, reward, done, _ = env.step(action)

            if show and not env.render(screen_width=720, screen_height=720):
                return
            if render_file is not None:
                writer.send(env.render(mode="rgb_array", screen_width=720, screen_height=720).copy(order='C'))
            if logger is not None:
                states.append(env.game_state())
            pbar.update()

    if logger is not None:
        logger.log(states)
    if render_file is not None:
        writer.close()
    if show:
        # Show final state
        while True:
            if not env.render(screen_width=720, screen_height=720):
                return
            plt.pause(0.01)  # Sleep


def show_logfile(log_file):
    """Render logfile to mp4"""
    from visualization import Spe_edAx
    from matplotlib.widgets import Slider

    def format_state(t):
        s = "Players:\n"
        s += "\n".join(str(p) for p in game.player_states[t]) + "\n"

        s += "\nActions:\n"
        if t + 1 < len(game.data):
            s += "\n".join(str(a) for a in game.infer_actions(t)) + "\n"
        else:
            s += "\n".join("win" if p.active else "inactive" for p in game.player_states[t]) + "\n"

        return s

    game = SavedGame.load(log_file)
    game.move_controlled_player_to_front()

    fig = plt.figure(figsize=(720 / 100, 720 / 100), dpi=100)
    ax1 = plt.subplot(1, 1, 1)
    viewer = Spe_edAx(fig, ax1, game.cell_states[0], game.player_states[0])

    plt.tight_layout()
    plt.subplots_adjust(bottom=0.1, right=0.6)
    slider = Slider(plt.axes([0.1, 0.025, 0.8, 0.03]), 't', 0, len(game.data) - 1, valinit=0, valstep=1, valfmt="%d")
    text_box = fig.text(0.61, 0.975, format_state(0), ha='left', va='top')

    def change_t(val):
        t = int(slider.val)
        viewer.update(game.cell_states[t], game.player_states[t])
        text_box.set_text(format_state(t))

    slider.on_changed(change_t)
    plt.show()


def render_logfile(log_file, fps=10):
    """Render logfile to mp4"""
    from visualization import Spe_edAx, render_video
    from imageio_ffmpeg import get_ffmpeg_exe
    import subprocess
    import tempfile

    def temp_file_name(suffix):
        """Create the name of a temp file with given suffix without opening it."""
        return Path(tempfile.gettempdir()) / (next(tempfile._get_candidate_names()) + suffix)

    game = SavedGame.load(log_file)
    game.move_controlled_player_to_front()

    fig = plt.figure(
        figsize=(720 / 100, 720 / 100),
        dpi=100,
        tight_layout=True,
    )
    ax = plt.subplot(1, 1, 1)
    viewer = Spe_edAx(fig, ax, game.cell_states[0], game.player_states[0])

    def frames():
        """Draw all game states"""
        for i in tqdm(range(len(game.cell_states)), desc=f"Rendering {log_file.name}"):
            viewer.update(game.cell_states[i], game.player_states[i])
            fig.canvas.draw()

            frame = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8).reshape(720, 720, 3)
            yield frame

    # Render video to temp file
    tmp_video = temp_file_name(".mp4")
    width, height = fig.canvas.get_width_height()
    render_video(tmp_video, frames(), width, height, fps=fps)

    # Create thumbnail in temp file
    tmp_thumbnail = temp_file_name(".jpg")
    plt.savefig(tmp_thumbnail)

    # Join both in log dir
    subprocess.run(
        [
            get_ffmpeg_exe(), "-i",
            str(tmp_video), "-i",
            str(tmp_thumbnail), "-y", "-map", "0", "-map", "1", "-c", "copy", "-disposition:v:1", "attached_pic", "-v",
            "warning",
            str(log_file.parent / (log_file.name[:-5] + ".mp4"))
        ]
    )

    # Cleanup
    plt.close(fig)
    tmp_video.unlink()
    tmp_thumbnail.unlink()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='spe_ed')
    parser.add_argument('mode', nargs='?', choices=[
        'play',
        'replay',
        'render_logdir',
        'plot',
    ], default="play")
    parser.add_argument('--show', action='store_true', help='Display game.')
    parser.add_argument('--render-file', type=str, default=None, help='File to render to. Should end with .mp4')
    parser.add_argument('--sim', action='store_true', help='Use simulator.')
    parser.add_argument('--log-file', type=str, default=None, help='Log file to load.')
    parser.add_argument('--log-dir', type=str, default=None, help='Directory for logs.')
    parser.add_argument('--upload', action='store_true', help='Upload generated log to cloud server.')
    parser.add_argument('--fps', type=int, default=10, help='FPS for rendering.')
    args = parser.parse_args()

    if args.mode == 'render_logdir':
        log_dir = Path(args.log_dir)
        if not log_dir.is_dir():
            logging.error(f"{log_dir} is not a directory")
            quit(1)

        for log_file in log_dir.iterdir():
            if not log_file.name.endswith(".json"):
                continue
            if (log_dir / (log_file.name[:-5] + ".mp4")).exists():
                continue
            render_logfile(log_file, fps=args.fps)
    elif args.mode == 'replay':
        show_logfile(args.log_file)
    elif args.mode == 'plot':
        from statistics import create_plots

        log_dir = Path(args.log_dir)
        if not log_dir.is_dir():
            logging.error(f"{log_dir} is not a directory")
            quit(1)

        create_plots(log_dir, log_dir.parent / "statistics.csv")
    else:
        # Create logger
        if args.log_dir is not None:
            logger_callbacks = []
            if args.upload:
                logger_callbacks.append(
                    CloudUploader(
                        os.environ["CLOUD_URL"],
                        os.environ["CLOUD_USER"],
                        os.environ["CLOUD_PASSWORD"],
                        remote_dir="logs/"
                    ).upload
                )
            logger = Spe_edLogger(args.log_dir, logger_callbacks)
        else:
            logger = None

        # Create environment
        if args.sim:
            env = SimulatedSpe_edEnv(40, 40, [HeuristicPolicy(PathLengthHeuristic()) for _ in range(5)])
        else:
            env = WebsocketEnv(os.environ["URL"], os.environ["KEY"])

        # Create policy
        pol = HeuristicPolicy(
            CompositeHeuristic(
                [
                    PathLengthHeuristic(20, 100),
                    RegionHeuristic(),
                    OpponentDistanceHeuristic(dist_threshold=16),
                    RandomHeuristic(),
                ],
                weights=[20, 1, 1e-3, 1e-4]
            )
        )

        repeat = not args.show and args.render_file is None

        while True:
            play(env, pol, show=args.show, render_file=args.render_file, fps=args.fps, logger=logger, silent=not repeat)
            if not repeat:
                break
