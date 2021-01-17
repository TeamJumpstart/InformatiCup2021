import argparse
import time
import multiprocessing as mp
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from environments import SimulatedSpe_edEnv, WebsocketEnv
from environments.logging import Spe_edLogger, CloudUploader
from environments.spe_ed import SavedGame
from policies import HeuristicPolicy, load_named_policy
from heuristics import PathLengthHeuristic
from tournament.tournament import run_tournament
import os
import logging
from pathlib import Path

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
)

default_window_size = (720, 720)


def play(env, pol, show=False, render_file=None, fps=10, logger=None, silent=True, window_size=default_window_size):
    obs = env.reset()

    if show and not env.render(screen_width=window_size[0], screen_height=window_size[1]):
        return
    if render_file is not None:  # Initialize video writer
        from imageio_ffmpeg import write_frames

        writer = write_frames(render_file, window_size, fps=fps, codec="libx264", quality=8)
        writer.send(None)  # seed the generator
        writer.send(
            env.render(mode="rgb_array", screen_width=window_size[0], screen_height=window_size[1]).copy(order='C')
        )
    if logger is not None:  # Log initial state
        states = [env.game_state()]
        time_limits = []

    done = False
    with tqdm(disable=silent) as pbar:
        while not done:
            action = pol.act(*obs)
            obs, reward, done, _ = env.step(action)

            if show and not env.render(screen_width=window_size[0], screen_height=window_size[1]):
                return
            if render_file is not None:
                writer.send(
                    env.render(mode="rgb_array", screen_width=window_size[0],
                               screen_height=window_size[1]).copy(order='C')
                )
            if logger is not None:
                states.append(env.game_state())
                if isinstance(env, WebsocketEnv):
                    time_limits.append(env.time_limit)
            pbar.update()

    if logger is not None:
        logger.log(states, time_limits)
    if render_file is not None:
        writer.close()
    if show:
        # Show final state
        while True:
            if not env.render(screen_width=window_size[0], screen_height=window_size[1]):
                return
            plt.pause(0.01)  # Sleep


def show_logfile(log_file, window_size=default_window_size):
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
    if game.you is not None:
        game.move_controlled_player_to_front()

    fig = plt.figure(figsize=(window_size[0] / 100, window_size[1] / 100), dpi=100)
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


def render_logfile(log_file, fps=10, silent=False, window_size=default_window_size):
    """Render logfile to mp4.

    Resulting .mp4 is placed alongside the .json file.

    Args:
        log_file: Log file to render.
        fps: FPS of generated video.
        silent: Show no progress bar.
    """
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
        figsize=(window_size[0] / 100, window_size[1] / 100),
        dpi=100,
        tight_layout=True,
    )
    ax = plt.subplot(1, 1, 1)
    viewer = Spe_edAx(fig, ax, game.cell_states[0], game.player_states[0])

    def frames():
        """Draw all game states"""
        for i in tqdm(range(len(game.cell_states)), desc=f"Rendering {log_file.name}", disable=silent):
            viewer.update(game.cell_states[i], game.player_states[i])
            fig.canvas.draw()

            frame = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8).reshape(window_size[0], window_size[1], 3)
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
    parser.add_argument(
        'mode',
        nargs='?',
        choices=['play', 'replay', 'render_logdir', 'plot', 'tournament', 'tournament-plot'],
        default="play"
    )
    parser.add_argument('--show', action='store_true', help='Display games using an updating matplotlib plot.')
    parser.add_argument('--render-file', type=str, default=None, help='File to render to. Should end with .mp4')
    parser.add_argument('--sim', action='store_true', help='The simulator environment runs a local simulation of Spe_ed instead of using the webserver.')
    parser.add_argument('--log-file', type=str, default=None, help='Path to a log file, used to load and replay games.')
    parser.add_argument('--log-dir', type=str, default=None, help='Directory for storing or retrieving logs.')
    parser.add_argument(
        '--t-config', type=str, default='./tournament/tournament_config.py', help='Path of the tournament config file containing which settings to run.'
    )
    parser.add_argument('--upload', action='store_true', help='Upload generated log to cloud server.')
    parser.add_argument('--fps', type=int, default=10, help='FPS for rendering.')
    parser.add_argument(
        '--cores', type=int, default=None, help='Number of cores for multiprocessing, default uses all.'
    )
    parser.add_argument('--repeat', type=bool, default=False, help='Play endlessly.')
    args = parser.parse_args()

    if args.mode == 'render_logdir':
        log_dir = Path(args.log_dir)
        if not log_dir.is_dir():
            logging.error(f"{log_dir} is not a directory")
            quit(1)

        log_files = []
        for log_file in log_dir.iterdir():
            if not log_file.name.endswith(".json"):
                continue
            if (log_dir / (log_file.name[:-5] + ".mp4")).exists():
                continue
            log_files.append(log_file)

        with mp.Pool(args.cores) as pool, tqdm(desc="Rendering games", total=len(log_files)) as pbar:
            for log_file in log_files:
                pool.apply_async(render_logfile, (log_file, args.fps, True), callback=lambda _: pbar.update())
            pool.close()
            pool.join()

    elif args.mode == 'replay':
        show_logfile(args.log_file)
    elif args.mode == 'tournament':
        from statistics import create_tournament_plots

        log_dir = Path(args.log_dir)
        run_tournament(args.show, log_dir, args.t_config, args.cores)
        create_tournament_plots(log_dir, log_dir.parent)
    elif args.mode == 'tournament-plot':
        from statistics import create_tournament_plots

        log_dir = Path(args.log_dir)
        if not log_dir.is_dir():
            logging.error(f"{log_dir} is not a directory")
            quit(1)

        create_tournament_plots(log_dir, log_dir.parent)
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
            env = SimulatedSpe_edEnv(40, 40, [HeuristicPolicy(PathLengthHeuristic(10)) for _ in range(5)])
        else:
            env = WebsocketEnv(os.environ["URL"], os.environ["KEY"], os.environ["TIME_URL"])

        # Create policy
        pol = load_named_policy("Garruk")

        while True:
            try:
                play(
                    env,
                    pol,
                    show=args.show,
                    render_file=args.render_file,
                    fps=args.fps,
                    logger=logger,
                    silent=args.repeat
                )
            except Exception:
                logging.exception("Exception during play")
                time.sleep(60)  # Sleep for a bit and try again

            if not args.repeat:
                break
