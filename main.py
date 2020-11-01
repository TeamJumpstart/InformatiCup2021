import argparse
import time
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from environments import SimulatedSpe_edEnv, WebsocketEnv
from environments.logging import Spe_edLogger, CloudUploader
from environments.spe_ed import SavedGame
from policies import RandomPolicy, RandomProbingPolicy
import os
import logging
from pathlib import Path

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
)


def simulate(env, pol):
    with tqdm() as pbar:
        runs, wins = 0, 0
        while True:
            obs = env.reset()
            done = False

            while not done:
                action = pol.act(*obs)
                obs, reward, done, _ = env.step(action)

            runs += 1
            if reward > 0:
                wins += 1
            pbar.update()
            pbar.set_description(f"{pol} {wins/runs:.2f}")


def show(env, pol, fps=None, keep_open=True):
    obs = env.reset()
    done = False

    start = time.time()
    with tqdm() as pbar:
        while not done:
            if fps is None or (time.time() - start) * fps < env.rounds:  # Only render if there is time
                if not env.render(screen_width=720, screen_height=720):
                    return
                if fps is not None and (time.time() - start) * fps < env.rounds:
                    plt.pause(1 / fps)  # Sleep

            action = pol.act(*obs)
            obs, reward, done, _ = env.step(action)
            pbar.update()

    if keep_open:
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


def render(env, pol, output_file, fps=10):
    """Render the execution of a given pilicy in an environment."""
    from imageio_ffmpeg import write_frames

    obs = env.reset()
    done = False

    writer = write_frames(output_file, (720, 720), fps=fps, codec="libx264", quality=8)
    writer.send(None)  # seed the generator
    writer.send(env.render(mode="rgb_array", screen_width=720, screen_height=720).copy(order='C'))

    with tqdm() as pbar:
        while not done:
            action = pol.act(*obs)
            obs, _, done, _ = env.step(action)
            writer.send(env.render(mode="rgb_array", screen_width=720, screen_height=720).copy(order='C'))
            pbar.update()
    writer.close()


def render_logfile(log_file, fps=10):
    """Render logfile to mp4"""
    from visualization import Spe_edAx, render_video

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

    width, height = fig.canvas.get_width_height()
    render_video(log_file.parent / (log_file.name[:-5] + ".mp4"), frames(), width, height, fps=fps)
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='spe_ed')
    parser.add_argument('mode', nargs='?', choices=[
        'play',
        'show',
        'render',
        'render_logdir',
    ], default="play")
    parser.add_argument('--render', type=str, default=None, help='File to render to. Should end with .mp4')
    parser.add_argument('--sim', action='store_true', help='Use simulator.')
    parser.add_argument('--log-file', type=str, default=None, help='Log file to load.')
    parser.add_argument('--log-dir', type=str, default="logs/", help='Directory for logs.')
    parser.add_argument('--upload', action='store_true', help='Upload generated log to cloud server.')
    parser.add_argument('--fps', type=int, default=10, help='FPS for showing or rendering.')
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
    elif args.mode == 'show' and args.log_file is not None:
        show_logfile(args.log_file)
    else:
        # Create environment
        if args.sim:
            env = SimulatedSpe_edEnv(40, 40, [RandomPolicy() for _ in range(5)])
        else:
            logging.basicConfig(level=logging.DEBUG)  # Reduce loglevel

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

            env = WebsocketEnv(
                os.environ["URL"],
                os.environ["KEY"],
                logger=Spe_edLogger(args.log_dir, logger_callbacks),
            )

        # Create policy
        pol = RandomProbingPolicy(20, 100, True)

        if args.mode == 'render':
            render(env, pol, args.render, fps=args.fps)
        elif args.mode == 'show':
            show(env, pol, fps=args.fps)
        else:
            simulate(env, pol)
