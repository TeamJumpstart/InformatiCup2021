import argparse
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from environments import SimulatedSpe_edEnv
from environments.logging import Spe_edLogger
from policies import *
from heuristics import PathLengthHeuristic, RandomHeuristic, CompositeHeuristic, RegionHeuristic, OpponentDistanceHeuristic
import logging

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
    if logger is not None:  # Log initial state
        states = [env.game_state()]

    done = False
    with tqdm(disable=not silent) as pbar:
        while not done:
            action = pol.act(*obs)
            obs, reward, done, _ = env.step(action)

            if show and not env.render(screen_width=720, screen_height=720):
                return
            if logger is not None:
                states.append(env.game_state())
            pbar.update()

    if logger is not None:
        logger.log(states)
    if show:
        # Show final state
        while True:
            if not env.render(screen_width=720, screen_height=720):
                return
            plt.pause(0.01)  # Sleep


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='spe_ed')
    parser.add_argument('--show', action='store_true', help='Display game.')
    parser.add_argument('--log-dir', type=str, default=None, help='Directory for logs.')
    args = parser.parse_args()

    # Create logger
    if args.log_dir is not None:
        logger_callbacks = []
        logger = Spe_edLogger(args.log_dir, logger_callbacks)
    else:
        logger = None

    # Create environment
    env = SimulatedSpe_edEnv(40, 40, [HeuristicPolicy(PathLengthHeuristic()) for _ in range(5)])

    # Create policy to test
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

    repeat = not args.show

    while True:
        play(env, pol, show=args.show, render_file=args.render_file, fps=args.fps, logger=logger, silent=not repeat)
        if not repeat:
            break
