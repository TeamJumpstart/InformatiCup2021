import argparse
import numpy as np
import matplotlib.pyplot as plt
import itertools as it
from tqdm import tqdm
import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from environments import SimulatedSpe_edEnv
from environments.logging import Spe_edLogger
from policies import RandomPolicy, HeuristicPolicy, SpiralPolicy
from heuristics import PathLengthHeuristic, RandomHeuristic, CompositeHeuristic, RegionHeuristic, OpponentDistanceHeuristic
import logging

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
POLICY_LIST = [
    {
        "name": "Composite",
        "pol": pol
    },
    {
        "name": "Random",
        "pol": RandomPolicy()
    },
    {
        "name": "Spiral",
        "pol": SpiralPolicy()
    },
    {
        "name": "PathLength",
        "pol": HeuristicPolicy(PathLengthHeuristic())
    },
    {
        "name": "Region",
        "pol": HeuristicPolicy(RegionHeuristic())
    },
    {
        "name": "OppDist",
        "pol": HeuristicPolicy(OpponentDistanceHeuristic())
    },
]
number_games = 1

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
)


def play_game(env, policy, show=False, fps=10, logger=None):
    obs = env.reset()

    if show and not env.render(screen_width=720, screen_height=720):
        return
    if logger is not None:  # Log initial state
        states = [env.game_state()]

    done = False
    while not done:
        action = policy["pol"].act(*obs)
        obs, reward, done, _ = env.step(action)

        if show and not env.render(screen_width=720, screen_height=720):
            return
        if logger is not None:
            states.append(env.game_state())

    if logger is not None:
        logger.log(states)
    if show:
        # Show final state
        while True:
            if not env.render(screen_width=720, screen_height=720):
                return
            plt.pause(0.01)  # Sleep


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='spe_ed tournament')
    parser.add_argument('--show', action='store_true', help='Display game.')
    parser.add_argument('--log-dir', type=str, default=None, help='Directory for logs.')
    args = parser.parse_args()

    # Create logger
    if args.log_dir is not None:
        logger_callbacks = []
        logger = Spe_edLogger(args.log_dir, logger_callbacks)
    else:
        logger = None

    # ToDo: create varying with, height, number of players and constellations

    with tqdm() as pbar:
        for number_players in range(2, 7):  # games with 2 to 6 players
            player_constellations = it.combinations(POLICY_LIST, number_players)  # maybe with replacements
            for constellation in player_constellations:
                # Create environment
                env = SimulatedSpe_edEnv(40, 40, [c["pol"] for c in constellation[1:]])
                for game in range(number_games):
                    play_game(env, constellation[0], show=args.show)

            pbar.update()
