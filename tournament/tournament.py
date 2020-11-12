import argparse
import numpy as np
import matplotlib.pyplot as plt
import itertools as it
from tqdm import tqdm
import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from environments import SimulatedSpe_edEnv
from environments.logging import TournamentLogger
from policies import RandomPolicy, HeuristicPolicy, SpiralPolicy
from heuristics import PathLengthHeuristic, RandomHeuristic, CompositeHeuristic, RegionHeuristic, OpponentDistanceHeuristic

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
width_height_pairs = [(30, 30), (50, 50)]


def play_game(env, policies, show=False, fps=10, logger=None):
    obs = env.reset()
    if show and not env.render(screen_width=720, screen_height=720):
        return
    if logger is not None:  # Log initial state
        states = [env.game_state()]

    done = False
    while not done:
        action = policies[0]["pol"].act(*obs)
        obs, reward, done, _ = env.step(action)

        if show and not env.render(screen_width=720, screen_height=720):
            return
        if logger is not None:
            states.append(env.game_state())

    if logger is not None:  # log states together with a mapping of player_id to policy
        policy_mapping = dict(
            zip([player_id for player_id, _ in env.game_state()["players"].items()], [pol["name"] for pol in policies])
        )
        logger.log(states, policy_mapping)
    if show:
        # Show final state
        while True:
            if not env.render(screen_width=720, screen_height=720):
                return
            plt.pause(0.01)  # Sleep


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='spe_ed tournament')
    # ToDo: add modes for playing and analysis
    parser.add_argument('--show', action='store_true', help='Display game.')
    parser.add_argument('--log-dir', type=str, default=None, help='Directory for logs.')
    args = parser.parse_args()

    # Create logger
    if args.log_dir is not None:
        logger = TournamentLogger(args.log_dir)
    else:
        logger = None

    with tqdm(total=5, desc="Number of players(2-6)", position=0) as player_number_pbar:
        for number_players in range(2, 7):  # games with 2 to 6 players
            player_constellations = list(it.combinations(POLICY_LIST, number_players))  # maybe with replacements
            with tqdm(
                total=len(player_constellations), desc="Combinations", position=number_players - 1
            ) as constellation_pbar:
                for constellation in player_constellations:
                    for (width, height) in width_height_pairs:
                        # Create environment
                        env = SimulatedSpe_edEnv(width, height, [c["pol"] for c in constellation[1:]])
                        for game in range(number_games):
                            play_game(env, constellation, show=args.show, logger=logger)
                    constellation_pbar.update()
            player_number_pbar.update()
