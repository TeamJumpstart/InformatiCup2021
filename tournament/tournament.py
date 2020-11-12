import argparse
import numpy as np
import matplotlib.pyplot as plt
import itertools as it
from tqdm import tqdm
import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from environments import SimulatedSpe_edEnv
from environments.logging import TournamentLogger
import tournament_config


def play_game(env, policies, show=False, fps=10, logger=None):
    """Simulate a single game with the given environment and policies"""
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
    if show:  # Show final state
        while True:
            if not env.render(screen_width=720, screen_height=720):
                return
            plt.pause(0.01)  # Sleep


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='spe_ed tournament')
    parser.add_argument('mode', nargs='?', choices=[
        'play',
        'analyze',
    ], default="play")
    parser.add_argument('--show', action='store_true', help='Display game.')
    parser.add_argument('--log-dir', type=str, default=None, help='Directory for logs.')
    args = parser.parse_args()

    # Create logger
    if args.log_dir is not None:
        logger = TournamentLogger(args.log_dir)
    else:
        logger = None

    if args.mode == "analyze":
        # ToDo
        pass
    else:  # play
        # games with 2 to 6 players
        with tqdm(total=5, desc="Number of players(2-6)", position=0) as player_number_pbar:
            for tournament_config.number_players in range(2, 7):
                player_constellations = list(
                    it.combinations(tournament_config.policy_list, tournament_config.number_players)
                )  # maybe with replacements
                # games with different policy combinations
                with tqdm(
                    total=len(player_constellations),
                    desc="Combinations",
                    position=tournament_config.number_players - 1
                ) as constellation_pbar:
                    for constellation in player_constellations:
                        # games with different map size
                        for (width, height) in tournament_config.width_height_pairs:
                            # Create environment
                            env = SimulatedSpe_edEnv(width, height, [c["pol"] for c in constellation[1:]])
                            # number of games to be played
                            for game in range(tournament_config.number_games):
                                play_game(env, constellation, show=args.show, logger=logger)
                        constellation_pbar.update()
                player_number_pbar.update()
