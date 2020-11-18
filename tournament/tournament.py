import matplotlib.pyplot as plt
import itertools as it
from tqdm import tqdm
from pathlib import Path
from environments.simulator import SimulatedSpe_edEnv
from environments.logging import TournamentLogger
import tournament.tournament_config


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


def run_tournament(show, log_dir):
    # Create logger
    if log_dir is not None:
        directory = Path(log_dir)
        if not directory.is_dir():
            directory.mkdir(parents=True, exist_ok=True)
        logger = TournamentLogger(log_dir)
    else:
        logger = None

    # games with 2 to 6 players
    with tqdm(total=5, desc="Number of players(2-6)", position=0) as player_number_pbar:
        for tournament.tournament_config.number_players in range(2, 7):
            player_constellations = list(
                it.combinations(tournament.tournament_config.policy_list, tournament.tournament_config.number_players)
            )  # maybe with replacements
            # games with different policy combinations
            with tqdm(
                total=len(player_constellations),
                desc="Combinations",
                position=tournament.tournament_config.number_players - 1
            ) as constellation_pbar:
                for constellation in player_constellations:
                    # games with different map size
                    for (width, height) in tournament.tournament_config.width_height_pairs:  # ToDo: add to file name
                        # do not run games when log already exists
                        log_file = directory / "_".join([pol_name for pol_name in dict(constellation).keys()]) / ".json"
                        if logger is not None and log_file.is_file():
                            continue
                        env = SimulatedSpe_edEnv(width, height, [c["pol"] for c in constellation[1:]])
                        # number of games to be played
                        for game in range(tournament.tournament_config.number_games):
                            play_game(env, constellation, show=show, logger=logger)
                    constellation_pbar.update()
            player_number_pbar.update()
