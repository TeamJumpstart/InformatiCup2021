import matplotlib.pyplot as plt
import itertools as it
import logging
import numpy as np
from tqdm.auto import tqdm
from pathlib import Path
import multiprocessing as mp
from environments.simulator import simulate
from environments.simulator import SimulatedSpe_edEnv
from environments.logging import TournamentLogger
from importlib.machinery import SourceFileLoader


class TournamentEnv(SimulatedSpe_edEnv):
    """TODO."""
    def __init__(self, width, height, policies, seed=None):
        """Initialize TournamentEnv.

        Args:
            width, height: TODO
            policies: TODO
            seed: TODO
        """
        SimulatedSpe_edEnv.__init__(self, width, height, policies[1:])
        self.policies = policies

    def step(self):
        """TODO."""
        actions = []
        for player in self.players:  # Compute actions of players
            if player.active:
                policy = self.policies[player.player_id - 1]
                obs = self._get_obs(player)
                actions.append(policy.act(*obs))
            else:
                actions.append("change_nothing")

        # Perform simulation step
        _, _, self.rounds, _ = simulate(self.cells, self.players, self.rounds, actions)

        done = sum(1 for p in self.players if p.active) < 2
        if done:
            for p in self.players:
                p.name = str(self.policies[p.player_id - 1])
        return done

    def game_state(self):
        """Get current game state as dict."""
        return {
            'width': self.width,
            'height': self.height,
            'cells': self.cells.tolist(),
            'players': dict(p.to_dict() for p in self.players),
            'you': None,
            'running': sum(1 for p in self.players if p.active) > 1,
        }


def play_game(width, height, policies, show=False, logger=None):
    """Simulate a single game with the given environment and policies."""
    try:
        env = TournamentEnv(width, height, policies)
        env.reset()

        if show and not env.render(screen_width=720, screen_height=720):
            return
        if logger is not None:  # Log initial state
            states = [env.game_state()]

        done = False
        while not done:
            done = env.step()

            if show and not env.render(screen_width=720, screen_height=720):
                return
            if logger is not None:
                states.append(env.game_state())

        if logger is not None:  # log states together with a mapping of player_id to policy
            logger.log(states)
        if show:  # Show final state
            while True:
                if not env.render(screen_width=720, screen_height=720):
                    return
                plt.pause(0.01)  # Sleep
    except Exception:
        logging.exception("Error during simulation")


def run_tournament(show, log_dir, tournament_config_file):
    """Run a sequence of games in different combinations of given policies and log their results."""
    # load config file
    config = SourceFileLoader('tournament_config', tournament_config_file).load_module()
    if len(config.policies) < 6:
        raise ValueError("Must provide at least 6 policies")

    # Create logger
    if log_dir is not None:
        directory = Path(log_dir)
        directory.mkdir(parents=True, exist_ok=True)
        logger = TournamentLogger(log_dir, config.policies)
    else:
        logger = None

    with mp.Pool() as pool, tqdm(desc="Simulating games", total=config.n_games) as pbar:
        for _ in range(config.n_games):
            n_players = np.random.choice(5, p=config.n_players_distribution) + 2
            width = np.random.randint(config.min_size, config.max_size + 1)
            height = np.random.randint(config.min_size, config.max_size + 1)
            constellation = np.random.choice(config.policies, size=n_players, replace=False)

            pool.apply_async(play_game, (width, height, constellation, show, logger), callback=lambda _: pbar.update())
        pool.close()
        pool.join()
