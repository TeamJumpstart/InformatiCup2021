import matplotlib.pyplot as plt
import logging
import numpy as np
from tqdm.auto import tqdm
from pathlib import Path
from uuid import uuid4
import json
import multiprocessing as mp
import pandas as pd
from environments.spe_ed import SavedGame
from environments.simulator import simulate, SimulatedSpe_edEnv
from importlib.machinery import SourceFileLoader


def init_stats_lock(l):
    """Helper to pass the lock to sub processes."""
    global stats_lock
    stats_lock = l


class TournamentLogger():
    def __init__(self, log_dir, write_logs):
        self.log_dir = Path(log_dir)
        self.csv_file = self.log_dir.parent / "statistics.csv"
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.write_logs = write_logs

    def log(self, states):
        """Handle the logging of a completed tournament game with a set of different policies.

        Args:
            states: List of game states in form of parsed json.
            policy_ids: The used policy IDs
        """
        log_file = self.log_dir / f"{uuid4().hex}.json"

        # Write log files
        if self.write_logs:
            with open(log_file, "w") as f:
                json.dump(states, f, separators=(',', ':'))

        # Append new statisics
        game = SavedGame(states)
        df = pd.DataFrame(
            [
                (
                    log_file.name[:-5],  # name of game
                    game.rounds,  # rounds
                    game.winner.name if game.winner is not None else None,  # winner
                    game.names[game.you - 1] if game.you is not None else None,  # you
                    game.names,  # names
                    game.width,
                    game.height,
                )
            ],
            columns=["uuid", "rounds", "winner", "you", "names", "width", "height"]
        )

        with stats_lock:  # Ensure only one process is writing
            df.to_csv(self.csv_file, mode='a', header=not self.csv_file.exists(), index=False)


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
        logger = TournamentLogger(log_dir, config.write_logs)
    else:
        logger = None

    with mp.Pool(
        initializer=init_stats_lock,  # Pass the stats lock to all spawned subprocesses
        initargs=(mp.Lock(), )
    ) as pool, tqdm(desc="Simulating games", total=config.n_games) as pbar:

        for _ in range(config.n_games):
            n_players = np.random.choice(5, p=config.n_players_distribution) + 2
            width = np.random.randint(config.min_size, config.max_size + 1)
            height = np.random.randint(config.min_size, config.max_size + 1)
            constellation = np.random.choice(config.policies, size=n_players, replace=False)

            pool.apply_async(play_game, (width, height, constellation, show, logger), callback=lambda _: pbar.update())
        pool.close()
        pool.join()
