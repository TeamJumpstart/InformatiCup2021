import matplotlib.pyplot as plt
import itertools as it
from tqdm import tqdm
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


def play_game(env, policy_names, game_suffix, show=False, logger=None):
    """Simulate a single game with the given environment and policies."""
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
        logger.log(states, policy_names, game_suffix)
    if show:  # Show final state
        while True:
            if not env.render(screen_width=720, screen_height=720):
                return
            plt.pause(0.01)  # Sleep


def run_tournament(show, log_dir, tournament_config_file):
    """Run a sequence of games in different combinations of given policies and log their results."""
    # Create logger
    if log_dir is not None:
        directory = Path(log_dir)
        directory.mkdir(parents=True, exist_ok=True)
        logger = TournamentLogger(log_dir)
    else:
        logger = None

    # load config file
    config = SourceFileLoader('tournament_config', tournament_config_file).load_module()
    # games with 2 to 6 players
    with mp.Pool() as pool, tqdm(total=5, desc="Number of players(2-6)", position=0) as player_number_pbar:
        for config.number_players in range(2, 7):
            player_constellations = list(
                it.combinations(config.policies, config.number_players)
            )  # maybe with replacements
            # games with different policy combinations
            with tqdm(
                total=len(player_constellations), desc="Combinations", position=config.number_players - 1
            ) as constellation_pbar:
                for constellation in player_constellations:
                    game_data = []
                    # games with different map size
                    for (width, height) in config.width_height_pairs:
                        # number of games to be played
                        for game_number in range(config.number_games):
                            policy_ids = [str(config.policies.index(pol)) for pol in constellation]
                            game_suffix = f"_w{width}h{height}_{game_number}.json"
                            # do not run games when log already exists
                            if logger is not None and (directory / ("_".join(policy_ids) + game_suffix)).is_file():
                                continue
                            environment = TournamentEnv(width, height, constellation)
                            environment.reset()
                            game_data.append([policy_ids, game_suffix, environment])
                    # parallelized execution using starmap (takes multiple parameters as opposed to map), async: faster
                    # but potentially out of order
                    pool.starmap_async(play_game, [(env, ids, suf, show, logger) for ids, suf, env in game_data]).get()
                    constellation_pbar.update()
            player_number_pbar.update()
        if logger is not None:
            logger.save_nick_names([str(pol) for pol in constellation])
