import logging
from pathlib import Path
from statistics.log_files import get_log_files

import numpy as np
import pandas as pd
from tqdm import tqdm

from environments.spe_ed import SavedGame


def fetch_statistics(log_dir, csv_file, key_column='date'):
    # Seach for unprocessed log files
    known_log_files = set(pd.read_csv(csv_file)[key_column]) if Path(csv_file).exists() else set()
    new_log_files = [
        f for f in get_log_files(log_dir) if f.name[:-5] not in known_log_files and f.name[:-5] != "_name_mapping"
    ]  # exclude name mapping

    if len(new_log_files) > 0:
        # Process new log files
        results = []
        for log_file in tqdm(new_log_files, desc="Parsing new log files"):
            try:
                game = SavedGame.load(log_file)
            except Exception:
                logging.exception(f"Failed to load {log_file}")
                continue

            results.append(
                (
                    log_file.name[:-5],  # name of game
                    game.rounds,  # rounds
                    game.winner.name if game.winner is not None else None,  # winner
                    game.names[game.you - 1] if game.you is not None else None,  # you
                    game.names,  # names
                    game.width,
                    game.height,
                )
            )

        # Append new statisics
        df = pd.DataFrame(results, columns=[key_column, "rounds", "winner", "you", "names", "width", "height"])
        df.to_csv(csv_file, mode='a', header=len(known_log_files) == 0, index=False)

    # Return all data from csv
    return pd.read_csv(
        csv_file,
        parse_dates=["date"] if key_column == "date" else None,
        converters={"names": lambda x: x.strip("[]").replace("'", "").split(", ")}
    )


def get_win_rate(policy, stats, number_of_players=None, matchup_opponent=None, grid_size=None):
    '''Returns overall win rate for the selected policy by default or specific for a given opponent or size of the grid.

    Args:
            policy: full name of policy
            stats: pandas df of statistics from fetch_statistics
            matchup_opponent: full name of policy to compare against
            grid_size: "small", "medium" or "large"

    '''
    relevant_games = stats[stats["names"].apply(lambda x: policy in x)]
    if number_of_players is not None:
        relevant_games = relevant_games[relevant_games["names"].map(len) == number_of_players]
    if matchup_opponent is not None:
        if matchup_opponent == policy:
            return None
        relevant_games = relevant_games[relevant_games["names"].str.contains(matchup_opponent, regex=False)]
    if grid_size is not None:
        relevant_games = relevant_games[pd.cut(
            relevant_games["width"] * relevant_games["height"],
            bins=[1681, 3132, 4030, 6400],  # 1/3 and 2/3 quantile of size distribution
            labels=["small", "medium", "large"]
        ) == grid_size]

    won = (relevant_games['winner'] == policy).agg(['mean', 'count', 'std'])
    return won["mean"], won["count"], won["std"]


def normalize_winrate(winrate, baseline):
    adjusted = winrate.copy()

    # Scale winrates below baseline to [0, 0.5]
    mask_lesser = winrate <= baseline
    adjusted[mask_lesser] = winrate[mask_lesser] / baseline / 2

    mask_greater = winrate > baseline
    adjusted[mask_greater] = ((winrate[mask_greater] - baseline) / (1 - baseline) + 1) / 2

    return adjusted
