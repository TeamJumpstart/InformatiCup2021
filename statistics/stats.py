from pathlib import Path
import pandas as pd
from tqdm import tqdm
from environments.spe_ed import SavedGame
from statistics.log_files import get_log_files


def fetch_statistics(log_dir, csv_file, tournament_mode=False):
    file_name_format = 'date'
    if tournament_mode:
        file_name_format = 'matchup'
    # Seach for unprocessed log files
    known_log_files = set(pd.read_csv(csv_file)[file_name_format]) if Path(csv_file).exists() else set()
    new_log_files = [f for f in get_log_files(log_dir) if f.name[:-5] not in known_log_files]

    if len(new_log_files) > 0:
        # Process new log files
        results = []
        for log_file in tqdm(new_log_files, desc="Parsing new log files"):
            game = SavedGame.load(log_file)

            results.append(
                (
                    log_file.name[:-5],  # order by file name
                    game.rounds,  # rounds
                    game.winner.name if game.winner is not None else None,  # winner
                    game.names[game.you - 1] if game.you is not None else None,  # you
                    game.names,  # names
                    game.width,
                    game.height,
                )
            )

        # Append new statisics
        df = pd.DataFrame(results, columns=[file_name_format, "rounds", "winner", "you", "names", "width", "height"])
        df.to_csv(csv_file, mode='a', header=len(known_log_files) == 0, index=False)

    # Return all data from csv
    return pd.read_csv(
        csv_file,
        parse_dates=[file_name_format],
        converters={"names": lambda x: x.strip("[]").replace("'", "").split(", ")}
    )


def get_win_rate(policy, stats, number_of_players=None, matchup_opponent=None, grid_size=None):
    '''Returns overall win rate for the selected policy by default or specific for a given opponent or size of the grid.

    Args:
            policy: full name of policy
            stats: pandas df of statistics from fetch_statistics
            matchup_opponent: full name of policy to compare against
            grid_size: width-height tuple

    '''
    relevant_games = stats[stats["names"].str.contains(policy, regex=False)]
    if number_of_players is not None:
        relevant_games = relevant_games[relevant_games["names"].map(len) == number_of_players]
    if matchup_opponent is not None:
        if matchup_opponent == policy:
            return None
        relevant_games = relevant_games[relevant_games["names"].str.contains(matchup_opponent, regex=False)]
    if grid_size is not None:
        relevant_games = relevant_games[(relevant_games["width"] == grid_size[0]) &
                                        (relevant_games["height"] == grid_size[1])]
    if len(relevant_games) == 0:
        return None
    return len([winner for winner in relevant_games["winner"] if winner is not None and winner == policy]
              ) / len(relevant_games)


def create_matchup_stats(policy_names, policy_nick_names, stats, csv_file):
    matchup_win_rates = []
    for policy_name in policy_names:
        policy_matchups = []
        for opponent_policy in policy_names:
            policy_matchups.append(get_win_rate(policy_name, stats, matchup_opponent=opponent_policy))
        matchup_win_rates.append(policy_matchups)
    df = pd.DataFrame(
        [[policy_nick_names[i], policy_names[i]] + matchup_win_rates[i] for i in range(len(policy_names))],
        columns=["Policy short", "Policy full"] + [f"vs{nick}" for nick in policy_nick_names]
    )
    df.to_csv(csv_file, mode='w', index=False)
