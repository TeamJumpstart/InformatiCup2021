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
