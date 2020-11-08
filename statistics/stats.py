from pathlib import Path
import pandas as pd
from tqdm import tqdm
from environments.spe_ed import SavedGame
from statistics.log_files import get_log_files


def fetch_statistics(log_dir, csv_file):
    # Seach for unprocessed log files
    known_log_files = set(pd.read_csv(csv_file)['date']) if Path(csv_file).exists() else set()
    new_log_files = [f for f in get_log_files(log_dir) if f.name[:-5] not in known_log_files]

    if len(new_log_files) > 0:
        # Process new log files
        results = []
        for log_file in tqdm(new_log_files, desc="Parsing new log files"):
            game = SavedGame.load(log_file)

            results.append(
                (
                    log_file.name[:-5],  # date
                    game.rounds,  # rounds
                    game.winner.name if game.winner is not None else None,  # winner
                    game.names[game.you - 1],  # you
                    game.names,  # names
                )
            )

        # Append new statisics
        df = pd.DataFrame(results, columns=["date", "rounds", "winner", "you", "names"])
        df.to_csv("data.csv", mode='a', header=len(known_log_files) == 0, index=False)

    # Return all data from csv
    return pd.read_csv(
        "data.csv", parse_dates=["date"], converters={"names": lambda x: x.strip("[]").replace("'", "").split(", ")}
    )
