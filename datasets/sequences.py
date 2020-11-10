import numpy as np
from pathlib import Path
from tqdm import tqdm
from environments import spe_ed
from environments.spe_ed import SavedGame
from statistics import get_log_files
from state_representation import windowed_abstraction


def create_sequences(log_dir, date):
    names = []
    data = {"names": names}
    for log_file in tqdm(get_log_files(log_dir, prefix=date)):
        game = SavedGame.load(log_file)
        for player_id in game.player_ids:
            # Compute abstraction window
            windows = windowed_abstraction(game, player_id, radius=5)

            # Infer actions
            actions = [
                spe_ed.actions.index(a) if a in spe_ed.actions else None for a in (
                    spe_ed.infer_action(game.player_states[t][player_id - 1], game.player_states[t + 1][player_id - 1])
                    for t in range(len(windows))
                )
            ]
            if any(a is None for a in actions):
                continue  # Skip sequences with illegal actions

            name = f"{log_file.name[:-5]}_{player_id}"
            names.append(name)
            data[f"{name}-state"] = windows
            data[f"{name}-action"] = actions

    np.savez_compressed(Path(log_dir).parent / "sequences" / date, **data)


def load_sequences(sequence_file):
    with np.load(sequence_file) as data:
        states = [data[f"{name}-state"] for name in data["names"]]
        actions = [data[f"{name}-action"] for name in data["names"]]
    return states, actions
