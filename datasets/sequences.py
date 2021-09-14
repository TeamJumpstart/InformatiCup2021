from pathlib import Path
from statistics import get_log_files

import numpy as np
from tqdm import tqdm

from environments import spe_ed
from environments.spe_ed import SavedGame
from state_representation import windowed_abstraction


def create_sequences(log_dir, date, radius=5):
    "Create a state/action sequences .npz file."
    names = []
    data = {"names": names}
    for log_file in tqdm(get_log_files(log_dir, prefix=date)):
        game = SavedGame.load(log_file)
        for player_id in game.player_ids:
            # Compute abstraction window
            windows = windowed_abstraction(game, player_id, radius=radius)

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


def load_sequence_file(sequence_file):
    """Read state/action sequences from .npz file."""
    with np.load(sequence_file) as data:
        states = [data[f"{name}-state"] for name in data["names"]]
        actions = [data[f"{name}-action"] for name in data["names"]]
    return states, actions


def load_sequence_dataset(sequence_dir):
    states = []
    actions = []
    for f in tqdm([f for f in Path(sequence_dir).iterdir() if f.name.endswith(".npz")], desc="Loading sequences"):
        s, a = load_sequence_file(f)
        states.extend(s)
        actions.extend(a)
    return states, actions
