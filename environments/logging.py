from datetime import datetime
import json
import logging
from pathlib import Path
import owncloud


class Spe_edLogger():
    def __init__(self, log_dir="logs/", callbacks=[]):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        self.callbacks = callbacks

    def log(self, states):
        """Handle the logging of a completed game.

        Args:
            states: List of game states in form of parsed json.
        """
        log_file = self.log_dir / f"{datetime.now():%Y%m%d-%H%M%S}.json"
        with open(log_file, "w") as f:
            json.dump(states, f, separators=(',', ':'))

        # Handle callbacks
        for callback in self.callbacks:
            try:
                callback(log_file)
            except Exception:
                logging.exception("Logging callback failed")

    def log_policy(self, policy):
        """Log the policy name of a game.

        Args:
            policy:  
        """
        log_file = self.log_dir / f"{datetime.now():%Y%m%d-%H%M%S}_policy.json"
        with open(log_file, "w") as f:
            json.dump(str(policy), f, separators=(',', ':'))


class CloudUploader():
    def __init__(self, url, user, password, remote_dir):
        self.url = url
        self.user = user
        self.password = password
        self.remote_dir = remote_dir

    def upload(self, log_file):
        oc = owncloud.Client(self.url)
        oc.login(self.user, self.password)

        try:  # Create folder
            oc.mkdir(self.remote_dir)
        except owncloud.HTTPResponseError:
            pass

        oc.put_file(self.remote_dir + log_file.name, str(log_file))

        oc.logout()


class TournamentLogger():
    def __init__(self, log_dir="logs/"):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)

    def log(self, states, policy_ids, game_suffix):
        """Handle the logging of a completed tournament game with a set of different policies.

        Args:
            states: List of game states in form of parsed json.
            policy_ids: The used policy IDs
        """
        combined_name = "_".join(policy_ids)
        log_file = self.log_dir / f"{combined_name}{game_suffix}"
        with open(log_file, "w") as f:
            json.dump(states, f, separators=(',', ':'))

    def save_nick_names(self, nick_names):
        name_mapping = {i: nick_names[i] for i in range(len(nick_names))}
        with open(self.log_dir / "_name_mapping.json", "w") as f:
            json.dump(name_mapping, f, separators=(',', ':'))
