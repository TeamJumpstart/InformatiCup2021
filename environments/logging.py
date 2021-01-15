from datetime import datetime
import json
import logging
from pathlib import Path
from uuid import uuid4
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
    def __init__(self, log_dir, policies):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.name_mapping = TournamentLogger.load_name_mapping(self.log_dir, policies)

    def log(self, states):
        """Handle the logging of a completed tournament game with a set of different policies.

        Args:
            states: List of game states in form of parsed json.
            policy_ids: The used policy IDs
        """
        log_file = self.log_dir / f"{uuid4().hex}.json"
        with open(log_file, "w") as f:
            json.dump(states, f, separators=(',', ':'))

    @staticmethod
    def load_name_mapping(log_dir, policies):
        name_mapping_file = log_dir / "_name_mapping.json"

        # Load current mapping
        if name_mapping_file.exists():
            with open(name_mapping_file, "r") as f:
                name_mapping = json.load(f)
        else:
            name_mapping = {}

        # Update missing entries
        changed = False
        for pol in policies:
            if str(pol) not in name_mapping:
                name_mapping[str(pol)] = len(name_mapping)
                changed = True

        # Write back
        if changed:
            with open(name_mapping_file, "w") as f:
                json.dump(name_mapping, f, indent=4)

        return name_mapping
