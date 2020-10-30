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
            json.dump(states, f)

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
