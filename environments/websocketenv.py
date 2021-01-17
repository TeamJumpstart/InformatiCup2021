import asyncio
from datetime import datetime, timedelta
import json
import logging
import requests
import time
import websockets
import numpy as np
from tqdm import tqdm
from environments.spe_ed import Player
from environments.spe_ed_env import Spe_edEnv


class WebsocketEnv(Spe_edEnv):
    """TODO."""
    def __init__(self, url, key, time_url):
        """Initialize WebsocketEnv.

        Args:
            url: Websocket URL of the server
            key: API key
            time_url: URL of time server
        """
        Spe_edEnv.__init__(self, 40, 40)  # Dummy dimensions

        self.url = url
        self.key = key
        self.rtt, self.time_offset = measure_server_time(time_url, n_probes=10)
        logging.info("RTT: {}, time offset: {time_offset}")

    def reset(self):
        """Build connection, save state, and return observation."""
        self.websocket = asyncio.get_event_loop().run_until_complete(self.connect())
        logging.info("Waiting for initial state...")
        asyncio.get_event_loop().run_until_complete(self.await_state())
        return self._get_obs(self.controlled_player)

    async def connect(self):
        """Build connection and return websocket connection."""
        logging.info("Client connecting")
        try:
            return await websockets.connect(f"{self.url}?key={self.key}", max_queue=None)
        except websockets.exceptions.InvalidStatusCode as e:
            if e.status_code == 429:  # Server rejects us
                logging.warn("Server rejected connection")

                # Sleep and try again
                await asyncio.sleep(60)
                return await self.connect()
            else:  # Other status code, not handled here
                raise

    async def await_state(self):
        """Wait for received game state and save state in class attributes."""
        state = json.loads(await self.websocket.recv())
        self.__game_state = state
        self.done = not state["running"]

        if not self.done:
            deadline = datetime.strptime(state['deadline'], "%Y-%m-%dT%H:%M:%S%z").timestamp() + self.time_offset
            self.time_limit = deadline - time.time()

        self.players = [Player.from_json(player_id, player_data) for player_id, player_data in state["players"].items()]
        self.width = state["width"]
        self.height = state["height"]
        self.cells = np.array(state["cells"])
        self.controlled_player = [player for player in self.players if int(player.player_id) == state["you"]][0]

        logging.info(
            f"Client received state (active={self.controlled_player.active}, running={not self.done}" +
            (f", time={self.time_limit:.02f})" if not self.done else ")")
        )

        if self.done:  # Close connection if done
            await self.websocket.close()

    def step(self, action):
        """Send action, save state and return observation."""
        # only send action if player is active
        if self.controlled_player.active is True:
            asyncio.get_event_loop().run_until_complete(self.send_action(action))

        asyncio.get_event_loop().run_until_complete(self.await_state())
        reward = 1 if self.done and self.controlled_player.active else 0
        return self._get_obs(self.controlled_player), reward, self.done, {}

    async def send_action(self, action):
        """Wait for send action."""
        await self.websocket.send(json.dumps({"action": action}))
        logging.info(f"Client sent action {action}")

    def game_state(self):
        """Get current game state as dict."""
        return self.__game_state


def measure_server_time(time_url='https://msoll.de/spe_ed_time', n_probes=10):
    """Measures roundtriptime time time diffference between local time and server time.

    Transform server time to local time by adding `time_offset`.

    Returns:
        rtt: Roundtrip time is seconds (includign TLS handshake)
        time_offset: Difference from server time to local time
    """
    time_deltas = []
    time_offsets = []
    last_server_time = None
    for _ in tqdm(range(10), desc="Measuring roundtrip time"):
        data = requests.get(time_url).json()
        now = time.time()

        # Parse servetime
        server_time = (
            datetime.strptime(data['time'], "%Y-%m-%dT%H:%M:%S%z") + timedelta(milliseconds=int(data['milliseconds']))
        ).timestamp()
        if last_server_time is not None:
            time_deltas.append((server_time - last_server_time))

        time_offsets.append(now - server_time)
        last_server_time = server_time

    return np.quantile(time_deltas, .9), np.mean(time_offsets)
