import numpy as np
from environments.spe_ed import Player
from environments.spe_ed_env import Spe_edEnv
import asyncio
import json
import websockets
from datetime import datetime
from pathlib import Path


class WebsocketEnv(Spe_edEnv):
    def __init__(self, url, key, log_path="./logs/"):
        Spe_edEnv.__init__(self, 40, 40)

        self.url = url
        self.key = key
        self.log_path = log_path
        Path(log_path).mkdir(parents=True, exist_ok=True)

    def reset(self):
        """Build connection, save state, and return observation"""
        self.states = []
        self.websocket = asyncio.get_event_loop().run_until_complete(self.connect())
        asyncio.get_event_loop().run_until_complete(self.await_state())
        return self._get_obs(self.controlled_player)

    async def connect(self):
        """Build connection and return websocket connection"""
        print("Waiting for initial state...", flush=True)
        return await websockets.connect(f"{self.url}?key={self.key}")

    async def await_state(self):
        """Wait for received game state and save state in class attributes"""
        state_json = await self.websocket.recv()
        self.states.append(state_json)
        state = json.loads(state_json)
        print("<", "state received", flush=True)

        self.players = [Player.from_json(player_id, player_data) for player_id, player_data in state["players"].items()]
        self.width = state["width"]
        self.height = state["height"]
        self.cells = np.array(state["cells"])
        self.controlled_player = [player for player in self.players if int(player.player_id) == state["you"]][0]
        self.done = not state["running"]

    def step(self, action):
        """Send action, save state and return observation"""
        # only send action if player is active
        if self.controlled_player.active is True:
            asyncio.get_event_loop().run_until_complete(self.send_action(action))

        asyncio.get_event_loop().run_until_complete(self.await_state())
        reward = 1 if self.done and self.controlled_player.active else 0
        if self.done:
            with open(self.log_path + datetime.now().strftime("%Y_%m_%d_%H_%M_%S"), "w") as file:
                json.dump(self.states, file)
        return self._get_obs(self.controlled_player), reward, self.done, {}

    async def send_action(self, action):
        """Wait for send action"""
        action_json = json.dumps({"action": action})
        print(">", action, flush=True)
        await self.websocket.send(action_json)
