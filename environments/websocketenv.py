import numpy as np
from environments.spe_ed import Spe_edEnv, Player
import asyncio
import json
import os
import websockets


class WebsocketEnv(Spe_edEnv):
    def __init__(self, width, height, opponent_policies, seed=None):
        Spe_edEnv.__init__(self, width, height)

        self.opponent_policies = opponent_policies
        self.seed(seed)
        self.url = os.environ["URL"]
        self.key = os.environ["KEY"]

    def reset(self):
        """Build connection, save state, and return observation"""
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
        state = json.loads(state_json)
        print("<", "state received")

        self.players = [
            Player(player_id=player_id, **player_data) for player_id, player_data in state["players"].items()
        ]
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
        return self._get_obs(self.controlled_player), reward, self.done, {}

    async def send_action(self, action):
        """Wait for send action"""
        action_json = json.dumps({"action": action})
        print(">", action)
        await self.websocket.send(action_json)
