import numpy as np
from environments.spe_ed import Spe_edEnv, Player, directions, cartesian_directions
import asyncio
import signal
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
        """Build connection and return observation"""
        asyncio.run(self.connect())
        asyncio.run(self.await_state())
        return self._get_obs(self.controlled_player)

    async def connect(self):
        """Build connection and save state"""
        print("Waiting for initial state...", flush=True)
        self.websocket = await websockets.connect(f"{self.url}?key={self.key}")

    async def await_state(self):
        state_json = await self.websocket.recv()
        state = json.loads(state_json)
        with open('state.json', 'w') as file:
            json.dump(state, file)
        print("<", state)

        self.players = [
            Player(player_id=player_id, **player_data) for player_id, player_data in state["players"].items()
        ]
        self.width = state["width"]
        self.height = state["height"]
        a = np.array(state["cells"])
        print(a.shape, a.dtype)
        del self.cells
        self.cells = np.array(state["cells"])
        del self.controlled_player
        print(self.players)
        self.controlled_player = [player for player in self.players if int(player.player_id) == state["you"]][0]
        self.done = state["running"]

    def step(self, action):
        # Sends actions to the server
        asyncio.run(self.send_action(action))
        print("Send complete")
        asyncio.run(self.await_state())

        reward = 1 if self.done and self.controlled_player.active else 0
        return self._get_obs(self.controlled_player), reward, self.done, {}

    async def send_action(self, action):
        action_json = json.dumps({"action": action})
        print(">", action)
        await self.websocket.send(action_json)
