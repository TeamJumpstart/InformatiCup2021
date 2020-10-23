import unittest
from environments import WebsocketEnv
import websockets
import asyncio
import json


class TestWebsocketEnvironment(unittest.TestCase):
    def setUp(self):
        self.url = "127.0.0.1"
        self.port = 8000
        self.key = ""
        self.step_counter = 0

        async def handler(self, websocket, path):
            file = open("tests/spe_ed-1603124417603.json", "r")
            states = json.load(file)
            file.close()
            await self.websocket_server.send(states[self.step_counter])
            message = await self.websocket_server.recv()
            self.assertEqual(json.loads(message), "change_nothing")
            self.step_counter += 1
            if self.step_counter >= len(states) - 1:
                self.new_loop.stop()

        self.new_loop = asyncio.new_event_loop()
        self.websocket_server = websockets.serve(handler, self.url, self.port, loop=self.new_loop)
        self.new_loop.run_in_executor(None, self.websocket_server)

        print("thread started")

    def tearDown(self):
        self.new_loop.stop()

    def dummy_test(self):
        self.assertEqual(1, 1)

    def test_connection(self):
        env = WebsocketEnv(f"ws://{self.url}:{self.port}", self.key)
        obs = env.reset()
        done = False
        while not done:
            action = "change_nothing"
            obs, reward, done, _ = env.step(action)
