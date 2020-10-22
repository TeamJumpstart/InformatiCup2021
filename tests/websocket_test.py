import unittest
from environments import WebsocketEnv
import websockets
import asyncio
import json


class TestWebsocketEnvironment(unittest.TestCase):
    def setUp(self):
        self.url = "localhost"
        self.port = 8000
        self.key = ""

        self.websocket_server = websockets.serve(self.handler, self.url, self.port)
        asyncio.get_event_loop().run_until_complete(self.websocket_server)
        asyncio.get_event_loop().run_forever()

    def tearDown(self):
        asyncio.get_event_loop().stop()

    def dummy_test(self):
        pass

    def test_connection(self):
        env = WebsocketEnv(self.url, self.key)
        obs = env.reset()
        done = False
        while not done:
            action = "change_nothing"
            obs, reward, done, _ = env.step(action)

    async def handler(self):
        file = open("test/spe_ed-1603124417603.json", "r")
        state_json = json.loads(file)
        file.close()
        await self.websocket_server.send(state_json)
        message = await self.websocket_server.recv()
        self.assertEqual(json.loads(message), "change_nothing")
