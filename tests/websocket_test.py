import unittest
from environments import WebsocketEnv
import websockets
import asyncio


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

    def test_connection(self):
        env = WebsocketEnv(self.url, self.key)

    async def handler(self):
        #send
        data = await self.websocket_server.recv()
