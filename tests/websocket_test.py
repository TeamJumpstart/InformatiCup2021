import asyncio
import json
import unittest

import websockets


class DummyServer:
    def __init__(self, url, port):
        self.url = url
        self.port = port
        self.step_counter = 0
        with open("tests/spe_ed-1603124417603.json") as file:
            self.states = json.load(file)
        # Start new thread from current loop
        self.__serving = asyncio.get_event_loop().run_in_executor(None, self.run)
        self.__starting = asyncio.get_event_loop().create_future()
        asyncio.get_event_loop().run_until_complete(self.__starting)

    async def handler(self, websocket, path):
        """Handler"""
        for state in self.states:
            print("Server: await send state")
            await websocket.send(json.dumps(self.states[self.step_counter]))
            print("Server: send state", flush=True)
            message = await websocket.recv()
            print("Server: received command")
            assert json.loads(message)["action"] == "change_nothing"
            self.step_counter += 1
            if self.step_counter >= len(self.states):
                print("counter exceeded")
                self.stop()
            print("Server: websocket open: ", websocket.open)

    async def serve(self):
        """Run server until stopped.
        Executed in loop of server thread.
        """
        async with websockets.serve(self.handler, self.url, self.port):
            print("Server is up")
            await self.__stop
            print("Server is down")

    def run(self):  # Executed in new thread
        print("Server is starting")
        loop = asyncio.new_event_loop()  # Create new loop
        self.__stop = loop.create_future()  # Create stop signal

        # Release __init__
        self.__starting.get_loop().call_soon_threadsafe(self.__starting.set_result, None)

        loop.run_until_complete(self.serve())  # Run serve in loop until stopped
        loop.close()
        print("Server stopped")

    def stop(self):
        # Send stop signal in server loop
        self.__stop.get_loop().call_soon_threadsafe(self.__stop.set_result, None)

        # Wait until server has actually stopped
        asyncio.get_event_loop().run_until_complete(self.__serving)


class TestWebsocketEnvironment(unittest.TestCase):
    def setUp(self):
        self.url = "127.0.0.1"
        self.port = 8000
        self.key = ""
        self.server = DummyServer(self.url, self.port)

    def tearDown(self):
        print("Stop")
        self.server.stop()
