# telemetry.py

"""
Telemetry module for Trackmania 2020 using OpenPlanet DataSender plugin.

Starts a WebSocket server to listen for /vstate snapshots (vehicle state) and
stores the latest snapshot for synchronous access via get_snapshot().
"""

import asyncio
import json
import logging
import threading
from typing import Optional

import websockets
from websockets.legacy.server import WebSocketServerProtocol

PORT = 8765
URI = f"ws://127.0.0.1:{PORT}/vstate"

logging.basicConfig(level=logging.INFO, format="[Telem] %(message)s")


class Telemetry:
    _snapshot: Optional[dict] = None
    _server_thread: Optional[threading.Thread] = None
    _loop: Optional[asyncio.AbstractEventLoop] = None

    @classmethod
    def get_snapshot(cls) -> Optional[dict]:
        """Return the latest vehicle snapshot (or None if no data yet)."""
        return cls._snapshot

    @classmethod
    def start_listener(cls):
        """Start the WebSocket server in a background thread."""
        if cls._server_thread and cls._server_thread.is_alive():
            logging.info("Telemetry listener already running")
            return

        cls._server_thread = threading.Thread(target=cls._run_loop, daemon=True)
        cls._server_thread.start()
        logging.info("Telemetry listener started")

    @classmethod
    def _run_loop(cls):
        cls._loop = asyncio.new_event_loop()
        asyncio.set_event_loop(cls._loop)
        cls._loop.run_until_complete(cls._main())
        cls._loop.run_forever()

    @classmethod
    async def _handler(cls, ws: WebSocketServerProtocol):
        path = getattr(getattr(ws, "request", None), "path", None) or getattr(ws, "path", "/vstate")
        if path != "/vstate":
            logging.warning("Reject connection on %s", path)
            await ws.close(code=1008, reason="unsupported path")
            return

        logging.info("Client connected to /vstate: %s", ws.remote_address)
        try:
            async for raw in ws:
                try:
                    data = json.loads(raw)
                except json.JSONDecodeError:
                    logging.error("Bad JSON: %.80s", raw)
                    continue

                # Store relevant vehicle state
                cls._snapshot = {
                    "spd": data.get("spd"),
                    "pos": data.get("pos"),
                    "rpm": data.get("rpm"),
                    "gear": data.get("gear"),
                    "slipFL": data.get("slipFL"),
                    "slipFR": data.get("slipFR"),
                    "slipRL": data.get("slipRL"),
                    "slipRR": data.get("slipRR"),
                }
        except websockets.ConnectionClosedOK:
            pass
        except websockets.ConnectionClosedError as exc:
            logging.warning("Connection closed with error: %s", exc)
        finally:
            logging.info("Client disconnected: %s", ws.remote_address)

    @classmethod
    async def _main(cls):
        server = await websockets.serve(
            cls._handler,
            "127.0.0.1",
            PORT,
            ping_interval=None,
            ping_timeout=None,
        )
        logging.info("Telemetry WebSocket server listening on ws://127.0.0.1:%d/vstate", PORT)
        return server


if __name__ == "__main__":
    Telemetry.start_listener()
    try:
        # Keep main thread alive
        while True:
            asyncio.sleep(1)
    except KeyboardInterrupt:
        logging.info("Telemetry server stopped")
