import asyncio
import websockets
import json
import logging
import time

# File handler for UI event log
event_log_handler = logging.FileHandler("event.log")
event_log_handler.setLevel(logging.INFO)
event_log_formatter = logging.Formatter("%(asctime)s [WebSocketEvent] %(message)s")
event_log_handler.setFormatter(event_log_formatter)

# Main handler for errors/info
main_log_handler = logging.FileHandler("websocket.log")
main_log_handler.setLevel(logging.INFO)
main_log_formatter = logging.Formatter("%(asctime)s [WebSocket] %(levelname)s: %(message)s")
main_log_handler.setFormatter(main_log_formatter)

# Root logger (for errors, info)
logging.basicConfig(
    level=logging.INFO,
    handlers=[main_log_handler, event_log_handler, logging.StreamHandler()]
)
ws_logger = logging.getLogger("WebSocketMain")
ws_event_logger = logging.getLogger("WebSocketEvent")
ws_event_logger.addHandler(event_log_handler)
ws_event_logger.propagate = False

class OKXWebSocket:
    def __init__(
        self,
        symbol: str = "BTC-USDT-SWAP",
        endpoint: str = "wss://ws.gomarket-cpp.goquant.io/ws/l2-orderbook/okx/BTC-USDT-SWAP"
    ):
        self.symbol = symbol
        self.endpoint = endpoint
        self._connection_status = "disconnected"
        self._message_counter = 0
        self._last_message_time = 0
        self._stop_event = asyncio.Event()

    @property
    def status(self) -> str:
        return self._connection_status

    @property
    def message_rate(self) -> float:
        now = time.time()
        elapsed = now - self._last_message_time if self._last_message_time else 1
        return self._message_counter / elapsed if elapsed > 0 else 0.0

    async def run(self, processor):
        while not self._stop_event.is_set():
            try:
                self._connection_status = "connecting"
                async with websockets.connect(
                    self.endpoint, ping_interval=30, open_timeout=30
                ) as ws:
                    self._connection_status = "connected"
                    ws_logger.info(f"Connected to {self.endpoint}")
                    self._message_counter = 0
                    self._last_message_time = time.time()
                    async for msg in ws:
                        self._message_counter += 1
                        self._last_message_time = time.time()
                        ws_event_logger.info(f"RAW: {msg}")

                        # Robust parsing and error logging
                        try:
                            orderbook = json.loads(msg)
                            if (
                                "timestamp" in orderbook
                                and "asks" in orderbook
                                and "bids" in orderbook
                            ):
                                ws_event_logger.info(
                                    f"VALID: timestamp={orderbook.get('timestamp')} best_bid={orderbook['bids'][0] if orderbook['bids'] else None} best_ask={orderbook['asks'][0] if orderbook['asks'] else None}"
                                )
                                processor.process(orderbook)
                            else:
                                logging.error(f"Processing failed: Invalid message format. Raw message: {orderbook}")
                                ws_event_logger.error(f"Processing failed: Invalid message format. Raw message: {orderbook}")
                        except Exception as e:
                            logging.error(f"Processing failed: {e}. Raw message: {msg}")
                            ws_event_logger.error(f"Processing failed: {e}. Raw message: {msg}")
            except Exception as e:
                self._connection_status = "disconnected"
                ws_logger.error(f"WebSocket error: {e}. Reconnecting in 5 seconds...")
                await asyncio.sleep(5)

    def stop(self):
        self._stop_event.set()