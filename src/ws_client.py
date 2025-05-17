import asyncio
import websockets
import json
import logging
from typing import Dict, List, Deque
from collections import deque 

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [WebSocket] %(levelname)s: %(message)s",
    handlers=[
        logging.FileHandler("websocket.log"),
        logging.StreamHandler()
    ]
)

class OKXWebSocket:
    def __init__(self, symbol: str = "BTC-USDT"):
        self.uri = "wss://ws.okx.com:8443/ws/v5/public"
        self.symbol = symbol
        self.orderbook: Dict[str, List[List[str]]] = {"bids": [], "asks": []}
        self._queue = asyncio.Queue(maxsize=1000)
        self._stop_event = asyncio.Event()
        self._connection_status = "disconnected"
        self._message_counter = 0

    @property
    def status(self) -> str:
        """Current connection status"""
        return self._connection_status

    @property
    def message_rate(self) -> float:
        """Messages per second"""
        return self._message_counter / 60 if self._message_counter > 0 else 0

    async def connect(self, max_retries: int = 5):
        retry_count = 0
        while not self._stop_event.is_set() and retry_count < max_retries:
            try:
                self._connection_status = "connecting"
                async with websockets.connect(self.uri, ping_interval=30) as ws:
                    self._connection_status = "connected"
                    await ws.send(json.dumps({
                        "op": "subscribe",
                        "args": [{"channel": "books-l2-tbt", "instId": self.symbol}]
                    }))
                    logging.info(f"Connected to {self.symbol} orderbook")
                    retry_count = 0
                    
                    while not self._stop_event.is_set():
                        try:
                            data = await asyncio.wait_for(ws.recv(), timeout=10)
                            self._message_counter += 1
                            await self._queue.put(data)
                        except (asyncio.TimeoutError, websockets.ConnectionClosed) as e:
                            logging.warning(f"Connection error: {e}")
                            break
            
            except Exception as e:
                retry_count += 1
                wait_time = min(2 ** retry_count, 30)
                logging.error(f"Retry {retry_count}/{max_retries} in {wait_time}s: {e}")
                await asyncio.sleep(wait_time)
        self._connection_status = "disconnected"

    async def consumer(self, processor):
        """Process messages with rate limiting"""
        while not self._stop_event.is_set():
            try:
                # Throttle if queue is overloaded
                if self._queue.qsize() > 10:
                    excess = self._queue.qsize() - 5
                    for _ in range(excess):
                        self._queue.get_nowait()
                    logging.warning(f"Dropped {excess} messages due to backlog")
                
                data = await self._queue.get()
                message = json.loads(data)
                processor.process(message)
            except json.JSONDecodeError:
                logging.error("Invalid JSON message")
            except Exception as e:
                logging.error(f"Consumer error: {e}")

    async def run(self, processor):
        """Start producer and consumer tasks"""
        await asyncio.gather(
            self.connect(),
            self.consumer(processor)
        )