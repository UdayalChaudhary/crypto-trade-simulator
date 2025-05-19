import numpy as np
import time
import logging
from typing import Deque, Optional
from collections import deque
from numba import jit

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [OrderBook] %(levelname)s: %(message)s"
)

@jit(nopython=True)
def calculate_slippage_numba(levels: np.ndarray, quantity: float) -> float:
        total_cost = 0.0
        remaining = quantity
        for i in range(levels.shape[0]):
            price, size = levels[i, 0], levels[i, 1]
            cost = price * min(size, remaining / price)
            total_cost += cost
            remaining -= cost
            if remaining <= 0:
                break
        return total_cost / quantity if quantity > 0 else 0.0

class OrderBookProcessor:
    def __init__(self, max_history: int = 1000):
        self.bids: np.ndarray = np.array([])
        self.asks: np.ndarray = np.array([])
        self.price_history: Deque[float] = deque(maxlen=max_history)
        self.latency_log: Deque[float] = deque(maxlen=100)
        self._mid_price: Optional[float] = None
        self._last_update = time.time()

    @property
    def is_fresh(self) -> bool:
        """Check if data is recent (within 5 seconds)"""
        return time.time() - self._last_update < 5

    def process(self, message: dict):
        start_time = time.perf_counter()

        try:
            # For GoMarket/OKX, expect direct bids/asks, not under "data"
            if not isinstance(message, dict) or "bids" not in message or "asks" not in message:
                logging.error(f"Processing failed: Invalid message format. Raw message: {message}")
                return

            # Convert string price/size to float and create numpy arrays
            self.bids = np.array([[float(p), float(s)] for p, s in message["bids"]], dtype=float)
            self.asks = np.array([[float(p), float(s)] for p, s in message["asks"]], dtype=float)

            # Validate order book integrity
            if len(self.bids) == 0 or len(self.asks) == 0:
                logging.warning("Processing warning: Received orderbook tick with empty bids or asks.")
                return
            if self.bids[0][0] >= self.asks[0][0]:
                logging.warning("Crossed market detected")
                return

            self._mid_price = (self.bids[0][0] + self.asks[0][0]) / 2
            self.price_history.append(self._mid_price)
            self._last_update = time.time()

        except Exception as e:
            logging.error(f"Processing failed: {e}. Raw message: {message}")
        finally:
            self.latency_log.append((time.perf_counter() - start_time) * 1000)


    def calculate_slippage(self, quantity_usd: float, is_buy: bool = True) -> float:
        levels = self.asks if is_buy else self.bids
        if len(levels) == 0:
            logging.warning("Empty order book in slippage calc")
            return 0.0

        # Reference price: best ask for buy, best bid for sell
        reference_price = levels[0][0]

        total_qty = 0.0
        total_cost = 0.0
        remaining = quantity_usd

        for price, size in levels:
            level_value = price * size
            trade_value = min(remaining, level_value)
            total_cost += trade_value
            total_qty += trade_value / price
            remaining -= trade_value
            if remaining <= 0:
                break

        average_execution_price = total_cost / quantity_usd if quantity_usd > 0 else reference_price

        slippage = (average_execution_price / reference_price - 1) * 100
        return slippage