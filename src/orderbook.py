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
            if not isinstance(message, dict) or "data" not in message:
                raise ValueError("Invalid message format")
            
            data = message["data"][0]
            self.bids = np.array(data["bids"], dtype=float)
            self.asks = np.array(data["asks"], dtype=float)
            
            # Validate order book integrity
            if len(self.bids) == 0 or len(self.asks) == 0:
                raise ValueError("Empty bids/asks")
            if self.bids[0][0] >= self.asks[0][0]:
                raise ValueError("Crossed market detected")
            
            self._mid_price = (self.bids[0][0] + self.asks[0][0]) / 2
            self.price_history.append(self._mid_price)
            self._last_update = time.time()
            
        except Exception as e:
            logging.error(f"Processing failed: {e}")
        finally:
            self.latency_log.append((time.perf_counter() - start_time) * 1000)

    @jit(nopython=True)
    def _calculate_slippage_numba(self, levels: np.ndarray, quantity: float) -> float:
        """Numba-optimized core slippage calculation"""
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

    def calculate_slippage(self, quantity_usd: float, is_buy: bool = True) -> float:
        """Safe slippage calculation with fallbacks"""
        levels = self.asks if is_buy else self.bids
        if len(levels) == 0:
            logging.warning("Empty order book in slippage calc")
            return 0.0
        
        try:
            total_cost = self._calculate_slippage_numba(levels, quantity_usd)
            return (total_cost / quantity_usd - 1) * 100  # Percentage slippage
        except Exception as e:
            logging.error(f"Slippage calc error: {e}")
            return 0.0