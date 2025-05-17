import sys
import os
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.orderbook import OrderBookProcessor

def test_process_valid_orderbook():
    processor = OrderBookProcessor()
    
    mock_msg = {
        "data": [{
            "bids": [["100.0", "2.0"]],
            "asks": [["101.0", "1.5"]]
        }]
    }

    processor.process(mock_msg)
    assert processor.bids.shape[0] > 0, "Bids should not be empty"
    assert processor.asks.shape[0] > 0, "Asks should not be empty"
    assert processor._mid_price == 100.5, "Mid price calculation is incorrect"


def test_slippage_with_mock_data():
    processor = OrderBookProcessor()
    mock_msg = {
        "data": [{
            "bids": [["99.0", "1.0"]],
            "asks": [["101.0", "2.0"], ["102.0", "5.0"]]
        }]
    }
    processor.process(mock_msg)
    slippage = processor.calculate_slippage(quantity_usd=100, is_buy=True)
    assert slippage >= 0, "Slippage should be non-negative"


def test_empty_orderbook_returns_zero_slippage():
    processor = OrderBookProcessor()
    processor.bids = np.array([])
    processor.asks = np.array([])

    slippage = processor.calculate_slippage(quantity_usd=50)
    assert slippage == 0.0, "Slippage should be zero on empty book"


def test_data_freshness_flag():
    processor = OrderBookProcessor()
    processor._last_update = 0  # simulate stale data
    assert not processor.is_fresh, "Data should be marked as stale"
