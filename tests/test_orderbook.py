import numpy as np
from src.orderbook import OrderBookProcessor

def test_process_valid_orderbook():
    processor = OrderBookProcessor()
    mock_msg = {
        "data": [{
            "bids": [["9500.0", "1.5"]],
            "asks": [["9501.0", "1.0"]]
        }]
    }
    processor.process(mock_msg)
    assert processor._mid_price > 0, "Mid price should be calculated"
    assert len(processor.bids) > 0
    assert len(processor.asks) > 0

def test_empty_book_slippage():
    processor = OrderBookProcessor()
    slippage = processor.calculate_slippage(100, is_buy=True)
    assert slippage == 0, "Slippage on empty book should be 0"

def test_freshness_check():
    processor = OrderBookProcessor()
    processor._last_update = 0  # simulate old timestamp
    assert processor.is_fresh is False, "Data should be stale"
