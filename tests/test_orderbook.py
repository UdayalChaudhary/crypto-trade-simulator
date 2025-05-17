import pytest
import numpy as np
from src.orderbook import OrderBookProcessor
from datetime import datetime, timedelta

@pytest.fixture
def order_book():
    """Fixture providing a clean OrderBookProcessor instance"""
    return OrderBookProcessor()

@pytest.fixture
def sample_data():
    """Sample order book data for testing"""
    return {
        "bids": [[100.0, 2.5], [99.5, 3.0], [99.0, 1.5]],
        "asks": [[101.0, 2.0], [101.5, 1.0], [102.0, 3.0]]
    }

def test_initial_state(order_book):
    """Test order book initializes empty"""
    assert len(order_book.bids) == 0
    assert len(order_book.asks) == 0
    assert order_book._mid_price is None

def test_process_snapshot(order_book, sample_data):
    """Test processing full snapshot"""
    msg = {"data": [{"bids": sample_data["bids"], "asks": sample_data["asks"]}]}
    order_book.process(msg)
    
    # Verify bids
    assert np.array_equal(order_book.bids, np.array(sample_data["bids"]))
    assert order_book.bids[0][0] == 100.0  # Best bid
    
    # Verify asks
    assert np.array_equal(order_book.asks, np.array(sample_data["asks"]))
    assert order_book.asks[0][0] == 101.0  # Best ask
    
    # Verify mid price
    assert order_book._mid_price == pytest.approx(100.5)

def test_empty_orderbook_handling(order_book):
    """Test empty order book messages"""
    with pytest.raises(ValueError):
        order_book.process({"invalid": "data"})
    
    # Empty bids/asks should raise error
    with pytest.raises(ValueError):
        order_book.process({"data": [{"bids": [], "asks": []}]})

def test_crossed_market_detection(order_book):
    """Test detection of crossed markets (bid > ask)"""
    crossed_data = {
        "data": [{"bids": [[101.0, 1.0]], "asks": [[100.0, 1.0]]}]
    }
    with pytest.raises(ValueError, match="Crossed market"):
        order_book.process(crossed_data)

def test_slippage_calculation(order_book, sample_data):
    """Test slippage calculations"""
    msg = {"data": [{"bids": sample_data["bids"], "asks": sample_data["asks"]}]}
    order_book.process(msg)
    
    # Test buy order slippage (using asks)
    slippage = order_book.calculate_slippage(5.0, is_buy=True)
    expected_avg_price = (101.0*2.0 + 101.5*1.0 + 102.0*2.0) / 5.0
    expected_slippage = (expected_avg_price - 101.0) / 101.0 * 100
    assert slippage == pytest.approx(expected_slippage, rel=1e-3)
    
    # Test empty book case
    empty_book = OrderBookProcessor()
    assert empty_book.calculate_slippage(1.0) == 0.0

def test_data_freshness(order_book, sample_data):
    """Test freshness tracking"""
    msg = {"data": [{"bids": sample_data["bids"], "asks": sample_data["asks"]}]}
    order_book.process(msg)
    
    assert order_book.is_fresh is True
    
    # Simulate stale data
    order_book._last_update = datetime.now() - timedelta(seconds=10)
    assert order_book.is_fresh is False

def test_latency_tracking(order_book, sample_data):
    """Test latency measurement"""
    msg = {"data": [{"bids": sample_data["bids"], "asks": sample_data["asks"]}]}
    
    # First process
    order_book.process(msg)
    assert len(order_book.latency_log) == 1
    assert 0 < order_book.latency_log[0] < 100  # Should take <100ms
    
    # Second process
    order_book.process(msg)
    assert len(order_book.latency_log) == 2

def test_numba_fallback(monkeypatch, order_book, sample_data):
    """Test fallback when Numba is unavailable"""
    # Simulate Numba failure
    monkeypatch.setattr("numba.jit", lambda **kwargs: lambda f: f)
    
    msg = {"data": [{"bids": sample_data["bids"], "asks": sample_data["asks"]}]}
    order_book.process(msg)
    
    # Should still work without Numba
    slippage = order_book.calculate_slippage(1.0)
    assert isinstance(slippage, float)

def test_price_history(order_book, sample_data):
    """Test price history tracking"""
    msg = {"data": [{"bids": sample_data["bids"], "asks": sample_data["asks"]}]}
    
    # Process multiple updates
    for _ in range(3):
        order_book.process(msg)
    
    assert len(order_book.price_history) == 3
    assert all(price == 100.5 for price in order_book.price_history)