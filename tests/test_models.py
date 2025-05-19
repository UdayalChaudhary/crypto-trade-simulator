import sys
import os
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.models import ExecutionModels


def test_calculate_volatility():
    prices = np.array([100, 101, 99, 102, 100])
    vol = ExecutionModels.calculate_volatility(prices)
    assert isinstance(vol, float)
    assert vol > 0, "Volatility should be positive"


def test_volatility_with_constant_prices():
    prices = np.array([100, 100, 100, 100])
    vol = ExecutionModels.calculate_volatility(prices)
    assert vol == 0.0, "Volatility should be zero when prices are constant"


def test_almgren_chriss_impact_typical():
    model = ExecutionModels()
    impact = model.almgren_chriss_impact(quantity=5000, volatility=0.02, daily_volume=1_000_000)
    assert impact > 0, "Market impact should be positive for typical input"


def test_almgren_chriss_zero_quantity():
    model = ExecutionModels()
    impact = model.almgren_chriss_impact(quantity=0, volatility=0.02, daily_volume=1_000_000)
    assert impact == 0, "Market impact should be zero for zero quantity"


def test_predict_maker_probability_range():
    model = ExecutionModels()
    prob = model.predict_maker_probability([100, 0.01])
    assert 0 <= prob <= 1, "Maker probability should be between 0 and 1"


def test_predict_maker_probability_invalid_input():
    model = ExecutionModels()
    # Provide malformed input
    prob = model.predict_maker_probability(["bad", None])
<<<<<<< HEAD
    assert 0 <= prob <= 1, "Model should fallback to 0.5 for invalid input"
=======
    assert 0 <= prob <= 1, "Model should fallback to 0.5 for invalid input"
>>>>>>> 8131dd2 (Refactored)
