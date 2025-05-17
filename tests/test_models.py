import numpy as np
from src.models import ExecutionModels

def test_volatility_calculation():
    prices = np.array([100, 101, 102, 103, 104])
    vol = ExecutionModels.calculate_volatility(prices)
    assert vol > 0, "Volatility should be greater than 0 for increasing prices"

def test_almgren_chriss_impact():
    model = ExecutionModels()
    impact = model.almgren_chriss_impact(quantity=10000, volatility=0.02, daily_volume=1_000_000)
    assert impact > 0, "Impact should be positive"

def test_maker_probability_default():
    model = ExecutionModels()
    prob = model.predict_maker_probability([100, 0.01])
    assert 0 <= prob <= 1, "Probability should be in [0, 1]"

def test_fallback_model_training():
    model = ExecutionModels()
    model._initialize_default_models()
    assert model.maker_model is not None
