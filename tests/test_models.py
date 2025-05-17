import pytest
import numpy as np
from src.models import ExecutionModels
from joblib import dump, load
import os

@pytest.fixture
def model(tmp_path):
    """Fixture providing a clean ExecutionModels instance with temp dir"""
    model_dir = tmp_path / "models"
    model_dir.mkdir()
    return ExecutionModels()

@pytest.fixture
def sample_data():
    """Generate test data for volatility and prediction"""
    return {
        "prices": np.array([100, 101, 100.5, 102, 101.5]),
        "features": [50, 0.1]  # quantity, volatility
    }

def test_volatility_calculation(model, sample_data):
    """Test volatility calculation from price series"""
    volatility = model.calculate_volatility(sample_data["prices"])
    expected = np.std(np.log(sample_data["prices"][1:]/sample_data["prices"][:-1])) * np.sqrt(365)
    assert volatility == pytest.approx(expected, rel=1e-3)

def test_volatility_edge_cases(model):
    """Test volatility with edge cases"""
    # Single price point
    assert model.calculate_volatility(np.array([100])) == 0.0
    
    # Constant price
    assert model.calculate_volatility(np.array([100, 100, 100])) == 0.0
    
    # Negative prices (should never happen but test robustness)
    with pytest.raises(ValueError):
        model.calculate_volatility(np.array([100, -101]))

def test_market_impact(model):
    """Test Almgren-Chriss impact model"""
    # Normal case
    impact = model.almgren_chriss_impact(10000, 0.2, 1_000_000)
    assert 0 < impact < 100  # Reasonable bounds
    
    # Edge cases
    assert model.almgren_chriss_impact(0, 0.2, 1_000_000) == 0.0
    assert model.almgren_chriss_impact(1, 0.0, 1_000_000) == 0.0

def test_maker_probability(model, sample_data):
    """Test maker/taker probability prediction"""
    prob = model.predict_maker_probability(sample_data["features"])
    assert 0 <= prob <= 1  # Valid probability range
    
    # Test with None model (should use default)
    original_model = model.maker_model
    model.maker_model = None
    assert model.predict_maker_probability(sample_data["features"]) == 0.5
    model.maker_model = original_model

def test_model_persistence(model, tmp_path, sample_data):
    """Test saving/loading models"""
    # Train a simple model
    X = np.array([[10, 0.1], [100, 0.2], [1000, 0.3]])
    y = np.array([0, 1, 1])
    model.maker_model.fit(X, y)
    
    # Save and reload
    model_path = tmp_path / "models" / "test_model.joblib"
    dump(model.maker_model, model_path)
    loaded_model = load(model_path)
    
    # Verify same predictions
    original_pred = model.maker_model.predict_proba([sample_data["features"]])[0]
    loaded_pred = loaded_model.predict_proba([sample_data["features"]])[0]
    assert np.allclose(original_pred, loaded_pred)

def test_feature_normalization(model, monkeypatch):
    """Test input normalization in predictions"""
    # Mock model to verify inputs
    def mock_predict_proba(X):
        assert 0 < X[0][0] < 1  # Quantity normalized
        assert 0 < X[0][1] < 1   # Volatility clipped
        return [[0.5]]
    
    monkeypatch.setattr(model.maker_model, "predict_proba", mock_predict_proba)
    model.predict_maker_probability([1_000_000, 2.0])  # Extreme values