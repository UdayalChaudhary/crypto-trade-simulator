import numpy as np
from numba import jit
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from joblib import dump, load
import os
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [Models] %(levelname)s: %(message)s"
)

class ExecutionModels:
    def __init__(self):
        os.makedirs("models", exist_ok=True)
        self.scaler = StandardScaler()
        self.maker_model = self._load_model("maker_model.joblib")
        self._initialize_default_models()

    def _load_model(self, filename: str):
        """Load a model from disk if it exists."""
        path = f"models/{filename}"
        if os.path.exists(path):
            logging.info(f"Loaded pre-trained model: {filename}")
            return load(path)
        return None

    def _initialize_default_models(self):
        """Fallback models if loading/training fails."""
        if self.maker_model is None:
            logging.warning("Initializing default logistic regression model for maker probability.")
            self.maker_model = LogisticRegression()
            # Train with dummy data
            X = np.array([[1, 0.1], [10, 0.2]])
            y = np.array([0, 1])
            self.maker_model.fit(X, y)

    @staticmethod
    @jit(nopython=True)
    def calculate_volatility(prices: np.ndarray) -> float:
        """Annualized volatility with safety checks."""
        if len(prices) < 2:
            return 0.0
        log_returns = np.log(prices[1:] / prices[:-1])
        return np.std(log_returns) * np.sqrt(365)

    def almgren_chriss_impact(self, quantity: float, volatility: float, daily_volume: float) -> float:
        """
        Calculate market impact using a simplified Almgren-Chriss model.
        Returns 0 if any required input is non-positive.
        """
        if quantity <= 0 or daily_volume <= 0 or volatility <= 0:
            return 0.0
        normalized_qty = np.clip(quantity / daily_volume, 1e-10, 1.0)  # Prevent division issues
        return 0.5 * volatility * np.sqrt(normalized_qty)

    def predict_maker_probability(self, features: list) -> float:
        """Normalized prediction with fallback in case of model error."""
        try:
            # Feature normalization
            norm_features = [
                np.log1p(features[0]) / 10.0,  # Quantity (log-scaled)
                np.clip(features[1], 0.001, 0.5)  # Volatility
            ]
            return self.maker_model.predict_proba([norm_features])[0][1]
        except Exception as e:
            logging.error(f"Prediction failed: {e}")
            return 0.5

    def save_models(self):
        """Persist models to disk."""
        dump(self.maker_model, "models/maker_model.joblib")
        logging.info("Models saved to disk")