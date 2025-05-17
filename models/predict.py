from joblib import load
import numpy as np
import logging

class MakerPredictor:
    def __init__(self):
        try:
            self.scaler = load("models/scaler.pkl")
            self.model = load("models/maker_model.joblib")
            logging.info("Models loaded successfully")
        except Exception as e:
            logging.error(f"Model loading failed: {e}")
            raise

    def preprocess_features(self, quantity: float, volatility: float) -> np.ndarray:
        """Normalize inputs for prediction"""
        features = np.array([
            np.log1p(quantity),  # Log transform
            np.clip(volatility, 0.001, 0.5)  # Bounded volatility
        ]).reshape(1, -1)
        
        return self.scaler.transform(features)

    def predict_maker_probability(self, quantity: float, volatility: float) -> float:
        """Predict probability of order being maker"""
        try:
            features = self.preprocess_features(quantity, volatility)
            return float(self.model.predict_proba(features)[0, 1])  # Probability of class 1 (maker)
        except Exception as e:
            logging.error(f"Prediction failed: {e}")
            return 0.5  # Fallback value

# Example usage
if __name__ == "__main__":
    predictor = MakerPredictor()
    print(predictor.predict_maker_probability(1000, 0.02))  # Sample prediction