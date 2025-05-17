import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from joblib import dump
import os

def generate_training_data():
    """Synthetic training data simulating real trading conditions"""
    np.random.seed(42)
    
    # Features: [log(quantity), volatility]
    X = np.column_stack([
        np.log1p(np.random.exponential(scale=1000, size=1000)),  # Log quantity
        np.clip(np.random.normal(loc=0.02, scale=0.01, size=1000), 0.001, 0.5)  # Volatility
    ])
    
    # Labels: 1=maker, 0=taker (based on heuristic rules)
    y = ((X[:, 0] > 5) & (X[:, 1] < 0.03)).astype(int)  # Large orders in stable markets tend to be makers
    
    return X, y

def train_and_save_models():
    # Create models directory if not exists
    os.makedirs("models", exist_ok=True)
    
    # Generate training data
    X, y = generate_training_data()
    
    # Train scaler
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Train logistic regression model
    model = LogisticRegression(
        penalty='l2',
        C=0.1,
        solver='lbfgs',
        class_weight='balanced'
    )
    model.fit(X_scaled, y)
    
    # Save models
    dump(scaler, "models/scaler.pkl")
    dump(model, "models/maker_model.joblib")
    print("Models trained and saved successfully")

if __name__ == "__main__":
    train_and_save_models()