import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.train import load_data
import joblib

def predict():
    _, X_test, _, _ = load_data()
    model_path = os.path.join(os.path.dirname(__file__), "model.joblib")
    model = joblib.load(model_path)
    predictions = model.predict(X_test[:5])
    print("Sample Predictions:", predictions)

if __name__ == "__main__":
    predict()
