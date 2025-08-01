import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.train import load_data, train
import joblib
from sklearn.linear_model import LinearRegression

import joblib
from sklearn.linear_model import LinearRegression
from src.train import load_data, train

def test_data_loading():
    X_train, X_test, y_train, y_test = load_data()
    assert X_train.shape[0] > 0 and X_test.shape[0] > 0

def test_model_training():
    train()
    model = joblib.load("src/model.joblib")
    assert isinstance(model, LinearRegression)
    assert hasattr(model, "coef_")

def test_r2_threshold():
    X_train, X_test, y_train, y_test = load_data()
    model = joblib.load("src/model.joblib")
    y_pred = model.predict(X_test)
    from sklearn.metrics import r2_score
    assert r2_score(y_test, y_pred) > 0.4  # example threshold
