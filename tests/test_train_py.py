import pytest
import numpy as np
import os
from sklearn.linear_model import LinearRegression
from src.train import load_data, train_model, evaluate_model
from src.utils import save_model, load_model

def test_data_loading():
    """Ensure data loads correctly"""
    X_train, X_test, y_train, y_test = load_data()
    assert X_train.shape[1] == 8  # California housing has 8 features
    assert len(X_train) == len(y_train)
    assert len(X_test) == len(y_test)
    assert not np.isnan(X_train).any()
    assert not np.isnan(y_train).any()

def test_model_training():
    """Ensure model trains correctly"""
    X_train, _, y_train, _ = load_data()
    model = train_model(X_train, y_train)
    assert isinstance(model, LinearRegression)
    assert hasattr(model, "coef_")
    assert hasattr(model, "intercept_")

def test_model_performance():
    """Ensure model meets minimum R² performance"""
    X_train, X_test, y_train, y_test = load_data()
    model = train_model(X_train, y_train)
    r2, _ = evaluate_model(model, X_test, y_test)
    assert r2 > 0.5  # basic threshold

def test_model_saving():
    """Ensure model saves and loads correctly"""
    X_train, _, y_train, _ = load_data()
    model = train_model(X_train, y_train)
    
    path = save_model(model, "test_model.joblib")
    loaded_model = load_model("test_model.joblib")
    
    assert np.allclose(model.coef_, loaded_model.coef_)
    assert np.isclose(model.intercept_, loaded_model.intercept_)
    
    # cleanup
    os.remove(path)

if __name__ == "__main__":
    pytest.main([__file__])
