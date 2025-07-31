import os
import joblib

def save_model(model, filename):
    """Save model to models directory"""
    os.makedirs("models", exist_ok=True)
    path = os.path.join("models", filename)
    joblib.dump(model, path)
    return path

def load_model(filename):
    """Load model from models directory"""
    path = os.path.join("models", filename)
    return joblib.load(path)
