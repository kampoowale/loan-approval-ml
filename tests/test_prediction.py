import pickle
import pandas as pd
from src.config import required_features


def test_model_loads():
    with open("models/best_model.pkl", "rb") as f:
        model = pickle.load(f)
    assert model is not None


def test_prediction_output():
    with open("models/best_model.pkl", "rb") as f:
        model = pickle.load(f)

    sample = {col: 1 for col in required_features}
    df = pd.DataFrame([sample])

    pred = model.predict(df)[0]
    assert pred in [0, 1]
