import pickle
import pandas as pd
from src.config import required_features, model_path


# Load model once (FastAPI will import this file)
with open(model_path, "rb") as f:
    model = pickle.load(f)


def validate_and_prepare_input(data: dict) -> pd.DataFrame:
    """
    Ensures the incoming JSON has all required fields,
    fills missing ones with None, and returns a DataFrame
    with correct column order.
    """

    df = pd.DataFrame([data])

    # Add missing columns
    for col in required_features:
        if col not in df.columns:
            df[col] = None

    # Reorder columns
    df = df[required_features]

    return df


def predict_single(data: dict) -> dict:
    """
    Takes a JSON-like dict, validates it, runs prediction,
    and returns both class and probability.
    """

    df = validate_and_prepare_input(data)

    pred_class = int(model.predict(df)[0])
    pred_prob = float(model.predict_proba(df)[0][1])

    return {
        "prediction": pred_class,
        "probability": pred_prob
    }
