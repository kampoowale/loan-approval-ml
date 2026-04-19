import pandas as pd
import pickle
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier

from src.config import categorical_features, numeric_features, param_grid, model_path


def load_data():
    df = pd.read_csv("data/raw/loan_data.csv")
    return df


def build_preprocessor():
    transformer = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features),
            ("num", "passthrough", numeric_features)
        ]
    )
    return transformer


def build_pipeline():
    preprocessor = build_preprocessor()
    pipeline = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("model", RandomForestClassifier())
    ])
    return pipeline


def train_model():
    df = load_data()

    X = df.drop("Loan_Status", axis=1)
    y = df["Loan_Status"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    pipeline = build_pipeline()

    grid_search = GridSearchCV(
        estimator=pipeline,
        param_grid=param_grid,
        cv=5,
        scoring="f1",
        n_jobs=-1
    )

    grid_search.fit(X_train, y_train)

    best_model = grid_search.best_estimator_

    with open(model_path, "wb") as f:
        pickle.dump(best_model, f)

    print("Training complete. Best model saved.")

    return best_model


if __name__ == "__main__":
    train_model()
