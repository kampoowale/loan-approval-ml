# src/config.py

# Categorical and numeric feature lists
categorical_features = [
    "Gender",
    "Married",
    "Dependents",
    "Education",
    "Self_Employed",
    "Property_Area"
]

numeric_features = [
    "ApplicantIncome",
    "CoapplicantIncome",
    "LoanAmount",
    "Loan_Amount_Term",
    "Credit_History"
]

# Combined list for prediction input
required_features = categorical_features + numeric_features

# Hyperparameter grid for GridSearchCV
param_grid = {
    "model__n_estimators": [100, 200],
    "model__max_depth": [5, 10, None],
    "model__min_samples_split": [2, 5]
}

# Path to save the trained model
model_path = "models/best_model.pkl"
