import pandas as pd
import numpy as np

np.random.seed(42)

rows = 500

gender = np.random.choice(["Male", "Female"], rows)
married = np.random.choice(["Yes", "No"], rows)
dependents = np.random.choice(["0", "1", "2", "3+"], rows)
education = np.random.choice(["Graduate", "Not Graduate"], rows)
self_employed = np.random.choice(["Yes", "No"], rows)

applicant_income = np.random.normal(
    5000, 2000, rows).clip(1500, 15000).astype(int)
coapplicant_income = np.random.normal(
    1500, 1000, rows).clip(0, 8000).astype(int)

loan_amount = np.random.normal(150, 50, rows).clip(50, 400).astype(int)
loan_term = np.random.choice([120, 180, 240, 300, 360], rows)

credit_history = np.random.choice([1.0, 0.0], rows, p=[0.8, 0.2])
property_area = np.random.choice(["Urban", "Semiurban", "Rural"], rows)

# Loan approval logic (realistic but synthetic)
loan_status = (
    (credit_history == 1.0).astype(int)
    + (applicant_income > 4000).astype(int)
    + (loan_amount < 200).astype(int)
)

loan_status = (loan_status >= 2).astype(int)

df = pd.DataFrame({
    "Gender": gender,
    "Married": married,
    "Dependents": dependents,
    "Education": education,
    "Self_Employed": self_employed,
    "ApplicantIncome": applicant_income,
    "CoapplicantIncome": coapplicant_income,
    "LoanAmount": loan_amount,
    "Loan_Amount_Term": loan_term,
    "Credit_History": credit_history,
    "Property_Area": property_area,
    "Loan_Status": loan_status
})

df.to_csv("data/raw/loan_data.csv", index=False)

print("Dataset generated: data/raw/loan_data.csv")
