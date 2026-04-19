from fastapi import FastAPI
from pydantic import BaseModel
from src.predict import predict_single

app = FastAPI(
    title="Loan Approval Prediction API",
    description="A machine learning API that predicts loan approval using a trained model.",
    version="1.0"
)


# Input schema for validation
class LoanApplication(BaseModel):
    Gender: str | None = None
    Married: str | None = None
    Dependents: str | None = None
    Education: str | None = None
    Self_Employed: str | None = None
    ApplicantIncome: float | None = None
    CoapplicantIncome: float | None = None
    LoanAmount: float | None = None
    Loan_Amount_Term: float | None = None
    Credit_History: float | None = None
    Property_Area: str | None = None


@app.get("/")
def home():
    return {"message": "Loan Approval Prediction API is running"}


@app.post("/predict")
def predict_loan(data: LoanApplication):
    data_dict = data.dict()
    result = predict_single(data_dict)
    return result
