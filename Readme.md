
**Loan Approval Prediction API**

A machine learning project that predicts whether a loan application will be approved.  
Built with a clean modular architecture, FastAPI, and a trained ML model.

**Features**

- End‑to‑end ML pipeline  
- Clean modular project structure  
- FastAPI prediction endpoint  
- Input validation using Pydantic  
- Model saved and loaded from `/models`  
- Basic test cases using pytest  
- Ready for deployment (Render, Railway, Azure, etc.)

**Project Structure**

loan-approval-ml/
│── data/
│   ├── raw/
│   ├── processed/
│
│── models/
│   ├── best_model.pkl
│
│── src/
│   ├── config.py
│   ├── preprocessing.py
│   ├── train.py
│   ├── predict.py
│
│── api/
│   ├── main.py
│
│── tests/
│   ├── test_prediction.py
│
│── notebooks/
│   ├── exploration.ipynb
│
│── README.md
│── requirements.txt

**How the Model Works**

- Preprocessing using `ColumnTransformer`  
- OneHotEncoding for categorical features  
- RandomForestClassifier inside a Pipeline  
- GridSearchCV for hyperparameter tuning  
- Model saved as `best_model.pkl

**Training the Model**

Run:
python src/train.py

This will:
- load data  
- preprocess  
- train  
- run GridSearchCV  
- save the best model  

**Running the API**

Start the FastAPI server:
uvicorn api.main:app --reload

Open Swagger UI:
http://127.0.0.1:8000/docs

**Example Request**

POST to `/predict`:

```json
{
  "Gender": "Male",
  "Married": "Yes",
  "Dependents": "1",
  "Education": "Graduate",
  "Self_Employed": "No",
  "ApplicantIncome": 5000,
  "CoapplicantIncome": 2000,
  "LoanAmount": 150,
  "Loan_Amount_Term": 360,
  "Credit_History": 1,
  "Property_Area": "Urban"
}
```

---

**Example Response**

```json
{
  "prediction": 1,
  "probability": 0.87
}
```

**Running Tests**

```
pytest
```

**Requirements**

Install dependencies:

```
pip install -r requirements.txt
```

---

**Technologies Used**

- Python  
- scikit‑learn  
- FastAPI  
- Pydantic  
- Uvicorn  
- Pandas  
- pytest  