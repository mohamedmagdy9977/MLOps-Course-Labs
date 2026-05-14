"""
Tests for the Churn Prediction API.

Run with:
    pytest tests/ -v
    pytest tests/ -v --cov=app --cov=main --cov-report=term-missing
"""

import csv
from pathlib import Path

import pytest

from app.churn_schema import ChurnCustomerRequest
from app.model_utils import predict_customer
from litestar.testing import TestClient
from main import app

_REPO_ROOT = Path(__file__).resolve().parent.parent
_CHURN_CSV = _REPO_ROOT / "Churn_Modelling.csv"

# Same customer as in Churn_Modelling.csv (CreditScore 619, France, Female, age 42, …).
# CSV has a larger EstimatedSalary; this payload matches the JSON shape you use in Swagger.
REAL_CUSTOMER_JSON = {
    "CreditScore": 619,
    "Geography": "France",
    "Gender": "Female",
    "Age": 42,
    "Tenure": 2,
    "Balance": 0,
    "NumOfProducts": 1,
    "HasCrCard": 1,
    "IsActiveMember": 1,
    "EstimatedSalary": 5,
}


def _load_matching_csv_row() -> dict:
    """Return API payload built from the CSV row that matches the example above (full CSV salary)."""
    if not _CHURN_CSV.is_file():
        pytest.skip("Churn_Modelling.csv not in repo root")
    with _CHURN_CSV.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if (
                row.get("CreditScore") == "619"
                and row.get("Geography") == "France"
                and row.get("Gender") == "Female"
                and row.get("Age") == "42"
                and row.get("Tenure") == "2"
                and row.get("Balance") in ("0", "0.0")
                and row.get("NumOfProducts") == "1"
            ):
                return {
                    "CreditScore": int(row["CreditScore"]),
                    "Geography": row["Geography"],
                    "Gender": row["Gender"],
                    "Age": int(row["Age"]),
                    "Tenure": int(row["Tenure"]),
                    "Balance": float(row["Balance"] or 0),
                    "NumOfProducts": int(row["NumOfProducts"]),
                    "HasCrCard": int(row["HasCrCard"]),
                    "IsActiveMember": int(row["IsActiveMember"]),
                    "EstimatedSalary": float(row["EstimatedSalary"]),
                }
    pytest.fail("Expected a matching row in Churn_Modelling.csv")


# ---------------------------------------------------------------------------
# Function Tests
# ---------------------------------------------------------------------------


def test_predict_customer_real_json_example():
    customer = ChurnCustomerRequest(**REAL_CUSTOMER_JSON)
    assert predict_customer(customer) == 0


def test_predict_customer_csv_native_salary():
    payload = _load_matching_csv_row()
    customer = ChurnCustomerRequest(**payload)
    assert predict_customer(customer) in (0, 1)


def test_predict_customer_high_risk():
    customer = ChurnCustomerRequest(
        CreditScore=400,
        Geography="Spain",
        Gender="Male",
        Age=72,
        Tenure=2,
        Balance=200_000.0,
        NumOfProducts=4,
        HasCrCard=0,
        IsActiveMember=0,
        EstimatedSalary=150_000.0,
    )
    assert predict_customer(customer) == 1


def test_predict_customer_label_is_binary():
    customer = ChurnCustomerRequest(
        CreditScore=600,
        Geography="Germany",
        Gender="Male",
        Age=40,
        Tenure=4,
        Balance=50_000.0,
        NumOfProducts=2,
        HasCrCard=1,
        IsActiveMember=0,
        EstimatedSalary=80_000.0,
    )
    assert predict_customer(customer) in (0, 1)


# ---------------------------------------------------------------------------
# Endpoint Tests
# ---------------------------------------------------------------------------


def test_post_predict_real_json():
    with TestClient(app=app) as client:
        response = client.post("/predict", json=REAL_CUSTOMER_JSON)
        assert response.status_code == 201
        assert response.json() == {"prediction": 0}


def test_post_predict_payload_from_csv_file():
    payload = _load_matching_csv_row()
    with TestClient(app=app) as client:
        response = client.post("/predict", json=payload)
        assert response.status_code == 201
        assert response.json()["prediction"] in (0, 1)


def test_get_health():
    with TestClient(app=app) as client:
        response = client.get("/health")
        assert response.status_code == 200
        assert response.json() == {"status": "healthy"}


def test_get_home():
    with TestClient(app=app) as client:
        response = client.get("/")
        assert response.status_code == 200
        assert response.json() == {"message": "Welcome to the Churn Prediction API"}


def test_invalid_input():
    with TestClient(app=app) as client:
        response = client.post(
            "/predict",
            json={
                "CreditScore": 800,
                "Geography": "NotACountry",
                "Gender": "Female",
                "Age": 25,
                "Tenure": 5,
                "Balance": 10_000.0,
                "NumOfProducts": 1,
                "HasCrCard": 1,
                "IsActiveMember": 1,
                "EstimatedSalary": 50_000.0,
            },
        )
        assert response.status_code == 400
        assert response.json() == {"detail": "Invalid input"}
