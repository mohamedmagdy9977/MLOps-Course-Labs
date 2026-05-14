"""
Model loading and prediction logic.

The model must be loaded ONCE at module level, NOT inside the predict function.
"""

from pathlib import Path

import joblib

from app.churn_schema import ChurnCustomerRequest
from app.preprocess import customer_to_matrix

_MODEL_PATH = Path(__file__).resolve().parent.parent / "data" / "model.joblib"
pipeline = joblib.load(_MODEL_PATH)


def predict_customer(customer: ChurnCustomerRequest) -> int:
    """Encode raw customer fields and return churn label (0 or 1)."""
    X = customer_to_matrix(customer)
    return int(pipeline.predict(X)[0])


if __name__ == "__main__":
    sample = ChurnCustomerRequest(
        CreditScore=650,
        Geography="France",
        Gender="Female",
        Age=42,
        Tenure=3,
        Balance=120_000.0,
        NumOfProducts=2,
        HasCrCard=1,
        IsActiveMember=1,
        EstimatedSalary=75_000.0,
    )
    print(f"Input:      {sample.model_dump()}")
    print(f"Prediction: {predict_customer(sample)}")
