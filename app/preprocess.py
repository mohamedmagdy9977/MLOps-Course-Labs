"""Map one raw customer JSON (validated by Pydantic) to the numeric row the model expects."""

from __future__ import annotations

import numpy as np

from app.churn_schema import ChurnCustomerRequest

_GEO = {"France": 0.0, "Germany": 1.0, "Spain": 2.0}
_GENDER = {"Female": 0.0, "Male": 1.0}


def customer_to_matrix(customer: ChurnCustomerRequest) -> np.ndarray:
    """
    Single-row float matrix ``(1, 10)`` — same order as in ``scripts/build_churn_pipeline.py``.

    Geography and Gender are ordinally encoded with fixed tables (not learned at inference).
    The saved sklearn pipeline applies scaling + classifier on this matrix.
    """
    d = customer.model_dump()
    row = [
        float(d["CreditScore"]),
        _GEO[d["Geography"]],
        _GENDER[d["Gender"]],
        float(d["Age"]),
        float(d["Tenure"]),
        float(d["Balance"]),
        float(d["NumOfProducts"]),
        float(d["HasCrCard"]),
        float(d["IsActiveMember"]),
        float(d["EstimatedSalary"]),
    ]
    return np.asarray([row], dtype=np.float64)
