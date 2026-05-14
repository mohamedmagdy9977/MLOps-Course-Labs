"""API request shape for raw bank churn rows (before encoding)."""

from typing import Literal

from pydantic import BaseModel, Field

Geography = Literal["France", "Germany", "Spain"]
Gender = Literal["Female", "Male"]


class ChurnCustomerRequest(BaseModel):
    """Customer fields as in the modelling CSV (no ``Exited`` — that is what we predict)."""

    CreditScore: int = Field(ge=0, le=1000)
    Geography: Geography
    Gender: Gender
    Age: int = Field(ge=18, le=100)
    Tenure: int = Field(ge=0, le=20)
    Balance: float = Field(ge=0)
    NumOfProducts: int = Field(ge=1, le=4)
    HasCrCard: int = Field(ge=0, le=1)
    IsActiveMember: int = Field(ge=0, le=1)
    EstimatedSalary: float = Field(ge=0)
