#!/usr/bin/env python3
"""Fit a sklearn pipeline on synthetic data and write ``data/model.joblib`` (run from repo root)."""

from pathlib import Path

import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def _synthetic_matrix(rng: np.random.Generator, n: int) -> np.ndarray:
    """Build ``(n, 10)`` design matrix with the same layout as ``app.preprocess.customer_to_matrix``."""
    credit = rng.integers(350, 900, size=n).astype(np.float64)
    geo = rng.integers(0, 3, size=n).astype(np.float64)
    gender = rng.integers(0, 2, size=n).astype(np.float64)
    age = rng.integers(18, 85, size=n).astype(np.float64)
    tenure = rng.integers(0, 11, size=n).astype(np.float64)
    balance = rng.uniform(0, 250_000, size=n).astype(np.float64)
    nprod = rng.integers(1, 5, size=n).astype(np.float64)
    card = rng.integers(0, 2, size=n).astype(np.float64)
    active = rng.integers(0, 2, size=n).astype(np.float64)
    salary = rng.uniform(10_000, 200_000, size=n).astype(np.float64)
    return np.column_stack([credit, geo, gender, age, tenure, balance, nprod, card, active, salary])


def main() -> None:
    root = Path(__file__).resolve().parent.parent
    data_dir = root / "data"
    data_dir.mkdir(exist_ok=True)

    rng = np.random.default_rng(42)
    n = 1_200
    X = _synthetic_matrix(rng, n)
    score = (X[:, 3] > 55) * 2 + (X[:, 5] > 120_000) * 2 + (X[:, 6] >= 3) + (X[:, 8] == 0) + rng.normal(0, 0.4, size=n)
    y = (score > 2.5).astype(int)

    pipeline = Pipeline(
        steps=[
            ("scale", StandardScaler()),
            ("clf", LogisticRegression(max_iter=1_000, random_state=42)),
        ]
    )
    pipeline.fit(X, y)

    out_path = data_dir / "model.joblib"
    joblib.dump(pipeline, out_path)
    print(f"Wrote {out_path}")

    low = np.array(
        [[800.0, 0.0, 0.0, 25.0, 5.0, 10_000.0, 1.0, 1.0, 1.0, 50_000.0]],
        dtype=np.float64,
    )
    high = np.array(
        [[400.0, 2.0, 1.0, 72.0, 2.0, 200_000.0, 4.0, 0.0, 0.0, 150_000.0]],
        dtype=np.float64,
    )
    print("sample_low_pred", int(pipeline.predict(low)[0]))
    print("sample_high_pred", int(pipeline.predict(high)[0]))


if __name__ == "__main__":
    main()
