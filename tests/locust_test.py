"""
Locust load tests for the Churn Prediction API.

Start the API first:
    litestar --app main:app run --host 0.0.0.0 --port 8000

Run Locust (web UI on http://localhost:8089):
    locust -f tests/locust_test.py --host http://localhost:8000

Headless example (10 users, 2/s spawn rate, 60s):
    locust -f tests/locust_test.py --host http://localhost:8000 \\
        --headless -u 10 -r 2 -t 60s
"""

from locust import HttpUser, between, task

# Sample customer payload (same shape as POST /predict in Swagger)
PREDICT_PAYLOAD = {
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

ALTERNATE_PAYLOAD = {
    "CreditScore": 650,
    "Geography": "Germany",
    "Gender": "Male",
    "Age": 40,
    "Tenure": 4,
    "Balance": 50_000.0,
    "NumOfProducts": 2,
    "HasCrCard": 1,
    "IsActiveMember": 0,
    "EstimatedSalary": 80_000.0,
}


class ChurnApiUser(HttpUser):
    """Simulates clients hitting the churn API."""

    wait_time = between(0.5, 2)

    @task(1)
    def get_home(self) -> None:
        with self.client.get("/", name="GET /", catch_response=True) as response:
            if response.status_code != 200:
                response.failure(f"Expected 200, got {response.status_code}")

    @task(2)
    def get_health(self) -> None:
        with self.client.get("/health", name="GET /health", catch_response=True) as response:
            if response.status_code != 200:
                response.failure(f"Expected 200, got {response.status_code}")
            elif response.json().get("status") != "healthy":
                response.failure("Unexpected health body")

    @task(5)
    def post_predict_sample(self) -> None:
        self._post_predict(PREDICT_PAYLOAD)

    @task(3)
    def post_predict_alternate(self) -> None:
        self._post_predict(ALTERNATE_PAYLOAD)

    def _post_predict(self, payload: dict) -> None:
        with self.client.post(
            "/predict",
            json=payload,
            name="POST /predict",
            catch_response=True,
        ) as response:
            if response.status_code != 201:
                response.failure(f"Expected 201, got {response.status_code}: {response.text}")
                return
            body = response.json()
            if body.get("prediction") not in (0, 1):
                response.failure(f"Invalid prediction: {body}")
