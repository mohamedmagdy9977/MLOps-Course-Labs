# Final Project Checklist

## Setup

- [ ] Fork <https://github.com/ishraq-hassan/MLOps-Course-Labs>
- [ ] Clone your fork locally
- [ ] Install dependencies: `uv sync` or `pip install -e ".[dev]"`
- [ ] Install pre-commit hooks: `uv run pre-commit install`
- [ ] Place your best churn model into `data/model.joblib`

## Logger (`app/logger_setup.py`)

- [ ] TODO 1: Set up basic logging with level INFO
- [ ] TODO 2: Create and return a named logger

## Model Utils (`app/model_utils.py`)

- [ ] TODO 1: Load your model at module level
- [ ] TODO 2: Implement `predict_churn()`
- [ ] TODO 3: Fill in sample features
- [ ] Verify: `uv run python -m app.model_utils`

## API (`main.py`)

- [ ] TODO 1: Define `ChurnRequest` fields
- [ ] TODO 2: Create `GET /`
- [ ] TODO 3: Create `GET /health`
- [ ] TODO 4: Create `POST /predict` with logging
- [ ] TODO 5: Register handlers in `Litestar(route_handlers=[...])`

## Run & Screenshot

- [ ] Start the server: `uv run litestar --app main:app run --reload`
- [ ] Open <http://localhost:8000/schema/swagger>
- [ ] **Take a screenshot of the Swagger UI**

## Tests (`tests/test_main.py`)

- [ ] TODO 1: Function test for `predict_churn`
- [ ] TODO 3: Endpoint test for `POST /predict`
- [ ] TODO 4: Endpoint test for `GET /health`
- [ ] TODO 5: Endpoint test for `GET /`
- [ ] Run: `uv run pytest tests/ -v --cov=app --cov=main --cov-report=term-missing`
- [ ] **Coverage is above 70%**
- [ ] **Take a screenshot of the results + coverage**

## Bonus

- [ ] TODO 2 (tests): Extra function test with edge cases
- [ ] TODO 6 (tests): Test invalid input returns 400
- [ ] Set up HyperDX for live logs

## Submit

- [ ] Commit the `uv.lock` file (**points will be deducted if missing**)
- [ ] Push to your fork
- [ ] **Upload** the Swagger UI screenshot
- [ ] **Upload** the test results and coverage screenshot
- [ ] **Upload** the link to your repo

---

## Quick Reference

```bash
uv run litestar --app main:app run --reload                              # start server
uv run pytest tests/ -v                                                  # run tests
uv run pytest tests/ -v --cov=app --cov=main --cov-report=term-missing   # with coverage
```

| Litestar vs FastAPI     | FastAPI            | Litestar                      |
| ----------------------- | ------------------ | ----------------------------- |
| Swagger UI              | `/docs`            | `/schema/swagger`             |
| Run command             | `uvicorn main:app` | `litestar --app main:app run` |
| Route decorators        | `@app.get("/")`    | `@get("/")`                   |
| App creation            | `FastAPI()`        | `Litestar(route_handlers=[])` |
| POST default status     | 200                | 201                           |
| Validation error status | 422                | 400                           |
