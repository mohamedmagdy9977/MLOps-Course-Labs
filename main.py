"""
Churn Prediction API

Run with:
    litestar --app main:app run --reload
Then open:
    http://localhost:8000/schema/swagger
"""

from litestar import Litestar, Request, Response, get, post
from litestar.exceptions import ValidationException
from litestar.status_codes import HTTP_201_CREATED

from app.churn_schema import ChurnCustomerRequest
from app.logger_setup import setup_logging
from app.model_utils import predict_customer

logger = setup_logging()


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------
@get("/", sync_to_thread=False)
def get_home() -> dict[str, str]:
    logger.info("Home endpoint accessed")
    return {"message": "Welcome to the Churn Prediction API"}


@get("/health", sync_to_thread=False)
def get_health() -> dict[str, str]:
    return {"status": "healthy"}


@post("/predict", status_code=HTTP_201_CREATED, sync_to_thread=False)
def post_predict(data: ChurnCustomerRequest) -> dict[str, int]:
    prediction = predict_customer(data)
    logger.info("Prediction request customer=%s prediction=%s", data.model_dump(), prediction)
    return {"prediction": prediction}


def validation_exception_handler(_request: Request, _exc: ValidationException) -> Response:
    return Response(content={"detail": "Invalid input"}, status_code=400)


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------
app = Litestar(
    route_handlers=[get_home, get_health, post_predict],
    exception_handlers={ValidationException: validation_exception_handler},
)
