import logging
from io import BytesIO

from fastapi import FastAPI, File, HTTPException, UploadFile
from PIL import Image
from starlette.requests import Request

from monitoring.logging_config import configure_logging
from monitoring.metrics import metrics, now
from src.api.inference import load_model, predict_image
from src.api.schemas import HealthResponse, PredictionResponse

configure_logging()
logger = logging.getLogger("cats-dogs-api")

app = FastAPI(title="Cats vs Dogs Classifier", version="1.0.0")
model = None


@app.on_event("startup")
def startup_event() -> None:
    global model
    try:
        model = load_model()
    except FileNotFoundError:
        logger.warning("Model artifact not found at startup; API will remain healthy but prediction is unavailable.")
        model = None


@app.middleware("http")
async def log_requests(request: Request, call_next):
    start = now()
    response = await call_next(request)
    latency = now() - start
    metrics.observe(latency)
    logger.info(
        "request path=%s method=%s status=%s latency_ms=%.2f",
        request.url.path,
        request.method,
        response.status_code,
        latency * 1000,
    )
    return response


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(status="ok", model_loaded=model is not None)


@app.get("/metrics")
def app_metrics() -> dict[str, float | int]:
    return {
        "request_count": metrics.request_count,
        "average_latency_seconds": metrics.average_latency_seconds,
    }


@app.post("/predict", response_model=PredictionResponse)
async def predict(file: UploadFile = File(...)) -> PredictionResponse:
    if model is None:
        raise HTTPException(status_code=503, detail="Model is not available. Restore or rebuild the trained checkpoint before calling /predict.")

    contents = await file.read()
    image = Image.open(BytesIO(contents))
    prediction = predict_image(model, image)
    logger.info("prediction filename=%s label=%s", file.filename, prediction["label"])
    return PredictionResponse(**prediction)
