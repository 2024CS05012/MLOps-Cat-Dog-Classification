from pydantic import BaseModel


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool


class PredictionResponse(BaseModel):
    label: str
    probabilities: dict[str, float]
