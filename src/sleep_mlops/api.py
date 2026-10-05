from __future__ import annotations

from pathlib import Path
from typing import Literal

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.routing import APIRoute
from pydantic import BaseModel, Field
from sleep_mlops.prepare import add_engineered_features
from sleep_mlops.sleep_analysis_api import router as sleep_analysis_router
from sleep_mlops.sleep_records_api import router as sleep_records_router
from sleep_mlops.sleep_store import database_status

try:
    from prometheus_fastapi_instrumentator import Instrumentator
except ImportError:  # pragma: no cover
    Instrumentator = None


MODEL_PATH = Path("models/sleep_quality_model.joblib")
DATASET_PATH = Path("datasets/sleep_mobile_stress_dataset_15000.csv")
TRAIN_PATH = Path("data/processed/train.csv")
TEST_PATH = Path("data/processed/test.csv")
app = FastAPI(
    title="Sleep Pattern Analysis API",
    version="2.0.0",
    description="Predicts sleep quality using the updated timing-aware sleep dataset and feature pipeline.",
)
# Recreate the endpoints on the app so FastAPI and Prometheus share concrete routes.
for router in (sleep_records_router, sleep_analysis_router):
    for route in router.routes:
        if isinstance(route, APIRoute):
            app.add_api_route(
                route.path,
                route.endpoint,
                methods=route.methods,
                response_model=route.response_model,
                status_code=route.status_code,
                tags=route.tags,
                name=route.name,
            )


class SleepFeatures(BaseModel):
    age: int = Field(..., ge=13, le=100)
    gender: Literal["Female", "Male", "Other"]
    occupation: str = Field(..., min_length=2)
    daily_screen_time_hours: float = Field(..., ge=0, le=24)
    phone_usage_before_sleep_minutes: float = Field(..., ge=0, le=720)
    sleep_duration_hours: float = Field(..., ge=0, le=24)
    stress_level: float = Field(..., ge=1, le=10)
    caffeine_intake_cups: float = Field(..., ge=0, le=30)
    physical_activity_minutes: float = Field(..., ge=0, le=1440)
    notifications_received_per_day: float = Field(..., ge=0, le=5000)
    mental_fatigue_score: float = Field(..., ge=1, le=10)


class PredictionResponse(BaseModel):
    predicted_sleep_quality_score: float
    interpretation: str
    model_path: str


def get_model():
    if not MODEL_PATH.exists():
        raise HTTPException(
            status_code=503,
            detail="Model artifact is missing. Run data preparation and training first.",
        )
    return joblib.load(MODEL_PATH)


def interpretation(score: float) -> str:
    if score < 4:
        return "poor predicted sleep quality"
    if score < 7:
        return "moderate predicted sleep quality"
    return "good predicted sleep quality"


@app.get("/")
def root() -> dict:
    return {"service": "sleep-pattern-analysis", "docs": "/docs"}


@app.get("/health")
def health() -> dict:
    database_configured, database_available = database_status()
    return {
        "status": "ok",
        "model_available": MODEL_PATH.exists(),
        "updated_dataset_available": DATASET_PATH.exists(),
        "processed_train_available": TRAIN_PATH.exists(),
        "processed_test_available": TEST_PATH.exists(),
        "sleep_database_configured": database_configured,
        "sleep_database_available": database_available,
    }


@app.get("/metadata")
def metadata() -> dict:
    """Return the dataset/model artifacts deployed with this backend."""
    result = {
        "dataset": str(DATASET_PATH),
        "target": "sleep_quality_score",
        "model": str(MODEL_PATH),
        "feature_pipeline": "src/sleep_mlops/prepare.py",
        "engineered_schema": "41 columns in processed train/test datasets",
    }
    if DATASET_PATH.exists():
        result["raw_rows"] = int(pd.read_csv(DATASET_PATH, usecols=["user_id"]).shape[0])
    if TRAIN_PATH.exists():
        result["train_rows"] = int(pd.read_csv(TRAIN_PATH, usecols=["user_id"]).shape[0])
    if TEST_PATH.exists():
        result["test_rows"] = int(pd.read_csv(TEST_PATH, usecols=["user_id"]).shape[0])
    return result


@app.post("/predict", response_model=PredictionResponse)
def predict(payload: SleepFeatures) -> PredictionResponse:
    model = get_model()
    row = pd.DataFrame([{**payload.model_dump(), "user_id": "api_request"}])
    # Apply the same feature engineering as during training
    row = add_engineered_features(row)
    score = float(model.predict(row)[0])
    score = max(1.0, min(10.0, score))
    return PredictionResponse(
        predicted_sleep_quality_score=round(score, 3),
        interpretation=interpretation(score),
        model_path=str(MODEL_PATH),
    )


if Instrumentator is not None:
    Instrumentator().instrument(app).expose(app)
