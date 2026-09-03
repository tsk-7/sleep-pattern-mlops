from __future__ import annotations

import os
from pathlib import Path
from typing import Literal

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

try:
    from prometheus_fastapi_instrumentator import Instrumentator
except ImportError:  # pragma: no cover
    Instrumentator = None


MODEL_PATH = Path("models/sleep_quality_model.joblib")
app = FastAPI(
    title="Sleep Pattern Analysis API",
    version="1.0.0",
    description="Predicts sleep quality from mobile-use, lifestyle, and demographic features.",
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        origin.strip()
        for origin in os.getenv(
            "CORS_ORIGINS",
            "http://localhost:5173,http://127.0.0.1:5173",
        ).split(",")
        if origin.strip()
    ],
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type"],
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
    return {"status": "ok", "model_available": MODEL_PATH.exists()}


@app.post("/predict", response_model=PredictionResponse)
def predict(payload: SleepFeatures) -> PredictionResponse:
    model = get_model()
    row = pd.DataFrame([payload.model_dump()])
    score = float(model.predict(row)[0])
    score = max(1.0, min(10.0, score))
    return PredictionResponse(
        predicted_sleep_quality_score=round(score, 3),
        interpretation=interpretation(score),
        model_path=str(MODEL_PATH),
    )


if Instrumentator is not None:
    Instrumentator().instrument(app).expose(app)
