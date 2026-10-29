
"""FastAPI service for the customer churn prediction model."""

import logging
import os
import secrets
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Literal

import joblib
import pandas as pd
from fastapi import Depends, FastAPI, Header, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, Field, model_validator

from src.drift_monitor import generate_drift_report


# --------------------------------------------------
# CONFIGURATION
# --------------------------------------------------

ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "artifacts" / "churn_pipeline.joblib"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)

logger = logging.getLogger("churn-api")

NUMERIC_FEATURES = [
    "SeniorCitizen",
    "tenure",
    "MonthlyCharges",
    "TotalCharges",
]

CATEGORICAL_FEATURES = [
    "gender",
    "Partner",
    "Dependents",
    "PhoneService",
    "MultipleLines",
    "InternetService",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
    "Contract",
    "PaperlessBilling",
    "PaymentMethod",
]

FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES

# Configure CHURN_API_KEY in the deployment environment.
# The fallback is for local development only.
API_KEY = os.getenv(
    "CHURN_API_KEY",
    "local-dev-key-change-me",
)

model_bundle = None


# --------------------------------------------------
# MODEL LIFECYCLE
# --------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load the trained model when the API starts."""
    global model_bundle

    if not API_KEY or API_KEY == "local-dev-key-change-me":
        logger.warning(
            "Using the development API key fallback. "
            "Configure CHURN_API_KEY before deployment."
        )

    if not MODEL_PATH.exists():
        raise RuntimeError(
            f"Model file not found: {MODEL_PATH}. "
            "Run src/train.py first."
        )

    try:
        loaded_bundle = joblib.load(MODEL_PATH)

        if (
            not isinstance(loaded_bundle, dict)
            or "pipeline" not in loaded_bundle
            or "threshold" not in loaded_bundle
            or "model_version" not in loaded_bundle
        ):
            raise RuntimeError(
                "Model artifact is missing required fields."
            )

        model_bundle = loaded_bundle

        logger.info(
            "Model loaded successfully: version=%s",
            model_bundle["model_version"],
        )

    except Exception:
        logger.exception("Failed to load the model artifact.")
        raise

    yield

    model_bundle = None
    logger.info("Model unloaded.")


# --------------------------------------------------
# FASTAPI APPLICATION
# --------------------------------------------------

app = FastAPI(
    title="Customer Churn Prediction API",
    description=(
        "Predict customer churn risk using a trained machine-learning "
        "pipeline and monitor feature drift using Population Stability "
        "Index (PSI). Prediction and drift-monitoring endpoints "
        "require an API key."
    ),
    version="1.1.0",
    lifespan=lifespan,
)

# CORS: allow the local React/Vite development frontend.
# Add your exact production frontend origin when deploying it.
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
    ],
    allow_credentials=False,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["Content-Type", "X-API-Key"],
)


# --------------------------------------------------
# REQUEST MODELS
# --------------------------------------------------

class CustomerFeatures(BaseModel):
    """Validated customer features used by the ML pipeline."""

    model_config = ConfigDict(extra="forbid")

    SeniorCitizen: Literal[0, 1]
    tenure: int = Field(ge=0, le=100)
    MonthlyCharges: float = Field(ge=0, le=1000)
    TotalCharges: float = Field(ge=0, le=100000)

    gender: str = Field(min_length=1, max_length=30)
    Partner: str = Field(min_length=1, max_length=30)
    Dependents: str = Field(min_length=1, max_length=30)
    PhoneService: str = Field(min_length=1, max_length=30)
    MultipleLines: str = Field(min_length=1, max_length=50)
    InternetService: str = Field(min_length=1, max_length=50)
    OnlineSecurity: str = Field(min_length=1, max_length=50)
    OnlineBackup: str = Field(min_length=1, max_length=50)
    DeviceProtection: str = Field(min_length=1, max_length=50)
    TechSupport: str = Field(min_length=1, max_length=50)
    StreamingTV: str = Field(min_length=1, max_length=50)
    StreamingMovies: str = Field(min_length=1, max_length=50)
    Contract: str = Field(min_length=1, max_length=50)
    PaperlessBilling: str = Field(min_length=1, max_length=30)
    PaymentMethod: str = Field(min_length=1, max_length=100)

    @model_validator(mode="after")
    def check_tenure_charges(self):
        if self.tenure == 0 and self.TotalCharges > 0:
            raise ValueError(
                "TotalCharges should be zero when tenure is zero."
            )
        return self


class PredictionRequest(BaseModel):
    """Request schema for a single customer prediction."""

    model_config = ConfigDict(extra="forbid")

    customer_id: str = Field(min_length=1, max_length=100)
    features: CustomerFeatures


class BatchPredictionRequest(BaseModel):
    """Request schema for batch predictions."""

    model_config = ConfigDict(extra="forbid")

    customers: list[PredictionRequest] = Field(
        min_length=1,
        max_length=100,
    )


class DriftRequest(BaseModel):
    """Request schema for PSI feature-drift monitoring."""

    model_config = ConfigDict(extra="forbid")

    customers: list[CustomerFeatures] = Field(
        min_length=1,
        max_length=1000,
    )


# --------------------------------------------------
# API KEY AUTHENTICATION
# --------------------------------------------------

def require_api_key(
    x_api_key: str | None = Header(
        default=None,
        alias="X-API-Key",
    ),
) -> None:
    """Reject missing or incorrect API keys."""

    if not x_api_key or not secrets.compare_digest(
        x_api_key,
        API_KEY,
    ):
        logger.warning(
            "Rejected request with missing or invalid API key."
        )

        raise HTTPException(
            status_code=401,
            detail="Missing or invalid API key.",
            headers={"WWW-Authenticate": "ApiKey"},
        )


# --------------------------------------------------
# SINGLE CUSTOMER PREDICTION
# --------------------------------------------------

def predict_one(request_data: PredictionRequest) -> dict:
    """Generate a churn prediction for one customer."""

    if model_bundle is None:
        raise HTTPException(
            status_code=503,
            detail="Model is not loaded.",
        )

    row = request_data.features.model_dump()
    frame = pd.DataFrame([row], columns=FEATURES)

    start = time.perf_counter()

    try:
        probabilities = model_bundle["pipeline"].predict_proba(
            frame
        )[0]

        churn_probability = float(probabilities[1])
        threshold = float(model_bundle["threshold"])

    except Exception as exc:
        logger.exception(
            "Prediction failed for customer_id=%s",
            request_data.customer_id,
        )

        raise HTTPException(
            status_code=500,
            detail="Prediction failed. Check the server logs.",
        ) from exc

    latency_ms = (time.perf_counter() - start) * 1000
    predicted_churn = churn_probability >= threshold

    risk_level = (
        "High"
        if churn_probability >= 0.60
        else "Medium"
        if churn_probability >= threshold
        else "Low"
    )

    result = {
        "customer_id": request_data.customer_id,
        "predicted_churn": bool(predicted_churn),
        "churn_probability": round(churn_probability, 4),
        "threshold": round(threshold, 4),
        "risk_level": risk_level,
        "model_version": model_bundle["model_version"],
        "latency_ms": round(latency_ms, 3),
    }

    logger.info(
        "prediction_completed customer_id=%s "
        "latency_ms=%.3f churn=%s",
        request_data.customer_id,
        latency_ms,
        predicted_churn,
    )

    return result


# --------------------------------------------------
# HEALTH CHECK
# --------------------------------------------------

@app.get("/health")
def health_check():
    """Public health endpoint for monitoring."""

    loaded = model_bundle is not None

    return JSONResponse(
        status_code=200 if loaded else 503,
        content={
            "status": "ok" if loaded else "not_ready",
            "model_loaded": loaded,
        },
    )


# --------------------------------------------------
# SINGLE PREDICTION ENDPOINT
# --------------------------------------------------

@app.post(
    "/predict",
    dependencies=[Depends(require_api_key)],
)
def predict(request_data: PredictionRequest):
    """Predict churn for a single customer."""

    return predict_one(request_data)


# --------------------------------------------------
# BATCH PREDICTION ENDPOINT
# --------------------------------------------------

@app.post(
    "/predict/batch",
    dependencies=[Depends(require_api_key)],
)
def predict_batch(request_data: BatchPredictionRequest):
    """Predict churn for a batch of 1–100 customers."""

    start = time.perf_counter()

    results = [
        predict_one(customer)
        for customer in request_data.customers
    ]

    total_latency_ms = (time.perf_counter() - start) * 1000

    logger.info(
        "batch_prediction_completed count=%s "
        "total_latency_ms=%.3f",
        len(results),
        total_latency_ms,
    )

    return {
        "count": len(results),
        "predictions": results,
        "total_latency_ms": round(total_latency_ms, 3),
    }


# --------------------------------------------------
# PSI DRIFT MONITORING ENDPOINT
# --------------------------------------------------

@app.post(
    "/monitor/drift",
    dependencies=[Depends(require_api_key)],
)
def monitor_drift(request_data: DriftRequest):
    """
    Compare incoming customer features with the reference dataset.

    Returns feature-level PSI scores and an overall drift assessment.
    """

    try:
        customers = [
            customer.model_dump()
            for customer in request_data.customers
        ]

        report = generate_drift_report(customers)

        logger.info(
            "drift_monitor_completed rows=%s overall_level=%s",
            report["current_rows"],
            report["overall_drift_level"],
        )

        return report

    except ValueError as exc:
        logger.warning(
            "Invalid drift-monitoring request: %s",
            exc,
        )

        raise HTTPException(
            status_code=422,
            detail=str(exc),
        ) from exc

    except Exception as exc:
        logger.exception("Drift monitoring failed.")

        raise HTTPException(
            status_code=500,
            detail="Drift monitoring failed. Check the server logs.",
        ) from exc
