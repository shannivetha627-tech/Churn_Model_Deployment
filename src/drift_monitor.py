
"""Population Stability Index (PSI) monitoring for customer churn features."""

from pathlib import Path

import numpy as np
import pandas as pd

from src.preprocessing import clean_frame

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "raw" / "telco_churn.csv"

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

PSI_LOW = 0.10
PSI_HIGH = 0.20
EPSILON = 1e-6


def calculate_psi(reference, current, numeric=False, bins=10):
    """Calculate PSI between reference and current feature distributions."""
    reference = pd.Series(reference).dropna()
    current = pd.Series(current).dropna()

    if reference.empty or current.empty:
        raise ValueError("Reference and current data must not be empty.")

    if numeric:
        reference_numeric = pd.to_numeric(reference, errors="coerce").dropna()
        current_numeric = pd.to_numeric(current, errors="coerce").dropna()

        if reference_numeric.empty or current_numeric.empty:
            raise ValueError("Numeric feature has no valid numeric values.")

        # Use reference quantiles as stable comparison boundaries.
        boundaries = np.unique(
            np.quantile(reference_numeric, np.linspace(0, 1, bins + 1))
        )

        if len(boundaries) < 2:
            categories = ["all"]
            reference_counts = [len(reference_numeric)]
            current_counts = [len(current_numeric)]
        else:
            boundaries[0] = -np.inf
            boundaries[-1] = np.inf

            reference_counts = pd.cut(
                reference_numeric,
                bins=boundaries,
                include_lowest=True,
            ).value_counts(sort=False).to_numpy()

            current_counts = pd.cut(
                current_numeric,
                bins=boundaries,
                include_lowest=True,
            ).value_counts(sort=False).to_numpy()
    else:
        reference_values = reference.astype(str)
        current_values = current.astype(str)

        categories = sorted(
            set(reference_values.unique()) | set(current_values.unique())
        )

        reference_counts = (
            reference_values.value_counts()
            .reindex(categories, fill_value=0)
            .to_numpy()
        )
        current_counts = (
            current_values.value_counts()
            .reindex(categories, fill_value=0)
            .to_numpy()
        )

    reference_pct = np.asarray(reference_counts, dtype=float)
    current_pct = np.asarray(current_counts, dtype=float)

    reference_pct /= reference_pct.sum()
    current_pct /= current_pct.sum()

    # Smoothing prevents log(0) for previously unseen categories.
    reference_pct = np.clip(reference_pct, EPSILON, None)
    current_pct = np.clip(current_pct, EPSILON, None)

    reference_pct /= reference_pct.sum()
    current_pct /= current_pct.sum()

    psi = np.sum(
        (current_pct - reference_pct)
        * np.log(current_pct / reference_pct)
    )

    return float(psi)


def assess_drift(psi):
    """Assign a monitoring label to a PSI value."""
    if psi < PSI_LOW:
        return "Low"
    if psi <= PSI_HIGH:
        return "Moderate"
    return "High"


def load_reference_data():
    """Load and validate the original customer feature dataset."""
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Reference dataset not found: {DATA_PATH}"
        )

    reference = pd.read_csv(DATA_PATH)
    reference = clean_frame(reference)

    missing = sorted(set(FEATURES) - set(reference.columns))
    if missing:
        raise ValueError(
            f"Reference dataset is missing features: {missing}"
        )

    return reference[FEATURES].copy()


def generate_drift_report(current_data):
    """Compare incoming customer rows with the training reference data."""
    if not current_data:
        raise ValueError("Provide at least one customer record.")

    current = pd.DataFrame(current_data)

    missing = sorted(set(FEATURES) - set(current.columns))
    if missing:
        raise ValueError(
            f"Incoming data is missing features: {missing}"
        )

    reference = load_reference_data()
    current = clean_frame(current[FEATURES].copy())

    feature_results = {}

    for feature in FEATURES:
        numeric = feature in NUMERIC_FEATURES

        psi = calculate_psi(
            reference[feature],
            current[feature],
            numeric=numeric,
        )

        feature_results[feature] = {
            "psi": round(psi, 6),
            "drift_level": assess_drift(psi),
        }

    high_count = sum(
        result["drift_level"] == "High"
        for result in feature_results.values()
    )
    moderate_count = sum(
        result["drift_level"] == "Moderate"
        for result in feature_results.values()
    )

    overall_level = (
        "High"
        if high_count > 0
        else "Moderate"
        if moderate_count > 0
        else "Low"
    )

    return {
        "reference_dataset": DATA_PATH.name,
        "reference_rows": len(reference),
        "current_rows": len(current),
        "overall_drift_level": overall_level,
        "thresholds": {
            "low_max": PSI_LOW,
            "moderate_max": PSI_HIGH,
            "high_min_exclusive": PSI_HIGH,
        },
        "features_with_high_drift": high_count,
        "features_with_moderate_drift": moderate_count,
        "feature_drift": feature_results,
    }
