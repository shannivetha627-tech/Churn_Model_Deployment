
"""Train and evaluate the telecom customer churn prediction model."""

import hashlib
import json
import platform
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from src.preprocessing import clean_frame
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "raw" / "telco_churn.csv"
ARTIFACT_DIR = ROOT / "artifacts"
ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

RANDOM_STATE = 42

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
def build_pipeline():
    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore")),
    ])

    preprocessing = ColumnTransformer([
        ("numeric", numeric_pipeline, NUMERIC_FEATURES),
        ("categorical", categorical_pipeline, CATEGORICAL_FEATURES),
    ])

    return Pipeline([
        ("clean", FunctionTransformer(clean_frame, validate=False)),
        ("preprocess", preprocessing),
        ("classifier", LogisticRegression(max_iter=2000)),
    ])


def main():
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Dataset not found: {DATA_PATH}")

    dataset_hash = hashlib.sha256(DATA_PATH.read_bytes()).hexdigest()
    data = pd.read_csv(DATA_PATH)

    required_columns = set(FEATURES + ["Churn"])
    missing_columns = required_columns - set(data.columns)

    if missing_columns:
        raise ValueError(
            f"Dataset is missing required columns: {sorted(missing_columns)}"
        )

    data = data.dropna(subset=["Churn"]).copy()
    data["Churn"] = data["Churn"].map({"No": 0, "Yes": 1})

    X = data[FEATURES].copy()
    y = data["Churn"].astype(int)

    # Reserve an untouched test set for final evaluation.
    X_development, X_test, y_development, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        stratify=y,
        random_state=RANDOM_STATE,
    )

    X_train, X_validation, y_train, y_validation = train_test_split(
        X_development,
        y_development,
        test_size=0.25,
        stratify=y_development,
        random_state=RANDOM_STATE,
    )

    pipeline = build_pipeline()

    search = GridSearchCV(
        estimator=pipeline,
        param_grid=[
            {
                "classifier": [LogisticRegression(max_iter=2000)],
                "classifier__C": [0.1, 1.0, 10.0],
                "classifier__class_weight": [None, "balanced"],
            },
            {
                "classifier": [
                    RandomForestClassifier(
                        random_state=RANDOM_STATE,
                        n_jobs=1,
                    )
                ],
                "classifier__n_estimators": [100, 200],
                "classifier__max_depth": [None, 12],
                "classifier__class_weight": [None, "balanced"],
            },
        ],
        scoring="roc_auc",
        cv=5,
        n_jobs=-1,
        refit=True,
        error_score="raise",
    )

    print("Training and tuning the model. Please wait...")
    search.fit(X_train, y_train)
    best_pipeline = search.best_estimator_

    # Business assumption: missing a churner costs 5 units;
    # sending an unnecessary retention offer costs 1 unit.
    validation_probabilities = best_pipeline.predict_proba(
        X_validation
    )[:, 1]

    thresholds = np.arange(0.05, 0.951, 0.01)
    threshold_costs = []

    for threshold in thresholds:
        predictions = (validation_probabilities >= threshold).astype(int)

        tn, fp, fn, tp = confusion_matrix(
            y_validation,
            predictions,
            labels=[0, 1],
        ).ravel()

        threshold_costs.append((5 * fn) + fp)

    best_index = int(np.argmin(threshold_costs))
    best_threshold = float(thresholds[best_index])

    # Final evaluation happens only on the untouched test set.
    test_probabilities = best_pipeline.predict_proba(X_test)[:, 1]
    test_predictions = (test_probabilities >= best_threshold).astype(int)

    tn, fp, fn, tp = confusion_matrix(
        y_test,
        test_predictions,
        labels=[0, 1],
    ).ravel()

    metrics = {
        "roc_auc": float(roc_auc_score(y_test, test_probabilities)),
        "accuracy": float(accuracy_score(y_test, test_predictions)),
        "precision": float(
            precision_score(y_test, test_predictions, zero_division=0)
        ),
        "recall": float(
            recall_score(y_test, test_predictions, zero_division=0)
        ),
        "f1": float(f1_score(y_test, test_predictions, zero_division=0)),
        "threshold": best_threshold,
        "test_confusion_matrix": {
            "tn": int(tn),
            "fp": int(fp),
            "fn": int(fn),
            "tp": int(tp),
        },
        "validation_cost_at_threshold": int(threshold_costs[best_index]),
        "missed_churner_cost": 5,
        "unnecessary_offer_cost": 1,
        "best_cv_roc_auc": float(search.best_score_),
        "best_parameters": {
            key: str(value) for key, value in search.best_params_.items()
        },
        "classification_report": classification_report(
            y_test,
            test_predictions,
            target_names=["No churn", "Churn"],
            output_dict=True,
            zero_division=0,
        ),
    }

    version = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")

    artifact_path = ARTIFACT_DIR / "churn_pipeline.joblib"
    metadata_path = ARTIFACT_DIR / "model_metadata.json"

    bundle = {
        "pipeline": best_pipeline,
        "threshold": best_threshold,
        "model_version": version,
        "feature_names": FEATURES,
    }

    joblib.dump(bundle, artifact_path)

    metadata = {
        "model_name": "telecom-customer-churn",
        "model_version": version,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "dataset": DATA_PATH.name,
        "dataset_sha256": dataset_hash,
        "dataset_rows": int(len(data)),
        "feature_names": FEATURES,
        "target": "Churn",
        "metrics": metrics,
        "libraries": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scikit_learn": sklearn.__version__,
            "joblib": joblib.__version__,
        },
        "random_state": RANDOM_STATE,
    }

    metadata_path.write_text(
        json.dumps(metadata, indent=2),
        encoding="utf-8",
    )

    print("\nTRAINING COMPLETED")
    print("Model:", artifact_path)
    print("Metadata:", metadata_path)
    print("Dataset SHA-256:", dataset_hash)
    print("Selected threshold:", best_threshold)
    print("Best CV ROC-AUC:", round(search.best_score_, 4))

    print("\nTest metrics:")
    for key in ["roc_auc", "accuracy", "precision", "recall", "f1"]:
        print(f"{key}: {metrics[key]:.4f}")

    print("\nTest confusion matrix:", metrics["test_confusion_matrix"])
    print("\nBest parameters:", search.best_params_)


if __name__ == "__main__":
    main()
