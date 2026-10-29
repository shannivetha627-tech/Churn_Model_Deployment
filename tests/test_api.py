
import os

import httpx
import pytest

BASE_URL = "http://127.0.0.1:8000"
API_KEY = os.getenv("CHURN_API_KEY", "ChurnProjectLocal2026")
HEADERS = {"X-API-Key": API_KEY}
TIMEOUT = 30


@pytest.fixture
def valid_customer():
    return {
        "customer_id": "TEST001",
        "features": {
            "gender": "Female",
            "SeniorCitizen": 0,
            "Partner": "Yes",
            "Dependents": "No",
            "tenure": 12,
            "PhoneService": "Yes",
            "MultipleLines": "No",
            "InternetService": "DSL",
            "OnlineSecurity": "Yes",
            "OnlineBackup": "No",
            "DeviceProtection": "No",
            "TechSupport": "No",
            "StreamingTV": "No",
            "StreamingMovies": "No",
            "Contract": "Month-to-month",
            "PaperlessBilling": "Yes",
            "PaymentMethod": "Electronic check",
            "MonthlyCharges": 70.35,
            "TotalCharges": "844.2",
        },
    }


def test_health_endpoint():
    response = httpx.get(f"{BASE_URL}/health", timeout=10)

    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert data["model_loaded"] is True


def test_health_endpoint_has_json_content_type():
    response = httpx.get(f"{BASE_URL}/health", timeout=10)

    assert "application/json" in response.headers["content-type"]


def test_batch_prediction(valid_customer):
    response = httpx.post(
        f"{BASE_URL}/predict/batch",
        json={"customers": [valid_customer]},
        headers=HEADERS,
        timeout=TIMEOUT,
    )

    assert response.status_code == 200
    data = response.json()
    assert data["count"] == 1
    assert len(data["predictions"]) == 1

    prediction = data["predictions"][0]
    assert prediction["customer_id"] == "TEST001"
    assert 0 <= prediction["churn_probability"] <= 1
    assert prediction["risk_level"] in {"Low", "Medium", "High"}


def test_batch_prediction_includes_model_version(valid_customer):
    response = httpx.post(
        f"{BASE_URL}/predict/batch",
        json={"customers": [valid_customer]},
        headers=HEADERS,
        timeout=TIMEOUT,
    )

    assert response.status_code == 200
    assert response.json()["predictions"][0]["model_version"]


def test_batch_prediction_includes_latency(valid_customer):
    response = httpx.post(
        f"{BASE_URL}/predict/batch",
        json={"customers": [valid_customer]},
        headers=HEADERS,
        timeout=TIMEOUT,
    )

    assert response.status_code == 200
    data = response.json()
    prediction = data["predictions"][0]

    # The current API returns latency_ms for each prediction.
    assert prediction["latency_ms"] >= 0


def test_batch_prediction_preserves_customer_id(valid_customer):
    valid_customer["customer_id"] = "CUSTOMER_ABC"

    response = httpx.post(
        f"{BASE_URL}/predict/batch",
        json={"customers": [valid_customer]},
        headers=HEADERS,
        timeout=TIMEOUT,
    )

    assert response.status_code == 200
    assert (
        response.json()["predictions"][0]["customer_id"]
        == "CUSTOMER_ABC"
    )


def test_batch_prediction_multiple_customers(valid_customer):
    second_customer = {
        "customer_id": "TEST002",
        "features": {
            **valid_customer["features"],
            "tenure": 24,
            "MonthlyCharges": 55.0,
            "TotalCharges": "1320.0",
        },
    }

    response = httpx.post(
        f"{BASE_URL}/predict/batch",
        json={"customers": [valid_customer, second_customer]},
        headers=HEADERS,
        timeout=TIMEOUT,
    )

    assert response.status_code == 200
    data = response.json()
    assert data["count"] == 2
    assert len(data["predictions"]) == 2
    assert [p["customer_id"] for p in data["predictions"]] == [
        "TEST001",
        "TEST002",
    ]


def test_batch_prediction_rejects_invalid_api_key():
    response = httpx.post(
        f"{BASE_URL}/predict/batch",
        json={"customers": []},
        headers={"X-API-Key": "WrongKey123"},
        timeout=10,
    )

    assert response.status_code == 401
    assert response.json()["detail"] == "Missing or invalid API key."


def test_batch_prediction_rejects_missing_api_key(valid_customer):
    response = httpx.post(
        f"{BASE_URL}/predict/batch",
        json={"customers": [valid_customer]},
        timeout=10,
    )

    assert response.status_code == 401


def test_single_prediction(valid_customer):
    response = httpx.post(
        f"{BASE_URL}/predict",
        json=valid_customer,
        headers=HEADERS,
        timeout=TIMEOUT,
    )

    assert response.status_code == 200
    data = response.json()
    assert data["customer_id"] == "TEST001"
    assert 0 <= data["churn_probability"] <= 1


def test_single_prediction_rejects_invalid_api_key(valid_customer):
    response = httpx.post(
        f"{BASE_URL}/predict",
        json=valid_customer,
        headers={"X-API-Key": "WrongKey123"},
        timeout=10,
    )

    assert response.status_code == 401


def test_single_prediction_rejects_missing_api_key(valid_customer):
    response = httpx.post(
        f"{BASE_URL}/predict",
        json=valid_customer,
        timeout=10,
    )

    assert response.status_code == 401


def test_batch_prediction_rejects_missing_customers_field():
    response = httpx.post(
        f"{BASE_URL}/predict/batch",
        json={},
        headers=HEADERS,
        timeout=10,
    )

    assert response.status_code == 422


def test_batch_prediction_rejects_non_list_customers():
    response = httpx.post(
        f"{BASE_URL}/predict/batch",
        json={"customers": "not-a-list"},
        headers=HEADERS,
        timeout=10,
    )

    assert response.status_code == 422


def test_single_prediction_rejects_missing_features():
    response = httpx.post(
        f"{BASE_URL}/predict",
        json={"customer_id": "TEST003"},
        headers=HEADERS,
        timeout=10,
    )

    assert response.status_code == 422


def test_batch_prediction_rejects_invalid_customer_features():
    response = httpx.post(
        f"{BASE_URL}/predict/batch",
        json={
            "customers": [
                {
                    "customer_id": "TEST_BAD",
                    "features": {"tenure": -5},
                }
            ]
        },
        headers=HEADERS,
        timeout=10,
    )

    assert response.status_code == 422
