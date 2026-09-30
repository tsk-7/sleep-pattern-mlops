from fastapi.testclient import TestClient

from sleep_mlops.api import app


client = TestClient(app)


def test_root():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json()["service"] == "sleep-pattern-analysis"


def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_metadata_reports_updated_dataset():
    response = client.get("/metadata")
    assert response.status_code == 200
    body = response.json()
    assert body["target"] == "sleep_quality_score"
    assert body["raw_rows"] == 15000
    assert body["train_rows"] == 12000
    assert body["test_rows"] == 3000


def test_validation_rejects_invalid_screen_time():
    payload = {
        "age": 30,
        "gender": "Female",
        "occupation": "Teacher",
        "daily_screen_time_hours": 30,
        "phone_usage_before_sleep_minutes": 30,
        "sleep_duration_hours": 7,
        "stress_level": 5,
        "caffeine_intake_cups": 2,
        "physical_activity_minutes": 40,
        "notifications_received_per_day": 100,
        "mental_fatigue_score": 5,
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 422
