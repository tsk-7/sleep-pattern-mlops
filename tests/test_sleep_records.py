from __future__ import annotations

from datetime import date, timedelta

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from sleep_mlops.api import app
from sleep_mlops.sleep_store import Base, get_db


@pytest.fixture
def client():
    engine = create_engine(
        "sqlite://",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(engine)
    session_factory = sessionmaker(bind=engine, expire_on_commit=False)

    def override_get_db():
        with session_factory() as session:
            yield session

    app.dependency_overrides[get_db] = override_get_db
    with TestClient(app) as test_client:
        yield test_client
    app.dependency_overrides.pop(get_db, None)
    Base.metadata.drop_all(engine)
    engine.dispose()


def record_payload(sleep_date: date | None = None) -> dict[str, str]:
    return {
        "user_id": "local-user",
        "sleep_date": (sleep_date or date.today() - timedelta(days=1)).isoformat(),
        "bedtime": "23:30",
        "actual_sleep_time": "23:45",
        "wake_up_time": "07:00",
        "get_up_time": "07:15",
        "notes": "Test record",
    }


def test_create_record_calculates_overnight_metrics(client: TestClient):
    response = client.post("/api/sleep-records", json=record_payload())

    assert response.status_code == 201, response.text
    body = response.json()
    assert body["sleep_duration_minutes"] == 435
    assert body["time_in_bed_minutes"] == 465
    assert body["sleep_onset_latency_minutes"] == 15
    assert body["get_up_latency_minutes"] == 15
    assert body["sleep_efficiency_percent"] == 93.5
    assert body["actual_sleep_time"] == "23:45"


def test_duplicate_record_for_user_and_date_is_rejected(client: TestClient):
    payload = record_payload()
    created = client.post("/api/sleep-records", json=payload)
    assert created.status_code == 201, created.text

    response = client.post("/api/sleep-records", json=payload)
    assert response.status_code == 400
    assert "already exists" in response.json()["detail"]


def test_invalid_times_and_unreasonable_duration_are_rejected(client: TestClient):
    invalid = record_payload()
    invalid["actual_sleep_time"] = "not-a-time"
    invalid_response = client.post("/api/sleep-records", json=invalid)
    assert invalid_response.status_code == 400, invalid_response.text

    too_long = record_payload()
    too_long["actual_sleep_time"] = "08:00"
    too_long["wake_up_time"] = "08:00"
    too_long["get_up_time"] = "08:15"
    long_response = client.post("/api/sleep-records", json=too_long)
    assert long_response.status_code == 400, long_response.text


def test_read_update_and_delete_sleep_record(client: TestClient):
    created = client.post("/api/sleep-records", json=record_payload()).json()
    record_id = created["id"]

    assert client.get(f"/api/sleep-records/{record_id}").status_code == 200
    assert len(client.get("/api/sleep-records/user/local-user").json()) == 1

    updated = record_payload()
    updated["notes"] = "Updated"
    assert client.put(f"/api/sleep-records/{record_id}", json=updated).json()["notes"] == "Updated"
    assert client.delete(f"/api/sleep-records/{record_id}").status_code == 204
    assert client.get(f"/api/sleep-records/{record_id}").status_code == 404


def test_pattern_analysis_returns_backend_aggregates_and_empty_state(client: TestClient):
    payload = record_payload()
    client.post("/api/sleep-records", json=payload)

    analysis = client.get("/api/sleep-analysis/pattern/local-user").json()
    assert analysis["record_count"] == 1
    assert analysis["average_sleep_duration_minutes"] == 435
    assert analysis["consistency_label"] == "More sleep records are needed for this analysis."
    assert analysis["records"][0]["sleep_efficiency_percent"] == 93.5
    assert client.get(f"/api/sleep-analysis/daily/local-user/{payload['sleep_date']}").json()["record"]
    missing = client.get(f"/api/sleep-analysis/daily/local-user/{date.today().isoformat()}").json()
    assert missing["record"] is None
    assert missing["message"] == "No sleep record for this day."


def test_weekly_monthly_and_weekend_comparison_use_saved_records(client: TestClient):
    today = date.today()
    weekday = next(today - timedelta(days=offset) for offset in range(7) if (today - timedelta(days=offset)).weekday() < 5)
    weekend = next(today - timedelta(days=offset) for offset in range(7) if (today - timedelta(days=offset)).weekday() >= 5)
    workday_record = record_payload(weekday)
    workday_record.update(
        bedtime="23:50",
        actual_sleep_time="00:05",
        wake_up_time="07:05",
        get_up_time="07:15",
    )
    weekend_record = record_payload(weekend)
    weekend_record.update(
        bedtime="00:10",
        actual_sleep_time="00:20",
        wake_up_time="07:20",
        get_up_time="07:30",
    )
    assert client.post("/api/sleep-records", json=workday_record).status_code == 201
    assert client.post("/api/sleep-records", json=weekend_record).status_code == 201

    weekly = client.get("/api/sleep-analysis/weekly/local-user").json()
    monthly = client.get("/api/sleep-analysis/monthly/local-user").json()
    comparison = client.get("/api/sleep-analysis/weekend-comparison/local-user").json()
    pattern = client.get("/api/sleep-analysis/pattern/local-user").json()

    assert weekly["record_count"] == monthly["record_count"] == 2
    assert weekly["bedtime_variability_minutes"] == pytest.approx(10, abs=0.2)
    assert comparison["social_jetlag_minutes"] == pytest.approx(15, abs=0.2)
    assert comparison["weekday_weekend_bedtime_difference_minutes"] == pytest.approx(20, abs=0.2)
    assert "Regular Sleeper" in pattern["pattern_labels"]


def test_out_of_order_sleep_events_are_rejected(client: TestClient):
    payload = record_payload()
    payload.update(
        bedtime="23:00",
        actual_sleep_time="22:00",
        wake_up_time="07:00",
        get_up_time="07:15",
    )

    response = client.post("/api/sleep-records", json=payload)
    assert response.status_code == 400
    assert "within 12 hours" in response.json()["detail"]
