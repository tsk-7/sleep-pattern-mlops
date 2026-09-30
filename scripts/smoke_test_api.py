from fastapi.testclient import TestClient
from sleep_mlops.api import app

payload = {
    'age': 30, 'gender': 'Female', 'occupation': 'Teacher',
    'daily_screen_time_hours': 4.5, 'phone_usage_before_sleep_minutes': 45,
    'sleep_duration_hours': 7.2, 'stress_level': 4.0, 'caffeine_intake_cups': 2,
    'physical_activity_minutes': 45, 'notifications_received_per_day': 120,
    'mental_fatigue_score': 4.0,
}
response = TestClient(app).post('/predict', json=payload)
print(response.status_code)
print(response.json())
assert response.status_code == 200
assert 1 <= response.json()['predicted_sleep_quality_score'] <= 10
