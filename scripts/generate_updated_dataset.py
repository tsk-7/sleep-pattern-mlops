from pathlib import Path
import pandas as pd
from sleep_mlops.prepare import add_sleep_timing_features, REQUIRED_COLUMNS

root = Path(__file__).resolve().parents[1]
source = root / 'datasets/sleep_mobile_stress_dataset_15000.csv'
out = root / 'datasets/sleep_mobile_stress_dataset_15000.csv'
legacy_out = root / 'data/raw/sleep_mobile_stress_dataset_15000.csv'
df = pd.read_csv(source)
updated = add_sleep_timing_features(df)
raw_columns = list(REQUIRED_COLUMNS) + ['date','bedtime','wake_time','last_caffeine_time','day_of_week','is_weekend']
# Preserve the documented source-column order, followed by the six new raw timing columns.
raw_columns = [c for c in [
    'user_id','age','gender','occupation','daily_screen_time_hours',
    'phone_usage_before_sleep_minutes','sleep_duration_hours','sleep_quality_score',
    'stress_level','caffeine_intake_cups','physical_activity_minutes',
    'notifications_received_per_day','mental_fatigue_score','date','bedtime',
    'wake_time','last_caffeine_time','day_of_week','is_weekend'] if c in updated]
out.parent.mkdir(parents=True, exist_ok=True)
if source.resolve() != out.resolve():
    updated[raw_columns].to_csv(out, index=False)
legacy_out.parent.mkdir(parents=True, exist_ok=True)
updated[raw_columns].to_csv(legacy_out, index=False)
print(f'Prepared {out} with shape {updated[raw_columns].shape}')
