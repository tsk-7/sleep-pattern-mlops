from pathlib import Path
import json
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / 'data/raw/sleep_mobile_stress_dataset_15000.csv'
TRAIN = ROOT / 'data/processed/train.csv'
TEST = ROOT / 'data/processed/test.csv'
REPORT = ROOT / 'reports/updated_dataset_validation.json'

required = {
    'user_id','age','gender','occupation','daily_screen_time_hours','phone_usage_before_sleep_minutes',
    'sleep_duration_hours','sleep_quality_score','stress_level','caffeine_intake_cups',
    'physical_activity_minutes','notifications_received_per_day','mental_fatigue_score',
    'date','bedtime','wake_time','last_caffeine_time','day_of_week','is_weekend',
    'bedtime_circular_sin','bedtime_circular_cos','wake_time_circular_sin','wake_time_circular_cos',
    'mid_sleep_time','mid_sleep_minutes','bedtime_variability','wake_time_variability',
    'social_jetlag_score','caffeine_clearance_window','work_schedule_mismatch',
    'late_night_digital_share','circadian_disruption_index','screen_to_sleep_ratio',
    'stress_fatigue_index','digital_intensity','activity_screen_balance','age_stress_impact',
    'notification_frequency','caffeine_activity_ratio','pre_sleep_disruption_ratio','lifestyle_balance_score'
}
raw_required = {
    'user_id','age','gender','occupation','daily_screen_time_hours','phone_usage_before_sleep_minutes',
    'sleep_duration_hours','sleep_quality_score','stress_level','caffeine_intake_cups',
    'physical_activity_minutes','notifications_received_per_day','mental_fatigue_score',
    'date','bedtime','wake_time','last_caffeine_time','day_of_week','is_weekend'
}

def mins(value):
    h, m = str(value).split(':')[:2]
    return int(h)*60+int(m)

def validate(df):
    if 'bedtime_circular_sin' not in df.columns:
        return {'rows': int(len(df)), 'columns': int(len(df.columns)), 'missing_cells': int(df.isna().sum().sum()),
                'duplicate_rows': int(df.duplicated().sum()), 'required_columns_missing': sorted(raw_required-set(df.columns))}, {}
    bed = df['bedtime'].map(mins)
    wake = df['wake_time'].map(mins)
    span = (wake-bed) % 1440
    expected = df['sleep_duration_hours']*60
    angle = 2*np.pi*bed/1440
    checks = {
        'rows': int(len(df)), 'columns': int(len(df.columns)),
        'missing_cells': int(df.isna().sum().sum()),
        'duplicate_rows': int(df.duplicated().sum()),
        'required_columns_missing': sorted(required-set(df.columns)),
        'sleep_duration_alignment_max_minutes': float(np.max(np.abs(span-expected))),
        'invalid_sleep_durations': int(((span < 60) | (span > 960)).sum()),
        'invalid_is_weekend': int((df['is_weekend'] != (pd.to_datetime(df['date']).dt.dayofweek >= 5).astype(int)).sum()),
        'invalid_circular_features': int((np.abs(df['bedtime_circular_sin']-np.sin(angle)) > 1e-8).sum() + (np.abs(df['bedtime_circular_cos']-np.cos(angle)) > 1e-8).sum()),
        'negative_caffeine_clearance': int((df['caffeine_clearance_window'] < 0).sum()),
        'invalid_social_jetlag': int(((df['social_jetlag_score'] < 0) | (df['social_jetlag_score'] > 720)).sum()),
        'invalid_late_night_share': int(((df['late_night_digital_share'] < 0) | (df['late_night_digital_share'] > 1)).sum()),
        'target_leakage_columns': sorted(set(df.columns) & {'target','target_encoded'}),
        'unique_users': int(df['user_id'].nunique()),
        'longitudinal_records_per_user_max': int(df.groupby('user_id').size().max()),
    }
    num = df.select_dtypes(include='number')
    corr = num.corr(numeric_only=True)['sleep_quality_score'].sort_values(ascending=False).to_dict()
    return checks, {k: float(v) if pd.notna(v) else None for k,v in corr.items()}

raw = pd.read_csv(RAW)
train = pd.read_csv(TRAIN)
test = pd.read_csv(TEST)
raw_checks, _ = validate(raw)
train_checks, corr = validate(train)
test_checks, _ = validate(test)
report = {'raw': raw_checks, 'train': train_checks, 'test': test_checks, 'target_correlations': corr,
          'train_test_user_overlap': int(len(set(train.user_id) & set(test.user_id))),
          'train_test_row_overlap': int(len(pd.merge(train, test, how='inner')))}
REPORT.parent.mkdir(exist_ok=True)
REPORT.write_text(json.dumps(report, indent=2), encoding='utf-8')
print(json.dumps(report, indent=2))
