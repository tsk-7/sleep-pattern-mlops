# Sleep Pattern Analysis System — Updated Dataset Package

This folder contains the reproducible data artifacts for the Sleep Pattern Analysis MLOps project. The original 15,000 observational records were retained; no target values were replaced and no target-derived predictors were introduced.

| File | Rows | Columns | Role |
|---|---:|---:|---|
| `sleep_mobile_stress_dataset_15000.csv` | 15,000 | 19 | Updated raw layer: the original 13 columns plus date, bedtime, wake time, last caffeine time, day of week, and weekend flag. |
| `sleep_dataset_train_engineered.csv` | 12,000 | 41 | Training layer with the raw columns, nine original engineered features, and the new timing/circadian features. |
| `sleep_dataset_test_engineered.csv` | 3,000 | 41 | Held-out testing layer with the same 41-column schema. |

## New feature definitions

The raw timing fields are generated deterministically from the existing user profile and a stable hash of `user_id`. Bedtime is linked to screen time, stress, fatigue, occupation, and bounded user-specific variation. Wake time is calculated as bedtime plus the observed `sleep_duration_hours`, modulo 24 hours. The date determines `day_of_week`, and `is_weekend` is 1 only for Saturday or Sunday.

For a clock value represented as minutes after midnight, `angle = 2π × minutes / 1440`. The circular features are `sin(angle)` and `cos(angle)` for bedtime and wake time. The midpoint is calculated using the overnight span `(wake − bedtime) mod 1440`: `mid_sleep_minutes = (bedtime + span / 2) mod 1440`.

`caffeine_clearance_window = ((bedtime_minutes − last_caffeine_minutes) mod 1440) / 60`.

`work_schedule_mismatch` is the circular absolute distance, in minutes, between mid-sleep and an occupation schedule midpoint. Doctors and freelancers use a flexible/shift midpoint of 04:00; students use 05:00; all other occupations use a daytime midpoint of 03:30. This is a transparent research heuristic, not a clinical schedule label.

`late_night_digital_share` uses the documented proxy `min(phone_usage_before_sleep_minutes, 180) / (daily_screen_time_hours × 60 + 0.00001)`, clipped to [0, 1], because event-level screen timestamps are not present.

`bedtime_variability` and `wake_time_variability` are circular rolling standard deviations over the previous seven chronological records for the same `user_id`. The supplied data contains one record per user, so the valid within-user estimate is 0. The same implementation supports repeated `user_id + date` records in future tracking exports. `social_jetlag_score` is the circular absolute difference between a user's mean weekday and mean weekend mid-sleep times; it is 0 when one side is unavailable.

`circadian_disruption_index = pre_sleep_disruption_ratio × bedtime_variability` exactly as specified.

The original nine engineered features remain unchanged: screen-to-sleep ratio, stress-fatigue index, digital intensity, activity-screen balance, age-stress impact, notification frequency, caffeine-activity ratio, pre-sleep disruption ratio, and lifestyle balance score. The target remains the observed `sleep_quality_score` and is never used to construct a predictor.

## Split and validation

The preparation pipeline uses an 80/20 split with `random_state=42`. Because the supplied file has one record per user, the random row split has no user overlap: 12,000 training users and 3,000 testing users. For a future longitudinal export, a user-level or time-based split should replace this row split to prevent historical/future leakage.

The generated validation report is stored at `reports/updated_dataset_validation.json`. The current run found zero missing cells, zero duplicate rows, no impossible sleep durations, no invalid weekend flags, no circular-feature mismatches, no negative caffeine-clearance windows, no invalid social-jetlag values, no out-of-range digital-share values, no target-leakage columns, and no train/test row or user overlap. Bedtime/wake-time spans match the stated duration within the one-minute rounding tolerance.
