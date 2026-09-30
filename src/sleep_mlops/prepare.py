from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
import yaml


REQUIRED_COLUMNS = {
    "user_id", "age", "gender", "occupation", "daily_screen_time_hours",
    "phone_usage_before_sleep_minutes", "sleep_duration_hours", "sleep_quality_score",
    "stress_level", "caffeine_intake_cups", "physical_activity_minutes",
    "notifications_received_per_day", "mental_fatigue_score",
}
TAU = 2 * np.pi
NEW_TIMING_COLUMNS = [
    "date", "bedtime", "wake_time", "last_caffeine_time", "day_of_week", "is_weekend",
    "bedtime_circular_sin", "bedtime_circular_cos", "wake_time_circular_sin",
    "wake_time_circular_cos", "mid_sleep_time", "mid_sleep_minutes", "bedtime_variability",
    "wake_time_variability", "social_jetlag_score", "caffeine_clearance_window",
    "work_schedule_mismatch", "late_night_digital_share", "circadian_disruption_index",
]


def load_config(path: str = "configs/config.yaml") -> dict:
    with open(path, "r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def _stable_int(value: object) -> int:
    return int(hashlib.sha256(str(value).encode("utf-8")).hexdigest()[:12], 16)


def _clock(minutes: float) -> str:
    minutes = int(round(minutes)) % 1440
    return f"{minutes // 60:02d}:{minutes % 60:02d}"


def _clock_minutes(value: str | float) -> float:
    if isinstance(value, (float, int, np.number)):
        return float(value) % 1440
    hour, minute = str(value).split(":")[:2]
    return (int(hour) * 60 + int(minute)) % 1440


def circular_distance(a: float, b: float) -> float:
    return abs((a - b + 720) % 1440 - 720)


def _circular_rolling_std(values: pd.Series, window: int = 7) -> pd.Series:
    """Rolling circular SD in minutes, grouped by one user's chronological records."""
    result = []
    for i in range(len(values)):
        sample = np.asarray(values.iloc[max(0, i - window + 1): i + 1], dtype=float)
        angles = sample / 1440 * TAU
        center = np.arctan2(np.sin(angles).mean(), np.cos(angles).mean())
        centered = ((angles - center + np.pi) % TAU) - np.pi
        result.append(float(np.std(centered, ddof=0) * 1440 / TAU))
    return pd.Series(result, index=values.index)


def _occupation_schedule(occupation: str) -> tuple[str, float]:
    """Return schedule class and expected midpoint (minutes after midnight)."""
    text = str(occupation).lower()
    if any(x in text for x in ("doctor", "freelancer")):
        return "shift_or_flexible", 240.0
    if "student" in text:
        return "late_day", 300.0
    return "daytime", 210.0


def add_sleep_timing_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add coherent timing features without changing the observed target.

    The supplied CSV has one row per user, so within-user rolling variability and
    weekday/weekend comparisons are zero when a comparison history is unavailable.
    If future data contains repeated user_id + date records, the same functions
    automatically calculate circular rolling statistics and social jetlag.
    """
    out = df.copy()
    if "date" not in out.columns:
        base = pd.Timestamp("2025-01-01")
        out["date"] = [
            (base + pd.Timedelta(days=_stable_int(uid) % 365)).strftime("%Y-%m-%d")
            for uid in out["user_id"]
        ]
    dates = pd.to_datetime(out["date"], errors="raise")
    out["date"] = dates.dt.strftime("%Y-%m-%d")
    out["day_of_week"] = dates.dt.dayofweek.astype(int)
    out["is_weekend"] = (out["day_of_week"] >= 5).astype(int)

    if "bedtime" not in out.columns or "wake_time" not in out.columns:
        bed_values, wake_values, caffeine_values = [], [], []
        for row in out.itertuples(index=False):
            # Deterministic, bounded variation linked to existing behavior.
            jitter = (_stable_int(row.user_id) % 61) - 30
            occupation_shift = {"Doctor": 20, "Freelancer": 35, "Student": 45}.get(row.occupation, 0)
            bed = 22 * 60 + 30 + 8 * float(row.stress_level) + 5 * float(row.mental_fatigue_score)
            bed += 10 * float(row.daily_screen_time_hours) + occupation_shift + jitter
            bed %= 1440
            wake = (bed + float(row.sleep_duration_hours) * 60) % 1440
            caffeine_gap = 150 + 20 * float(row.caffeine_intake_cups) + max(0, jitter) / 3
            caffeine = (bed - caffeine_gap) % 1440
            bed_values.append(_clock(bed)); wake_values.append(_clock(wake)); caffeine_values.append(_clock(caffeine))
        out["bedtime"] = bed_values
        out["wake_time"] = wake_values
        out["last_caffeine_time"] = caffeine_values
    elif "last_caffeine_time" not in out.columns:
        out["last_caffeine_time"] = out["bedtime"]

    out["_bed_minutes"] = out["bedtime"].map(_clock_minutes)
    out["_wake_minutes"] = out["wake_time"].map(_clock_minutes)
    out["_caffeine_minutes"] = out["last_caffeine_time"].map(_clock_minutes)
    out["bedtime_circular_sin"] = np.sin(TAU * out["_bed_minutes"] / 1440)
    out["bedtime_circular_cos"] = np.cos(TAU * out["_bed_minutes"] / 1440)
    out["wake_time_circular_sin"] = np.sin(TAU * out["_wake_minutes"] / 1440)
    out["wake_time_circular_cos"] = np.cos(TAU * out["_wake_minutes"] / 1440)
    sleep_span = (out["_wake_minutes"] - out["_bed_minutes"]) % 1440
    out["mid_sleep_minutes"] = (out["_bed_minutes"] + sleep_span / 2) % 1440
    out["mid_sleep_time"] = out["mid_sleep_minutes"].map(_clock)
    out["caffeine_clearance_window"] = ((out["_bed_minutes"] - out["_caffeine_minutes"]) % 1440) / 60

    order = out.sort_values(["user_id", "date"]).copy()
    order["bedtime_variability"] = order.groupby("user_id", sort=False)["_bed_minutes"].transform(_circular_rolling_std)
    order["wake_time_variability"] = order.groupby("user_id", sort=False)["_wake_minutes"].transform(_circular_rolling_std)
    weekday_mid = order["mid_sleep_minutes"].where(order["is_weekend"].eq(0)).groupby(order["user_id"]).transform("mean")
    weekend_mid = order["mid_sleep_minutes"].where(order["is_weekend"].eq(1)).groupby(order["user_id"]).transform("mean")
    order["social_jetlag_score"] = [
        circular_distance(w, e) if pd.notna(w) and pd.notna(e) else 0.0
        for w, e in zip(weekday_mid, weekend_mid)
    ]
    out = order.sort_index()

    out["work_schedule_mismatch"] = [
        circular_distance(mid, _occupation_schedule(occupation)[1])
        for mid, occupation in zip(out["mid_sleep_minutes"], out["occupation"])
    ]
    # Proxy from the observed pre-sleep phone use, capped to a 3-hour window.
    late_minutes = np.minimum(out["phone_usage_before_sleep_minutes"].astype(float), 180.0)
    out["late_night_digital_share"] = (late_minutes / (out["daily_screen_time_hours"] * 60 + 1e-5)).clip(0, 1)
    out["circadian_disruption_index"] = out["pre_sleep_disruption_ratio"] * out["bedtime_variability"] if "pre_sleep_disruption_ratio" in out else 0.0
    out = out.drop(columns=["_bed_minutes", "_wake_minutes", "_caffeine_minutes"])
    return out


def add_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    """Apply original nine features, then the reproducible timing features."""
    out = df.copy()
    out["screen_to_sleep_ratio"] = out["daily_screen_time_hours"] / (out["sleep_duration_hours"] + 1e-5)
    out["stress_fatigue_index"] = out["stress_level"] * out["mental_fatigue_score"]
    out["digital_intensity"] = (out["daily_screen_time_hours"] * 60 + out["phone_usage_before_sleep_minutes"]) / 60.0
    out["activity_screen_balance"] = out["physical_activity_minutes"] / (out["daily_screen_time_hours"] * 60 + 1e-5)
    out["age_stress_impact"] = out["age"] * out["stress_level"]
    out["notification_frequency"] = out["notifications_received_per_day"] / (out["daily_screen_time_hours"] + 0.1)
    out["caffeine_activity_ratio"] = out["caffeine_intake_cups"] / (out["physical_activity_minutes"] + 1.0)
    out["pre_sleep_disruption_ratio"] = out["phone_usage_before_sleep_minutes"] / (out["sleep_duration_hours"] + 0.1)
    out["lifestyle_balance_score"] = (out["physical_activity_minutes"] / 60.0) - out["caffeine_intake_cups"]
    out = add_sleep_timing_features(out)
    out["circadian_disruption_index"] = out["pre_sleep_disruption_ratio"] * out["bedtime_variability"]
    return out


def prepare_data(config_path: str = "configs/config.yaml") -> tuple[pd.DataFrame, pd.DataFrame]:
    from sklearn.model_selection import train_test_split

    config = load_config(config_path)
    data_cfg = config["data"]
    df = pd.read_csv(data_cfg["raw_path"])
    missing_columns = REQUIRED_COLUMNS.difference(df.columns)
    if missing_columns:
        raise ValueError(f"Missing required columns: {sorted(missing_columns)}")
    if df.empty or df[data_cfg["target"]].isna().any():
        raise ValueError("The input dataset is empty or the target contains missing values.")
    df = add_engineered_features(df).drop_duplicates().reset_index(drop=True)
    train_df, test_df = train_test_split(df, test_size=float(data_cfg["test_size"]), random_state=int(config["project"]["random_state"]))
    Path(data_cfg["train_path"]).parent.mkdir(parents=True, exist_ok=True)
    train_df.to_csv(data_cfg["train_path"], index=False)
    test_df.to_csv(data_cfg["test_path"], index=False)
    return train_df, test_df


if __name__ == "__main__":
    train, test = prepare_data()
    print(f"Prepared train={train.shape}, test={test.shape}")
    print(f"Added columns: {len(set(train.columns) - REQUIRED_COLUMNS)}")
