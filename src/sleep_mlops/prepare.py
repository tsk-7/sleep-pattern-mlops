from __future__ import annotations

from pathlib import Path

import pandas as pd
import yaml
from sklearn.model_selection import train_test_split


REQUIRED_COLUMNS = {
    "user_id",
    "age",
    "gender",
    "occupation",
    "daily_screen_time_hours",
    "phone_usage_before_sleep_minutes",
    "sleep_duration_hours",
    "sleep_quality_score",
    "stress_level",
    "caffeine_intake_cups",
    "physical_activity_minutes",
    "notifications_received_per_day",
    "mental_fatigue_score",
}


def load_config(path: str = "configs/config.yaml") -> dict:
    with open(path, "r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def prepare_data(config_path: str = "configs/config.yaml") -> tuple[pd.DataFrame, pd.DataFrame]:
    config = load_config(config_path)
    data_cfg = config["data"]
    raw_path = Path(data_cfg["raw_path"])
    df = pd.read_csv(raw_path)

    missing_columns = REQUIRED_COLUMNS.difference(df.columns)
    if missing_columns:
        raise ValueError(f"Missing required columns: {sorted(missing_columns)}")
    if df.empty:
        raise ValueError("The input dataset is empty.")
    if df[data_cfg["target"]].isna().any():
        raise ValueError("The target column contains missing values.")

    # user_id identifies a row and is intentionally excluded from model features.
    df = df.drop_duplicates().reset_index(drop=True)
    train_df, test_df = train_test_split(
        df,
        test_size=float(data_cfg["test_size"]),
        random_state=int(config["project"]["random_state"]),
    )

    Path(data_cfg["train_path"]).parent.mkdir(parents=True, exist_ok=True)
    train_df.to_csv(data_cfg["train_path"], index=False)
    test_df.to_csv(data_cfg["test_path"], index=False)
    return train_df, test_df


if __name__ == "__main__":
    train, test = prepare_data()
    print(f"Prepared train={train.shape}, test={test.shape}")
