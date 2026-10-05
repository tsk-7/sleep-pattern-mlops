from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class SleepRecordRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    user_id: str = Field(min_length=1, max_length=128)
    sleep_date: str
    bedtime: str
    actual_sleep_time: str
    wake_up_time: str
    get_up_time: str
    notes: str | None = Field(default=None, max_length=2000)


class SleepRecordResponse(BaseModel):
    id: int
    user_id: str
    sleep_date: str
    date: str
    bedtime: str
    actual_sleep_time: str
    wake_up_time: str
    wake_time: str
    get_up_time: str
    notes: str | None
    sleep_duration_minutes: int
    time_in_bed_minutes: int
    sleep_onset_latency_minutes: int
    get_up_latency_minutes: int
    sleep_efficiency_percent: float
    sleep_duration_hours: float
    mid_sleep_time: str
    mid_sleep_minutes: float
    bedtime_trend_minutes: int
    bedtime_deviation_minutes: float | None = None
    wake_time_deviation_minutes: float | None = None
    is_weekend: bool
    source: str
    created_at: str | None
    updated_at: str | None


class DailySleepAnalysisResponse(BaseModel):
    sleep_date: str
    record: SleepRecordResponse | None
    message: str | None


class SleepAnalysisResponse(BaseModel):
    period_days: int | None = None
    record_count: int
    average_bedtime_minutes: float | None
    average_wake_time_minutes: float | None
    average_sleep_duration_minutes: float | None
    average_mid_sleep_minutes: float | None
    bedtime_variability_minutes: float | None
    wake_time_variability_minutes: float | None
    sleep_duration_variability_minutes: float | None
    consistency_index: int | None
    consistency_label: str
    weekday_mid_sleep_minutes: float | None
    weekend_mid_sleep_minutes: float | None
    social_jetlag_minutes: float | None
    weekday_weekend_bedtime_difference_minutes: float | None
    weekday_weekend_wake_time_difference_minutes: float | None
    weekday_weekend_mid_sleep_difference_minutes: float | None
    weekday_weekend_sleep_duration_difference_minutes: float | None
    weekday_bedtime_minutes: float | None
    weekend_bedtime_minutes: float | None
    weekday_wake_time_minutes: float | None
    weekend_wake_time_minutes: float | None
    weekday_sleep_duration_minutes: float | None
    weekend_sleep_duration_minutes: float | None
    pattern_labels: list[str]
    records: list[SleepRecordResponse]


class WeekendComparisonResponse(BaseModel):
    record_count: int
    weekday_mid_sleep_minutes: float | None
    weekend_mid_sleep_minutes: float | None
    social_jetlag_minutes: float | None
    weekday_weekend_bedtime_difference_minutes: float | None
    weekday_weekend_wake_time_difference_minutes: float | None
    weekday_weekend_mid_sleep_difference_minutes: float | None
    weekday_weekend_sleep_duration_difference_minutes: float | None
    weekday_bedtime_minutes: float | None
    weekend_bedtime_minutes: float | None
    weekday_wake_time_minutes: float | None
    weekend_wake_time_minutes: float | None
    weekday_sleep_duration_minutes: float | None
    weekend_sleep_duration_minutes: float | None
