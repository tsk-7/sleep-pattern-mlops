from __future__ import annotations

import math
from typing import Any

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import select
from sqlalchemy.orm import Session

from sleep_mlops.sleep_schemas import (
    DailySleepAnalysisResponse,
    SleepAnalysisResponse,
    WeekendComparisonResponse,
)
from sleep_mlops.sleep_store import (
    SleepRecordModel,
    _parse_date,
    get_db,
    get_records,
    serialize_record,
)


router = APIRouter(prefix="/api/sleep-analysis", tags=["sleep analysis"])
DAY_MINUTES = 24 * 60


def _wrap(value: float) -> float:
    return value % DAY_MINUTES


def _circular_mean(values: list[float]) -> float | None:
    if not values:
        return None
    angles = [(_wrap(value) / DAY_MINUTES) * math.tau for value in values]
    sine = sum(math.sin(angle) for angle in angles) / len(angles)
    cosine = sum(math.cos(angle) for angle in angles) / len(angles)
    return _wrap(math.atan2(sine, cosine) * DAY_MINUTES / math.tau)


def _circular_sd(values: list[float]) -> float | None:
    if len(values) < 2:
        return None
    angles = [(_wrap(value) / DAY_MINUTES) * math.tau for value in values]
    sine = sum(math.sin(angle) for angle in angles) / len(angles)
    cosine = sum(math.cos(angle) for angle in angles) / len(angles)
    resultant = min(1.0, math.hypot(sine, cosine))
    return math.sqrt(-2 * math.log(max(resultant, math.ulp(1.0)))) * DAY_MINUTES / math.tau


def _difference(first: float | None, second: float | None) -> float | None:
    if first is None or second is None:
        return None
    distance = abs(_wrap(first) - _wrap(second))
    return min(distance, DAY_MINUTES - distance)


def _mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def _summary(records: list[SleepRecordModel]) -> dict[str, Any]:
    items = [serialize_record(record) for record in records]
    weekdays = [item for item in items if not item["is_weekend"]]
    weekends = [item for item in items if item["is_weekend"]]

    def mean_clock(source: list[dict[str, Any]], key: str) -> float | None:
        return _circular_mean([_clock_minutes(item[key]) for item in source])

    bedtime_values = [_clock_minutes(item["bedtime"]) for item in items]
    wake_values = [_clock_minutes(item["wake_up_time"]) for item in items]
    duration_values = [float(item["sleep_duration_minutes"]) for item in items]
    # Mid-sleep averages also cross midnight, so average them on a circular clock.
    weekday_mid = _circular_mean([float(item["mid_sleep_minutes"]) for item in weekdays])
    weekend_mid = _circular_mean([float(item["mid_sleep_minutes"]) for item in weekends])
    bedtime_sd = _circular_sd(bedtime_values)
    wake_sd = _circular_sd(wake_values)
    duration_sd = _population_sd(duration_values)
    social_jetlag = _difference(weekday_mid, weekend_mid)
    average_mid_sleep = _circular_mean([float(item["mid_sleep_minutes"]) for item in items])
    average_bedtime = mean_clock(items, "bedtime")
    average_wake_time = mean_clock(items, "wake_up_time")

    for item in items:
        item["bedtime_deviation_minutes"] = _difference(
            _clock_minutes(item["bedtime"]), average_bedtime
        )
        item["wake_time_deviation_minutes"] = _difference(
            _clock_minutes(item["wake_up_time"]), average_wake_time
        )

    if bedtime_sd is None or wake_sd is None or duration_sd is None:
        consistency_label = "More sleep records are needed for this analysis."
    elif bedtime_sd <= 30 and wake_sd <= 30 and duration_sd <= 30:
        consistency_label = "Consistent Schedule"
    elif bedtime_sd <= 60 and wake_sd <= 60 and duration_sd <= 60:
        consistency_label = "Moderately Variable Schedule"
    else:
        consistency_label = "Irregular Schedule"
    consistency_index = (
        None
        if bedtime_sd is None or wake_sd is None
        else max(0, min(100, round(100 - (bedtime_sd + wake_sd) / 4)))
    )

    labels: list[str] = []
    average_duration = _mean(duration_values)
    if len(items) >= 2:
        if consistency_label == "Consistent Schedule":
            labels.append("Regular Sleeper")
        elif consistency_label == "Moderately Variable Schedule":
            labels.append("Moderately Consistent")
        elif consistency_label == "Irregular Schedule":
            labels.append("Irregular Schedule")
        if average_bedtime is not None and (average_bedtime >= 23 * 60 or average_bedtime < 4 * 60):
            labels.append("Late Sleeper")
        elif average_bedtime is not None and 19 * 60 <= average_bedtime < 22 * 60:
            labels.append("Early Sleeper")
        if social_jetlag is not None and social_jetlag >= 60:
            labels.append("Weekend Shifter")
        if average_duration is not None and average_duration < 7 * 60:
            labels.append("Short Sleep Pattern")
        elif average_duration is not None and average_duration > 9 * 60:
            labels.append("Long Sleep Pattern")

    return {
        "record_count": len(items),
        "average_bedtime_minutes": average_bedtime,
        "average_wake_time_minutes": average_wake_time,
        "average_sleep_duration_minutes": _mean(duration_values),
        "average_mid_sleep_minutes": average_mid_sleep,
        "bedtime_variability_minutes": bedtime_sd,
        "wake_time_variability_minutes": wake_sd,
        "sleep_duration_variability_minutes": duration_sd,
        "consistency_index": consistency_index,
        "consistency_label": consistency_label,
        "weekday_mid_sleep_minutes": weekday_mid,
        "weekend_mid_sleep_minutes": weekend_mid,
        "social_jetlag_minutes": social_jetlag,
        "weekday_weekend_bedtime_difference_minutes": _difference(
            mean_clock(weekdays, "bedtime"), mean_clock(weekends, "bedtime")
        ),
        "weekday_weekend_wake_time_difference_minutes": _difference(
            mean_clock(weekdays, "wake_up_time"), mean_clock(weekends, "wake_up_time")
        ),
        "weekday_weekend_mid_sleep_difference_minutes": social_jetlag,
        "weekday_weekend_sleep_duration_difference_minutes": _duration_difference(weekdays, weekends),
        "weekday_bedtime_minutes": mean_clock(weekdays, "bedtime"),
        "weekend_bedtime_minutes": mean_clock(weekends, "bedtime"),
        "weekday_wake_time_minutes": mean_clock(weekdays, "wake_up_time"),
        "weekend_wake_time_minutes": mean_clock(weekends, "wake_up_time"),
        "weekday_sleep_duration_minutes": _mean([float(item["sleep_duration_minutes"]) for item in weekdays]),
        "weekend_sleep_duration_minutes": _mean([float(item["sleep_duration_minutes"]) for item in weekends]),
        "consistency_label": consistency_label,
        "pattern_labels": labels,
        "records": items,
    }


def _clock_minutes(value: str) -> float:
    hour, minute = (int(part) for part in value.split(":"))
    return hour * 60 + minute


def _population_sd(values: list[float]) -> float | None:
    if len(values) < 2:
        return None
    average = sum(values) / len(values)
    return math.sqrt(sum((value - average) ** 2 for value in values) / len(values))


def _duration_difference(weekdays: list[dict[str, Any]], weekends: list[dict[str, Any]]) -> float | None:
    weekday_mean = _mean([float(item["sleep_duration_minutes"]) for item in weekdays])
    weekend_mean = _mean([float(item["sleep_duration_minutes"]) for item in weekends])
    if weekday_mean is None or weekend_mean is None:
        return None
    return abs(weekday_mean - weekend_mean)


@router.get("/daily/{user_id}/{sleep_date}", response_model=DailySleepAnalysisResponse)
def daily_analysis(
    user_id: str,
    sleep_date: str,
    session: Session = Depends(get_db),
) -> dict[str, Any]:
    parsed_date = _parse_date(sleep_date)
    record = session.scalar(
        select(SleepRecordModel).where(
            SleepRecordModel.user_id == user_id,
            SleepRecordModel.sleep_date == parsed_date,
        )
    )
    return {
        "sleep_date": parsed_date.isoformat(),
        "record": serialize_record(record) if record else None,
        "message": None if record else "No sleep record for this day.",
    }


@router.get("/weekly/{user_id}", response_model=SleepAnalysisResponse)
def weekly_analysis(user_id: str, session: Session = Depends(get_db)) -> dict[str, Any]:
    records = get_records(session, user_id, 7)
    return {"period_days": 7, **_summary(records)}


@router.get("/monthly/{user_id}", response_model=SleepAnalysisResponse)
def monthly_analysis(user_id: str, session: Session = Depends(get_db)) -> dict[str, Any]:
    records = get_records(session, user_id, 30)
    return {"period_days": 30, **_summary(records)}


@router.get("/pattern/{user_id}", response_model=SleepAnalysisResponse)
def pattern_analysis(user_id: str, session: Session = Depends(get_db)) -> dict[str, Any]:
    records = get_records(session, user_id, 90)
    return {"period_days": 90, **_summary(records)}


@router.get("/weekend-comparison/{user_id}", response_model=WeekendComparisonResponse)
def weekend_comparison(user_id: str, session: Session = Depends(get_db)) -> dict[str, Any]:
    summary = _summary(get_records(session, user_id, 90))
    return {
        "record_count": summary["record_count"],
        "weekday_mid_sleep_minutes": summary["weekday_mid_sleep_minutes"],
        "weekend_mid_sleep_minutes": summary["weekend_mid_sleep_minutes"],
        "social_jetlag_minutes": summary["social_jetlag_minutes"],
        "weekday_weekend_bedtime_difference_minutes": summary["weekday_weekend_bedtime_difference_minutes"],
        "weekday_weekend_wake_time_difference_minutes": summary["weekday_weekend_wake_time_difference_minutes"],
        "weekday_weekend_mid_sleep_difference_minutes": summary["weekday_weekend_mid_sleep_difference_minutes"],
        "weekday_weekend_sleep_duration_difference_minutes": summary["weekday_weekend_sleep_duration_difference_minutes"],
        "weekday_bedtime_minutes": summary["weekday_bedtime_minutes"],
        "weekend_bedtime_minutes": summary["weekend_bedtime_minutes"],
        "weekday_wake_time_minutes": summary["weekday_wake_time_minutes"],
        "weekend_wake_time_minutes": summary["weekend_wake_time_minutes"],
        "weekday_sleep_duration_minutes": summary["weekday_sleep_duration_minutes"],
        "weekend_sleep_duration_minutes": summary["weekend_sleep_duration_minutes"],
    }
