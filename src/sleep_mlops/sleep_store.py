from __future__ import annotations

import os
import re
from collections.abc import Generator
from datetime import date, datetime, time, timedelta
from functools import lru_cache
from threading import Lock
from typing import Any

from fastapi import HTTPException
from sqlalchemy import Date, DateTime, Index, Integer, String, Time, UniqueConstraint, create_engine, func, select
from sqlalchemy.exc import IntegrityError, SQLAlchemyError
from sqlalchemy.orm import DeclarativeBase, Mapped, Session, mapped_column, sessionmaker


class Base(DeclarativeBase):
    pass


class SleepRecordModel(Base):
    __tablename__ = "sleep_records"
    __table_args__ = (
        UniqueConstraint("user_id", "sleep_date", name="uq_sleep_records_user_date"),
        Index("ix_sleep_records_user_date", "user_id", "sleep_date"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    user_id: Mapped[str] = mapped_column(String(128), nullable=False)
    sleep_date: Mapped[date] = mapped_column(Date, nullable=False)
    bedtime: Mapped[time] = mapped_column(Time, nullable=False)
    actual_sleep_time: Mapped[time] = mapped_column(Time, nullable=False)
    wake_up_time: Mapped[time] = mapped_column(Time, nullable=False)
    get_up_time: Mapped[time] = mapped_column(Time, nullable=False)
    notes: Mapped[str | None] = mapped_column(String(2000), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, nullable=False, server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(
        DateTime, nullable=False, server_default=func.now(), onupdate=func.now()
    )


_schema_lock = Lock()
_schema_initialized = False


@lru_cache(maxsize=1)
def _session_factory() -> sessionmaker[Session]:
    database_url = os.getenv("SLEEP_DATABASE_URL")
    if not database_url:
        raise HTTPException(
            status_code=503,
            detail="Sleep database is not configured. Set SLEEP_DATABASE_URL for the MySQL database.",
        )
    try:
        engine = create_engine(database_url, pool_pre_ping=True)
    except (SQLAlchemyError, ValueError) as error:
        raise HTTPException(status_code=503, detail="Sleep database configuration is invalid.") from error
    return sessionmaker(bind=engine, expire_on_commit=False)


def database_status() -> tuple[bool, bool]:
    if not os.getenv("SLEEP_DATABASE_URL"):
        return False, False
    try:
        with _session_factory().kw["bind"].connect():
            return True, True
    except (HTTPException, SQLAlchemyError):
        return True, False


def get_db() -> Generator[Session, None, None]:
    global _schema_initialized
    factory = _session_factory()
    if not _schema_initialized:
        with _schema_lock:
            if not _schema_initialized:
                try:
                    Base.metadata.create_all(factory.kw["bind"])
                    _schema_initialized = True
                except SQLAlchemyError as error:
                    raise HTTPException(
                        status_code=503,
                        detail="Could not connect to the sleep database. Check SLEEP_DATABASE_URL and ensure sleep_pattern_db exists.",
                    ) from error
    session = factory()
    try:
        yield session
    except SQLAlchemyError as error:
        session.rollback()
        raise HTTPException(status_code=503, detail="The sleep database request failed.") from error
    finally:
        session.close()


def _parse_date(value: str) -> date:
    if not isinstance(value, str) or re.fullmatch(r"\d{4}-\d{2}-\d{2}", value) is None:
        raise HTTPException(status_code=400, detail="sleep_date must be a valid YYYY-MM-DD date.")
    try:
        parsed = date.fromisoformat(value)
    except ValueError as error:
        raise HTTPException(status_code=400, detail="sleep_date must be a valid YYYY-MM-DD date.") from error
    if parsed > date.today():
        raise HTTPException(status_code=400, detail="sleep_date cannot be in the future.")
    return parsed


def _parse_time(value: str, field: str) -> time:
    if not isinstance(value, str) or re.fullmatch(r"(?:[01]\d|2[0-3]):[0-5]\d", value) is None:
        raise HTTPException(status_code=400, detail=f"{field} must be a valid 24-hour time (HH:MM).")
    try:
        parsed = time.fromisoformat(value)
    except ValueError as error:
        raise HTTPException(status_code=400, detail=f"{field} must be a valid 24-hour time (HH:MM).") from error
    if parsed.tzinfo is not None:
        raise HTTPException(status_code=400, detail=f"{field} must not include a timezone.")
    return parsed.replace(second=0, microsecond=0)


def _minutes(value: time) -> int:
    return value.hour * 60 + value.minute


def _ordered_minutes(values: list[time]) -> list[int]:
    result = [_minutes(values[0])]
    for current in values[1:]:
        minute = _minutes(current)
        while minute < result[-1]:
            minute += 24 * 60
        result.append(minute)
    return result


def calculate_metrics(
    bedtime: time,
    actual_sleep_time: time,
    wake_up_time: time,
    get_up_time: time,
) -> dict[str, int | float]:
    bedtime_at, actual_sleep_at, wake_at, get_up_at = _ordered_minutes(
        [bedtime, actual_sleep_time, wake_up_time, get_up_time]
    )
    sleep_duration = wake_at - actual_sleep_at
    time_in_bed = get_up_at - bedtime_at
    onset_latency = actual_sleep_at - bedtime_at
    get_up_latency = get_up_at - wake_at

    if sleep_duration <= 0 or sleep_duration > 24 * 60:
        raise HTTPException(status_code=400, detail="Sleep duration must be greater than 0 and no more than 24 hours.")
    if onset_latency > 12 * 60:
        raise HTTPException(status_code=400, detail="Actual sleep time must follow bedtime within 12 hours.")
    if get_up_latency > 12 * 60:
        raise HTTPException(status_code=400, detail="Get-up time must follow wake-up time within 12 hours.")
    if time_in_bed <= 0 or time_in_bed > 24 * 60:
        raise HTTPException(status_code=400, detail="Time in bed must be greater than 0 and no more than 24 hours.")
    if sleep_duration > time_in_bed:
        raise HTTPException(status_code=400, detail="Sleep duration cannot exceed time in bed.")
    return {
        "sleep_duration_minutes": sleep_duration,
        "time_in_bed_minutes": time_in_bed,
        "sleep_onset_latency_minutes": onset_latency,
        "get_up_latency_minutes": get_up_latency,
        "sleep_efficiency_percent": round(sleep_duration / time_in_bed * 100, 1),
    }


def normalize_record_input(payload: dict[str, Any]) -> dict[str, Any]:
    user_id = payload.get("user_id")
    if not isinstance(user_id, str) or not user_id.strip() or len(user_id) > 128:
        raise HTTPException(status_code=400, detail="user_id must be a non-empty string of at most 128 characters.")
    notes = payload.get("notes")
    if notes is not None and (not isinstance(notes, str) or len(notes) > 2000):
        raise HTTPException(status_code=400, detail="notes must be no longer than 2000 characters.")
    parsed = {
        "user_id": user_id.strip(),
        "sleep_date": _parse_date(payload.get("sleep_date", "")),
        "bedtime": _parse_time(payload.get("bedtime", ""), "bedtime"),
        "actual_sleep_time": _parse_time(payload.get("actual_sleep_time", ""), "actual_sleep_time"),
        "wake_up_time": _parse_time(payload.get("wake_up_time", ""), "wake_up_time"),
        "get_up_time": _parse_time(payload.get("get_up_time", ""), "get_up_time"),
        "notes": notes.strip() if isinstance(notes, str) and notes.strip() else None,
    }
    calculate_metrics(
        parsed["bedtime"], parsed["actual_sleep_time"], parsed["wake_up_time"], parsed["get_up_time"]
    )
    return parsed


def serialize_record(record: SleepRecordModel) -> dict[str, Any]:
    metrics = calculate_metrics(
        record.bedtime, record.actual_sleep_time, record.wake_up_time, record.get_up_time
    )
    sleep_minutes = int(metrics["sleep_duration_minutes"])
    mid_sleep_minutes = (
        _minutes(record.actual_sleep_time) + sleep_minutes / 2
    ) % (24 * 60)
    bedtime_minutes = _minutes(record.bedtime)
    return {
        "id": record.id,
        "user_id": record.user_id,
        "sleep_date": record.sleep_date.isoformat(),
        "date": record.sleep_date.isoformat(),
        "bedtime": record.bedtime.strftime("%H:%M"),
        "actual_sleep_time": record.actual_sleep_time.strftime("%H:%M"),
        "wake_up_time": record.wake_up_time.strftime("%H:%M"),
        "wake_time": record.wake_up_time.strftime("%H:%M"),
        "get_up_time": record.get_up_time.strftime("%H:%M"),
        "notes": record.notes,
        **metrics,
        "sleep_duration_hours": sleep_minutes / 60,
        "mid_sleep_time": f"{int(mid_sleep_minutes // 60):02d}:{int(mid_sleep_minutes % 60):02d}",
        "source": "backend",
        "is_weekend": record.sleep_date.weekday() >= 5,
        "mid_sleep_minutes": mid_sleep_minutes,
        "bedtime_trend_minutes": bedtime_minutes + (24 * 60 if bedtime_minutes < 10 * 60 else 0),
        "bedtime_deviation_minutes": None,
        "wake_time_deviation_minutes": None,
        "daily_screen_time_hours": None,
        "phone_usage_before_sleep_minutes": None,
        "last_caffeine_time": None,
        "caffeine_clearance_window_hours": None,
        "caffeine_intake_cups": None,
        "physical_activity_minutes": None,
        "stress_level": None,
        "mental_fatigue_score": None,
        "notifications_received_per_day": None,
        "created_at": record.created_at.isoformat() if record.created_at else None,
        "updated_at": record.updated_at.isoformat() if record.updated_at else None,
    }


def get_records(session: Session, user_id: str, days: int | None = None) -> list[SleepRecordModel]:
    statement = select(SleepRecordModel).where(SleepRecordModel.user_id == user_id)
    if days is not None:
        cutoff = date.today() - timedelta(days=days - 1)
        statement = statement.where(SleepRecordModel.sleep_date >= cutoff)
    return list(session.scalars(statement.order_by(SleepRecordModel.sleep_date)).all())


def record_by_id(session: Session, record_id: int) -> SleepRecordModel:
    record = session.get(SleepRecordModel, record_id)
    if record is None:
        raise HTTPException(status_code=404, detail="Sleep record not found.")
    return record


def save_record(session: Session, values: dict[str, Any], record: SleepRecordModel | None = None) -> SleepRecordModel:
    if record is None:
        existing = session.scalar(
            select(SleepRecordModel.id).where(
                SleepRecordModel.user_id == values["user_id"],
                SleepRecordModel.sleep_date == values["sleep_date"],
            )
        )
        if existing is not None:
            raise HTTPException(status_code=400, detail="A sleep record already exists for this user and date.")
        record = SleepRecordModel(**values)
        session.add(record)
    else:
        for key, value in values.items():
            setattr(record, key, value)
    try:
        session.commit()
        session.refresh(record)
    except IntegrityError as error:
        session.rollback()
        raise HTTPException(status_code=400, detail="A sleep record already exists for this user and date.") from error
    except SQLAlchemyError as error:
        session.rollback()
        raise HTTPException(status_code=503, detail="Could not save the sleep record to the database.") from error
    return record


def delete_record(session: Session, record: SleepRecordModel) -> None:
    try:
        session.delete(record)
        session.commit()
    except SQLAlchemyError as error:
        session.rollback()
        raise HTTPException(status_code=503, detail="Could not delete the sleep record from the database.") from error
