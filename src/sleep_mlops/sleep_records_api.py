from __future__ import annotations

import json
from datetime import date
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import ValidationError
from sqlalchemy import select
from sqlalchemy.orm import Session

from sleep_mlops.sleep_schemas import SleepRecordRequest, SleepRecordResponse
from sleep_mlops.sleep_store import (
    SleepRecordModel,
    delete_record,
    get_db,
    get_records,
    normalize_record_input,
    record_by_id,
    save_record,
    serialize_record,
)


router = APIRouter(prefix="/api/sleep-records", tags=["sleep records"])


async def _request_payload(request: Request) -> dict[str, Any]:
    try:
        payload = await request.json()
    except (json.JSONDecodeError, UnicodeDecodeError) as error:
        raise HTTPException(status_code=400, detail="Request body must be valid JSON.") from error
    if not isinstance(payload, dict):
        raise HTTPException(status_code=400, detail="Request body must be a JSON object.")
    return payload


def _validated_input(payload: dict[str, Any]) -> dict[str, Any]:
    try:
        request = SleepRecordRequest.model_validate(payload)
    except ValidationError as error:
        details = "; ".join(
            f"{'.'.join(str(part) for part in issue['loc'])}: {issue['msg']}"
            for issue in error.errors()
        )
        raise HTTPException(status_code=400, detail=f"Invalid sleep record: {details}") from error
    return normalize_record_input(request.model_dump())


@router.post("", status_code=201, response_model=SleepRecordResponse)
async def create_sleep_record(request: Request, session: Session = Depends(get_db)) -> dict[str, Any]:
    values = _validated_input(await _request_payload(request))
    record = save_record(session, values)
    return serialize_record(record)


@router.get("/user/{user_id}", response_model=list[SleepRecordResponse])
def list_sleep_records(
    user_id: str,
    days: int | None = None,
    session: Session = Depends(get_db),
) -> list[dict[str, Any]]:
    if days is not None and not 1 <= days <= 3660:
        raise HTTPException(status_code=400, detail="days must be between 1 and 3660.")
    return [serialize_record(record) for record in get_records(session, user_id, days)]


@router.get("/{record_id}", response_model=SleepRecordResponse)
def get_sleep_record(record_id: int, session: Session = Depends(get_db)) -> dict[str, Any]:
    return serialize_record(record_by_id(session, record_id))


@router.put("/{record_id}", response_model=SleepRecordResponse)
async def update_sleep_record(
    record_id: int,
    request: Request,
    session: Session = Depends(get_db),
) -> dict[str, Any]:
    record = record_by_id(session, record_id)
    values = _validated_input(await _request_payload(request))
    return serialize_record(save_record(session, values, record))


@router.delete("/{record_id}", status_code=204)
def remove_sleep_record(record_id: int, session: Session = Depends(get_db)) -> None:
    delete_record(session, record_by_id(session, record_id))
