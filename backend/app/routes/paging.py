"""Keyset pagination: stable, fast pages for large libraries (no OFFSET scans)."""

from __future__ import annotations

import base64
import uuid
from datetime import datetime
from typing import Any

from sqlalchemy import and_, or_

from app.deps import api_error


def encode_cursor(value: datetime | None, row_id: uuid.UUID) -> str:
    raw = f"{value.isoformat() if value else ''}|{row_id}"
    return base64.urlsafe_b64encode(raw.encode()).decode()


def decode_cursor(cursor: str) -> tuple[datetime | None, uuid.UUID]:
    try:
        raw = base64.urlsafe_b64decode(cursor.encode()).decode()
        stamp, row_id = raw.split("|", 1)
        return (datetime.fromisoformat(stamp) if stamp else None), uuid.UUID(row_id)
    except (ValueError, UnicodeDecodeError) as exc:
        raise api_error(400, "bad_cursor", "Invalid page cursor.") from exc


def after(column: Any, id_column: Any, cursor: str, descending: bool):
    value, row_id = decode_cursor(cursor)
    if descending:
        return or_(column < value, and_(column == value, id_column < row_id))
    return or_(column > value, and_(column == value, id_column > row_id))
