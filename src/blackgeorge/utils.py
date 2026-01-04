from datetime import UTC, datetime
from uuid import uuid4


def new_id() -> str:
    return uuid4().hex


def utc_now() -> datetime:
    return datetime.now(UTC)
