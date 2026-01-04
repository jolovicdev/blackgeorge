from datetime import datetime
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class Event(BaseModel):
    event_id: str
    type: str
    timestamp: datetime
    run_id: str
    source: str
    payload: dict[str, Any] = Field(default_factory=dict)

    model_config = ConfigDict(frozen=True)
