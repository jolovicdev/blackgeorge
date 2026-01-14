from dataclasses import asdict, is_dataclass
from typing import Any, cast

from pydantic import BaseModel, ConfigDict, field_serializer


class ToolCall(BaseModel):
    id: str
    name: str
    arguments: dict[str, Any]
    result: Any | None = None
    error: str | None = None

    model_config = ConfigDict(frozen=True)

    @field_serializer("result")
    def _serialize_result(self, value: Any) -> Any:
        if value is None:
            return None
        if isinstance(value, BaseModel):
            return value.model_dump(mode="json", warnings=False)
        if is_dataclass(value) and not isinstance(value, type):
            return asdict(cast(Any, value))
        return value
