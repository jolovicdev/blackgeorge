from typing import Any

from pydantic import BaseModel, ConfigDict, field_serializer

from blackgeorge.core.serialization import to_json_value


class ToolCall(BaseModel):
    id: str
    name: str
    arguments: dict[str, Any]
    result: Any | None = None
    error: str | None = None

    model_config = ConfigDict(frozen=True)

    @field_serializer("result")
    def _serialize_result(self, value: Any) -> Any:
        return to_json_value(value)
