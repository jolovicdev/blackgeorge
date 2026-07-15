from dataclasses import asdict, is_dataclass
from typing import Any, cast

from pydantic import BaseModel


def to_json_value(value: Any) -> Any:
    if isinstance(value, BaseModel):
        return to_json_value(value.model_dump(mode="json", warnings=False))
    if is_dataclass(value) and not isinstance(value, type):
        return to_json_value(asdict(cast(Any, value)))
    if isinstance(value, dict):
        return {key: to_json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_json_value(item) for item in value]
    return value
