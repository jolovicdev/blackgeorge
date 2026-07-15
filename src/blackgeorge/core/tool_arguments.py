import json
from typing import Any


def parse_tool_arguments(value: Any, raw_limit: int = 100) -> tuple[dict[str, Any], str | None]:
    if value is None or value == "":
        return {}, None
    if isinstance(value, dict):
        return dict(value), None
    if not isinstance(value, str):
        return {}, f"Unsupported tool arguments type: {type(value).__name__}"
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError as exc:
        return {}, f"Invalid JSON in tool arguments: {exc}. Raw: {value[:raw_limit]}"
    if not isinstance(parsed, dict):
        return {}, "Tool arguments JSON must decode to an object"
    return parsed, None
