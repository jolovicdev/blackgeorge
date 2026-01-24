import importlib
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, TypeAdapter, field_serializer, field_validator

from blackgeorge.core.message import Message
from blackgeorge.utils import new_id


def _schema_ref(value: Any) -> dict[str, str] | None:
    if isinstance(value, type) and issubclass(value, BaseModel):
        return {"kind": "model", "module": value.__module__, "qualname": value.__qualname__}
    if isinstance(value, TypeAdapter):
        inner = getattr(value, "_type", None)
        if isinstance(inner, type) and issubclass(inner, BaseModel):
            return {
                "kind": "adapter_model",
                "module": inner.__module__,
                "qualname": inner.__qualname__,
            }
    return None


def _resolve_ref(module: str, qualname: str) -> Any | None:
    try:
        target: Any = importlib.import_module(module)
    except Exception:
        return None
    for part in qualname.split("."):
        target = getattr(target, part, None)
        if target is None:
            return None
    return target


def _parse_schema_string(value: str) -> Any | None:
    if value.startswith("<class '") and value.endswith("'>"):
        path = value[len("<class '") : -2]
        module, _, qualname = path.rpartition(".")
        if module and qualname:
            return _resolve_ref(module, qualname)
    return None


class Job(BaseModel):
    id: str = Field(default_factory=new_id)
    input: Any
    expected_output: str | None = None
    tools_override: list[Any] | None = None
    response_schema: type[BaseModel] | TypeAdapter[Any] | None = None
    constraints: dict[str, Any] = Field(default_factory=dict)
    metadata: dict[str, Any] = Field(default_factory=dict)
    initial_messages: list[Message] | None = None
    thinking: dict[str, Any] | None = None
    drop_params: bool | None = None
    extra_body: dict[str, Any] | None = None

    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)

    @field_validator("response_schema", mode="before")
    @classmethod
    def _deserialize_response_schema(cls, value: Any) -> Any:
        if value is None:
            return None
        if isinstance(value, (TypeAdapter, type)):
            return value
        if isinstance(value, dict):
            module = value.get("module")
            qualname = value.get("qualname")
            if isinstance(module, str) and isinstance(qualname, str):
                resolved = _resolve_ref(module, qualname)
                if isinstance(resolved, type) and issubclass(resolved, BaseModel):
                    if value.get("kind") == "adapter_model":
                        return TypeAdapter(resolved)
                    return resolved
            return None
        if isinstance(value, str):
            resolved = _parse_schema_string(value)
            if isinstance(resolved, type) and issubclass(resolved, BaseModel):
                return resolved
            return None
        return None

    @field_serializer("response_schema", when_used="json")
    def _serialize_response_schema(self, value: Any) -> Any:
        if value is None:
            return None
        ref = _schema_ref(value)
        if ref is not None:
            return ref
        return str(value)

    @field_serializer("tools_override", when_used="json")
    def _serialize_tools_override(self, value: Any) -> Any:
        if value is None:
            return None
        return [getattr(tool, "name", str(tool)) for tool in value]
