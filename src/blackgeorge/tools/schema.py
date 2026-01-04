import inspect
from typing import Any, cast, get_type_hints

from pydantic import BaseModel, create_model


def build_input_model(fn: Any) -> type[BaseModel]:
    signature = inspect.signature(fn)
    hints = get_type_hints(fn)
    fields: dict[str, Any] = {}

    for name, param in signature.parameters.items():
        if param.kind in {inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD}:
            continue
        annotation = hints.get(name, Any)
        default = param.default if param.default is not inspect.Parameter.empty else ...
        fields[name] = (annotation, default)

    model_name = f"{fn.__name__.title()}Input"
    model = create_model(model_name, **fields)
    return cast(type[BaseModel], model)


def build_schema(input_model: type[BaseModel]) -> dict[str, Any]:
    return input_model.model_json_schema()
