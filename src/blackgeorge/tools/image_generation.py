import json
from collections.abc import Mapping, Sequence
from typing import cast

import litellm

from blackgeorge.tools.base import ToolResult
from blackgeorge.tools.decorators import tool


def _value(obj: object, key: str) -> object | None:
    if isinstance(obj, Mapping):
        return cast(Mapping[str, object], obj).get(key)
    return getattr(obj, key, None)


def _extract_image_generation_response(response: object) -> dict[str, str]:
    data = _value(response, "data")
    if not isinstance(data, Sequence) or isinstance(data, str | bytes):
        return {}
    if len(data) == 0:
        return {}

    image = data[0]
    result: dict[str, str] = {}

    url = _value(image, "url")
    if isinstance(url, str) and url:
        result["url"] = url

    b64_json = _value(image, "b64_json")
    if isinstance(b64_json, str) and b64_json:
        result["b64_json"] = b64_json

    revised_prompt = _value(image, "revised_prompt")
    if isinstance(revised_prompt, str) and revised_prompt:
        result["revised_prompt"] = revised_prompt

    return result


def _extract_image_url_from_content_blocks(content: object) -> str | None:
    if not isinstance(content, Sequence) or isinstance(content, str | bytes):
        return None
    for block in content:
        block_type = _value(block, "type")
        image_url = _value(block, "image_url")
        if (
            block_type not in {"image_url", "input_image", "output_image"}
            and image_url is None
            and _value(block, "url") is None
        ):
            continue
        if isinstance(image_url, str) and image_url:
            return image_url
        if image_url is not None:
            nested_url = _value(image_url, "url")
            if isinstance(nested_url, str) and nested_url:
                return nested_url
        direct_url = _value(block, "url")
        if isinstance(direct_url, str) and direct_url:
            return direct_url
    return None


def _extract_chat_image_response(response: object) -> dict[str, str]:
    choices = _value(response, "choices")
    if not isinstance(choices, Sequence) or isinstance(choices, str | bytes):
        return {}
    if len(choices) == 0:
        return {}

    message = _value(choices[0], "message")
    images = _value(message, "images")
    result: dict[str, str] = {}

    if isinstance(images, Sequence) and not isinstance(images, str | bytes) and len(images) > 0:
        first_image = images[0]
        image_url = _value(first_image, "image_url")
        url: object | None = None
        if image_url is not None:
            url = _value(image_url, "url")
        if url is None:
            url = _value(first_image, "url")
        if isinstance(url, str) and url:
            result["url"] = url

    content = _value(message, "content")
    if "url" not in result:
        if isinstance(content, str) and content.startswith("data:image/"):
            result["url"] = content
        else:
            block_url = _extract_image_url_from_content_blocks(content)
            if isinstance(block_url, str) and block_url:
                result["url"] = block_url

    return result


def _generate_via_chat_completion(prompt: str, model: str) -> dict[str, str]:
    response = litellm.completion(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        modalities=["image", "text"],
    )
    return _extract_chat_image_response(response)


async def _agenerate_via_chat_completion(prompt: str, model: str) -> dict[str, str]:
    response = await litellm.acompletion(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        modalities=["image", "text"],
    )
    return _extract_chat_image_response(response)


def _tool_result(payload: dict[str, str]) -> ToolResult:
    compact = dict(payload)
    url = compact.get("url")
    if isinstance(url, str) and url.startswith("data:image/"):
        compact["url"] = "[data-url omitted]"
    b64_json = compact.get("b64_json")
    if isinstance(b64_json, str) and b64_json:
        compact["b64_json"] = "[b64_json omitted]"
    return ToolResult(
        content=json.dumps(compact, ensure_ascii=True, default=str),
        data=payload,
    )


@tool(description="Generate an image from a text prompt using an AI model")
def generate_image(
    prompt: str,
    model: str = "openrouter/google/gemini-3-pro-image-preview",
    size: str = "1024x1024",
    quality: str = "standard",
) -> ToolResult:
    generation_error: Exception | None = None
    try:
        response = litellm.image_generation(
            prompt=prompt,
            model=model,
            size=size,
            quality=quality,
        )
        result = _extract_image_generation_response(response)
        if result:
            return _tool_result(result)
    except Exception as exc:
        generation_error = exc

    result = _generate_via_chat_completion(prompt=prompt, model=model)
    if result:
        return _tool_result(result)
    if generation_error is not None:
        raise generation_error
    return _tool_result({})


@tool(description="Generate an image asynchronously from a text prompt")
async def agenerate_image(
    prompt: str,
    model: str = "openrouter/google/gemini-3-pro-image-preview",
    size: str = "1024x1024",
    quality: str = "standard",
) -> ToolResult:
    generation_error: Exception | None = None
    try:
        response = await litellm.aimage_generation(
            prompt=prompt,
            model=model,
            size=size,
            quality=quality,
        )
        result = _extract_image_generation_response(response)
        if result:
            return _tool_result(result)
    except Exception as exc:
        generation_error = exc

    result = await _agenerate_via_chat_completion(prompt=prompt, model=model)
    if result:
        return _tool_result(result)
    if generation_error is not None:
        raise generation_error
    return _tool_result({})
