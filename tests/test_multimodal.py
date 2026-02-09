import json

import pytest

from blackgeorge import Job, encode_file
from blackgeorge.tools.base import ToolResult
from blackgeorge.tools.image_generation import agenerate_image, generate_image
from blackgeorge.worker_messages import render_input


def test_multimodal_message_with_image_url() -> None:
    content = [
        {"type": "text", "text": "What's in this image?"},
        {"type": "image_url", "image_url": {"url": "https://example.com/image.jpg"}},
    ]

    job = Job(input=content)
    assert job.input == content


def test_render_input_preserves_multimodal_blocks() -> None:
    content = [
        {"type": "text", "text": "What is in this image?"},
        {"type": "image_url", "image_url": {"url": "https://example.com/cat.png"}},
    ]

    rendered = render_input(content)
    assert isinstance(rendered, list)
    assert rendered == content


def test_render_input_serializes_non_multimodal_list() -> None:
    content = ["a", 1, {"x": "y"}]

    rendered = render_input(content)
    assert isinstance(rendered, str)
    assert json.loads(rendered) == content


def test_multimodal_message_with_video() -> None:
    content = [
        {"type": "text", "text": "Summarize this video"},
        {"type": "video_url", "video_url": {"url": "https://youtube.com/watch?v=test"}},
    ]

    job = Job(input=content)
    assert job.input == content


def test_encode_file_creates_data_url(tmp_path) -> None:
    test_file = tmp_path / "test.jpg"
    test_file.write_bytes(b"fake image data")

    encoded = encode_file(str(test_file))

    assert encoded.startswith("data:image/jpeg;base64,")
    assert "ZmFrZSBpbWFnZSBkYXRh" in encoded


def test_encode_file_with_custom_mime() -> None:
    import tempfile

    with tempfile.NamedTemporaryFile(mode="wb", suffix=".bin", delete=False) as f:
        f.write(b"test data")
        path = f.name

    try:
        encoded = encode_file(path, mime_type="application/custom")
        assert encoded.startswith("data:application/custom;base64,")
    finally:
        import os

        os.unlink(path)


def test_multimodal_message_with_multiple_images() -> None:
    content = [
        {"type": "text", "text": "Compare these images"},
        {"type": "image_url", "image_url": {"url": "https://example.com/img1.jpg"}},
        {"type": "image_url", "image_url": {"url": "https://example.com/img2.jpg"}},
    ]

    job = Job(input=content)
    assert job.input == content
    assert len([c for c in content if c["type"] == "image_url"]) == 2


def test_multimodal_message_with_pdf() -> None:
    content = [
        {"type": "text", "text": "Summarize this document"},
        {
            "type": "file",
            "file": {
                "file_data": "data:application/pdf;base64,base64data",
                "filename": "doc.pdf",
            },
        },
    ]

    job = Job(input=content)
    assert job.input == content


def test_generate_image_uses_image_generation_response(monkeypatch) -> None:
    class Image:
        def __init__(self) -> None:
            self.url = "https://example.com/generated.png"
            self.b64_json = None
            self.revised_prompt = "revised"

    class Response:
        def __init__(self) -> None:
            self.data = [Image()]

    def fake_image_generation(**_kwargs):
        return Response()

    def fake_completion(**_kwargs):
        raise AssertionError("completion fallback should not be called")

    monkeypatch.setattr("litellm.image_generation", fake_image_generation)
    monkeypatch.setattr("litellm.completion", fake_completion)

    result = generate_image.callable(prompt="cat")
    assert isinstance(result, ToolResult)
    assert isinstance(result.data, dict)
    assert result.data["url"] == "https://example.com/generated.png"
    assert result.data["revised_prompt"] == "revised"


def test_generate_image_falls_back_to_chat_completion(monkeypatch) -> None:
    calls: list[dict[str, object]] = []

    class EmptyResponse:
        def __init__(self) -> None:
            self.data: list[object] = []

    def fake_image_generation(**_kwargs):
        return EmptyResponse()

    def fake_completion(**kwargs):
        calls.append(kwargs)
        return {
            "choices": [
                {
                    "message": {
                        "images": [
                            {
                                "type": "image_url",
                                "index": 0,
                                "image_url": {"url": "data:image/png;base64,abc"},
                            }
                        ]
                    }
                }
            ]
        }

    monkeypatch.setattr("litellm.image_generation", fake_image_generation)
    monkeypatch.setattr("litellm.completion", fake_completion)

    result = generate_image.callable(prompt="cat")
    assert isinstance(result, ToolResult)
    assert isinstance(result.data, dict)
    assert result.data["url"] == "data:image/png;base64,abc"
    assert result.content is not None
    assert "[data-url omitted]" in result.content
    assert len(calls) == 1
    assert calls[0]["modalities"] == ["image", "text"]


@pytest.mark.asyncio
async def test_agenerate_image_falls_back_to_chat_completion(monkeypatch) -> None:
    calls: list[dict[str, object]] = []

    class EmptyResponse:
        def __init__(self) -> None:
            self.data: list[object] = []

    async def fake_aimage_generation(**_kwargs):
        return EmptyResponse()

    async def fake_acompletion(**kwargs):
        calls.append(kwargs)
        return {
            "choices": [
                {
                    "message": {
                        "images": [
                            {
                                "type": "image_url",
                                "index": 0,
                                "image_url": {"url": "data:image/png;base64,xyz"},
                            }
                        ]
                    }
                }
            ]
        }

    monkeypatch.setattr("litellm.aimage_generation", fake_aimage_generation)
    monkeypatch.setattr("litellm.acompletion", fake_acompletion)

    result = await agenerate_image.callable(prompt="cat")
    assert isinstance(result, ToolResult)
    assert isinstance(result.data, dict)
    assert result.data["url"] == "data:image/png;base64,xyz"
    assert result.content is not None
    assert "[data-url omitted]" in result.content
    assert len(calls) == 1
    assert calls[0]["modalities"] == ["image", "text"]
