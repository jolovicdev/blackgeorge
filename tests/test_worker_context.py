from blackgeorge.core.message import Message
from blackgeorge.worker_context import message_summary_text


def test_message_summary_text_includes_input_text_blocks() -> None:
    message = Message(
        role="user",
        content=[
            {"type": "input_text", "text": "keep this instruction"},
            {"type": "input_image", "image_url": {"url": "https://example.com/image.png"}},
        ],
    )

    summary = message_summary_text(message)
    assert summary == "user: keep this instruction"


def test_message_summary_text_keeps_multimodal_marker_without_text_blocks() -> None:
    message = Message(
        role="user",
        content=[
            {"type": "input_image", "image_url": {"url": "https://example.com/image.png"}},
        ],
    )

    summary = message_summary_text(message)
    assert summary == "user: [multimodal message]"
