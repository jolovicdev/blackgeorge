import os

import pytest

LIVE_MODEL = "deepseek/deepseek-chat"


def pytest_collection_modifyitems(items: list[pytest.Item]) -> None:
    if os.environ.get("DEEPSEEK_API_KEY"):
        return
    skip = pytest.mark.skip(reason="DEEPSEEK_API_KEY not set")
    for item in items:
        if "live" in item.path.parts:
            item.add_marker(skip)
