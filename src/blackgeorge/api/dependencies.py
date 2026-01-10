from collections.abc import AsyncGenerator

from blackgeorge.api.config import APIConfig
from blackgeorge.desk import Desk

_config: APIConfig | None = None
_desk: Desk | None = None


def get_config() -> APIConfig:
    global _config
    if _config is None:
        _config = APIConfig()
    return _config


async def get_desk() -> AsyncGenerator[Desk, None]:
    global _desk
    if _desk is None:
        config = get_config()
        _desk = Desk(
            model=config.default_model,
            storage_dir=config.storage_dir,
            temperature=config.default_temperature,
            max_tokens=config.max_tokens,
            stream=config.default_stream,
        )
    yield _desk
