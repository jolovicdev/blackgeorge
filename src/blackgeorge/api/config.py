from pydantic_settings import BaseSettings, SettingsConfigDict


class APIConfig(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="BLACKGEORGE_API_")

    title: str = "Blackgeorge API"
    version: str = "0.1.0"
    cors_origins: list[str] = ["*"]

    default_model: str = "openai/gpt-4"
    default_temperature: float | None = None
    max_tokens: int | None = None
    default_stream: bool = False

    storage_dir: str = ".blackgeorge"


_config: APIConfig | None = None


def get_config() -> APIConfig:
    global _config
    if _config is None:
        _config = APIConfig()
    return _config
