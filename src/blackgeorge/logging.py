import json
import logging
from datetime import UTC, datetime
from typing import Any


class StructuredLogger:
    def __init__(self, name: str, level: int = logging.INFO) -> None:
        self.logger = logging.getLogger(name)
        self.logger.setLevel(level)
        self.context: dict[str, Any] = {}

        if not self.logger.handlers:
            handler = logging.StreamHandler()
            handler.setLevel(level)
            self.logger.addHandler(handler)

    def with_context(self, **kwargs: Any) -> "StructuredLogger":
        new_logger = StructuredLogger(self.logger.name, self.logger.level)
        new_logger.context = {**self.context, **kwargs}
        new_logger.logger = self.logger
        return new_logger

    def _format(self, level: str, message: str, **kwargs: Any) -> str:
        record = {
            "timestamp": datetime.now(UTC).isoformat(),
            "level": level,
            "message": message,
            **self.context,
            **kwargs,
        }
        return json.dumps(record, default=str)

    def debug(self, message: str, **kwargs: Any) -> None:
        self.logger.debug(self._format("DEBUG", message, **kwargs))

    def info(self, message: str, **kwargs: Any) -> None:
        self.logger.info(self._format("INFO", message, **kwargs))

    def warning(self, message: str, **kwargs: Any) -> None:
        self.logger.warning(self._format("WARNING", message, **kwargs))

    def error(self, message: str, **kwargs: Any) -> None:
        self.logger.error(self._format("ERROR", message, **kwargs))

    def critical(self, message: str, **kwargs: Any) -> None:
        self.logger.critical(self._format("CRITICAL", message, **kwargs))


def get_logger(name: str, level: int = logging.INFO) -> StructuredLogger:
    return StructuredLogger(name, level)
