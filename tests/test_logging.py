import json
import logging
from io import StringIO

from blackgeorge.logging import StructuredLogger, get_logger


def test_structured_logger_basic() -> None:
    log_stream = StringIO()
    handler = logging.StreamHandler(log_stream)

    logger = StructuredLogger("test", logging.INFO)
    logger.logger.handlers.clear()
    logger.logger.addHandler(handler)

    logger.info("Test message", key1="value1", key2=42)

    output = log_stream.getvalue()
    assert output

    log_data = json.loads(output.strip())
    assert log_data["level"] == "INFO"
    assert log_data["message"] == "Test message"
    assert log_data["key1"] == "value1"
    assert log_data["key2"] == 42
    assert "timestamp" in log_data


def test_structured_logger_with_context() -> None:
    log_stream = StringIO()
    handler = logging.StreamHandler(log_stream)

    base_logger = StructuredLogger("test", logging.INFO)
    base_logger.logger.handlers.clear()
    base_logger.logger.addHandler(handler)

    logger = base_logger.with_context(run_id="abc123", worker="TestWorker")
    logger.info("Test message", tool="read_file")

    output = log_stream.getvalue()
    log_data = json.loads(output.strip())

    assert log_data["run_id"] == "abc123"
    assert log_data["worker"] == "TestWorker"
    assert log_data["tool"] == "read_file"
    assert log_data["message"] == "Test message"


def test_get_logger() -> None:
    logger = get_logger("test_module")
    assert isinstance(logger, StructuredLogger)
    assert logger.logger.name == "test_module"
