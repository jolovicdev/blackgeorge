# Logging

Blackgeorge provides a structured logger that outputs JSON-formatted logs with context support. This is useful for debugging, monitoring, and log aggregation systems.

## Getting a logger

Use the `get_logger` function to create a logger:

```python
from blackgeorge.logging import get_logger

logger = get_logger("my_app")
logger.info("Application started")
```

## Log levels

The logger supports standard Python logging levels:

```python
import logging

from blackgeorge.logging import get_logger

logger = get_logger("my_app", level=logging.DEBUG)

logger.debug("Detailed debug information")
logger.info("General information")
logger.warning("Warning message")
logger.error("Error occurred")
logger.critical("Critical failure")
```

## Structured output

All log messages are formatted as JSON:

```python
logger.info("User logged in", user_id="123", ip_address="192.168.1.1")
```

Output:
```json
{"timestamp": "2024-01-19T10:30:45.123456", "level": "INFO", "message": "User logged in", "user_id": "123", "ip_address": "192.168.1.1"}
```

## Context support

Create loggers with additional context that persists across all log calls:

```python
from blackgeorge.logging import get_logger

# Base logger with context
logger = get_logger("my_app").with_context(
    service="payment-processor",
    version="1.0.0",
    environment="production"
)

# Context is automatically included
logger.info("Processing payment", payment_id="abc123")
```

Output:
```json
{
  "timestamp": "2024-01-19T10:30:45.123456",
  "level": "INFO",
  "message": "Processing payment",
  "service": "payment-processor",
  "version": "1.0.0",
  "environment": "production",
  "payment_id": "abc123"
}
```

### Extending context

Create new loggers with additional context from an existing logger:

```python
base_logger = get_logger("my_app").with_context(service="api")

# Add request-specific context
request_logger = base_logger.with_context(
    request_id="req-456",
    user_id="user-789"
)

logger.info("Handling request")
```

## Using with Blackgeorge components

### Logging in tools

```python
from blackgeorge.tools import tool
from blackgeorge.logging import get_logger

logger = get_logger("tools").with_context(component="file_operations")

@tool()
def read_file(file_path: str) -> str:
    logger.info("Reading file", file_path=file_path)
    try:
        with open(file_path) as f:
            content = f.read()
        logger.info("File read successfully", file_path=file_path, size=len(content))
        return content
    except Exception as e:
        logger.error("Failed to read file", file_path=file_path, error=str(e))
        raise
```

### Logging in event handlers

```python
from blackgeorge import Desk
from blackgeorge.logging import get_logger

logger = get_logger("events").with_context(component="event_monitor")

def on_tool_completed(event):
    tool_name = event.payload.get("tool_name")
    logger.info("Tool completed", tool=tool_name, run_id=event.run_id)

desk = Desk(model="openai/gpt-5-nano")
desk.event_bus.subscribe("tool.completed", on_tool_completed)
```

### Logging in custom stores

```python
from blackgeorge.store import RunStore
from blackgeorge.logging import get_logger

logger = get_logger("stores").with_context(component="custom_store")

class CustomRunStore(RunStore):
    def create_run(self, run_id: str, input_payload: dict) -> None:
        logger.debug("Creating run", run_id=run_id)
        # Implementation
        logger.info("Run created", run_id=run_id)
```

## Log levels and filtering

Control log verbosity by setting the log level:

```python
import logging

from blackgeorge.logging import get_logger

# Only WARNING and above will be logged
logger = get_logger("my_app", level=logging.WARNING)

logger.debug("This won't be logged")
logger.info("Neither will this")
logger.warning("This will be logged")
logger.error("So will this")
```

## Configuring handlers

The `StructuredLogger` uses Python's standard logging module. Configure handlers for different output destinations:

```python
import logging
from blackgeorge.logging import get_logger

# Get the underlying logger
logger = get_logger("my_app")
underlying = logger.logger

# Add a file handler
file_handler = logging.FileHandler("app.log")
file_handler.setLevel(logging.DEBUG)
underlying.addHandler(file_handler)

# Add a stream handler with different level
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
underlying.addHandler(console_handler)
```

## Timestamps

All log entries include UTC timestamps in ISO 8601 format:

```python
logger.info("Event occurred")
# {"timestamp": "2024-01-19T10:30:45.123456+00:00", "level": "INFO", "message": "Event occurred"}
```

## Data types

The logger handles various data types in context:

```python
logger.info(
    "Complex data",
    integer=42,
    floating=3.14,
    boolean=True,
    none_value=None,
    list=[1, 2, 3],
    dict={"key": "value"},
    custom_object=str(CustomClass())
)
```

## Error logging

Log exceptions with context:

```python
try:
    risky_operation()
except Exception as e:
    logger.error("Operation failed", error=str(e), error_type=type(e).__name__)
    raise
```

## Performance considerations

1. **JSON serialization**: The logger uses `json.dumps` with `default=str` for non-serializable objects
2. **Handler selection**: Use appropriate log levels to avoid excessive logging
3. **Async operations**: For high-throughput scenarios, consider offloading log processing

## Integration with observability platforms

The JSON output format is compatible with most log aggregation systems:

### ELK Stack (Elasticsearch, Logstash, Kibana)

```python
logger.info("API request", method="GET", path="/api/users", status=200)
```

### Splunk

```python
logger.info("Transaction completed", transaction_id="abc123", amount=99.99)
```

### Datadog

```python
logger.info("Metric recorded", metric_name="request.duration", value=123.45, tags=["api:v1"])
```

## StructuredLogger API

### Creating a logger

```python
from blackgeorge.logging import StructuredLogger

logger = StructuredLogger("my_app", level=logging.INFO)
```

### Methods

| Method | Description |
|--------|-------------|
| `debug(message, **kwargs)` | Log debug message |
| `info(message, **kwargs)` | Log info message |
| `warning(message, **kwargs)` | Log warning message |
| `error(message, **kwargs)` | Log error message |
| `critical(message, **kwargs)` | Log critical message |
| `with_context(**kwargs)` | Return new logger with additional context |

### Logger properties

| Property | Description |
|----------|-------------|
| `logger` | The underlying Python logger |
| `context` | The current context dictionary |

## Examples

### Request logging

```python
from blackgeorge.logging import get_logger

logger = get_logger("api").with_context(service="user-service")

def handle_request(request):
    request_logger = logger.with_context(
        request_id=request.id,
        user_id=request.user_id
    )

    request_logger.info("Request started", path=request.path)
    try:
        response = process_request(request)
        request_logger.info("Request completed", status=response.status_code)
        return response
    except Exception as e:
        request_logger.error("Request failed", error=str(e))
        raise
```

### Tool execution tracking

```python
from blackgeorge.tools import tool
from blackgeorge.logging import get_logger

logger = get_logger("tools").with_context(component="database")

@tool()
def query_database(sql: str) -> list[dict]:
    logger.info("Executing query", sql=sql)
    try:
        results = execute_sql(sql)
        logger.info("Query completed", row_count=len(results))
        return results
    except Exception as e:
        logger.error("Query failed", sql=sql, error=str(e))
        raise
```

### Multi-component logging

```python
from blackgeorge.logging import get_logger

# Create base logger for the application
base_logger = get_logger("myapp").with_context(
    app="myapp",
    version="1.0.0"
)

# Create component-specific loggers
db_logger = base_logger.with_context(component="database")
api_logger = base_logger.with_context(component="api")
worker_logger = base_logger.with_context(component="worker")

# All logs include app and version context
db_logger.info("Connected to database")
api_logger.info("Received request")
worker_logger.info("Processing job")
```

## Best practices

1. **Use descriptive log levels**: Reserve ERROR for failures, WARNING for potential issues
2. **Include relevant context**: Add structured data that helps with debugging
3. **Use consistent field names**: Use `user_id` not `userId` or `userIdentifier`
4. **Avoid sensitive data**: Don't log passwords, tokens, or personal information
5. **Log at appropriate granularity**: Debug logs should be disableable in production
6. **Use context wisely**: Share context across related log calls
7. **Handle serialization**: Use `str()` for complex objects that can't be JSON-serialized
