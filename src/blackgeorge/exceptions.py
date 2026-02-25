class BlackgeorgeError(Exception):
    pass


class ContextLimitError(BlackgeorgeError):
    def __init__(
        self,
        message: str,
        model_registered: bool,
        summary_attempted: bool,
    ) -> None:
        self.model_registered = model_registered
        self.summary_attempted = summary_attempted
        super().__init__(message)


class ToolExecutionError(BlackgeorgeError):
    def __init__(
        self,
        tool_name: str,
        message: str,
        original_exception: Exception | None = None,
    ) -> None:
        self.tool_name = tool_name
        self.original_exception = original_exception
        super().__init__(f"Tool '{tool_name}' failed: {message}")


class ToolValidationError(BlackgeorgeError):
    def __init__(self, tool_name: str, message: str) -> None:
        self.tool_name = tool_name
        super().__init__(f"Tool '{tool_name}' validation failed: {message}")


class ToolTimeoutError(BlackgeorgeError):
    def __init__(self, tool_name: str, timeout_seconds: float) -> None:
        self.tool_name = tool_name
        self.timeout_seconds = timeout_seconds
        super().__init__(f"Tool '{tool_name}' timed out after {timeout_seconds}s")


class EventHandlerError(BlackgeorgeError):
    def __init__(self, event_type: str, handler_error: Exception) -> None:
        self.event_type = event_type
        self.handler_error = handler_error
        super().__init__(f"Event handler for '{event_type}' failed: {handler_error}")


class RunnerNotRegisteredError(BlackgeorgeError):
    def __init__(self, runner_type: str, runner_name: str) -> None:
        self.runner_type = runner_type
        self.runner_name = runner_name
        super().__init__(f"{runner_type} '{runner_name}' not registered")


class StreamingUnsupportedError(BlackgeorgeError):
    def __init__(self, model: str, provider: str | None = None) -> None:
        self.model = model
        self.provider = provider
        super().__init__(f"Streaming not supported for model '{model}'")
