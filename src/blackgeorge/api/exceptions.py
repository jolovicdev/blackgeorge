class APIException(Exception):
    def __init__(self, status_code: int, detail: str, errors: list[str] | None = None):
        self.status_code = status_code
        self.detail = detail
        self.errors = errors or []
        super().__init__(detail)


class WorkerNotFoundError(APIException):
    def __init__(self, worker_name: str):
        super().__init__(404, f"Worker '{worker_name}' not found")


class WorkforceNotFoundError(APIException):
    def __init__(self, workforce_name: str):
        super().__init__(404, f"Workforce '{workforce_name}' not found")


class RunNotFoundError(APIException):
    def __init__(self, run_id: str):
        super().__init__(404, f"Run '{run_id}' not found")


class InvalidResumeError(APIException):
    def __init__(self, reason: str):
        super().__init__(400, f"Cannot resume run: {reason}")
