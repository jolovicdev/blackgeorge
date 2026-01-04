from typing import Any

import instructor


class InstructorClientPool:
    def __init__(self) -> None:
        self._sync: dict[str, Any] = {}
        self._async: dict[str, Any] = {}

    def get(self, model: str, async_client: bool) -> Any:
        if async_client:
            if model not in self._async:
                self._async[model] = instructor.from_provider(
                    f"litellm/{model}",
                    async_client=True,
                )
            return self._async[model]
        if model not in self._sync:
            self._sync[model] = instructor.from_provider(f"litellm/{model}")
        return self._sync[model]


instructor_clients = InstructorClientPool()
