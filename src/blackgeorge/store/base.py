from dataclasses import dataclass
from datetime import datetime
from typing import Any

from blackgeorge.core.event import Event
from blackgeorge.core.types import RunStatus
from blackgeorge.store.state import RunState


@dataclass(frozen=True)
class RunRecord:
    run_id: str
    status: RunStatus
    input: Any
    output: str | None
    output_json: Any | None
    created_at: datetime
    updated_at: datetime
    state: RunState | None


class RunStore:
    def create_run(self, run_id: str, input_payload: Any) -> None:
        raise NotImplementedError

    def update_run(
        self,
        run_id: str,
        status: RunStatus,
        output: str | None,
        output_json: Any | None,
        state: RunState | None,
    ) -> None:
        raise NotImplementedError

    def get_run(self, run_id: str) -> RunRecord | None:
        raise NotImplementedError

    def add_event(self, event: Event) -> None:
        raise NotImplementedError

    def get_events(self, run_id: str) -> list[Event]:
        raise NotImplementedError
