from typing import Any

from blackgeorge.core.event import Event
from blackgeorge.core.types import RunStatus
from blackgeorge.store.base import RunRecord, RunStore
from blackgeorge.store.state import RunState
from blackgeorge.utils import utc_now


class InMemoryRunStore(RunStore):
    def __init__(self) -> None:
        self._runs: dict[str, RunRecord] = {}
        self._events: dict[str, list[Event]] = {}

    def create_run(self, run_id: str, input_payload: Any) -> None:
        now = utc_now()
        self._runs[run_id] = RunRecord(
            run_id=run_id,
            status="running",
            input=input_payload,
            output=None,
            output_json=None,
            created_at=now,
            updated_at=now,
            state=None,
        )
        self._events[run_id] = []

    def update_run(
        self,
        run_id: str,
        status: RunStatus,
        output: str | None,
        output_json: Any | None,
        state: RunState | None,
    ) -> None:
        record = self._runs.get(run_id)
        now = utc_now()
        if record is None:
            self._runs[run_id] = RunRecord(
                run_id=run_id,
                status=status,
                input=None,
                output=output,
                output_json=output_json,
                created_at=now,
                updated_at=now,
                state=state,
            )
            return
        self._runs[run_id] = RunRecord(
            run_id=record.run_id,
            status=status,
            input=record.input,
            output=output,
            output_json=output_json,
            created_at=record.created_at,
            updated_at=now,
            state=state,
        )

    def get_run(self, run_id: str) -> RunRecord | None:
        return self._runs.get(run_id)

    def add_event(self, event: Event) -> None:
        self._events.setdefault(event.run_id, []).append(event)

    def get_events(self, run_id: str) -> list[Event]:
        return list(self._events.get(run_id, []))
