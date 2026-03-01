from typing import Any

from blackgeorge.config import RunConfig
from blackgeorge.core.job import Job
from blackgeorge.core.report import Report
from blackgeorge.store.state import RunState
from blackgeorge.tools.base import Tool
from blackgeorge.tools.registry import Toolbelt
from blackgeorge.worker_runner import WorkerRunner


class Worker:
    def __init__(
        self,
        name: str,
        tools: list[Tool] | None = None,
        model: str | None = None,
        instructions: str | None = None,
        memory_scope: str | None = None,
    ) -> None:
        self.name = name
        self.model = model
        self.instructions = instructions
        self.toolbelt = Toolbelt(tools)
        self.memory_scope = memory_scope or f"worker:{name}"

    def tools(self) -> list[Tool]:
        return self.toolbelt.list()

    def _runner(self) -> WorkerRunner:
        return WorkerRunner(self.name, self.toolbelt, self.instructions)

    def run(self, config: RunConfig, job: Job) -> tuple[Report, RunState | None]:
        return self._runner().run(config, job, self.model)

    async def arun(self, config: RunConfig, job: Job) -> tuple[Report, RunState | None]:
        return await self._runner().arun(config, job, self.model)

    def resume(
        self, config: RunConfig, state: RunState, decision_or_input: Any
    ) -> tuple[Report, RunState | None]:
        return self._runner().resume(config, state, decision_or_input, self.model)

    async def aresume(
        self, config: RunConfig, state: RunState, decision_or_input: Any
    ) -> tuple[Report, RunState | None]:
        return await self._runner().aresume(config, state, decision_or_input, self.model)
