from typing import Any

from pydantic import BaseModel, Field

from blackgeorge.core.types import WorkforceMode


class WorkerCreateRequest(BaseModel):
    name: str
    model: str | None = None
    instructions: str | None = None
    tools: list[str] = Field(default_factory=list)
    memory_scope: str | None = None


class WorkforceCreateRequest(BaseModel):
    name: str
    workers: list[str]
    mode: WorkforceMode = "managed"
    manager: str | None = None


class JobCreateRequest(BaseModel):
    input: dict[str, Any] | str
    expected_output: str | None = None
    response_schema: dict[str, Any] | None = None
    constraints: dict[str, Any] = Field(default_factory=dict)
    metadata: dict[str, Any] = Field(default_factory=dict)
    stream: bool | None = None


class ResumeRequest(BaseModel):
    decision: bool | None = None
    input: dict[str, Any] | str | None = None
