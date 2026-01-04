from pydantic import BaseModel


class ChangeReport(BaseModel):
    summary: str
    files_changed: list[str]
    tests_run: list[str]
    risks: list[str]
    next_steps: list[str]
