import asyncio
from collections.abc import Callable
from dataclasses import dataclass

from pydantic import BaseModel

from blackgeorge.async_utils import ensure_not_running_loop
from blackgeorge.core.job import Job
from blackgeorge.core.report import Report
from blackgeorge.desk import Desk
from blackgeorge.worker import Worker
from blackgeorge.workforce import Workforce


@dataclass(frozen=True)
class EvalCase:
    name: str
    job: Job
    contains: tuple[str, ...] = ()
    check: Callable[[Report], bool] | None = None


@dataclass(frozen=True)
class EvalResult:
    name: str
    passed: bool
    failures: tuple[str, ...]
    report: Report
    score: float | None = None


class JudgeScore(BaseModel):
    score: float
    reasoning: str


async def aevaluate(
    desk: Desk,
    runner: Worker | Workforce,
    cases: list[EvalCase],
    *,
    judge: Worker | None = None,
    rubric: str | None = None,
    min_score: float = 0.7,
) -> list[EvalResult]:
    if (judge is None) != (rubric is None):
        raise ValueError("judge and rubric must be provided together")
    results: list[EvalResult] = []
    for case in cases:
        report = await desk.arun(runner, case.job)
        score = None
        if judge is not None and rubric is not None and report.status == "completed":
            judge_report = await desk.arun(
                judge,
                Job(
                    input={
                        "rubric": rubric,
                        "task": case.job.input,
                        "answer": report.content,
                    },
                    expected_output="A score from 0.0 to 1.0 with reasoning",
                    response_schema=JudgeScore,
                ),
            )
            score = 0.0
            if judge_report.status == "completed" and judge_report.data is not None:
                score = float(judge_report.data.score)
        failures: list[str] = []
        if report.status != "completed":
            failures.append(f"status: expected completed, got {report.status}")
        content = report.content or ""
        for needle in case.contains:
            if needle not in content:
                failures.append(f"missing content substring: {needle!r}")
        if case.check is not None and not case.check(report):
            failures.append("check returned False")
        if score is not None and score < min_score:
            failures.append(f"score {score:.2f} below {min_score:.2f}")
        results.append(EvalResult(case.name, not failures, tuple(failures), report, score))
    return results


def evaluate(
    desk: Desk,
    runner: Worker | Workforce,
    cases: list[EvalCase],
    *,
    judge: Worker | None = None,
    rubric: str | None = None,
    min_score: float = 0.7,
) -> list[EvalResult]:
    ensure_not_running_loop("evaluate", "aevaluate")
    return asyncio.run(
        aevaluate(desk, runner, cases, judge=judge, rubric=rubric, min_score=min_score)
    )
